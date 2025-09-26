import math
from typing import Optional

import torch
import torch.nn as nn

from .config import OctodiffConfig


def _make_causal_mask(batch_size: int, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)


def _expand_attention_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    inverted = (1.0 - attention_mask.float()) * -1e4
    return inverted[:, None, None, :].to(dtype=dtype)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("Rotary embedding requires an even head dimension")
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            seq_len = x.shape[-2]
        if seq_len > self.max_position_embeddings:
            positions = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        else:
            positions = torch.arange(self.max_position_embeddings, device=x.device, dtype=self.inv_freq.dtype)[:seq_len]
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(dtype=x.dtype)
        sin = emb.sin().to(dtype=x.dtype)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    query = (query * cos) + (_rotate_half(query) * sin)
    key = (key * cos) + (_rotate_half(key) * sin)
    return query, key


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    bsz, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, num_kv_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(bsz, num_kv_heads * n_rep, seq_len, head_dim)


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    device = timesteps.device
    dtype = timesteps.dtype
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=device, dtype=dtype) / half)
    args = timesteps[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


class TimeEmbedding(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        emb = timestep_embedding(timesteps, self.hidden_size)
        return self.proj(emb)


class LogLinearDiffusionSchedule:
    def __init__(self, sigma_min: float, sigma_max: float):
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self._log_sigma_min = math.log(self.sigma_min)
        self._log_sigma_max = math.log(self.sigma_max)

    def sigma(self, timesteps: torch.Tensor) -> torch.Tensor:
        return torch.exp(
            timesteps * (self._log_sigma_max - self._log_sigma_min) + self._log_sigma_min
        )


class OctodiffAttention(nn.Module):
    def __init__(self, config: OctodiffConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.head_dim = self.hidden_size // self.num_heads
        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.size()

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value = value.view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(query, seq_len)
        query, key = _apply_rotary(query, key, cos, sin)

        key = _repeat_kv(key, self.num_key_value_groups)
        value = _repeat_kv(value, self.num_key_value_groups)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask

        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(value.dtype)
        attn_output = torch.matmul(probs, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        return self.out_proj(attn_output)


class OctodiffMLP(nn.Module):
    def __init__(self, config: OctodiffConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(hidden_states)))


class OctodiffDecoderLayer(nn.Module):
    def __init__(self, config: OctodiffConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = OctodiffAttention(config)
        self.mlp = OctodiffMLP(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_input = self.norm1(hidden_states)
        attn_output = self.attn(attn_input, attention_mask=attention_mask)
        hidden_states = hidden_states + attn_output

        mlp_input = self.norm2(hidden_states)
        mlp_output = self.mlp(mlp_input)
        return hidden_states + mlp_output


class OctodiffModel(nn.Module):
    def __init__(self, config: OctodiffConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([OctodiffDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed(self, input_ids: Optional[torch.LongTensor] = None, inputs_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Pass exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            return self.embed_tokens(input_ids)
        return inputs_embeds

    def forward_internal(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        time_emb: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.size()
        causal_mask = _make_causal_mask(batch_size, seq_len, hidden_states.device, hidden_states.dtype)
        if attention_mask is not None:
            expanded = _expand_attention_mask(attention_mask, hidden_states.dtype)
            causal_mask = causal_mask + expanded

        hidden_states = hidden_states + time_emb[:, None, :]

        for layer in self.layers:
            if self.training and hasattr(self, "_use_gradient_checkpointing") and self._use_gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, causal_mask, use_reentrant=False
                )
            else:
                hidden_states = layer(hidden_states, attention_mask=causal_mask)

        return self.norm(hidden_states)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        time_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if time_emb is None:
            raise ValueError("time_emb must be provided for diffusion forward")
        hidden_states = self.embed(input_ids=input_ids, inputs_embeds=inputs_embeds)
        return self.forward_internal(hidden_states, attention_mask, time_emb)


class OctodiffForDiffusionLM(nn.Module):
    def __init__(self, config: OctodiffConfig):
        super().__init__()
        self.config = config
        self.model = OctodiffModel(config)
        self.time_embed = TimeEmbedding(config.hidden_size)
        self.schedule = LogLinearDiffusionSchedule(config.sigma_min, config.sigma_max)
        self.noise_head = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def _broadcast_sigma(self, sigma: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        return sigma.view(-1, 1, 1).to(hidden_states.dtype)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        base_hidden = self.model.embed(input_ids=input_ids, inputs_embeds=inputs_embeds)
        batch_size = base_hidden.size(0)
        device = base_hidden.device

        if timesteps is None:
            timesteps = torch.rand(batch_size, device=device, dtype=base_hidden.dtype)
        if noise is None:
            noise = torch.randn_like(base_hidden)

        sigma = self.schedule.sigma(timesteps)
        sigma_broadcast = self._broadcast_sigma(sigma, base_hidden)
        noisy_hidden = base_hidden + sigma_broadcast * noise

        time_emb = self.time_embed(timesteps)
        hidden_states = self.model.forward_internal(noisy_hidden, attention_mask, time_emb)
        pred_noise = self.noise_head(hidden_states)
        denoised = noisy_hidden - sigma_broadcast * pred_noise
        logits = self.lm_head(denoised)
        loss = torch.mean((pred_noise - noise) ** 2)

        if not return_dict:
            return logits, loss

        return {
            "logits": logits,
            "loss": loss,
            "predicted_noise": pred_noise,
            "denoised": denoised,
            "sigma": sigma,
        }

    @torch.no_grad()
    def sample(
        self,
        prompt_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        guidance_scale: Optional[float] = None,
    ) -> torch.LongTensor:
        self.eval()
        device = prompt_ids.device
        if attention_mask is None:
            attention_mask = torch.ones_like(prompt_ids, device=device)
        if num_steps is None:
            num_steps = self.config.denoise_steps
        guidance = guidance_scale if guidance_scale is not None else self.config.guidance_scale

        generated = prompt_ids
        attn_mask = attention_mask

        for _ in range(max_new_tokens):
            pad_id = self.config.pad_token_id if self.config.pad_token_id is not None else self.config.eos_token_id
            next_placeholder = torch.full(
                (generated.size(0), 1), pad_id, device=device, dtype=generated.dtype
            )
            generated = torch.cat([generated, next_placeholder], dim=1)
            attn_mask = torch.cat([attn_mask, torch.ones_like(next_placeholder, dtype=attn_mask.dtype)], dim=1)

            base_hidden = self.model.embed(generated)
            latents = base_hidden.clone()
            latents[:, -1, :] = torch.randn_like(latents[:, -1, :]) * self.schedule.sigma_max * temperature

            diffusion_range = torch.linspace(1.0, 0.0, steps=num_steps, device=device, dtype=base_hidden.dtype)

            for t in diffusion_range:
                timestep = torch.full((generated.size(0),), t, device=device, dtype=base_hidden.dtype)
                sigma = self.schedule.sigma(timestep)
                sigma_broadcast = self._broadcast_sigma(sigma, latents)
                time_emb = self.time_embed(timestep)

                hidden_states = self.model.forward_internal(latents, attn_mask, time_emb)
                pred_noise = self.noise_head(hidden_states)

                if guidance != 1.0:
                    clean_hidden = self.model.forward_internal(base_hidden, attn_mask, time_emb)
                    clean_pred = self.noise_head(clean_hidden)
                    pred_noise = pred_noise + guidance * (pred_noise - clean_pred)

                latents = latents - sigma_broadcast * pred_noise
                latents[:, :-1, :] = base_hidden[:, :-1, :]

            timestep_zero = torch.zeros(generated.size(0), device=device, dtype=base_hidden.dtype)
            time_emb_zero = self.time_embed(timestep_zero)
            final_hidden = self.model.forward_internal(latents, attn_mask, time_emb_zero)
            logits = self.lm_head(final_hidden[:, -1, :])
            logits = logits / max(temperature, 1e-5)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated[:, -1] = next_token.squeeze(-1)

            if (generated[:, -1] == self.config.eos_token_id).all():
                break

        return generated
