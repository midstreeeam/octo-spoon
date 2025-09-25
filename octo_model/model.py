import math
from typing import Optional

import torch
import torch.nn as nn

from .config import OctoConfig


def _make_causal_mask(batch_size: int, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)


def _expand_attention_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    # attention_mask is expected to be 1 for tokens and 0 for padding
    inverted = (1.0 - attention_mask.float()) * -1e4
    return inverted[:, None, None, :].to(dtype=dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("Rotary embedding requires an even head dimension.")
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            seq_len = x.shape[-2]
        if seq_len > self.max_position_embeddings:
            seq_positions = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        else:
            seq_positions = torch.arange(self.max_position_embeddings, device=x.device, dtype=self.inv_freq.dtype)[:seq_len]

        freqs = torch.einsum("i,j->ij", seq_positions, self.inv_freq)
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
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


class OctoAttention(nn.Module):
    def __init__(self, config: OctoConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("`hidden_size` must be divisible by `num_attention_heads`.")
        self.head_dim = self.hidden_size // self.num_heads
        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError("`num_attention_heads` must be divisible by `num_key_value_heads`.")
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=getattr(config, "rope_theta", 10000.0),
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.size()

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(query, seq_len)
        query, key = _apply_rotary(query, key, cos, sin)

        key = _repeat_kv(key, self.num_key_value_groups)
        value = _repeat_kv(value, self.num_key_value_groups)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask

        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(value.dtype)
        attn_output = torch.matmul(probs, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.out_proj(attn_output)


class OctoMLP(nn.Module):
    def __init__(self, config: OctoConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(hidden_states)))


class OctoDecoderLayer(nn.Module):
    def __init__(self, config: OctoConfig):
        super().__init__()
        self.attn = OctoAttention(config)
        self.mlp = OctoMLP(config)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_input = self.norm1(hidden_states)
        attn_output = self.attn(attn_input, attention_mask=attention_mask)
        hidden_states = hidden_states + attn_output

        mlp_input = self.norm2(hidden_states)
        mlp_output = self.mlp(mlp_input)
        return hidden_states + mlp_output


class OctoModel(nn.Module):
    def __init__(self, config: OctoConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([OctoDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Pass exactly one of `input_ids` or `inputs_embeds`.")

        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds

        batch_size, seq_len, _ = hidden_states.size()
        causal_mask = _make_causal_mask(batch_size, seq_len, hidden_states.device, hidden_states.dtype)
        if attention_mask is not None:
            expanded = _expand_attention_mask(attention_mask, hidden_states.dtype)
            causal_mask = causal_mask + expanded

        for layer in self.layers:
            if self.training and hasattr(self, '_use_gradient_checkpointing') and self._use_gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, causal_mask, use_reentrant=False
                )
            else:
                hidden_states = layer(hidden_states, attention_mask=causal_mask)

        return self.norm(hidden_states)


class OctoForCausalLM(nn.Module):
    def __init__(self, config: OctoConfig):
        super().__init__()
        self.config = config
        self.model = OctoModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return {"logits": logits, "loss": loss}
