from typing import Optional

from transformers import PretrainedConfig


class OctodiffConfig(PretrainedConfig):
    """Configuration for the Octodiff diffusion LM.

    The schedule terms are easy to mix up, so the distinctions are documented here:

    * ``diffusion_steps`` – how many discrete noise levels we sample during training
      when perturbing the targets. Increasing this gives the denoiser more variety
      to learn from, but also raises training cost.
    * ``denoise_steps`` – number of denoising iterations to run by default at
      inference time when a caller does not supply ``num_steps``.
    """

    model_type = "octodiff"

    def __init__(
        self,
        vocab_size: int = 50257, #gpt2 vocab size
        hidden_size: int = 1024,
        intermediate_size: int = 4096,
        num_hidden_layers: int = 16,
        num_attention_heads: int = 16,
        num_key_value_heads: Optional[int] = None,
        max_position_embeddings: int = 4096,
        rope_theta: float = 10000.0,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        diffusion_steps: int = 16,
        sigma_min: float = 1e-3,
        sigma_max: float = 1.0,
        denoise_steps: int = 32,
        guidance_scale: float = 1.0,
        tie_word_embeddings: bool = False,
        pad_token_id: Optional[int] = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.diffusion_steps = max(1, diffusion_steps)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(max(sigma_max, sigma_min + 1e-6))
        self.denoise_steps = max(1, denoise_steps)
        self.guidance_scale = float(guidance_scale)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        if num_key_value_heads is None:
            num_key_value_heads = max(1, num_attention_heads // 4)
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads for GQA")
        self.num_key_value_heads = num_key_value_heads
