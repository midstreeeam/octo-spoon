from typing import Optional
from transformers import PretrainedConfig


class OctoConfig(PretrainedConfig):
    model_type = "octo"

    def __init__(
        self,
        vocab_size: int = 16000,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_key_value_heads: Optional[int] = None,
        max_position_embeddings: int = 4096,
        rms_norm_eps: float = 1e-6,
        pad_token_id: Optional[int] = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        attention_bias: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias

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
            raise ValueError("`num_attention_heads` must be divisible by `num_key_value_heads` for grouped attention.")
        self.num_key_value_heads = num_key_value_heads
