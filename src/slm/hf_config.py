"""Hugging Face configuration for the native SLM-Tavenel architecture."""

from transformers import PretrainedConfig


class SlmTavenelConfig(PretrainedConfig):
    model_type = "slm_tavenel"

    def __init__(
        self,
        vocab_size: int = 8192,
        d_model: int = 96,
        n_layers: int = 4,
        n_heads: int = 3,
        n_kv_heads: int = 1,
        ffn_hidden: int = 256,
        max_seq_len: int = 1024,
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-6,
        tie_embeddings: bool = True,
        qk_norm: bool = True,
        attn_gate: str = "headwise",
        window_pattern: list[int] | None = None,
        init_std: float = 0.02,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.ffn_hidden = ffn_hidden
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.norm_eps = norm_eps
        self.tie_embeddings = tie_embeddings
        self.qk_norm = qk_norm
        self.attn_gate = attn_gate
        self.window_pattern = list(window_pattern or [])
        self.init_std = init_std
        kwargs.setdefault("is_decoder", True)
        kwargs.setdefault("is_encoder_decoder", False)
        kwargs.setdefault("tie_word_embeddings", tie_embeddings)
        kwargs.setdefault("use_cache", False)
        super().__init__(**kwargs)
        if d_model % n_heads or n_heads % n_kv_heads or (d_model // n_heads) % 2:
            raise ValueError("Invalid head dimensions in SLM-Tavenel config")
        if attn_gate not in ("none", "headwise", "elementwise"):
            raise ValueError(f"Unsupported attention gate: {attn_gate}")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads
