from dataclasses import dataclass


@dataclass
class GPTConfig:
    context_window: int = 1024
    vocab_size: int = 50257
    n_layers: int = 12
    n_heads: int = 12
    n_embedding_size: int = 768
