import torch
import torch.nn as nn
from torch.nn import functional as F

from .layers import GPT2Block
from .settings import GPTConfig


class GPT2(nn.Module):
    def __init__(self, config: GPTConfig):
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(50257, config.n_embedding_size),
                wpe=nn.Embedding(50257, config.n_embedding_size),
                h=nn.ModuleList(GPT2Block(config) for _ in range(config.n_layers)),
                ln_f=nn.LayerNorm(config.n_embedding_size),
            )
        )

        self.lm_head = nn.Linear(config.n_embedding_size, config.vocab_size, bias=False)
