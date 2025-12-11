import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from .settings import GPTConfig


class GPT2Attention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embedding_size % config.n_heads == 0
        self.c_attn = nn.Linear(config.n_embedding_size, 3 * config.n_embedding_size)
        self.c_proj = nn.Linear(config.n_embedding_size, config.n_embedding_size)
        self.c_proj.NANOGPT_SCALE_INIT = 1.0
        # self.attn_dropout = nn.Dropout(0.1)
        # self.resid_dropout = nn.Dropout(0.1)
        self.n_head = config.n_heads
        self.n_embd = config.n_heads

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.context_window, config.context_window)).view(
                1, 1, config.context_window, config.context_window
            ),
        )
        self.config = config

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.config.n_embedding_size, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # attention = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # attention = attention.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # attention = F.softmax(attention, dim=-1)
        # attention = self.attn_dropout(attention)
        # y = attention @ v
        # USING FLASH ATTENTION
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class GPT2MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embedding_size, config.n_embedding_size * 4)
        self.c_proj = nn.Linear(4 * config.n_embedding_size, config.n_embedding_size)
        self.c_proj.NANOGPT_SCALE_INIT = 1.0
        self.act = nn.GELU(approximate="tanh")
        # self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        # x = self.dropout(x)
        x = self.c_proj(x)
        return x


class GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embedding_size)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embedding_size)
        self.mlp = GPT2MLP(config)

    def forward(self, x):
        # x = self.ln_1(x)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
