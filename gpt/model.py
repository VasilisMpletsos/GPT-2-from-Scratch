import time

import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from .layers import GPT2Block
from .settings import GPTConfig


class GPT2(nn.Module):
    def __init__(self, config: GPTConfig = GPTConfig()):
        super().__init__()

        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(50257, config.n_embedding_size),
                wpe=nn.Embedding(1024, config.n_embedding_size),
                h=nn.ModuleList(GPT2Block(config) for _ in range(config.n_layers)),
                ln_f=nn.LayerNorm(config.n_embedding_size),
            )
        )

        self.lm_head = nn.Linear(config.n_embedding_size, config.vocab_size, bias=False)

        # This shares the same weights for lm_head and wte as used in the original gpt-2 implementation but i dont want to use it
        # self.transformer.wte.weight = self.lm_head.weight

        # Customly initialize layer parameters
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std = (2 * self.config.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        B, T = x.size()
        assert T <= self.config.context_window, (
            f"The context window must be less than {self.config.context_window + 1}"
        )

        positional_number = torch.arange(0, T, dtype=torch.long, device=x.device)
        pos_embed = self.transformer.wpe(positional_number)
        tok_embed = self.transformer.wte(x)
        tokens = pos_embed + tok_embed

        for block in self.transformer.h:
            tokens = block(tokens)

        tokens = self.transformer.ln_f(tokens)
        logits = self.lm_head(tokens)
        return logits

    @classmethod
    def load_weights_from_pretrained(cls, hf_model_name: str):
        assert hf_model_name in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        hf_model = AutoModelForCausalLM.from_pretrained(
            f"openai-community/{hf_model_name}"
        )

        models_configs = {
            "gpt2": dict(n_layers=12, n_heads=12, n_embedding_size=768),
            "gpt2-medium": dict(n_layers=24, n_heads=16, n_embedding_size=1024),
            "gpt2-large": dict(n_layers=36, n_heads=20, n_embedding_size=1280),
            "gpt2-xl": dict(n_layers=48, n_heads=25, n_embedding_size=1600),
        }

        config_args = models_configs[hf_model_name]
        config_args["vocab_size"] = 50257
        config_args["context_window"] = 1024

        config = GPTConfig(**config_args)
        model = GPT2(config)

        state_dict = model.state_dict()
        state_dict_keys = state_dict.keys()
        state_dict_hf = hf_model.state_dict()
        state_dict_hf_keys = state_dict_hf.keys()

        state_dict_keys = [
            key for key in state_dict_keys if not key.endswith(".attn.bias")
        ]
        state_dict_hf_keys = [
            key for key in state_dict_hf_keys if not key.endswith(".attn.bias")
        ]
        state_dict_hf_keys = [
            key for key in state_dict_hf_keys if not key.endswith(".attn.masked_bias")
        ]

        transposed_key_weights = [
            ".attn.c_attn.weight",
            ".attn.c_proj.weight",
            ".mlp.c_fc.weight",
            ".mlp.c_proj.weight",
        ]

        assert len(state_dict_hf_keys) == len(state_dict_keys)

        pbar = tqdm(state_dict_hf_keys, desc="Copying weights")
        for key in pbar:
            pbar.set_description(f"Processing {key=}")
            # time.sleep(1)
            if any(key.endswith(keyT) for keyT in transposed_key_weights):
                with torch.no_grad():
                    state_dict[key].copy_(state_dict_hf[key].T)
            else:
                with torch.no_grad():
                    state_dict[key].copy_(state_dict_hf[key])

        return model
