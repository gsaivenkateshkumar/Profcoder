"""A small, randomly initialized causal Transformer for CPU experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from model_lab.tokenizer import VOCAB_SIZE


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = VOCAB_SIZE
    context_length: int = 256
    width: int = 192
    heads: int = 4
    layers: int = 4
    feed_forward_width: int = 768

    def __post_init__(self) -> None:
        if (
            self.vocab_size != VOCAB_SIZE
            or self.context_length < 2
            or self.width < 1
            or self.heads < 1
            or self.width % self.heads != 0
            or self.layers < 1
            or self.feed_forward_width < 1
        ):
            raise ValueError("invalid CPU model configuration")

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


class CausalAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = config.heads
        self.head_width = config.width // config.heads
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.projection = nn.Linear(config.width, config.width)
        mask = torch.tril(
            torch.ones(config.context_length, config.context_length, dtype=torch.bool)
        )
        self.register_buffer(
            "causal_mask", mask.view(1, 1, config.context_length, config.context_length),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(x).split(width, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch, length, self.heads, self.head_width).transpose(1, 2)

        q, k, v = map(split_heads, (q, k, v))
        scores = (q @ k.transpose(-2, -1)) * (self.head_width ** -0.5)
        scores = scores.masked_fill(
            ~self.causal_mask[:, :, :length, :length], torch.finfo(scores.dtype).min
        )
        weights = F.softmax(scores, dim=-1)
        attended = weights @ v
        return self.projection(attended.transpose(1, 2).contiguous().view(batch, length, width))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.width)
        self.attention = CausalAttention(config)
        self.feed_forward_norm = nn.LayerNorm(config.width)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.width, config.feed_forward_width),
            nn.GELU(),
            nn.Linear(config.feed_forward_width, config.width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.feed_forward(self.feed_forward_norm(x))


class ByteTransformer(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.width)
        self.position_embedding = nn.Embedding(config.context_length, config.width)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.layers))
        self.final_norm = nn.LayerNorm(config.width)
        self.output = nn.Linear(config.width, config.vocab_size, bias=False)
        self.apply(self._initialize)
        self.output.weight = self.token_embedding.weight

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None):
        if tokens.ndim != 2 or not 0 < tokens.size(1) <= self.config.context_length:
            raise ValueError("input must have shape (batch, 1..context_length)")
        if tokens.device.type != "cpu":
            raise ValueError("this experiment runs on CPU only")
        if targets is not None and targets.shape != tokens.shape:
            raise ValueError("targets must match input shape")
        positions = torch.arange(tokens.size(1), device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        logits = self.output(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.config.vocab_size), targets.reshape(-1))
        return logits, loss
