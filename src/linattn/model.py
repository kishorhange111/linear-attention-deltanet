"""A small LM whose token mixer is one of: softmax attention, linear attention, gated linear
attention, DeltaNet. Everything else (MLP, norms, widths) is identical.

Mixer details, following the papers' stable choices:
- linear attention: feature map φ = elu + 1 on q and k (Katharopoulos et al. 2020);
- gated: a per-head scalar decay α_t = σ(w·x_t)^(1/τ) with τ = 16, so α starts near 1
  (Gated Linear Attention, Yang et al. 2023);
- DeltaNet: q and k L2-normalised (needed so I − β k kᵀ is a contraction), and a learning
  rate β_t = σ(w·x_t) ∈ (0, 1) per head and token.
Each recurrent mixer's output goes through a per-head RMSNorm before the output projection.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .ops import delta_chunked, linear_chunked


@dataclass
class Config:
    vocab: int = 256
    d: int = 128
    heads: int = 2
    layers: int = 2
    ctx: int = 256
    mixer: str = "deltanet"  # softmax | linear | gated | deltanet
    chunk: int = 16


class Mixer(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c, self.h, self.hd = c, c.heads, c.d // c.heads
        self.qkv = nn.Linear(c.d, 3 * c.d, bias=False)
        self.o = nn.Linear(c.d, c.d, bias=False)
        if c.mixer in ("gated", "deltanet"):
            self.gate = nn.Linear(c.d, c.heads)
        self.norm = nn.RMSNorm(self.hd) if c.mixer != "softmax" else None

    def forward(self, x):
        b, L, d = x.shape
        q, k, v = self.qkv(x).view(b, L, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        m = self.c.mixer
        if m == "softmax":
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            pad = (-L) % self.c.chunk
            g = self.gate(x).transpose(1, 2) if m != "linear" else None  # (b, h, L)
            if m == "linear":
                q, k = F.elu(q) + 1, F.elu(k) + 1
            elif m == "deltanet":
                q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
            if pad:
                q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
                g = F.pad(g, (0, pad)) if g is not None else None
            q, k, v = q.float(), k.float(), v.float()
            if m == "deltanet":
                y, _ = delta_chunked(q, k, v, torch.sigmoid(g.float()), self.c.chunk)
            else:
                la = F.logsigmoid(g.float()) / 16 if g is not None else None
                y, _ = linear_chunked(q, k, v, la, self.c.chunk)
            y = self.norm(y[:, :, :L].to(x.dtype))
        return self.o(y.transpose(1, 2).reshape(b, L, d))


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(c.d), nn.RMSNorm(c.d)
        self.mix = Mixer(c)
        self.mlp = nn.Sequential(
            nn.Linear(c.d, 4 * c.d, bias=False), nn.GELU(), nn.Linear(4 * c.d, c.d, bias=False)
        )

    def forward(self, x):
        x = x + self.mix(self.n1(x))
        return x + self.mlp(self.n2(x))


class LM(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.vocab, c.d)
        self.pos = nn.Embedding(c.ctx, c.d) if c.mixer == "softmax" else None
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        self.norm = nn.RMSNorm(c.d)
        self.head = nn.Linear(c.d, c.vocab, bias=False)
        for mod in self.modules():
            if isinstance(mod, (nn.Linear, nn.Embedding)):
                nn.init.normal_(mod.weight, std=0.02)

    def forward(self, idx):
        x = self.emb(idx)
        if self.pos is not None:
            x = x + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))
