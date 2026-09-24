"""Minimal pre-LN transformer blocks built on ``scaled_dot_product_attention``.

Masks are boolean with True meaning "may attend", broadcastable to
``[batch, heads, queries, keys]``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def rope(x, pos):
    """Rotary position embedding for ``x`` of shape [B, heads, L, dh]."""
    half = x.shape[-1] // 2
    freq = 1.0 / (10000 ** (torch.arange(half, device=x.device, dtype=torch.float) / half))
    ang = pos[:, None].float() * freq[None]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.o = nn.Linear(d, d)

    def project_kv(self, mem, pos=None):
        B, L, _ = mem.shape
        k, v = self.kv(mem).view(B, L, 2, self.h, self.dh).permute(2, 0, 3, 1, 4)
        return (k if pos is None else rope(k, pos)), v

    def forward(self, x, mem=None, mask=None, kv=None, past=None, pos=None):
        """``pos`` (self-attention only) applies rotary embeddings to q and k."""
        B, L, D = x.shape
        q = self.q(x).view(B, L, self.h, self.dh).transpose(1, 2)
        if pos is not None:
            q = rope(q, pos)
        k, v = kv if kv is not None else self.project_kv(x if mem is None else mem, pos)
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(out.transpose(1, 2).reshape(B, L, D))


class MLP(nn.Module):
    def __init__(self, d, ff):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    """Bidirectional self-attention + MLP."""

    def __init__(self, d, heads, ff):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn, self.mlp = Attention(d, heads), MLP(d, ff)

    def forward(self, x, mask, pos=None):
        x = x + self.attn(self.n1(x), mask=mask, pos=pos)
        return x + self.mlp(self.n2(x))


class WorkspaceBlock(nn.Module):
    """Self-attention inside one question's workspace, cross-attention into the
    shared context memory, then an MLP."""

    def __init__(self, d, heads, ff):
        super().__init__()
        self.n1, self.n2, self.n3, self.nm = (nn.LayerNorm(d) for _ in range(4))
        self.self_attn, self.cross_attn = Attention(d, heads), Attention(d, heads)
        self.mlp = MLP(d, ff)

    def forward(self, w, mem, self_mask, cross_mask):
        w = w + self.self_attn(self.n1(w), mask=self_mask)
        w = w + self.cross_attn(self.n2(w), mem=self.nm(mem), mask=cross_mask)
        return w + self.mlp(self.n3(w))


class DecoderBlock(nn.Module):
    """Causal self-attention + MLP, with optional prefix key/value cache."""

    def __init__(self, d, heads, ff):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn, self.mlp = Attention(d, heads), MLP(d, ff)

    def forward(self, x, mask, past=None, return_kv=False, pos=None):
        h = self.n1(x)
        kv = self.attn.project_kv(h, pos)
        x = x + self.attn(h, mask=mask, kv=kv, past=past, pos=pos)
        x = x + self.mlp(self.n2(x))
        return (x, kv) if return_kv else x
