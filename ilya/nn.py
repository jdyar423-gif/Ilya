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


def prev_ids(ids):
    """Id of the token *before* each position (-1 at the start / after padding)."""
    out = torch.full_like(ids, -1)
    out[:, 1:] = ids[:, :-1]
    return out.masked_fill(ids < 0, -1)


def identity_matrix(q_ids, k_ids, exclude_diagonal=False, k_prev=None):
    """Symbol-identity channels for the exact-match attention prior, as a bool
    tensor [B, 2, Lq, Lk]:

    * channel 0: query token i and key token j are the same symbol;
    * channel 1: query token i equals the token *preceding* key j (an
      induction prior: "attend to what followed an earlier occurrence of me").

    Negative ids (padding / slots without a symbol) never match.
    """
    k_prev = prev_ids(k_ids) if k_prev is None else k_prev
    valid = (q_ids >= 0)[:, :, None]
    same = (q_ids[:, :, None] == k_ids[:, None, :]) & valid
    if exclude_diagonal:
        same = same & ~torch.eye(q_ids.shape[1], k_ids.shape[1], dtype=torch.bool,
                                 device=q_ids.device)[None]
    after = (q_ids[:, :, None] == k_prev[:, None, :]) & valid
    return torch.stack([same, after], dim=1)


class Attention(nn.Module):
    """Multi-head attention. With ``ident=True`` each head also learns scalar
    bonuses for attending to positions that hold the same symbol as the query,
    or that directly follow such a symbol (see :func:`identity_matrix`). They
    stand in for the retrieval circuits a pretrained model already has, start
    at zero, and are given identically to Ilya and to the Jev replica."""

    def __init__(self, d, heads, ident=False):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.o = nn.Linear(d, d)
        self.beta = nn.Parameter(torch.zeros(heads, 2)) if ident else None

    def project_kv(self, mem, pos=None):
        B, L, _ = mem.shape
        k, v = self.kv(mem).view(B, L, 2, self.h, self.dh).permute(2, 0, 3, 1, 4)
        return (k if pos is None else rope(k, pos)), v

    def forward(self, x, mem=None, mask=None, kv=None, past=None, pos=None, same=None):
        """``pos`` (self-attention only) applies rotary embeddings to q and k;
        ``same`` [B, 2, Lq, Lk] holds the identity channels for the exact-match bonus."""
        B, L, D = x.shape
        q = self.q(x).view(B, L, self.h, self.dh).transpose(1, 2)
        if pos is not None:
            q = rope(q, pos)
        k, v = kv if kv is not None else self.project_kv(x if mem is None else mem, pos)
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        if self.beta is not None and same is not None:
            # x10 reparametrisation: Adam moves a parameter by ~lr per step, which
            # would otherwise keep the bonus far too small to sharpen attention.
            bias = torch.einsum("hc,bcqk->bhqk", 10.0 * self.beta, same.to(q.dtype))
            if mask is not None:
                bias = bias.masked_fill(~mask, float("-inf"))
            mask = bias
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

    def __init__(self, d, heads, ff, ident=False):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn, self.mlp = Attention(d, heads, ident), MLP(d, ff)

    def forward(self, x, mask, pos=None, same=None):
        x = x + self.attn(self.n1(x), mask=mask, pos=pos, same=same)
        return x + self.mlp(self.n2(x))


class WorkspaceBlock(nn.Module):
    """Self-attention inside one question's workspace, cross-attention into the
    shared context memory, then an MLP."""

    def __init__(self, d, heads, ff, ident=False):
        super().__init__()
        self.n1, self.n2, self.n3, self.nm = (nn.LayerNorm(d) for _ in range(4))
        self.self_attn, self.cross_attn = Attention(d, heads, ident), Attention(d, heads, ident)
        self.mlp = MLP(d, ff)

    def forward(self, w, mem, self_mask, cross_mask, same_self=None, same_cross=None, n_query=None):
        """Only the first ``n_query`` workspace slots read the context (all if None)."""
        w = w + self.self_attn(self.n1(w), mask=self_mask, same=same_self)
        n = w.shape[1] if n_query is None else n_query
        if same_cross is not None:
            same_cross = same_cross[:, :, :n]
        read = self.cross_attn(self.n2(w[:, :n]), mem=self.nm(mem), mask=cross_mask, same=same_cross)
        w = w + torch.cat([read, torch.zeros_like(w[:, n:])], dim=1)
        return w + self.mlp(self.n3(w))


class DecoderBlock(nn.Module):
    """Causal self-attention + MLP, with optional prefix key/value cache."""

    def __init__(self, d, heads, ff, ident=False):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn, self.mlp = Attention(d, heads, ident), MLP(d, ff)

    def forward(self, x, mask, past=None, return_kv=False, pos=None, same=None):
        h = self.n1(x)
        kv = self.attn.project_kv(h, pos)
        x = x + self.attn(h, mask=mask, kv=kv, past=past, pos=pos, same=same)
        x = x + self.mlp(self.n2(x))
        return (x, kv) if return_kv else x
