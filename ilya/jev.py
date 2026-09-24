"""A faithful replica of the *public* Jev-style decision mechanism.

TypeSafe has not disclosed Jev's architecture. What is public (and what every
open re-implementation - simple-jev, jev-visual, OpenJev, AnyJev - does) is:
a causal language model reads ``context + question + labelled options``; the
context is a shared prefix computed once; each question is an isolated suffix;
the decision is the softmax over the option-label logits at the answer slot,
trained with a proper scoring rule so the probabilities are calibrated
in-distribution. This module implements exactly that, including prefix KV
caching, so the comparison with Ilya isolates the paradigm, not the budget.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .nn import DecoderBlock


class JevReplica(nn.Module):
    def __init__(self, vocab_size, d=128, heads=4, ff=512, layers=4, max_len=256, rope=False):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, d=d, heads=heads, ff=ff, layers=layers,
                           max_len=max_len, rope=rope)
        self.rope = rope
        self.tok = nn.Embedding(vocab_size, d)
        self.pos = None if rope else nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(DecoderBlock(d, heads, ff) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        nn.init.normal_(self.tok.weight, std=0.02)

    def _embed(self, ids, start=0):
        pos = torch.arange(start, start + ids.shape[1], device=ids.device)
        x = self.tok(ids)
        return (x, pos) if self.rope else (x + self.pos(pos), None)

    def _label_logits(self, h, labels, label_mask):
        # LM head tied to the input embedding, restricted to the option labels.
        w = self.tok(labels)                                  # [B, K, d]
        logits = torch.einsum("bd,bkd->bk", self.norm(h), w)
        return logits.masked_fill(~label_mask, float("-inf"))

    def forward(self, b):
        ids, valid = b["ids"], b["mask"]
        L = ids.shape[1]
        x, pos = self._embed(ids)
        causal = torch.ones(L, L, dtype=torch.bool, device=ids.device).tril()
        mask = causal[None, None] & valid[:, None, None, :]
        for blk in self.blocks:
            x = blk(x, mask, pos=pos)
        h = x[torch.arange(ids.shape[0]), b["ans_pos"]]
        return self._label_logits(h, b["labels"], b["label_mask"])

    # ------------------------------------------------ shared-prefix inference
    @torch.no_grad()
    def encode_prefix(self, prefix_ids):
        """Run the shared prefix once; returns per-layer (k, v)."""
        L = prefix_ids.shape[1]
        x, pos = self._embed(prefix_ids)
        mask = torch.ones(L, L, dtype=torch.bool, device=prefix_ids.device).tril()[None, None]
        cache = []
        for blk in self.blocks:
            x, kv = blk(x, mask, return_kv=True, pos=pos)
            cache.append(kv)
        return cache

    @torch.no_grad()
    def score_suffixes(self, cache, suffix_ids, suffix_valid, ans_pos, labels, label_mask):
        """Score a batch of question suffixes against one cached prefix."""
        M, S = suffix_ids.shape
        P = cache[0][0].shape[2]
        x, pos = self._embed(suffix_ids, start=P)
        causal = torch.ones(S, S, dtype=torch.bool, device=x.device).tril()[None, None] & suffix_valid[:, None, None, :]
        mask = torch.cat([torch.ones(M, 1, S, P, dtype=torch.bool, device=x.device), causal], dim=-1)
        for blk, (k, v) in zip(self.blocks, cache):
            x = blk(x, mask, past=(k.expand(M, -1, -1, -1), v.expand(M, -1, -1, -1)), pos=pos)
        h = x[torch.arange(M), ans_pos]
        return self._label_logits(h, labels, label_mask)
