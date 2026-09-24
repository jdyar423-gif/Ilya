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

from .nn import DecoderBlock, identity_matrix, prev_ids


class JevReplica(nn.Module):
    def __init__(self, vocab_size, d=128, heads=4, ff=512, layers=4, max_len=256, rope=False,
                 ident=False, readout="name"):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, d=d, heads=heads, ff=ff, layers=layers,
                           max_len=max_len, rope=rope, ident=ident, readout=readout)
        self.readout = readout  # used by the batching code, see batching.jev_labels
        self.rope, self.ident = rope, ident
        self.tok = nn.Embedding(vocab_size, d)
        self.pos = None if rope else nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(DecoderBlock(d, heads, ff, ident) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

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
        same = None
        if self.ident:
            sym = ids.masked_fill(~valid, -1)
            same = identity_matrix(sym, sym, exclude_diagonal=True)
        for blk in self.blocks:
            x = blk(x, mask, pos=pos, same=same)
        h = x[torch.arange(ids.shape[0]), b["ans_pos"]]
        return self._label_logits(h, b["labels"], b["label_mask"])

    # ------------------------------------------- shared-prefix training path
    def forward_grouped(self, b):
        """Same logits as :meth:`forward`, but every distinct context (prefix)
        is computed once and its keys/values are shared by all the question
        suffixes that use it - with gradients, so training also amortises the
        context. ``b`` comes from :func:`ilya.batching.collate_jev_grouped`."""
        assert self.rope, "the grouped path needs rotary positions"
        pre, pre_valid, idx = b["prefix"], b["prefix_mask"], b["ctx_index"]
        suf, suf_valid = b["suffix"], b["suffix_mask"]
        Bc, P = pre.shape
        B, S = suf.shape
        dev = pre.device
        plen = pre_valid.sum(1)
        pre_sym = pre.masked_fill(~pre_valid, -1)
        suf_sym = suf.masked_fill(~suf_valid, -1)
        # prefix pass (causal, right-padded)
        x, _ = self._embed(pre)
        pos_p = torch.arange(P, device=dev)
        mask_p = torch.ones(P, P, dtype=torch.bool, device=dev).tril()[None, None] & pre_valid[:, None, None, :]
        same_p = identity_matrix(pre_sym, pre_sym, exclude_diagonal=True) if self.ident else None
        kvs = []
        for blk in self.blocks:
            x, kv = blk(x, mask_p, return_kv=True, pos=pos_p, same=same_p)
            kvs.append(kv)
        # suffix pass: row r continues its own prefix at position plen[r]
        pos_s = plen[idx][:, None] + torch.arange(S, device=dev)[None]
        y = self.tok(suf)
        causal = torch.ones(S, S, dtype=torch.bool, device=dev).tril()[None, None] & suf_valid[:, None, None, :]
        mask_s = torch.cat([pre_valid[idx][:, None, None, :].expand(B, 1, S, P), causal], dim=-1)
        same_s = None
        if self.ident:
            keys = torch.cat([pre_sym[idx], suf_sym], dim=1)
            last = pre_sym[idx].gather(1, (plen[idx] - 1)[:, None])
            s_prev = torch.cat([last, suf_sym[:, :-1]], dim=1).masked_fill(suf_sym < 0, -1)
            k_prev = torch.cat([prev_ids(pre_sym)[idx], s_prev], dim=1)
            same_s = identity_matrix(suf_sym, keys, k_prev=k_prev)
            same_s[:, 0, :, P:] &= ~torch.eye(S, dtype=torch.bool, device=dev)[None]
        for blk, (k, v) in zip(self.blocks, kvs):
            y = blk(y, mask_s, past=(k[idx], v[idx]), pos=pos_s, same=same_s)
        h = y[torch.arange(B, device=dev), b["ans_pos"]]
        return self._label_logits(h, b["labels"], b["label_mask"])

    # ------------------------------------------------ shared-prefix inference
    @torch.no_grad()
    def encode_prefix(self, prefix_ids):
        """Run the shared prefix once; returns per-layer (k, v)."""
        L = prefix_ids.shape[1]
        x, pos = self._embed(prefix_ids)
        mask = torch.ones(L, L, dtype=torch.bool, device=prefix_ids.device).tril()[None, None]
        same = identity_matrix(prefix_ids, prefix_ids, exclude_diagonal=True) if self.ident else None
        cache = []
        for blk in self.blocks:
            x, kv = blk(x, mask, return_kv=True, pos=pos, same=same)
            cache.append(kv)
        return cache

    @torch.no_grad()
    def score_suffixes(self, cache, suffix_ids, suffix_valid, ans_pos, labels, label_mask, prefix_ids=None):
        """Score a batch of question suffixes against one cached prefix
        (``prefix_ids`` [1, P] is needed when the model uses identity bias)."""
        M, S = suffix_ids.shape
        P = cache[0][0].shape[2]
        same = None
        if self.ident:
            sym = suffix_ids.masked_fill(~suffix_valid, -1)
            keys = torch.cat([prefix_ids.expand(M, -1), sym], dim=1)
            same = identity_matrix(sym, keys)
            same[:, 0, :, P:] &= ~torch.eye(S, dtype=torch.bool, device=sym.device)[None]
        x, pos = self._embed(suffix_ids, start=P)
        causal = torch.ones(S, S, dtype=torch.bool, device=x.device).tril()[None, None] & suffix_valid[:, None, None, :]
        mask = torch.cat([torch.ones(M, 1, S, P, dtype=torch.bool, device=x.device), causal], dim=-1)
        for blk, (k, v) in zip(self.blocks, cache):
            x = blk(x, mask, past=(k.expand(M, -1, -1, -1), v.expand(M, -1, -1, -1)), pos=pos, same=same)
        h = x[torch.arange(M), ans_pos]
        return self._label_logits(h, labels, label_mask)
