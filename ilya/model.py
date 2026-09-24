"""Ilya: an open-world, anytime, presentation-invariant typed decision model.

Architecture (one context, many isolated questions):

* **Context stream** - the context is embedded once, passed through a prelude
  block and then refined by a *weight-tied* bidirectional block that is looped
  T times. It never sees the questions, so it is shared by all of them.
* **Question workspaces** - each question gets its own workspace
  ``[instruction tokens ; NONE ; candidate_1 .. candidate_K]``. Workspaces
  never attend to each other, so a question's answer cannot depend on which
  other questions were asked in the same request.
* **Candidates as a set** - candidate tokens carry no positional encoding and
  are embedded from their rubric (their name is only used when no rubric is
  given). Output probabilities are therefore *exactly* invariant to option
  order and to misleading option names.
* **Open-world readout** - every question has an implicit NONE outcome
  competing with the candidates, so "none of these / out of scope" is always
  expressible, including for yes/no (noul) and ordered (score) questions.
* **Anytime beliefs** - a readout after every loop iteration is trained with a
  strictly proper scoring rule, so each iteration's output is a calibrated
  belief given the computation so far. Inference halts per question once the
  belief is confident, and can run more iterations than it was trained with.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .nn import MLP, EncoderBlock, WorkspaceBlock, identity_matrix, prev_ids


class Ilya(nn.Module):
    def __init__(self, vocab_size, d=128, heads=4, ff=512, max_ctx=192, max_instr=16,
                 rope=False, ident=False):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, d=d, heads=heads, ff=ff,
                           max_ctx=max_ctx, max_instr=max_instr, rope=rope, ident=ident)
        self.rope, self.ident = rope, ident
        self.tok = nn.Embedding(vocab_size, d)
        self.pos_ctx = None if rope else nn.Embedding(max_ctx, d)
        self.pos_instr = nn.Embedding(max_instr, d)
        self.qtype = nn.Embedding(3, d)
        self.role = nn.Embedding(3, d)  # 0 instruction, 1 answer/NONE slot, 2 candidate
        self.none_emb = nn.Parameter(torch.randn(d) * 0.02)
        self.sem = MLP(d, ff)
        self.ord = nn.Linear(2, d, bias=False)
        self.prelude = EncoderBlock(d, heads, ff, ident)
        self.ctx_core = EncoderBlock(d, heads, ff, ident)
        self.ws_core = WorkspaceBlock(d, heads, ff, ident)
        self.c_norm, self.w_norm, self.out_norm = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        # The looped state is LayerNorm-ed every iteration; the re-injected
        # inputs are normalised to the same scale so they are not drowned.
        self.c_in, self.w_in = nn.LayerNorm(d), nn.LayerNorm(d)
        self.cand_head = nn.Linear(d, 1)
        self.none_head = nn.Linear(d, 1)
        self.ans_q, self.cand_k = nn.Linear(d, d), nn.Linear(d, d)
        self.ans_out = nn.Linear(d, d)  # tied "copy" readout into token-embedding space
        for m in self.modules():  # small, uniform init so no embedding drowns token identity
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        nn.init.normal_(self.ord.weight, std=0.02)

    # ------------------------------------------------------------------ inputs
    def ctx_positions(self, ctx):
        return torch.arange(ctx.shape[1], device=ctx.device) if self.rope else None

    def embed_context(self, ctx, ctx_mask, same=None):
        c = self.tok(ctx)
        if not self.rope:
            c = c + self.pos_ctx(torch.arange(ctx.shape[1], device=ctx.device))
        return self.prelude(c, ctx_mask[:, None, None, :], pos=self.ctx_positions(ctx), same=same)

    @staticmethod
    def workspace_ids(b):
        """Symbol id of every workspace slot (-1 = no symbol): instruction
        tokens, the answer slot, and each candidate's (first) semantic token."""
        B = b["instr"].shape[0]
        instr = b["instr"].masked_fill(~b["instr_mask"], -1)
        ans = torch.full((B, 1), -1, dtype=torch.long, device=instr.device)
        cand = b["sem"][:, :, 0].masked_fill(~b["cand_mask"], -1)
        return torch.cat([instr, ans, cand], dim=1)

    def embed_workspace(self, b):
        B, Li = b["instr"].shape
        K = b["sem"].shape[1]
        qt = self.qtype(b["qtype"])[:, None, :]
        pos = torch.arange(Li, device=b["instr"].device)
        instr = self.tok(b["instr"]) + self.pos_instr(pos) + self.role.weight[0]
        none = (self.none_emb + self.role.weight[1]).expand(B, 1, -1)
        m = b["sem_mask"].unsqueeze(-1).float()
        pooled = (self.tok(b["sem"]) * m).sum(2) / m.sum(2).clamp(min=1.0)
        cands = pooled + self.sem(pooled) + self.ord(b["ordinal"]) + self.role.weight[2]
        w = torch.cat([instr, none, cands], dim=1) + qt
        valid = torch.cat([b["instr_mask"], torch.ones(B, 1, dtype=torch.bool, device=w.device),
                           b["cand_mask"]], dim=1)
        self._sem_emb = pooled  # semantic token embedding of each candidate, for the copy readout
        return w, valid, Li

    # ----------------------------------------------------------------- forward
    def readout(self, w, Li, cand_mask):
        """NONE logit from the answer slot. Each candidate's logit adds three
        per-candidate terms: a tied *copy* score between the answer slot and the
        candidate's own semantic token embedding (the analogue of an LM head,
        but over the rubric), a bilinear match with the candidate's workspace
        state, and a candidate-only term. All are per-candidate functions of
        shared state, so the readout stays exactly permutation-equivariant."""
        h = self.out_norm(w)
        ans, cand = h[:, Li], h[:, Li + 1:]
        none = self.none_head(ans)
        copy = torch.einsum("bd,bkd->bk", self.ans_out(ans), self._sem_emb)
        match = torch.einsum("bd,bkd->bk", self.ans_q(ans), self.cand_k(cand)) / ans.shape[-1] ** 0.5
        logits = (copy + match + self.cand_head(cand).squeeze(-1)).masked_fill(~cand_mask, float("-inf"))
        return torch.cat([none, logits], dim=1)

    def forward(self, b, T=8, ctx_index=None, halt=None):
        """Return logits of shape ``[T, B, 1 + K]`` (index 0 = NONE).

        ``ctx_index`` maps each question row to a context row, so many questions
        can share one encoded context. ``halt`` (a probability threshold) makes
        halted questions keep their belief; the per-question number of
        iterations used is stored in ``self.last_iterations``.
        """
        ctx_mask = b["ctx_mask"]
        if ctx_index is None:
            ctx_index = b.get("ctx_index")
        if ctx_index is None:
            ctx_index = torch.arange(b["instr"].shape[0], device=b["instr"].device)
        same_c = same_w = same_x = None
        if self.ident:
            c_ids = b["ctx"].masked_fill(~ctx_mask, -1)
            w_ids = self.workspace_ids(b)
            same_c = identity_matrix(c_ids, c_ids, exclude_diagonal=True)
            Li_ = b["instr"].shape[1]
            w_prev = prev_ids(w_ids)
            w_prev[:, Li_:] = -1  # the answer slot and candidates form a set, not a sequence
            same_w = identity_matrix(w_ids, w_ids, exclude_diagonal=True, k_prev=w_prev)
            same_x = identity_matrix(w_ids, c_ids[ctx_index])
        c_e = self.c_in(self.embed_context(b["ctx"], ctx_mask, same_c))
        w_e, w_valid, Li = self.embed_workspace(b)
        w_e = self.w_in(w_e)
        c_mask4 = ctx_mask[:, None, None, :]
        self_mask = w_valid[:, None, None, :]
        cross_mask = ctx_mask[ctx_index][:, None, None, :]
        c_pos = self.ctx_positions(b["ctx"])
        c = torch.zeros_like(c_e)
        w = torch.zeros_like(w_e)
        outs = []
        done = torch.zeros(w_e.shape[0], dtype=torch.bool, device=w_e.device)
        iters = torch.full((w_e.shape[0],), T, dtype=torch.long, device=w_e.device)
        frozen = None
        for t in range(T):
            c = self.c_norm(self.ctx_core(c + c_e, c_mask4, pos=c_pos, same=same_c))
            # Evidence is retrieved by the question and answer slots only; the
            # candidates are hypotheses scored against it and never search the
            # context themselves (no option-driven "is my value anywhere?" shortcut).
            w = self.w_norm(self.ws_core(w + w_e, c[ctx_index], self_mask, cross_mask, same_w, same_x,
                                         n_query=Li + 1))
            logits = self.readout(w, Li, b["cand_mask"])
            if halt is not None:
                frozen = logits if frozen is None else torch.where(done[:, None], frozen, logits)
                newly = (~done) & (frozen.softmax(-1).max(-1).values >= halt) & (t >= 1)
                iters = torch.where(newly, torch.full_like(iters, t + 1), iters)
                done = done | newly
                outs.append(frozen)
                if bool(done.all()):  # every question has halted: stop computing
                    outs += [frozen] * (T - t - 1)
                    break
            else:
                outs.append(logits)
        self.last_iterations = iters
        return torch.stack(outs)
