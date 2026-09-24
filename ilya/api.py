"""Typed decision API (Jev-compatible request shape, richer answers).

    engine = Engine.load("checkpoints/ilya.pt")
    engine.decide(
        state="alpha red cube large . beta left alpha .",
        questions={
            "color": {"type": "choice", "instructions": "what color is alpha ?",
                      "criteria": {"x1": "crimson", "x2": "azure"}},
            "size":  {"type": "score", "instructions": "how big is alpha ?",
                      "criteria": ["tiny", "small", "medium", "large", "huge"]},
            "order": {"type": "noul", "instructions": "is beta left of alpha ?"},
        })

Every answer carries: the decision (or ``None`` when NONE wins), the full
distribution including the implicit ``NONE`` outcome, the number of loop
iterations actually used, and - when the engine was calibrated - a conformal
prediction set and a certified ``accept`` flag. Declared constraints are
imposed exactly through :mod:`ilya.coherence`.

The model is trained on the synthetic world in :mod:`ilya.world`; text is
tokenised against that vocabulary (unknown words become ``[UNK]``).
"""
from __future__ import annotations

import re

import numpy as np
import torch

from .batching import collate_ilya
from .coherence import condition
from .model import Ilya
from .world import VOCAB, Candidate, Example, Question

NONE = "NONE"


def tokenize(text):
    if isinstance(text, (list, tuple)):
        return [str(t).lower() for t in text]
    return re.findall(r"\[[a-z]+\]|[a-z0-9_]+|[?.:;]", str(text).lower())


def build_question(spec):
    qtype = spec["type"]
    instr = tokenize(spec.get("instructions") or "")
    crit = spec.get("criteria")
    if qtype == "choice":
        if not isinstance(crit, dict) or len(crit) < 2:
            raise ValueError("choice questions need a criteria object with at least 2 options")
        cands = [Candidate(str(k), tokenize(k), tokenize(v) if v else []) for k, v in crit.items()]
    elif qtype == "score":
        if not isinstance(crit, list) or len(crit) < 2:
            raise ValueError("score questions need an ordered criteria list with at least 2 levels")
        k = len(crit)
        cands = [Candidate(str(i), tokenize(v), [], ordinal=i / (k - 1)) for i, v in enumerate(crit)]
    elif qtype == "noul":
        # The released checkpoint was trained with bare true/false outcomes;
        # noul criteria are accepted for API compatibility but not used.
        cands = [Candidate("true", ["true"], []), Candidate("false", ["false"], [])]
    else:
        raise ValueError(f"unknown question type {qtype!r}")
    return Question(qtype, spec.get("family", qtype), instr, cands, gold=-1)


class Engine:
    def __init__(self, model: Ilya, temperature=1.0, qhat=None, gate=None,
                 max_iters=16, halt=0.97):
        self.model = model.eval()
        self.temperature = temperature
        self.qhat = qhat          # conformal threshold (optional)
        self.gate = gate          # certified confidence threshold (optional)
        self.max_iters = max_iters
        self.halt = halt

    @classmethod
    def load(cls, path, **kw):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = Ilya(**ck["config"])
        model.load_state_dict(ck["state"])
        cal = ck.get("calibration", {})
        opts = {k: cal[k] for k in ("temperature", "qhat", "gate") if k in cal}
        return cls(model, **{**opts, **kw})

    @torch.no_grad()
    def beliefs(self, state, questions, max_iters=None):
        qids = list(questions)
        qs = [build_question(questions[q]) for q in qids]
        ctx = tokenize(state)
        exs = [Example(ctx, q) for q in qs]          # one shared context object
        b = collate_ilya(exs)
        logits = self.model(b, T=max_iters or self.max_iters, halt=self.halt)[-1]
        probs = torch.softmax(logits / self.temperature, -1).numpy()
        iters = self.model.last_iterations.tolist()
        return qids, qs, probs, iters

    def decide(self, state, questions, constraints=None, max_iters=None):
        qids, qs, probs, iters = self.beliefs(state, questions, max_iters)
        beliefs = {}
        for qid, q, p in zip(qids, qs, probs):
            labels = [NONE] + [c.value for c in q.cands]
            beliefs[qid] = dict(zip(labels, p[:len(labels)].tolist()))
        cond = condition(beliefs, constraints) if constraints else None
        answers = {}
        for qid, q, it in zip(qids, qs, iters):
            dist = cond["marginals"][qid] if cond else beliefs[qid]
            dist = {k: float(v) for k, v in dist.items()}
            top = cond["map"][qid] if cond else max(dist, key=dist.get)
            answers[qid] = self._format(q, dist, top, it)
            if cond:
                answers[qid]["support"] = cond["support"][qid]
        return {"answers": answers}

    def _format(self, q, dist, top, iters):
        p_none = dist[NONE]
        inscope = {k: v for k, v in dist.items() if k != NONE}
        z = sum(inscope.values()) or 1.0
        out = {"type": q.qtype, "none": p_none, "iterations": int(iters),
               "probabilities": inscope, "confidence": dist.get(top, 0.0) if top else 0.0}
        decided = None if top in (None, NONE) else top
        if q.qtype == "choice":
            out["choice"] = decided
        elif q.qtype == "score":
            out["score"] = None if decided is None else sum(int(k) * v / z for k, v in inscope.items())
            out["level"] = None if decided is None else int(decided)
        else:
            out["noul"] = None if decided is None else inscope["true"] / z
        if self.qhat is not None:
            out["set"] = [k for k, v in dist.items() if v >= 1.0 - self.qhat]
        if self.gate is not None:
            out["accept"] = bool(out["confidence"] >= self.gate)
        return out
