"""Turn world examples into tensors for Ilya and for the Jev-paradigm replica."""
from __future__ import annotations

import torch

from .world import VOCAB, DIGITS, LETTERS

QTYPES = {"choice": 0, "score": 1, "noul": 2}


def _pad(seqs, value):
    n = max(1, max((len(s) for s in seqs), default=1))
    return torch.tensor([s + [value] * (n - len(s)) for s in seqs], dtype=torch.long)


def semantic_tokens(cand):
    """Rubric-authoritative binding: an option's meaning is its rubric when one is
    given, otherwise its name. The name is an identifier and never reaches the
    model when a rubric exists, so a misleading name cannot change the answer."""
    return cand.rubric if cand.rubric else cand.name


def collate_ilya(examples, vocab=VOCAB):
    """Batch for Ilya. Target 0 means NONE; target k+1 means candidate k.

    Examples that share the same context *object* share one encoded context
    row (``ctx_index`` maps each question to its context row)."""
    rows, ctx_index = {}, []
    for ex in examples:
        ctx_index.append(rows.setdefault(id(ex.context), len(rows)))
    uniq = {}
    for ex in examples:
        uniq.setdefault(id(ex.context), ex.context)
    ctx = [vocab.encode(c) for c in uniq.values()]
    instr = [vocab.encode(ex.question.instr) for ex in examples]
    K = max(len(ex.question.cands) for ex in examples)
    S = max(len(semantic_tokens(c)) for ex in examples for c in ex.question.cands)
    sem = torch.full((len(examples), K, S), vocab.pad, dtype=torch.long)
    cand_mask = torch.zeros(len(examples), K, dtype=torch.bool)
    ordinal = torch.zeros(len(examples), K, 2)
    for b, ex in enumerate(examples):
        for k, c in enumerate(ex.question.cands):
            toks = vocab.encode(semantic_tokens(c))
            sem[b, k, :len(toks)] = torch.tensor(toks)
            cand_mask[b, k] = True
            if c.ordinal is not None:
                ordinal[b, k] = torch.tensor([1.0, c.ordinal])
    ctx_t, instr_t = _pad(ctx, vocab.pad), _pad(instr, vocab.pad)
    return {
        "ctx": ctx_t, "ctx_mask": ctx_t != vocab.pad, "ctx_index": torch.tensor(ctx_index),
        "instr": instr_t, "instr_mask": instr_t != vocab.pad,
        "qtype": torch.tensor([QTYPES[ex.question.qtype] for ex in examples]),
        "sem": sem, "sem_mask": sem != vocab.pad, "cand_mask": cand_mask,
        "ordinal": ordinal,
        "target": torch.tensor([ex.question.gold + 1 for ex in examples]),
    }


def jev_labels(q, with_none):
    if q.qtype == "choice":
        labels = LETTERS[:len(q.cands)]
    elif q.qtype == "score":
        labels = DIGITS[:len(q.cands)]
    else:
        labels = [c.name[0] for c in q.cands]  # "true" / "false"
    return labels + (["[NOTA]"] if with_none else [])


def jev_suffix(q, with_none):
    """Question suffix in the Jev style: options are introduced by positional
    labels and the answer is read from the label logits at the [ANS] slot."""
    toks = ["[SEP]"] + list(q.instr) + ["[SEP]"]
    labels = jev_labels(q, with_none)
    for lab, c in zip(labels, q.cands):
        toks += [lab] + list(c.name) + ([":"] + list(c.rubric) if c.rubric else []) + [";"]
    if with_none:
        toks += ["[NOTA]", "[NOTA]", ";"]
    return toks + ["[ANS]"], labels


def collate_jev(examples, with_none, vocab=VOCAB):
    """Batch for the Jev replica. Target is the index of the gold label, the
    [NOTA] option for out-of-scope items when ``with_none``, else -100 (the
    closed-world model has no way to express NONE)."""
    seqs, labels, targets, prefix = [], [], [], []
    for ex in examples:
        suffix, labs = jev_suffix(ex.question, with_none)
        seq = ["[BOS]"] + list(ex.context) + suffix
        seqs.append(vocab.encode(seq))
        prefix.append(1 + len(ex.context))
        labels.append(vocab.encode(labs))
        g = ex.question.gold
        targets.append(g if g >= 0 else (len(labs) - 1 if with_none else -100))
    ids = _pad(seqs, vocab.pad)
    lab = _pad(labels, vocab.pad)
    return {
        "ids": ids, "mask": ids != vocab.pad,
        "ans_pos": torch.tensor([len(s) - 1 for s in seqs]),
        "labels": lab, "label_mask": lab != vocab.pad,
        "target": torch.tensor(targets), "prefix_len": torch.tensor(prefix),
    }
