"""Coherence by conditioning.

Questions are answered in isolation (so an answer never depends on which other
questions happen to be in the request). Logical relations *declared* by the
caller are then imposed exactly, by conditioning the product of the isolated
beliefs on the constraints being satisfied::

    P(x | C)  ∝  prod_m p_m(x_m) * 1[C(x)]

The constraint graph is split into connected components; each component's
joint is enumerated exactly as a dense tensor. The result has three parts:

* ``marginals`` - coherent per-question probabilities,
* ``map``       - the most probable *joint* assignment, which satisfies every
                  constraint by construction (independent argmaxes need not),
* ``support``   - ``P(C)`` under the isolated beliefs, i.e. how much of the
                  model's own belief mass is consistent with what the caller
                  knows. A low value is a built-in contradiction alarm.

The layer is model-agnostic: it works on any categorical beliefs, including
the Jev replica's.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Constraint:
    questions: tuple            # question ids, in the predicate's argument order
    predicate: Callable         # vectorised over broadcast arrays of outcome labels
    name: str = ""


def implies(q1, v1, q2, v2):
    return Constraint((q1, q2), lambda a, b: (a != v1) | (b == v2), f"{q1}={v1} -> {q2}={v2}")


def iff(q1, v1, q2, v2):
    return Constraint((q1, q2), lambda a, b: (a == v1) == (b == v2), f"{q1}={v1} <-> {q2}={v2}")


def excludes(q1, v1, q2, v2):
    return Constraint((q1, q2), lambda a, b: ~((a == v1) & (b == v2)), f"not({q1}={v1} & {q2}={v2})")


def count_equals(count_q, literal_qs, value_of, true_label="true"):
    """``count_q``'s value equals the number of ``literal_qs`` answered
    ``true_label``. ``value_of`` maps a count label to an int (or None for
    labels such as NONE that make the constraint unsatisfiable)."""
    def pred(c, *lits):
        total = sum((x == true_label).astype(int) for x in lits)
        vals = np.vectorize(lambda s: -1 if value_of(s) is None else value_of(s), otypes=[int])(c)
        return vals == total
    return Constraint((count_q, *literal_qs), pred, f"{count_q} = #{true_label}")


def _components(qids, constraints):
    parent = {q: q for q in qids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for c in constraints:
        for q in c.questions[1:]:
            parent[find(q)] = find(c.questions[0])
    groups = {}
    for q in qids:
        groups.setdefault(find(q), []).append(q)
    return list(groups.values())


def condition(beliefs, constraints, max_states=1 << 22):
    """``beliefs``: {qid: {label: prob}}. Returns dict with marginals, map, support."""
    constraints = list(constraints or [])
    for c in constraints:
        missing = [q for q in c.questions if q not in beliefs]
        if missing:
            raise KeyError(f"constraint {c.name or c} refers to unknown questions {missing}")
    marg, amap, support = {}, {}, {}
    for comp in _components(list(beliefs), constraints):
        cons = [c for c in constraints if c.questions[0] in comp]
        labels = [np.array(list(beliefs[q].keys()), dtype=object) for q in comp]
        probs = [np.array(list(beliefs[q].values()), dtype=float) for q in comp]
        if not cons:
            q = comp[0]
            p = probs[0] / probs[0].sum()
            marg[q] = dict(zip(labels[0], p))
            amap[q] = labels[0][int(np.argmax(p))]
            support[q] = 1.0
            continue
        shape = [len(p) for p in probs]
        if int(np.prod(shape)) > max_states:
            raise ValueError(f"constraint component {comp} has {int(np.prod(shape))} joint states")
        n = len(comp)
        joint = np.ones(shape)
        for i, p in enumerate(probs):
            joint = joint * (p / p.sum()).reshape([-1 if j == i else 1 for j in range(n)])
        axis = {q: i for i, q in enumerate(comp)}
        for c in cons:
            grids = [labels[axis[q]].reshape([-1 if j == axis[q] else 1 for j in range(n)])
                     for q in c.questions]
            mask = np.broadcast_to(np.asarray(c.predicate(*grids), dtype=bool), shape)
            joint = joint * mask
        z = float(joint.sum())
        for q in comp:
            support[q] = z
        if z <= 0.0:  # the declared constraints are impossible under the beliefs
            for i, q in enumerate(comp):
                marg[q] = dict(zip(labels[i], probs[i] / probs[i].sum()))
                amap[q] = None
            continue
        joint /= z
        best = np.unravel_index(int(np.argmax(joint)), shape)
        for i, q in enumerate(comp):
            other = tuple(j for j in range(n) if j != i)
            marg[q] = dict(zip(labels[i], joint.sum(axis=other)))
            amap[q] = labels[i][best[i]]
    return {"marginals": marg, "map": amap, "support": support}


def violates(assignment, constraints):
    """True if a concrete assignment {qid: label} breaks any constraint."""
    for c in constraints:
        args = [np.array([assignment[q]], dtype=object) for q in c.questions]
        if not bool(np.asarray(c.predicate(*args)).all()):
            return True
    return False
