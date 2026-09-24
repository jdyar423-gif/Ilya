import itertools
import random

import numpy as np
import pytest

from ilya.coherence import condition, count_equals, excludes, iff, implies, violates


def _brute(beliefs, constraints):
    qs = list(beliefs)
    marg = {q: {k: 0.0 for k in beliefs[q]} for q in qs}
    z, best, best_p = 0.0, None, -1.0
    for combo in itertools.product(*[list(beliefs[q].items()) for q in qs]):
        assign = {q: lab for q, (lab, _) in zip(qs, combo)}
        if violates(assign, constraints):
            continue
        p = float(np.prod([pr for _, pr in combo]))
        z += p
        for q in qs:
            marg[q][assign[q]] += p
        if p > best_p:
            best, best_p = assign, p
    return {q: {k: v / z for k, v in m.items()} for q, m in marg.items()}, best, z


def _random_beliefs(rng, n_q):
    out = {}
    for i in range(n_q):
        labels = ["NONE", "a", "b", "c"][: rng.randint(2, 4)]
        p = np.array([rng.random() + 0.05 for _ in labels])
        out[f"q{i}"] = dict(zip(labels, p / p.sum()))
    return out


@pytest.mark.parametrize("seed", range(20))
def test_conditioning_matches_brute_force(seed):
    rng = random.Random(seed)
    beliefs = _random_beliefs(rng, 5)
    qs = list(beliefs)
    cons = []
    for _ in range(3):  # every constraint touches q0, so they form one component
        a, b = "q0", rng.choice(qs[1:])
        va, vb = rng.choice(list(beliefs[a])), rng.choice(list(beliefs[b]))
        cons.append(rng.choice([implies, iff, excludes])(a, va, b, vb))
    out = condition(beliefs, cons)
    marg, best, z = _brute(beliefs, cons)
    assert abs(out["support"]["q0"] - z) < 1e-9
    if z == 0:
        return
    for q in qs:
        for k in beliefs[q]:
            assert abs(out["marginals"][q][k] - marg[q][k]) < 1e-9
    assert not violates(out["map"], cons)
    joint = lambda a: np.prod([beliefs[q][a[q]] for q in qs])
    assert abs(joint(out["map"]) - joint(best)) < 1e-12


def test_count_constraint_and_coherent_map():
    beliefs = {"n": {"zero": 0.1, "one": 0.2, "two": 0.7},
               "x": {"true": 0.4, "false": 0.6}, "y": {"true": 0.45, "false": 0.55}}
    cons = [count_equals("n", ["x", "y"], lambda s: ["zero", "one", "two"].index(s))]
    independent = {q: max(b, key=b.get) for q, b in beliefs.items()}
    assert violates(independent, cons)          # "two" but no true literal
    out = condition(beliefs, cons)
    assert not violates(out["map"], cons)
    assert out["support"]["n"] < 1.0


def test_unsatisfiable_constraints_are_reported():
    beliefs = {"a": {"true": 1.0, "false": 0.0}, "b": {"true": 1.0, "false": 0.0}}
    out = condition(beliefs, [excludes("a", "true", "b", "true")])
    assert out["support"]["a"] == 0.0 and out["map"]["a"] is None


def test_unknown_question_is_rejected():
    with pytest.raises(KeyError):
        condition({"a": {"x": 1.0}}, [implies("a", "x", "zzz", "y")])
