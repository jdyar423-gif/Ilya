import torch

from ilya.api import Engine, tokenize
from ilya.coherence import iff
from ilya.model import Ilya
from ilya.world import OOS_KINDS_TEST, SYNONYMS, VOCAB, make_split


def test_splits_are_deterministic_and_labelled():
    a, b = make_split("id", 50, seed=1), make_split("id", 50, seed=1)
    assert [x.context for x in a] == [x.context for x in b]
    assert all(ex.question.gold >= 0 for ex in a)
    for s in OOS_KINDS_TEST:
        assert all(ex.question.gold == -1 for ex in make_split(s, 50, seed=2))


def test_misleading_split_names_lie_but_rubrics_tell_the_truth():
    for ex in make_split("misleading", 100, seed=4):
        for c in ex.question.cands:
            assert c.name != [c.value]
            assert c.rubric[0] in SYNONYMS[c.value] + [c.value]


def test_engine_returns_typed_answers():
    torch.manual_seed(0)
    eng = Engine(Ilya(len(VOCAB), d=64, ff=128), max_iters=4, qhat=0.5, gate=0.9)
    out = eng.decide(
        "alpha red cube large . beta left alpha .",
        {"color": {"type": "choice", "instructions": "what color is alpha?",
                   "criteria": {"x1": "crimson", "x2": "azure"}},
         "size": {"type": "score", "instructions": "how big is alpha ?",
                  "criteria": ["tiny", "small", "medium", "large", "huge"]},
         "is_red": {"type": "noul", "instructions": "is alpha red ?"}},
        constraints=[iff("color", "x1", "is_red", "true")])["answers"]
    assert set(out) == {"color", "size", "is_red"}
    for a in out.values():
        assert 0.0 <= a["none"] <= 1.0 and 1 <= a["iterations"] <= 4
        assert abs(a["none"] + sum(a["probabilities"].values()) - 1) < 1e-5
        assert "set" in a and "accept" in a and "support" in a
    assert tokenize("what color is alpha?") == ["what", "color", "is", "alpha", "?"]
