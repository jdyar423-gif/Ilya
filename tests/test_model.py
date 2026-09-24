import random

import pytest
import torch

from ilya.batching import collate_ilya, collate_jev, jev_suffix
from ilya.jev import JevReplica
from ilya.model import Ilya
from ilya.world import VOCAB, Candidate, Example, Question, make_split, sample_group

torch.manual_seed(0)
MODEL = Ilya(len(VOCAB), d=64, ff=128, rope=True).eval()


def _choice_examples(n=24):
    return [ex for ex in make_split("id", 200, seed=3) if ex.question.qtype == "choice"][:n]


def _clone(ex, cands, gold):
    q = ex.question
    return Example(ex.context, Question(q.qtype, q.family, q.instr, cands, gold, q.meta))


@torch.no_grad()
def test_option_order_invariance_is_exact():
    exs = _choice_examples()
    rng = random.Random(0)
    perms, shuffled = [], []
    for ex in exs:
        order = list(range(len(ex.question.cands)))
        rng.shuffle(order)
        perms.append(order)
        shuffled.append(_clone(ex, [ex.question.cands[i] for i in order], order.index(ex.question.gold)))
    a = MODEL(collate_ilya(exs), T=5)[-1]
    b = MODEL(collate_ilya(shuffled), T=5)[-1]
    for i, order in enumerate(perms):
        assert torch.allclose(a[i, 0], b[i, 0], atol=1e-5)
        for new, old in enumerate(order):
            assert torch.allclose(a[i, 1 + old], b[i, 1 + new], atol=1e-5)


@torch.no_grad()
def test_misleading_names_cannot_change_beliefs():
    exs = make_split("misleading", 32, seed=5)
    renamed = [_clone(ex, [Candidate(c.value, ["c0"], c.rubric, c.ordinal) for c in ex.question.cands],
                      ex.question.gold) for ex in exs]
    a = MODEL(collate_ilya(exs), T=4)[-1]
    b = MODEL(collate_ilya(renamed), T=4)[-1]
    assert torch.allclose(a, b, atol=1e-6)


@torch.no_grad()
def test_questions_are_isolated_and_context_is_shared():
    group = sample_group(random.Random(1), 6)
    together = MODEL(collate_ilya(group), T=4)[-1]
    assert collate_ilya(group)["ctx"].shape[0] == 1  # one encoded context for six questions
    for i, ex in enumerate(group):
        alone = MODEL(collate_ilya([Example(list(ex.context), ex.question)]), T=4)[-1]
        k = alone.shape[1]
        assert torch.allclose(together[i, :k], alone[0], atol=1e-5)


@torch.no_grad()
def test_halting_records_iterations():
    exs = make_split("id", 16, seed=9)
    MODEL(collate_ilya(exs), T=6, halt=0.0)
    assert (MODEL.last_iterations == 2).all()
    MODEL(collate_ilya(exs), T=6, halt=1.1)
    assert (MODEL.last_iterations == 6).all()


@pytest.mark.parametrize("rope", [False, True])
@torch.no_grad()
def test_jev_prefix_cache_matches_full_forward(rope):
    torch.manual_seed(0)
    jev = JevReplica(len(VOCAB), d=64, ff=128, layers=2, rope=rope).eval()
    group = sample_group(random.Random(2), 5)
    for with_none in (False, True):
        b = collate_jev(group, with_none)
        full = jev(b)
        cache = jev.encode_prefix(torch.tensor([VOCAB.encode(["[BOS]"] + group[0].context)]))
        suf = [VOCAB.encode(jev_suffix(ex.question, with_none)[0]) for ex in group]
        S = max(map(len, suf))
        ids = torch.tensor([s + [VOCAB.pad] * (S - len(s)) for s in suf])
        cached = jev.score_suffixes(cache, ids, ids != VOCAB.pad, torch.tensor([len(s) - 1 for s in suf]),
                                    b["labels"], b["label_mask"])
        assert torch.allclose(full, cached, atol=1e-4)
