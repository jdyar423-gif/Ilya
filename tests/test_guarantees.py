import numpy as np

from ilya.guarantees import bayes_action, certify_gate, conformal_qhat, fit_temperature


def test_certified_gate_controls_true_selective_risk():
    # Calibrated confidences c ~ U(0.5, 1), correct ~ Bernoulli(c):
    # the true selective risk at threshold lam is (1 - lam) / 2.
    rng = np.random.default_rng(0)
    eps, delta, trials, bad, certified = 0.1, 0.1, 300, 0, 0
    for _ in range(trials):
        c = rng.uniform(0.5, 1.0, 3000)
        ok = rng.random(3000) < c
        lam = certify_gate(c, ok, eps, delta)
        if lam is not None:
            certified += 1
            bad += (1 - lam) / 2 > eps
    assert certified > trials * 0.9
    assert bad / trials <= delta


def test_conformal_sets_cover():
    rng = np.random.default_rng(1)
    K, n = 4, 4000
    logits = rng.normal(size=(n, K)) * 2
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = np.array([rng.choice(K, p=row) for row in p])
    qhat = conformal_qhat(p[:2000], y[:2000], alpha=0.1)
    sets = p[2000:] >= 1 - qhat
    assert sets[np.arange(2000), y[2000:]].mean() >= 0.88


def test_temperature_recovers_overconfidence():
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(3000, 3)) * 1.5
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = np.array([rng.choice(3, p=row) for row in p])
    t = fit_temperature(logits * 3.0, y)
    assert 2.5 < t < 3.6


def test_bayes_action_escalates_when_unsure():
    cost = [[0, 10], [10, 0]]
    assert bayes_action([0.95, 0.05], cost, escalate_cost=2) == 0
    assert bayes_action([0.5, 0.5], cost, escalate_cost=2) == -1
