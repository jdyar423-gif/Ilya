"""Distribution-free guarantees on top of calibrated beliefs.

* ``fit_temperature`` - post-hoc temperature scaling (applied to every model).
* ``conformal_qhat`` / ``prediction_sets`` - split-conformal prediction sets
  whose marginal coverage is at least ``1 - alpha`` on exchangeable data.
* ``certify_gate`` - a Learn-then-Test selective gate: the returned confidence
  threshold ``lam`` satisfies, with probability at least ``1 - delta`` over
  the calibration draw, ``P(wrong | confidence >= lam) <= eps``. A hosted
  System One model offers a confidence score; this turns it into a contract.
* ``bayes_action`` - cost-aware decision with an explicit escalate action.
"""
from __future__ import annotations

import math

import numpy as np
import torch


def fit_temperature(logits, targets, iters=200):
    """Temperature minimising NLL. ``logits`` [N, K] (may contain -inf), ``targets`` [N]."""
    logits = torch.as_tensor(logits, dtype=torch.float)
    logits = logits.masked_fill(torch.isinf(logits), -1e4)  # -inf padding would give NaN gradients
    targets = torch.as_tensor(targets)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=iters)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / log_t.exp(), targets)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp().clamp(0.05, 20.0))


def conformal_qhat(probs, targets, alpha):
    """Split-conformal threshold on the score 1 - p(true outcome)."""
    probs, targets = np.asarray(probs), np.asarray(targets)
    n = len(targets)
    scores = 1.0 - probs[np.arange(n), targets]
    level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def prediction_sets(probs, qhat):
    return np.asarray(probs) >= 1.0 - qhat


def _log_binom_cdf(k, n, p):
    """log P(Binomial(n, p) <= k), computed stably."""
    if k >= n:
        return 0.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return -math.inf
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
             + i * math.log(p) + (n - i) * math.log1p(-p) for i in range(k + 1)]
    m = max(terms)
    return m + math.log(sum(math.exp(t - m) for t in terms))


DEFAULT_GRID = (0.9999, 0.999, 0.995, 0.99, 0.98, 0.97, 0.95, 0.93, 0.9, 0.85,
                0.8, 0.75, 0.7, 0.6, 0.5)


def certify_gate(confidence, correct, eps, delta, grid=DEFAULT_GRID):
    """Learn-then-Test with a Bonferroni correction over a fixed grid.

    For each threshold the null "selective risk > eps" is rejected when the
    binomial p-value of the observed errors among accepted calibration items
    is below ``delta / len(grid)``. Returns the smallest certified threshold
    (maximum coverage) or ``None`` if no threshold can be certified.
    """
    conf = np.asarray(confidence, dtype=float)
    ok = np.asarray(correct, dtype=bool)
    certified = []
    for lam in grid:
        acc = conf >= lam
        n = int(acc.sum())
        if n == 0:
            continue
        k = int((~ok[acc]).sum())
        if math.exp(_log_binom_cdf(k, n, eps)) <= delta / len(grid):
            certified.append(lam)
    return min(certified) if certified else None


def bayes_action(probs, cost, escalate_cost):
    """Pick the action minimising expected cost.

    ``cost[a][o]`` is the cost of answering ``a`` when the truth is ``o``;
    returns ``-1`` when escalating (fixed cost ``escalate_cost``) is cheaper.
    """
    expected = np.asarray(cost, dtype=float) @ np.asarray(probs, dtype=float)
    a = int(np.argmin(expected))
    return -1 if escalate_cost < expected[a] else a
