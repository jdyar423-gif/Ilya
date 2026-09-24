"""Calibrate a trained Ilya checkpoint for the API.

    python -m ilya.calibrate --ckpt checkpoints/ilya.pt

Fits the temperature, a split-conformal threshold and a Learn-then-Test gate
on fresh data (seeds disjoint from every evaluation split) and stores them in
the checkpoint, so :class:`ilya.api.Engine` answers carry ``set`` and ``accept``.
"""
from __future__ import annotations

import argparse

import torch

from .evaluate import OOS_SPLITS, load, predict, probs_of
from .guarantees import certify_gate, conformal_qhat, fit_temperature
from .world import make_split


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/ilya.pt")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args(argv)
    kind, model, ck = load(args.ckpt)
    assert kind == "ilya", "the API serves Ilya checkpoints"
    T = ck["args"]["t_max"]
    pool = make_split("id", int(0.7 * args.n), seed=5001)
    for i, s in enumerate(OOS_SPLITS):
        pool += make_split(s, int(0.06 * args.n), seed=5101 + i)
    lg, y, _ = predict(model, kind, pool, T=T, halt=0.97)
    temp = fit_temperature(lg, y)
    p = probs_of(lg, temp)
    conf, correct = p.max(1), p.argmax(1) == y.numpy()
    cal = {"temperature": temp, "qhat": conformal_qhat(p, y.numpy(), args.alpha),
           "gate": certify_gate(conf, correct, args.eps, args.delta),
           "eps": args.eps, "delta": args.delta, "alpha": args.alpha, "n": len(pool)}
    ck["calibration"] = cal
    torch.save(ck, args.ckpt)
    print(cal)


if __name__ == "__main__":
    main()
