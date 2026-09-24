"""Head-to-head evaluation: Ilya vs the Jev-paradigm replica.

    python -m ilya.evaluate --ilya checkpoints/ilya.pt --jev checkpoints/jev.pt \
        --jev_none checkpoints/jev_none.pt --out results

Writes ``results/results.json`` and a human-readable ``results/REPORT.md``.
Every split is generated from a fixed seed that is disjoint from training.
Every model's probabilities are temperature-scaled on its own calibration
split before any metric is computed, so calibration comparisons are fair.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch

from .batching import collate_ilya, collate_jev, jev_suffix
from .coherence import condition, count_equals, iff, violates
from .guarantees import certify_gate, conformal_qhat, fit_temperature
from .jev import JevReplica
from .model import Ilya
from .world import NUMBERS, OOS_KINDS_TEST as OOS_SPLITS, VOCAB, Example, make_bundle, make_split, sample_group


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cls = Ilya if ck["kind"] == "ilya" else JevReplica
    model = cls(**ck["config"])
    model.load_state_dict(ck["state"])
    return ck["kind"], model.eval(), ck


# ----------------------------------------------------------------- inference
@torch.no_grad()
def predict(model, kind, examples, T=16, halt=0.97, bs=256, all_iters=False):
    """Logits in the common layout ``[NONE, cand_1..cand_K]`` padded with -inf.

    Returns (logits [N, 1+Kmax] or [T, N, 1+Kmax] if ``all_iters``, targets, iterations).
    """
    Kmax = max(len(ex.question.cands) for ex in examples)
    outs, iters = [], []
    for i in range(0, len(examples), bs):
        chunk = examples[i:i + bs]
        if kind == "ilya":
            b = collate_ilya(chunk)
            lg = model(b, T=T, halt=None if all_iters else halt)
            lg = lg if all_iters else lg[-1]
            iters += model.last_iterations.tolist()
        else:
            b = collate_jev(chunk, with_none=(kind == "jev_none"), readout=model.readout)
            raw = model(b)
            lg = torch.full((len(chunk), raw.shape[1] + (0 if kind == "jev_none" else 1)), float("-inf"))
            for r, ex in enumerate(chunk):
                K = len(ex.question.cands)
                lg[r, 1:1 + K] = raw[r, :K]
                if kind == "jev_none":
                    lg[r, 0] = raw[r, K]
            iters += [0] * len(chunk)
        pad = Kmax + 1 - lg.shape[-1]
        if pad > 0:
            lg = torch.cat([lg, torch.full((*lg.shape[:-1], pad), float("-inf"))], dim=-1)
        outs.append(lg)
    logits = torch.cat(outs, dim=-2)
    targets = torch.tensor([ex.question.gold + 1 for ex in examples])
    return logits, targets, np.array(iters)


def probs_of(logits, temp):
    return torch.softmax(logits / temp, -1).numpy()


# ------------------------------------------------------------------- metrics
def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def auroc(pos, neg):
    """P(score_pos > score_neg) with ties counted half."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sorted_v = allv[order]
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def summarize(p, y):
    n = len(y)
    pred = p.argmax(1)
    conf = p.max(1)
    correct = pred == y
    pt = np.clip(p[np.arange(n), y], 1e-12, 1)
    onehot = np.zeros_like(p)
    onehot[np.arange(n), y] = 1
    return {"n": int(n), "acc": float(correct.mean()), "nll": float(-np.log(pt).mean()),
            "brier": float(((p - onehot) ** 2).sum(1).mean()), "ece": ece(conf, correct),
            "conf_auroc": auroc(conf[correct], conf[~correct]) if 0 < correct.sum() < n else None}


def r(x, k=4):
    return None if x is None else round(float(x), k)


# --------------------------------------------------------------- experiments
def calibrate(model, kind, cfg):
    cal = make_split("cal", cfg["n_cal"], seed=101)
    if kind == "jev":  # the closed-world model cannot represent NONE
        cal = [ex for ex in cal if ex.question.gold >= 0]
    lg, y, _ = predict(model, kind, cal, T=cfg["T"], halt=cfg["halt"])
    return fit_temperature(lg, y)


def exp_id(model, kind, temp, cfg):
    exs = make_split("id", cfg["n_id"], seed=202)
    lg, y, it = predict(model, kind, exs, T=cfg["T"], halt=cfg["halt"])
    p = probs_of(lg, temp)
    out = {"all": summarize(p, y.numpy()), "mean_iterations": float(it.mean()) if kind == "ilya" else None}
    fams = sorted({ex.question.family for ex in exs})
    out["by_family"] = {}
    for f in fams:
        idx = np.array([ex.question.family == f for ex in exs])
        out["by_family"][f] = summarize(p[idx], y.numpy()[idx])
    drop = np.array([bool(ex.question.meta.get("dropped")) for ex in exs])
    out["answerable"] = summarize(p[~drop], y.numpy()[~drop])
    out["unanswerable"] = summarize(p[drop], y.numpy()[drop])
    return out, (p, y.numpy())


def exp_oos(model, kind, temp, cfg, id_scores):
    res = {}
    id_p, _ = id_scores
    id_ood = id_p[:, 0] if kind != "jev" else 1 - id_p.max(1)
    for split in OOS_SPLITS:
        exs = make_split(split, cfg["n_oos"], seed=303 + len(split))
        lg, y, _ = predict(model, kind, exs, T=cfg["T"], halt=cfg["halt"])
        p = probs_of(lg, temp)
        pred, conf = p.argmax(1), p.max(1)
        ood = p[:, 0] if kind != "jev" else 1 - conf
        res[split] = {"none_rate": float((pred == 0).mean()),
                      "confident_wrong@0.9": float(((pred != 0) & (conf >= 0.9)).mean()),
                      "mean_conf_when_answering": float(conf[pred != 0].mean()) if (pred != 0).any() else None,
                      "ood_auroc": auroc(ood, id_ood)}
    return res


def exp_shuffle(model, kind, temp, cfg):
    rng = random.Random(404)
    base = [ex for ex in make_split("id", cfg["n_id"], seed=202) if ex.question.qtype == "choice"]
    perm_exs, perms = [], []
    for ex in base:
        q = ex.question
        order = list(range(len(q.cands)))
        rng.shuffle(order)
        cands = [q.cands[i] for i in order]
        gold = order.index(q.gold) if q.gold >= 0 else -1
        perm_exs.append(Example(ex.context, type(q)(q.qtype, q.family, q.instr, cands, gold, q.meta)))
        perms.append(order)
    p1 = probs_of(predict(model, kind, base, T=cfg["T"], halt=cfg["halt"])[0], temp)
    p2 = probs_of(predict(model, kind, perm_exs, T=cfg["T"], halt=cfg["halt"])[0], temp)
    flips, tvs = 0, []
    for a, b, order in zip(p1, p2, perms):
        K = len(order)
        back = np.zeros(K + 1)
        back[0] = b[0]
        for new_pos, old in enumerate(order):
            back[1 + old] = b[1 + new_pos]
        flips += int(a[:K + 1].argmax() != back.argmax())
        tvs.append(0.5 * np.abs(a[:K + 1] - back).sum())
    return {"n": len(base), "flip_rate": flips / len(base), "mean_tv": float(np.mean(tvs)),
            "max_tv": float(np.max(tvs))}


def exp_misleading(model, kind, temp, cfg):
    exs = make_split("misleading", cfg["n_mis"], seed=505)
    honest = []
    for ex in exs:  # same questions, but every option is named after its own value
        q = ex.question
        cands = [type(c)(c.value, [c.value], c.rubric, c.ordinal) for c in q.cands]
        honest.append(Example(ex.context, type(q)(q.qtype, q.family, q.instr, cands, q.gold, q.meta)))
    lg1, y, _ = predict(model, kind, honest, T=cfg["T"], halt=cfg["halt"])
    lg2, _, _ = predict(model, kind, exs, T=cfg["T"], halt=cfg["halt"])
    s1, s2 = summarize(probs_of(lg1, temp), y.numpy()), summarize(probs_of(lg2, temp), y.numpy())
    pred2 = probs_of(lg2, temp).argmax(1)
    # How often does the model pick the option whose *name* is the true value?
    name_hits = np.mean([ex.question.cands[k - 1].name == [ex.question.cands[ex.question.gold].value]
                         if k > 0 else False for ex, k in zip(exs, pred2)])
    return {"honest_names_acc": s1["acc"], "misleading_names_acc": s2["acc"],
            "drop": s1["acc"] - s2["acc"], "followed_name_rate": float(name_hits)}


def exp_long(model, kind, temp, cfg):
    exs = make_split("long", cfg["n_long"], seed=606)
    hops = np.array([ex.question.meta["hops"] for ex in exs])
    y = np.array([ex.question.gold + 1 for ex in exs])
    res = {}
    settings = ([("T=%d" % t, t, None) for t in cfg["long_T"]] + [("anytime", cfg["T_long"], cfg["halt"])]
                if kind == "ilya" else [("fixed", None, None)])
    for name, T, halt in settings:
        lg, _, it = predict(model, kind, exs, T=T or 1, halt=halt)
        pred = probs_of(lg, temp).argmax(1)
        res[name] = {"acc": float((pred == y).mean()),
                     "by_hops": {int(h): float((pred[hops == h] == y[hops == h]).mean()) for h in sorted(set(hops))}}
        if kind == "ilya":
            res[name]["mean_iterations"] = float(it.mean())
            res[name]["iterations_by_hops"] = {int(h): float(it[hops == h].mean()) for h in sorted(set(hops))}
    return res


def exp_iterations(model, cfg):
    exs = make_split("id", cfg["n_id"], seed=202)
    lg, y, _ = predict(model, "ilya", exs, T=cfg["T"], all_iters=True)
    return [{"t": t + 1, **{k: summarize(torch.softmax(lg[t], -1).numpy(), y.numpy())[k]
                            for k in ("acc", "nll", "ece")}} for t in range(lg.shape[0])]


def exp_coherence(model, kind, temp, cfg):
    rng = random.Random(707)
    res = {}
    for bkind in ("color", "count"):
        stats = {"indep_violation": 0, "cond_violation": 0, "indep_acc": [], "cond_acc": [],
                 "indep_nll": [], "cond_nll": [], "support": []}
        bundles = [make_bundle(rng, bkind) for _ in range(cfg["n_bundles"])]
        flat = [Example(scene.context, q) for scene, qs, _ in bundles for q in qs]
        p_all = probs_of(predict(model, kind, flat, T=cfg["T"], halt=cfg["halt"])[0], temp)
        pos = 0
        for scene, qs, links in bundles:
            beliefs, gold = {}, {}
            for i, q in enumerate(qs):
                labels = ["NONE"] + [c.value for c in q.cands]
                pr = p_all[pos + i][:len(labels)]
                keep = [k for k in range(len(labels)) if not (kind == "jev" and k == 0)]
                beliefs[f"q{i}"] = {labels[k]: float(pr[k]) for k in keep}
                gold[f"q{i}"] = labels[q.gold + 1]
            pos += len(qs)
            cons = []
            for link in links:
                if link[0] == "iff_value":
                    cons.append(iff(f"q{link[1]}", link[2], f"q{link[3]}", "true"))
                else:
                    cons.append(count_equals(f"q{link[1]}", [f"q{j}" for j in link[2]],
                                             lambda s: NUMBERS.index(s) if s in NUMBERS else None))
            indep = {q: max(b, key=b.get) for q, b in beliefs.items()}
            out = condition(beliefs, cons)
            stats["indep_violation"] += violates(indep, cons)
            stats["cond_violation"] += out["map"]["q0"] is None or violates(out["map"], cons)
            stats["support"].append(out["support"]["q0"])
            for q in beliefs:
                stats["indep_acc"].append(indep[q] == gold[q])
                stats["cond_acc"].append(out["map"][q] == gold[q])
                stats["indep_nll"].append(-np.log(max(beliefs[q].get(gold[q], 0.0), 1e-12)))
                stats["cond_nll"].append(-np.log(max(out["marginals"][q].get(gold[q], 0.0), 1e-12)))
        n = len(bundles)
        res[bkind] = {"bundles": n,
                      "independent_violation_rate": stats["indep_violation"] / n,
                      "conditioned_violation_rate": stats["cond_violation"] / n,
                      "independent_acc": float(np.mean(stats["indep_acc"])),
                      "conditioned_acc": float(np.mean(stats["cond_acc"])),
                      "independent_nll": float(np.mean(stats["indep_nll"])),
                      "conditioned_nll": float(np.mean(stats["cond_nll"])),
                      "mean_support": float(np.mean(stats["support"]))}
    return res


def deployment_pool(cfg):
    """A deployment mix: 70% in-distribution, 30% spread over five kinds of
    out-of-scope traffic (two of which never appear in any training data)."""
    n = cfg["n_pool"]
    pool = make_split("id", int(0.7 * n), seed=808)
    for i, s in enumerate(OOS_SPLITS):
        pool += make_split(s, int(0.06 * n), seed=909 + i)
    return pool


def exp_guarantees(model, kind, temp, cfg, eps=0.05, delta=0.1, alpha=0.05, splits=200):
    pool = deployment_pool(cfg)
    lg, y, _ = predict(model, kind, pool, T=cfg["T"], halt=cfg["halt"])
    p = probs_of(lg, temp)
    valid = torch.isfinite(lg).numpy()  # outcomes the model can express (padding / NONE for closed models excluded)
    y = y.numpy()
    conf, correct = p.max(1), p.argmax(1) == y
    rng = np.random.default_rng(0)
    cov, risk, viol, none_cert, set_cov, set_size = [], [], 0, 0, [], []
    for _ in range(splits):
        perm = rng.permutation(len(y))
        c, t = perm[:len(y) // 2], perm[len(y) // 2:]
        lam = certify_gate(conf[c], correct[c], eps, delta)
        if lam is None:
            none_cert += 1
            cov.append(0.0)
        else:
            acc = conf[t] >= lam
            cov.append(float(acc.mean()))
            rk = float((~correct[t][acc]).mean()) if acc.any() else 0.0
            risk.append(rk)
            viol += rk > eps
        qh = conformal_qhat(p[c], y[c], alpha)
        sets = (p[t] >= 1 - qh) & valid[t]
        set_cov.append(float(sets[np.arange(len(t)), y[t]].mean()))
        set_size.append(float(sets.sum(1).mean()))
    return {"eps": eps, "delta": delta, "alpha": alpha, "pool": len(y),
            "raw_acc_on_pool": float(correct.mean()),
            "certified_coverage": float(np.mean(cov)),
            "test_selective_risk": float(np.mean(risk)) if risk else None,
            "risk_violation_freq": viol / max(1, splits - none_cert),
            "no_certificate_freq": none_cert / splits,
            "conformal_coverage": float(np.mean(set_cov)), "conformal_set_size": float(np.mean(set_size))}


@torch.no_grad()
def exp_speed(model, kind, cfg, M=32, reps=7):
    rng = random.Random(1111)
    groups = [sample_group(rng, M) for _ in range(reps + 2)]
    times, iters = [], []
    for g in groups:
        t0 = time.perf_counter()
        if kind == "ilya":
            model(collate_ilya(g), T=cfg["T"], halt=cfg["halt"])
            iters.append(float(model.last_iterations.float().mean()))
        else:
            with_none = kind == "jev_none"
            prefix = torch.tensor([VOCAB.encode(["[BOS]"] + g[0].context)])
            cache = model.encode_prefix(prefix)
            b = collate_jev(g, with_none, readout=model.readout)
            suf = [VOCAB.encode(jev_suffix(ex.question, with_none, model.readout)[0]) for ex in g]
            S = max(map(len, suf))
            ids = torch.tensor([s + [VOCAB.pad] * (S - len(s)) for s in suf])
            model.score_suffixes(cache, ids, ids != VOCAB.pad, torch.tensor([len(s) - 1 for s in suf]),
                                 b["labels"], b["label_mask"], prefix)
        times.append(time.perf_counter() - t0)
    ms = float(np.median(times[2:]) * 1000)
    return {"questions_per_call": M, "ms_per_call": ms, "decisions_per_sec": M / ms * 1000,
            "mean_iterations": float(np.mean(iters)) if iters else None}


def run(paths, cfg):
    results = {"config": cfg, "models": {}}
    for name, path in paths.items():
        if not path:
            continue
        kind, model, ck = load(path)
        t0 = time.time()
        if kind == "ilya" and cfg["T"] is None:  # evaluate inside the trained loop range
            cfg = dict(cfg, T=ck["args"]["t_max"])
        temp = calibrate(model, kind, cfg)
        res = {"params": ck["params"], "train_steps": ck["args"]["steps"], "temperature": temp,
               "max_iterations": cfg["T"] if kind == "ilya" else None, "train_log": ck.get("log", [])}
        res["id"], id_scores = exp_id(model, kind, temp, cfg)
        res["oos"] = exp_oos(model, kind, temp, cfg, id_scores)
        res["shuffle"] = exp_shuffle(model, kind, temp, cfg)
        res["misleading"] = exp_misleading(model, kind, temp, cfg)
        res["long"] = exp_long(model, kind, temp, cfg)
        res["coherence"] = exp_coherence(model, kind, temp, cfg)
        res["guarantees"] = exp_guarantees(model, kind, temp, cfg)
        res["speed"] = exp_speed(model, kind, cfg)
        if kind == "ilya":
            res["iterations"] = exp_iterations(model, cfg)
        res["eval_minutes"] = (time.time() - t0) / 60
        results["models"][name] = res
        print(f"[{name}] done in {res['eval_minutes']:.1f} min", flush=True)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ilya")
    ap.add_argument("--jev")
    ap.add_argument("--jev_none")
    ap.add_argument("--out", default="results")
    ap.add_argument("--scale", type=float, default=1.0, help="shrink every split (for smoke runs)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--readme", help="also refresh the results block of this README")
    ap.add_argument("--merge", action="store_true", help="merge into an existing results.json")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    scale = args.scale
    cfg = {"T": None, "halt": 0.97, "T_long": 32, "long_T": [2, 4, 8, 16, 24, 32],
           "n_cal": int(3000 * scale), "n_id": int(3000 * scale), "n_oos": int(1000 * scale),
           "n_mis": int(1500 * scale), "n_long": int(2800 * scale), "n_bundles": int(500 * scale),
           "n_pool": int(4000 * scale)}
    results = run({"ilya": args.ilya, "jev": args.jev, "jev_none": args.jev_none}, cfg)
    os.makedirs(args.out, exist_ok=True)
    prev = os.path.join(args.out, "results.json")
    if args.merge and os.path.exists(prev):  # keep models evaluated in an earlier call
        with open(prev) as f:
            old = json.load(f)
        results["models"] = {**old["models"], **results["models"]}
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    from .report import update_readme, write_report
    write_report(results, os.path.join(args.out, "REPORT.md"))
    if args.readme:
        update_readme(results, args.readme)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
