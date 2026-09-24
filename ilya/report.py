"""Render ``results.json`` as a markdown report."""
from __future__ import annotations

NAMES = {"ilya": "Ilya", "jev": "Jev-replica (closed world)", "jev_none": "Jev-replica + explicit NOTA option"}


def pct(x):
    return "–" if x is None else f"{100 * x:.1f}%"


def num(x, k=3):
    return "–" if x is None else f"{x:.{k}f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def write_report(results, path):
    M = results["models"]
    names = [n for n in ("ilya", "jev", "jev_none") if n in M]
    L = ["# Ilya vs Jev-paradigm — head-to-head report", "",
         "All numbers are produced by `python -m ilya.evaluate`; every split is seeded and disjoint "
         "from the training stream. Probabilities are temperature-scaled on each model's own "
         "calibration split before any metric is computed.", ""]

    L += ["## Budget", "", table(["model", "params", "train steps", "temperature", "ms / call (32 q)", "decisions / s", "mean loop iters"],
          [[NAMES[n], f"{M[n]['params']:,}", M[n]["train_steps"], num(M[n]["temperature"], 2),
            num(M[n]["speed"]["ms_per_call"], 1), num(M[n]["speed"]["decisions_per_sec"], 0),
            num(M[n]["speed"]["mean_iterations"], 1)] for n in names]), ""]

    steps = sorted({r["step"] for n in names for r in M[n].get("train_log", [])})
    if steps:
        curve = {n: {r["step"]: r["val_acc"] for r in M[n].get("train_log", [])} for n in names}
        L += ["## Learning curves (validation accuracy during training, same data stream)", "",
              table(["model"] + [str(s_) for s_ in steps],
                    [[NAMES[n]] + [pct(curve[n].get(s_)) for s_ in steps] for n in names]), ""]

    L += ["## 1. In-distribution decisions", "", table(
        ["model", "acc", "NLL", "Brier", "ECE", "conf AUROC", "acc (answerable)", "NLL (unanswerable)"],
        [[NAMES[n], pct(M[n]["id"]["all"]["acc"]), num(M[n]["id"]["all"]["nll"]), num(M[n]["id"]["all"]["brier"]),
          num(M[n]["id"]["all"]["ece"]), num(M[n]["id"]["all"]["conf_auroc"]),
          pct(M[n]["id"]["answerable"]["acc"]), num(M[n]["id"]["unanswerable"]["nll"])] for n in names]), ""]
    fams = list(M[names[0]]["id"]["by_family"])
    L += ["Accuracy by question family:", "", table(["model"] + fams,
          [[NAMES[n]] + [pct(M[n]["id"]["by_family"][f]["acc"]) for f in fams] for n in names]), ""]

    oos = list(M[names[0]]["oos"])
    L += ["## 2. Open world: out-of-scope inputs (correct answer = NONE)", "",
          "`weather` and `soup` never appear in any training data.", "",
          table(["model"] + [f"{s[4:]} NONE-rate" for s in oos],
                [[NAMES[n]] + [pct(M[n]["oos"][s]["none_rate"]) for s in oos] for n in names]), "",
          table(["model"] + [f"{s[4:]} confident-wrong" for s in oos],
                [[NAMES[n]] + [pct(M[n]["oos"][s]["confident_wrong@0.9"]) for s in oos] for n in names]), "",
          table(["model"] + [f"{s[4:]} OOD AUROC" for s in oos],
                [[NAMES[n]] + [num(M[n]["oos"][s]["ood_auroc"]) for s in oos] for n in names]), ""]

    L += ["## 3. Presentation invariance", "", table(
        ["model", "option-shuffle flip rate", "mean TV shift", "max TV shift",
         "acc honest names", "acc misleading names", "picked-by-name rate"],
        [[NAMES[n], pct(M[n]["shuffle"]["flip_rate"]), num(M[n]["shuffle"]["mean_tv"], 4), num(M[n]["shuffle"]["max_tv"], 4),
          pct(M[n]["misleading"]["honest_names_acc"]), pct(M[n]["misleading"]["misleading_names_acc"]),
          pct(M[n]["misleading"]["followed_name_rate"])] for n in names]), ""]

    L += ["## 4. Multi-hop depth (7–8 objects; training used ≤ 6 objects)", ""]
    rows = []
    for n in names:
        for setting, v in M[n]["long"].items():
            rows.append([f"{NAMES[n]} [{setting}]", pct(v["acc"])] +
                        [pct(v["by_hops"].get(h, v["by_hops"].get(str(h)))) for h in range(1, 8)] +
                        [num(v.get("mean_iterations"), 1)])
    L += [table(["model", "overall"] + [f"{h} hops" for h in range(1, 8)] + ["iters"], rows), ""]

    if "iterations" in M.get("ilya", {}):
        it = M["ilya"]["iterations"]
        L += ["## 5. Anytime beliefs (Ilya, in-distribution, no temperature)", "",
              table(["iteration"] + [str(x["t"]) for x in it],
                    [["acc"] + [pct(x["acc"]) for x in it], ["NLL"] + [num(x["nll"]) for x in it],
                     ["ECE"] + [num(x["ece"]) for x in it]]), ""]

    L += ["## 6. Coherence under declared constraints", ""]
    rows = []
    for n in names:
        for b, v in M[n]["coherence"].items():
            rows.append([NAMES[n], b, pct(v["independent_violation_rate"]), pct(v["conditioned_violation_rate"]),
                         pct(v["independent_acc"]), pct(v["conditioned_acc"]),
                         num(v["independent_nll"]), num(v["conditioned_nll"]), num(v["mean_support"])])
    L += [table(["model", "bundle", "violations (isolated)", "violations (conditioned)", "acc (isolated)",
                 "acc (conditioned)", "NLL (isolated)", "NLL (conditioned)", "mean support"], rows), ""]

    g0 = M[names[0]]["guarantees"]
    L += ["## 7. Certified decisions on a deployment mix (70% ID + 30% out-of-scope)", "",
          f"Learn-then-Test gate with ε = {g0['eps']}, δ = {g0['delta']}; conformal sets with α = {g0['alpha']}; "
          "200 random calibration/test splits.", "",
          table(["model", "raw acc", "certified coverage", "test selective risk", "risk > ε freq",
                 "no-certificate freq", "conformal coverage", "mean set size"],
                [[NAMES[n], pct(M[n]["guarantees"]["raw_acc_on_pool"]), pct(M[n]["guarantees"]["certified_coverage"]),
                  pct(M[n]["guarantees"]["test_selective_risk"]), pct(M[n]["guarantees"]["risk_violation_freq"]),
                  pct(M[n]["guarantees"]["no_certificate_freq"]), pct(M[n]["guarantees"]["conformal_coverage"]),
                  num(M[n]["guarantees"]["conformal_set_size"], 2)] for n in names]), ""]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def summary_rows(M):
    """Headline rows (label, key-fn) for the README summary."""
    oos = ["oos_absent", "oos_missing", "oos_recipe", "oos_weather", "oos_soup"]
    novel = ["oos_weather", "oos_soup"]

    def long_overall(v):
        long = v["long"]
        return long["anytime" if "anytime" in long else "fixed"]["acc"]

    return [
        ("分布内准确率", lambda v: pct(v["id"]["all"]["acc"])),
        ("分布内 ECE（越低越好）", lambda v: num(v["id"]["all"]["ece"])),
        ("越界输入判为 NONE 的比例（5 类平均）", lambda v: pct(_mean([v["oos"][s]["none_rate"] for s in oos]))),
        ("越界输入上的高置信错误（≥0.9，5 类平均）", lambda v: pct(_mean([v["oos"][s]["confident_wrong@0.9"] for s in oos]))),
        ("全新越界类型检测 AUROC（训练中从未出现）", lambda v: num(_mean([v["oos"][s]["ood_auroc"] for s in novel]))),
        ("选项洗牌后判决翻转率", lambda v: pct(v["shuffle"]["flip_rate"])),
        ("误导性选项名下的准确率", lambda v: pct(v["misleading"]["misleading_names_acc"])),
        ("按名称而非 rubric 作答的比例", lambda v: pct(v["misleading"]["followed_name_rate"])),
        ("约束违反率：隔离判决 → 条件化后（颜色束）",
         lambda v: f"{pct(v['coherence']['color']['independent_violation_rate'])} → {pct(v['coherence']['color']['conditioned_violation_rate'])}"),
        ("约束违反率：隔离判决 → 条件化后（计数束）",
         lambda v: f"{pct(v['coherence']['count']['independent_violation_rate'])} → {pct(v['coherence']['count']['conditioned_violation_rate'])}"),
        ("认证覆盖率（部署混合，风险 ≤5% @ 90%）", lambda v: pct(v["guarantees"]["certified_coverage"])),
        ("长链泛化：7–8 个物体的关系题（训练最多 6 个）", lambda v: pct(long_overall(v))),
        ("吞吐（每次调用 32 题，决策/秒，CPU）", lambda v: num(v["speed"]["decisions_per_sec"], 0)),
    ]


def update_readme(results, path):
    M = results["models"]
    names = [n for n in ("ilya", "jev", "jev_none") if n in M]
    cn = {"ilya": "Ilya", "jev": "Jev 复刻（封闭世界）", "jev_none": "Jev 复刻 + 显式 NOTA 选项"}
    head = ["指标"] + [cn[n] for n in names]
    rows = [[label] + [fn(M[n]) for n in names] for label, fn in summary_rows(M)]
    block = ("<!-- RESULTS -->\n### 结果摘要（自动生成，详见 [`results/REPORT.md`](results/REPORT.md)）\n\n"
             + table(head, rows) + "\n<!-- /RESULTS -->")
    s = open(path).read()
    a, b = s.index("<!-- RESULTS -->"), s.index("<!-- /RESULTS -->") + len("<!-- /RESULTS -->")
    with open(path, "w") as f:
        f.write(s[:a] + block + s[b:])
