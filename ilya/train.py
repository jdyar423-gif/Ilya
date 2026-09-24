"""Train Ilya or the Jev-paradigm replica on the synthetic world.

    python -m ilya.train --model ilya     --steps 8000 --out checkpoints/ilya.pt
    python -m ilya.train --model jev      --steps 8000 --out checkpoints/jev.pt
    python -m ilya.train --model jev_none --steps 8000 --out checkpoints/jev_none.pt

Both families see the same generator, batch size, optimiser, schedule and
number of steps. The closed-world ``jev`` model is trained on in-scope data
only (it has no way to express NONE); ``jev_none`` always receives an explicit
"none of these" option and is trained on the same out-of-scope data as Ilya.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time

import torch
import torch.nn.functional as F

from .batching import collate_ilya, collate_jev
from .jev import JevReplica
from .model import Ilya
from .world import VOCAB, make_split, sample_group

P_OOS = 0.2


def build(kind, **kw):
    if kind == "ilya":
        return Ilya(len(VOCAB), **kw)
    return JevReplica(len(VOCAB), **kw)


def iteration_weights(T):
    w = torch.arange(1, T + 1, dtype=torch.float)
    return w / w.sum()


def ilya_loss(model, batch, T):
    logits = model(batch, T=T)                                   # [T, B, 1+K]
    tgt = batch["target"].expand(T, -1).reshape(-1)
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt, reduction="none").view(T, -1)
    return (ce.mean(1) * iteration_weights(T)).sum(), logits[-1]


def jev_loss(model, batch):
    logits = model(batch)
    return F.cross_entropy(logits, batch["target"], ignore_index=-100), logits


@torch.no_grad()
def quick_eval(model, kind, examples, T=8):
    model.eval()
    correct = 0
    for i in range(0, len(examples), 256):
        chunk = examples[i:i + 256]
        if kind == "ilya":
            b = collate_ilya(chunk)
            pred = model(b, T=T)[-1].argmax(-1)
        else:
            b = collate_jev(chunk, with_none=(kind == "jev_none"), readout=model.readout)
            pred = model(b).argmax(-1)
        correct += (pred == b["target"]).sum().item()
    model.train()
    return correct / len(examples)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ilya", "jev", "jev_none"], required=True)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--group", type=int, default=4, help="questions per context")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4, help="Jev replica depth")
    ap.add_argument("--t_min", type=int, default=6, help="Ilya loop iterations (sampled per step)")
    ap.add_argument("--t_max", type=int, default=10)
    ap.add_argument("--rope", action="store_true", help="rotary positions (both model families)")
    ap.add_argument("--ident", action="store_true", help="identity-aware attention (both model families)")
    ap.add_argument("--readout", choices=["name", "label"], default="name",
                    help="Jev replica: read option-name tokens or positional letter labels")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--curriculum", type=float, default=0.5,
                    help="fraction of training over which rich option names, out-of-scope data and "
                         "longer loops are ramped in (identical for every model); 0 disables it")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    rng = random.Random(1000 + args.seed)
    kw = dict(d=args.d, ff=4 * args.d, rope=args.rope, ident=args.ident)
    if args.model != "ilya":
        kw.update(layers=args.layers, readout=args.readout)
    model = build(args.model, **kw)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.model}: {n_params:,} parameters", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup) *
                                              (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps)))))
    val = make_split("id", 512, seed=77)
    p_oos = 0.0 if args.model == "jev" else P_OOS
    log, t0, run = [], time.time(), 0.0
    def ramp(start, width):  # 0 -> 1 between (start) and (start + width) of the curriculum span
        if args.curriculum <= 0:
            return 1.0
        return min(1.0, max(0.0, (step / args.steps - start * args.curriculum) / (width * args.curriculum)))

    for step in range(1, args.steps + 1):
        # Curriculum: first plain option names, in-scope data and short loops;
        # then ramp in rubric/code names, out-of-scope data and longer loops.
        rich, oos_scale, loop_scale = ramp(0.2, 0.4), ramp(0.5, 0.5), ramp(0.5, 0.5)
        exs = [ex for _ in range(args.batch // args.group)
               for ex in sample_group(rng, args.group, p_oos=p_oos * oos_scale, p_rich_names=rich)]
        if args.model == "ilya":
            t_hi = args.t_min + round(loop_scale * (args.t_max - args.t_min))
            loss, _ = ilya_loss(model, collate_ilya(exs), rng.randint(args.t_min, t_hi))
        else:
            loss, _ = jev_loss(model, collate_jev(exs, with_none=(args.model == "jev_none"),
                                                  readout=args.readout))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % args.eval_every == 0 or step == args.steps:
            acc = quick_eval(model, args.model, val)
            rec = {"step": step, "loss": round(run, 4), "val_acc": round(acc, 4),
                   "minutes": round((time.time() - t0) / 60, 2)}
            log.append(rec)
            print(json.dumps(rec), flush=True)
            torch.save({"kind": args.model, "config": model.config, "state": model.state_dict(),
                        "args": vars(args), "log": log, "params": n_params}, args.out)


if __name__ == "__main__":
    main()
