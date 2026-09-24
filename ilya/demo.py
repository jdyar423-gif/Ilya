"""Side-by-side demo of the failure modes Jev is documented to have.

    python -m ilya.demo --ckpt checkpoints/ilya.pt
"""
from __future__ import annotations

import argparse
import json

from .api import Engine
from .coherence import iff

SCENE = ("alpha red cube large . beta blue ball tiny . gamma green cone small . "
         "alpha left beta . beta left gamma .")


def show(title, out):
    print(f"\n=== {title}")
    print(json.dumps(out["answers"], indent=1, default=lambda x: round(x, 4)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/ilya.pt")
    args = ap.parse_args(argv)
    eng = Engine.load(args.ckpt)

    show("1. an ordinary request: one context, several typed questions", eng.decide(SCENE, {
        "color": {"type": "choice", "instructions": "what color is alpha ?",
                  "criteria": {"red": None, "blue": None, "green": None}},
        "size": {"type": "score", "instructions": "how big is beta ?",
                 "criteria": ["tiny", "small", "medium", "large", "huge"]},
        "order": {"type": "noul", "instructions": "is alpha left of gamma ?"}}))

    show("2. out of scope: a cake recipe, and a question about an object that does not exist",
         {"answers": {**eng.decide("flour sugar egg . bake oven heat . stir butter milk .", {
             "recipe": {"type": "choice", "instructions": "what color is alpha ?",
                        "criteria": {"red": None, "blue": None}}})["answers"],
             **eng.decide(SCENE, {"ghost": {"type": "choice", "instructions": "what shape is zeta ?",
                                            "criteria": {"cube": None, "ball": None}}})["answers"]}})

    show("3. misleading option names: the rubric is the contract, the name is only an id",
         eng.decide(SCENE, {"color": {"type": "choice", "instructions": "what color is gamma ?",
                                      "criteria": {"red": "emerald", "green": "scarlet", "blue": "cobalt"}}}))

    show("4. declared logic is enforced exactly", eng.decide(SCENE, {
        "color": {"type": "choice", "instructions": "what color is beta ?",
                  "criteria": {"red": None, "blue": None}},
        "is_blue": {"type": "noul", "instructions": "is beta blue ?"}},
        constraints=[iff("color", "blue", "is_blue", "true")]))


if __name__ == "__main__":
    main()
