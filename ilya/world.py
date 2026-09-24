"""A synthetic, fully-observable-to-the-generator world for typed decisions.

Every context describes N objects standing on a line. Each object has a colour,
a shape and a size; adjacent objects are linked by ``A left B .`` sentences that
are shuffled, so answering "is X left of Y ?" for distant objects requires
composing several facts (multi-hop). Some attribute facts are deliberately
dropped from the context, so a calibrated model must sometimes be unsure.

Because the generator knows the ground truth, every property we care about
(accuracy, calibration, open-world abstention, order invariance, name/rubric
binding, coherence under constraints, multi-hop depth) is measurable exactly.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

NAMES = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
         "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi"]
COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "black", "white"]
COLOR_SYN = {"red": ["crimson", "scarlet"], "blue": ["azure", "cobalt"],
             "green": ["emerald", "jade"], "yellow": ["golden", "lemon"],
             "purple": ["violet", "lilac"], "orange": ["amber", "tangerine"],
             "black": ["ebony", "jet"], "white": ["ivory", "snow"]}
SHAPES = ["cube", "ball", "cone", "ring", "pyramid", "cylinder"]
SHAPE_SYN = {"cube": ["box", "block"], "ball": ["sphere", "orb"],
             "cone": ["funnel", "spike"], "ring": ["hoop", "loop"],
             "pyramid": ["tetra", "wedge"], "cylinder": ["tube", "pipe"]}
SIZES = ["tiny", "small", "medium", "large", "huge"]
SIZE_SYN = {"tiny": ["minuscule"], "small": ["little"], "medium": ["average"],
            "large": ["big"], "huge": ["giant"]}
NUMBERS = ["zero", "one", "two", "three", "four", "five", "six"]
CODES = [f"c{i}" for i in range(12)]
# Out-of-scope vocabularies. RECIPE is used for outlier exposure during training;
# WEATHER never appears in training and probes genuinely novel inputs.
RECIPE = ["flour", "sugar", "egg", "bake", "oven", "stir", "butter", "milk",
          "whisk", "dough", "salt", "cream", "cake", "pan", "heat", "mix"]
WEATHER = ["rain", "cloud", "wind", "storm", "sunny", "fog", "snowfall",
           "thunder", "humid", "breeze", "drizzle", "frost"]
FUNC = ["is", "left", "of", "what", "color", "shape", "how", "big", "many", "?",
        ".", "true", "false"]
LETTERS = [chr(ord("A") + i) for i in range(10)]
DIGITS = [str(i) for i in range(10)]
SPECIAL = ["[PAD]", "[UNK]", "[BOS]", "[SEP]", "[ANS]", "[NOTA]", ":", ";"]

SYNONYMS = {**COLOR_SYN, **SHAPE_SYN, **SIZE_SYN}
FAMILY_VALUES = {"color": COLORS, "shape": SHAPES}


class Vocab:
    def __init__(self):
        words = (SPECIAL + FUNC + NAMES + COLORS + SHAPES + SIZES + NUMBERS + CODES
                 + RECIPE + WEATHER + LETTERS + DIGITS
                 + sorted({s for v in SYNONYMS.values() for s in v}))
        seen, self.itos = set(), []
        for w in words:
            if w not in seen:
                seen.add(w)
                self.itos.append(w)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad = self.stoi["[PAD]"]
        self.unk = self.stoi["[UNK]"]

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        return [self.stoi.get(t, self.unk) for t in tokens]


VOCAB = Vocab()


@dataclass
class Candidate:
    value: str                 # semantic value ("red", "true", "three", ...)
    name: list                 # identifier tokens shown to the model
    rubric: list               # description tokens (may be empty)
    ordinal: float | None = None  # normalised rank for ordered (score) levels


@dataclass
class Question:
    qtype: str                 # "choice" | "score" | "noul"
    family: str                # color | shape | size | count | attr | rel
    instr: list
    cands: list
    gold: int                  # index into cands, or -1 for NONE (out of scope)
    meta: dict = field(default_factory=dict)


@dataclass
class Example:
    context: list
    question: Question


@dataclass
class Obj:
    name: str
    color: str
    shape: str
    size: str


@dataclass
class Scene:
    objs: list
    order: list                # order[i] = index of the object at position i
    dropped: set               # {(obj_index, attribute)}
    context: list

    def pos(self, i):
        return self.order.index(i)

    def by_name(self, name):
        for i, o in enumerate(self.objs):
            if o.name == name:
                return i
        return None


def make_scene(rng: random.Random, n_min=3, n_max=6, p_drop=0.15) -> Scene:
    n = rng.randint(n_min, n_max)
    names = rng.sample(NAMES, n)
    objs = [Obj(nm, rng.choice(COLORS), rng.choice(SHAPES), rng.choice(SIZES)) for nm in names]
    order = list(range(n))
    rng.shuffle(order)
    dropped = set()
    sentences = []
    for i, o in enumerate(objs):
        words = [o.name]
        for attr in ("color", "shape", "size"):
            if rng.random() < p_drop:
                dropped.add((i, attr))
            else:
                words.append(getattr(o, attr))
        if len(words) > 1:
            sentences.append(words + ["."])
    for a, b in zip(order, order[1:]):
        sentences.append([objs[a].name, "left", objs[b].name, "."])
    rng.shuffle(sentences)
    context = [w for s in sentences for w in s]
    return Scene(objs, order, dropped, context)


def foreign_context(rng: random.Random, words, n_sent=None):
    n_sent = n_sent or rng.randint(4, 9)
    out = []
    for _ in range(n_sent):
        out += [rng.choice(words) for _ in range(rng.randint(2, 4))] + ["."]
    return out


def soup_context(rng: random.Random, length=None):
    pool = COLORS + SHAPES + SIZES + ["is", "left", "of", ".", "?", "true", "false"] + NUMBERS
    return [rng.choice(pool) for _ in range(length or rng.randint(12, 40))]


# --------------------------------------------------------------------------
# Candidate naming. The *rubric* is the semantic contract of an option; the
# *name* is the identifier the caller gets back. Jev-style systems have been
# shown to follow the name instead of the rubric bound to it; the "misleading"
# policy (test only) probes exactly that.
# --------------------------------------------------------------------------
def _describe(rng, value, policy, pool=None):
    syn = SYNONYMS.get(value, [])
    if policy == "misleading":
        others = [v for v in pool if v != value]
        return [rng.choice(others)], [rng.choice(syn + [value])]
    if policy == "code":
        return [rng.choice(CODES)], [rng.choice(syn + [value])]
    name = [value] if policy == "canonical" or not syn else [rng.choice(syn)]
    rubric = [rng.choice(syn + [value])] if rng.random() < 0.5 else []
    return name, rubric


def _name_policy(rng, allow_code=True):
    r = rng.random()
    if r < 0.45:
        return "canonical"
    if r < 0.75 and allow_code:
        return "code"
    return "synonym"


def _choice_cands(rng, true_value, family, policy, k=None, include_true=True):
    values = FAMILY_VALUES[family]
    k = k or rng.randint(2, min(6, len(values) - (0 if include_true else 1)))
    others = [v for v in values if v != true_value]
    chosen = rng.sample(others, k - 1 if include_true else k)
    if include_true:
        chosen.append(true_value)
    rng.shuffle(chosen)
    cands, used_codes = [], set()
    for v in chosen:
        name, rubric = _describe(rng, v, policy, values)
        if policy == "code":  # identifiers must be unique within a question
            while name[0] in used_codes:
                name = [rng.choice(CODES)]
            used_codes.add(name[0])
        cands.append(Candidate(v, name, rubric))
    if policy == "misleading":
        # Every option is named after a *different* option's value: a derangement.
        vals = [c.value for c in cands]
        for _ in range(100):
            perm = vals[:]
            rng.shuffle(perm)
            if all(a != b for a, b in zip(perm, vals)):
                break
        for c, nm in zip(cands, perm):
            c.name = [nm]
            if not c.rubric or c.rubric == [nm]:
                c.rubric = [rng.choice(SYNONYMS[c.value])]
    gold = [c.value for c in cands].index(true_value) if true_value in chosen else -1
    return cands, gold


def _noul_cands():
    return [Candidate("true", ["true"], []), Candidate("false", ["false"], [])]


def _ordered_cands(rng, levels):
    k = len(levels)
    out = []
    for i, v in enumerate(levels):
        syn = SYNONYMS.get(v, [])
        rubric = [rng.choice(syn)] if syn and rng.random() < 0.5 else []
        out.append(Candidate(v, [v], rubric, ordinal=i / (k - 1)))
    return out


FAMILY_WEIGHTS = {"color": 0.18, "shape": 0.14, "size": 0.12, "count": 0.10,
                  "attr": 0.16, "rel": 0.30}


def ask(rng: random.Random, scene: Scene, family: str, *, name_policy=None,
        subject=None, oos=None, rel_dist=None) -> Question:
    """Ask one question about ``scene``.

    ``oos`` selects an out-of-scope variant whose correct answer is NONE:
    ``"absent"`` (subject not in the scene) or ``"missing"`` (the true value is
    not among the offered options).
    """
    objs = scene.objs
    present = [o.name for o in objs]
    if oos == "absent" and family == "count":  # counts have no subject to remove
        family = rng.choice(["color", "shape", "size", "attr", "rel"])
    if oos == "absent":
        subj_name = rng.choice([n for n in NAMES if n not in present])
        subj = None
    else:
        subj = rng.randrange(len(objs)) if subject is None else subject
        subj_name = objs[subj].name
    meta = {"oos": oos}
    policy = name_policy or _name_policy(rng)

    if family in ("color", "shape"):
        true_value = getattr(objs[subj], family) if subj is not None else rng.choice(FAMILY_VALUES[family])
        cands, gold = _choice_cands(rng, true_value, family, policy,
                                    include_true=(oos != "missing"))
        if subj is None:
            gold = -1
        if subj is not None:
            meta["dropped"] = (subj, family) in scene.dropped
        meta["policy"] = policy
        return Question("choice", family, ["what", family, "is", subj_name, "?"], cands, gold, meta)

    if family == "size":
        cands = _ordered_cands(rng, SIZES)
        gold = SIZES.index(objs[subj].size) if subj is not None else -1
        if subj is not None:
            meta["dropped"] = (subj, "size") in scene.dropped
        return Question("score", "size", ["how", "big", "is", subj_name, "?"], cands, gold, meta)

    if family == "count":
        c = rng.choice([o.color for o in objs]) if rng.random() < 0.6 else rng.choice(COLORS)
        n = sum(o.color == c for o in objs)
        cands = _ordered_cands(rng, NUMBERS)
        meta["dropped"] = any((i, "color") in scene.dropped for i in range(len(objs)))
        return Question("score", "count", ["how", "many", c, "?"], cands,
                        n if n < len(NUMBERS) else -1, meta)

    if family == "attr":
        attr = rng.choice(["color", "shape"])
        values = FAMILY_VALUES[attr]
        if subj is not None and rng.random() < 0.5:
            v = getattr(objs[subj], attr)
        else:
            v = rng.choice(values)
        gold = -1 if subj is None else (0 if getattr(objs[subj], attr) == v else 1)
        if subj is not None:
            meta["dropped"] = (subj, attr) in scene.dropped
        return Question("noul", "attr", ["is", subj_name, v, "?"], _noul_cands(), gold, meta)

    if family == "rel":
        if subj is None:  # absent subject: pair it with a present object
            other = rng.choice(present)
            a, b = (subj_name, other) if rng.random() < 0.5 else (other, subj_name)
            return Question("noul", "rel", ["is", a, "left", "of", b, "?"], _noul_cands(), -1,
                            {**meta, "hops": None})
        n = len(objs)
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
        if rel_dist is not None:
            pairs = [(i, j) for i, j in pairs if abs(scene.pos(i) - scene.pos(j)) in rel_dist]
            if not pairs:
                pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
        i, j = rng.choice(pairs)
        d = scene.pos(j) - scene.pos(i)
        return Question("noul", "rel", ["is", objs[i].name, "left", "of", objs[j].name, "?"],
                        _noul_cands(), 0 if d > 0 else 1, {**meta, "hops": abs(d)})
    raise ValueError(family)


def _pick_family(rng, families=None):
    fams = families or list(FAMILY_WEIGHTS)
    w = [FAMILY_WEIGHTS[f] for f in fams]
    return rng.choices(fams, weights=w, k=1)[0]


def sample_example(rng: random.Random, *, p_oos=0.0, n_range=(3, 6), families=None) -> Example:
    """Training/ID distribution. With ``p_oos`` > 0 a fraction of examples are
    out-of-scope (absent subject, missing true option, or a foreign recipe
    context) and must be answered NONE."""
    scene = make_scene(rng, *n_range)
    family = _pick_family(rng, families)
    if p_oos > 0 and rng.random() < p_oos:
        kind = rng.choices(["absent", "missing", "recipe"], weights=[0.4, 0.3, 0.3])[0]
        if kind == "missing":
            family = rng.choice(["color", "shape"])
        if kind == "recipe":
            q = ask(rng, scene, family, oos="absent")
            q.meta["oos"] = "recipe"
            return Example(foreign_context(rng, RECIPE), q)
        return Example(scene.context, ask(rng, scene, family, oos=kind))
    return Example(scene.context, ask(rng, scene, family))


def sample_group(rng: random.Random, n_q=4, *, p_oos=0.0, n_range=(3, 6)) -> list:
    """Several questions about one context (the "encode once, decide many"
    regime). The examples share the same context list object, which the
    batching code uses to encode the context once. The marginal mix of
    in-scope and out-of-scope questions matches :func:`sample_example`."""
    scene = make_scene(rng, *n_range)
    if p_oos > 0 and rng.random() < 0.3 * p_oos:  # a foreign context: everything is out of scope
        ctx = foreign_context(rng, RECIPE)
        out = []
        for _ in range(n_q):
            q = ask(rng, scene, _pick_family(rng), oos="absent")
            q.meta["oos"] = "recipe"
            out.append(Example(ctx, q))
        return out
    p_q = 0.7 * p_oos / (1 - 0.3 * p_oos) if p_oos > 0 else 0.0
    out = []
    for _ in range(n_q):
        family = _pick_family(rng)
        if rng.random() < p_q:
            kind = rng.choices(["absent", "missing"], weights=[4, 3])[0]
            if kind == "missing":
                family = rng.choice(["color", "shape"])
            out.append(Example(scene.context, ask(rng, scene, family, oos=kind)))
        else:
            out.append(Example(scene.context, ask(rng, scene, family)))
    return out


OOS_KINDS_TEST = ["oos_absent", "oos_missing", "oos_recipe", "oos_weather", "oos_soup"]


def make_split(name: str, n: int, seed: int) -> list:
    """Fixed evaluation splits (seeded, disjoint from the training stream by seed)."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        if name == "id":
            out.append(sample_example(rng))
        elif name == "cal":
            out.append(sample_example(rng, p_oos=0.2))
        elif name in ("oos_absent", "oos_missing"):
            scene = make_scene(rng)
            fam = rng.choice(["color", "shape"]) if name == "oos_missing" else _pick_family(rng)
            out.append(Example(scene.context, ask(rng, scene, fam, oos=name[4:])))
        elif name in ("oos_recipe", "oos_weather", "oos_soup"):
            scene = make_scene(rng)
            q = ask(rng, scene, _pick_family(rng), oos="absent")
            q.meta["oos"] = name[4:]
            if name == "oos_soup":
                ctx = soup_context(rng)
            else:
                ctx = foreign_context(rng, RECIPE if name == "oos_recipe" else WEATHER)
            out.append(Example(ctx, q))
        elif name == "misleading":
            scene = make_scene(rng)
            fam = rng.choice(["color", "shape"])
            out.append(Example(scene.context, ask(rng, scene, fam, name_policy="misleading")))
        elif name == "long":
            scene = make_scene(rng, 7, 8, p_drop=0.0)
            d = rng.randint(1, 7)
            out.append(Example(scene.context, ask(rng, scene, "rel", rel_dist={d})))
        else:
            raise ValueError(name)
    return out


# --------------------------------------------------------------------------
# Coherence bundles: several questions about one scene whose answers are
# logically linked. Returned with declarative constraints (see coherence.py).
# --------------------------------------------------------------------------
def make_bundle(rng: random.Random, kind: str):
    scene = make_scene(rng)
    if kind == "color":
        subj = rng.randrange(len(scene.objs))
        q_choice = ask(rng, scene, "color", subject=subj, name_policy="canonical")
        qs = [q_choice]
        for c in q_choice.cands:
            gold = 0 if scene.objs[subj].color == c.value else 1
            qs.append(Question("noul", "attr", ["is", scene.objs[subj].name, c.value, "?"],
                               _noul_cands(), gold, {"dropped": (subj, "color") in scene.dropped}))
        links = [("iff_value", 0, c.value, k + 1) for k, c in enumerate(q_choice.cands)]
        return scene, qs, links
    if kind == "count":
        c = rng.choice([o.color for o in scene.objs])
        n = sum(o.color == c for o in scene.objs)
        q_count = Question("score", "count", ["how", "many", c, "?"], _ordered_cands(rng, NUMBERS), n,
                           {"dropped": any((i, "color") in scene.dropped for i in range(len(scene.objs)))})
        qs = [q_count]
        for i, o in enumerate(scene.objs):
            qs.append(Question("noul", "attr", ["is", o.name, c, "?"], _noul_cands(),
                               0 if o.color == c else 1, {"dropped": (i, "color") in scene.dropped}))
        links = [("count_equals", 0, list(range(1, len(qs))))]
        return scene, qs, links
    raise ValueError(kind)
