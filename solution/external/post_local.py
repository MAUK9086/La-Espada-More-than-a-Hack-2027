"""E1 after p3-local: the clean Stage-2 refiner and the D1 decoder settings on the 103-feature base.

File structure
--------------
- POST_DIR: everything this experiment writes, in the gitignored cache, apart from the 82-feature
  caches of the earlier rounds (cache/dropout_speedfix, cache/folds, cache/refiner_clean), which stay
  reproducible.
- _point(): sends dropout_robustness, fold_cache, refiner and refiner_clean to POST_DIR.
- rows(), folds(): training rows (r1 dropout) and one P3 model per held-out match, with the submitted
  feature set (team_shape.LOCAL_GROUPS = ball, near3).
- cache(): out-of-fold probabilities on kinematic test inputs (fold_cache.build).
- pairs(), predict(), score(): refiner_clean's leak-free Stage-2, 28 pair models, declared alphas.
- decoder(): D1 decoder settings on the p3-local folds, and on the refined probabilities.
- final(): the Stage-2 model on all 8 folds, for the held-out run of a declared candidate.

Declared before running. Base: the fold cache must reproduce p3-local,
0.6945 / 0.6763, or score() stops. E1a: hgb Stage-2 at alpha 0.5, 0.75, 1.0; the best on 8 matches
is the candidate, with a nested estimate. E1b: D1 (alpha 0.65, C_SWITCH 12, W_NEAR 0 in the decoder,
islands 35 keep) and the same with C_SWITCH 10, on the p3-local probabilities. The decoder on refined
probabilities is information only. To advance: about +0.01 over 8 matches and at least +0.005 on DFL.

    .venv/Scripts/python solution/external/post_local.py rows
    .venv/Scripts/python solution/external/post_local.py folds
    .venv/Scripts/python solution/external/post_local.py cache
    .venv/Scripts/python solution/external/post_local.py pairs
    .venv/Scripts/python solution/external/post_local.py predict
    .venv/Scripts/python solution/external/post_local.py score
    .venv/Scripts/python solution/external/post_local.py decoder
    .venv/Scripts/python solution/external/post_local.py final <alpha>
"""

import itertools
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import common  # noqa: E402

common.FLAGS_ENABLED = False

import learned  # noqa: E402
import run  # noqa: E402
import team_shape  # noqa: E402
from common import CACHE_DIR, REPORT_DIR  # noqa: E402
from external import (diagnose_errors, dropout_robustness, fold_cache, lomo, refiner,  # noqa: E402
                      refiner_clean, review_ablation)

# Params
POST_DIR = os.path.join(CACHE_DIR, "post_local")
GROUPS = ("ball", "near3")
BASELINE = (0.6945, 0.6763)          # p3-local, LOMO r1 -> kinematic
ALPHAS = (0.5, 0.75, 1.0)            # E1a, declared
DECODERS = {                          # E1b, declared; also applied to refined probabilities (info)
    "D1_cs12": {"w_near": 0.0, "alpha": 0.65, "c_switch": 12.0, "island": (35, 15)},
    "D1_cs10": {"w_near": 0.0, "alpha": 0.65, "c_switch": 10.0, "island": (35, 15)},
}
REFINER_REPORT = os.path.join(REPORT_DIR, "post_local_refiner.md")
DECODER_REPORT = os.path.join(REPORT_DIR, "post_local_decoder.md")
FINAL_PATH = os.path.join(POST_DIR, "refiner_final.joblib")


def _point():
    """Send every helper to POST_DIR and the feature set GROUPS."""
    global REFINER_REPORT, DECODER_REPORT, FINAL_PATH, BASELINE
    if os.path.basename(POST_DIR) != "post_local":
        tag = os.path.basename(POST_DIR)
        REFINER_REPORT = os.path.join(REPORT_DIR, tag + "_refiner.md")
        DECODER_REPORT = os.path.join(REPORT_DIR, tag + "_decoder.md")
        FINAL_PATH = os.path.join(POST_DIR, "refiner_final.joblib")
        lomo_path = os.path.join(POST_DIR, "lomo.json")
        if os.path.isfile(lomo_path):
            with open(lomo_path, encoding="utf-8") as handle:
                BASELINE = tuple(json.load(handle)["baseline"])
    team_shape.LOCAL_GROUPS = GROUPS
    dropout_robustness.OUT_DIR = POST_DIR
    fold_cache.FOLD_DIR = os.path.join(POST_DIR, "folds")
    refiner.ALPHAS, refiner.INFO_ALPHAS, refiner.BASELINE = ALPHAS, (), BASELINE
    refiner.FINAL_PATH = FINAL_PATH
    refiner_clean.OUT_DIR = os.path.join(POST_DIR, "refiner_clean")
    refiner_clean.OUT_PATH = REFINER_REPORT
    refiner_clean.DETAIL_PATH = os.path.join(POST_DIR, "refiner_detail.json")
    refiner_clean.SCREEN, refiner_clean.SAVE_REFINED = False, True
    os.makedirs(refiner_clean.OUT_DIR, exist_ok=True)
    os.makedirs(fold_cache.FOLD_DIR, exist_ok=True)


def rows():
    """r1 training rows of every match, one match in memory at a time."""
    dropout_robustness._use("r1")
    os.makedirs(dropout_robustness._path("r1", ""), exist_ok=True)
    for name, loader in lomo.calibration_matches():
        target = dropout_robustness._path("r1", name + "_rows.npz")
        if os.path.isfile(target):
            continue
        _, _, labels, _, x_home, x_swap = dropout_robustness._match_inputs(loader)
        x, y = learned.training_rows(x_home, x_swap, labels)
        np.savez(target, x=x, y=y)
        print("  rows {} ({} features)".format(name, x.shape[1]), flush=True)


def folds():
    """One P3 model per held-out match, trained on the other 7."""
    import joblib
    names = [n for n, _ in lomo.calibration_matches()]
    for held in names:
        target = dropout_robustness._path("r1", "model_{}.joblib".format(held))
        if os.path.isfile(target):
            continue
        train = []
        for name in names:
            if name != held:
                with np.load(dropout_robustness._path("r1", name + "_rows.npz")) as z:
                    train.append((z["x"], z["y"]))
        joblib.dump(learned.fit(train), target)
        print("  model without {}".format(held), flush=True)


def decoder():
    """E1b on the p3-local folds; the same decoders on refined probabilities for information."""
    loaded = fold_cache.load()
    names = [f["name"] for f in loaded]
    dfl = [i for i, n in enumerate(names) if n.startswith("dfl_")]
    with open(refiner_clean.DETAIL_PATH, encoding="utf-8") as handle:
        detail = json.load(handle)["details"]
    best = max(ALPHAS, key=lambda a: np.mean([r["spell"] for r in detail["hgb_a{}".format(a)]]))
    rows_out, cache = {}, {}
    rows_out["P3 (p3-local)"] = [diagnose_errors.breakdown(f["labels"], review_ablation.decode_variant(
        f, {}, cache)) for f in loaded]
    for key, settings in DECODERS.items():
        rows_out[key] = [diagnose_errors.breakdown(f["labels"], review_ablation.decode_variant(
            f, settings, cache)) for f in loaded]
    refined = [dict(f, probs=np.load(os.path.join(refiner_clean.OUT_DIR, "refined_{}.npy".format(
        f["name"]))).astype(np.float64)) for f in loaded]
    rows_out["refiner a{} (info)".format(best)] = [diagnose_errors.breakdown(f["labels"],
        review_ablation.decode_variant(f, {"alpha": best}, cache)) for f in refined]
    for key, settings in DECODERS.items():
        info = dict(settings, alpha=best)
        rows_out["refiner a{} + {} (info)".format(best, key)] = [diagnose_errors.breakdown(
            f["labels"], review_ablation.decode_variant(f, info, cache)) for f in refined]
    base = np.array([r["spell"] for r in rows_out["P3 (p3-local)"]])
    if (round(base.mean(), 4), round(base[dfl].mean(), 4)) != BASELINE:
        raise SystemExit("decoder base does not reproduce {}".format(BASELINE))
    head = ("| setting | spellF1 (8) | change | spellF1 (DFL) | change | better | predicted | short | "
            "split_dead | split_team | merged | per match |")
    lines = ["# E1b: D1 decoder settings on p3-local", "",
             "Fold cache r1 -> kinematic, 103 features. Decoder settings change the decoding only.", "",
             head, "|" + "---|" * (head.count("|") - 1)]
    for key, rs in rows_out.items():
        s = np.array([r["spell"] for r in rs])
        lines.append("| {} | {:.4f} | {:+.4f} | {:.4f} | {:+.4f} | {}/8 | {}/{} | {} | {} |".format(
            key, s.mean(), s.mean() - base.mean(), s[dfl].mean(), s[dfl].mean() - base[dfl].mean(),
            int(np.sum(s > base + 1e-9)), sum(r["n_pred"] for r in rs), sum(r["n_gt"] for r in rs),
            " | ".join(str(sum(r["cats"][c] for r in rs)) for c in refiner.CATEGORIES),
            " ".join("{:.4f}".format(v) for v in s)))
    with open(DECODER_REPORT, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def nested():
    """E1a selection effect: choose the alpha on 7 matches, score it on the 8th."""
    with open(refiner_clean.DETAIL_PATH, encoding="utf-8") as handle:
        data = json.load(handle)
    names, detail = data["folds"], data["details"]
    spell = {a: np.array([r["spell"] for r in detail["hgb_a{}".format(a)]]) for a in ALPHAS}
    base = np.array([r["spell"] for r in detail["P3"]])
    gains = []
    for i in range(len(names)):
        others = [j for j in range(len(names)) if j != i]
        pick = max(ALPHAS, key=lambda a: spell[a][others].mean())
        gains.append((names[i], pick, spell[pick][i] - base[i]))
    g = np.array([x[2] for x in gains])
    dfl = [i for i, n in enumerate(names) if n.startswith("dfl_")]
    lines = ["", "## Nested estimate", "", "| held-back match | alpha chosen on the other 7 | gain |", "|---|---|---|"]
    lines += ["| {} | {} | {:+.4f} |".format(n, a, v) for n, a, v in gains]
    lines.append("\nMean {:+.4f} over 8 matches, {:+.4f} on DFL, {}/8 positive.".format(
        g.mean(), g[dfl].mean(), int(np.sum(g > 1e-9))))
    flips = np.array(list(itertools.product((1, -1), repeat=8)))
    for a in ALPHAS:
        d = spell[a] - base
        p = np.mean(np.abs((flips * d).mean(1)) >= abs(d.mean()) - 1e-12)
        lines.append("alpha {}: mean {:+.4f}, SD {:.4f}, lowest {:+.4f}, sign-flip p = {:.3f}".format(
            a, d.mean(), d.std(ddof=1), d.min(), p))
    with open(REFINER_REPORT, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def lomo_score():
    """LOMO of the fold cache's own P3 probabilities at alpha 0.5, the base for every later phase."""
    loaded = fold_cache.load()
    names = [f["name"] for f in loaded]
    dfl = [i for i, n in enumerate(names) if n.startswith("dfl_")]
    rows = [diagnose_errors.breakdown(f["labels"], run.decode_evidence(
        run.evidence(f["f"], f["blocks"]), f["blocks"], f["probs"], 0.5)) for f in loaded]
    s = np.array([r["spell"] for r in rows])
    cats = {c: sum(r["cats"][c] for r in rows) for c in refiner.CATEGORIES}
    lines = ["# LOMO of the fold cache, groups {}".format(" ".join(GROUPS)), "",
             "Trained r1, tested kinematic, alpha 0.5. spellF1 {:.4f} over 8 matches, {:.4f} on DFL; "
             "predicted {}/{}; {}; edge error {:.2f} / {:.2f} s.".format(
                 s.mean(), s[dfl].mean(), sum(r["n_pred"] for r in rows), sum(r["n_gt"] for r in rows),
                 ", ".join("{} {}".format(k, v) for k, v in cats.items()),
                 np.mean([r["start_abs_med"] for r in rows]), np.mean([r["end_abs_med"] for r in rows])),
             "", "| match | spellF1 |", "|---|---|"] + [
        "| {} | {:.4f} |".format(n, v) for n, v in zip(names, s)]
    with open(os.path.join(POST_DIR, "lomo.json"), "w", encoding="utf-8") as handle:
        json.dump({"baseline": [round(float(s.mean()), 4), round(float(s[dfl].mean()), 4)],
                   "per_match": dict(zip(names, map(float, s))), "cats": cats}, handle)
    with open(os.path.join(REPORT_DIR, os.path.basename(POST_DIR) + "_lomo.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def final(alpha):
    """The Stage-2 model on all 8 folds of the p3-local cache, for a declared candidate."""
    refiner.FINAL_ALPHA = alpha
    refiner.train_final()


def apply_sets(args):
    """Apply every '--set NAME=VALUE' to team_shape (e.g. CHAIN_EXACT=True, NEAR3_DEDUP_M=0.8) and
    return the remaining arguments."""
    import ast
    rest = []
    i = 0
    while i < len(args):
        if args[i] == "--set":
            name, value = args[i + 1].split("=", 1)
            if not hasattr(team_shape, name):
                raise SystemExit("team_shape has no {}".format(name))
            setattr(team_shape, name, ast.literal_eval(value))
            print("  team_shape.{} = {!r}".format(name, getattr(team_shape, name)), flush=True)
            i += 2
        else:
            rest.append(args[i])
            i += 1
    return rest


def main():
    """CLI entry point: [--groups g ...] [--dir name] phase [alpha]; one phase per call, resumable."""
    global GROUPS, POST_DIR
    args = apply_sets(sys.argv[1:])
    if "--dir" in args:
        i = args.index("--dir")
        POST_DIR = os.path.join(CACHE_DIR, args[i + 1])
        args = args[:i] + args[i + 2:]
    if "--groups" in args:
        i = args.index("--groups")
        j = i + 1
        while j < len(args) and args[j] in ("ball", "near3", "gap", "impulse", "bounds", "chain", "bridge", "near1"):
            j += 1
        GROUPS = tuple(args[i + 1:j])
        args = args[:i] + args[j:]
    _point()
    started = time.time()
    sources = lomo.calibration_matches()
    names = [name for name, _ in sources]
    phase = args[0]
    {"rows": rows, "folds": folds, "cache": lambda: fold_cache.build("r1", "k"), "lomo": lomo_score,
     "pairs": lambda: refiner_clean.fit_pairs(names),
     "predict": lambda: refiner_clean.predict(sources, names),
     "score": lambda: (refiner_clean.score(), nested()), "decoder": decoder,
     "final": lambda: final(float(args[1]))}[phase]()
    print("{} done in {:.0f} s".format(phase, time.time() - started), flush=True)


if __name__ == "__main__":
    main()
