"""The Stage-2 refiner without the stacking leak: stage-1 models that excluded both matches.

File structure
--------------
- fit_pairs(): one P3 model per pair of matches, trained on the other 6, on saved `r1` rows.
- predict(): each match's probabilities under every model that excluded it, kinematic test inputs,
  one match in memory at a time.
- score(): for each held-back match i, Stage-2 trains on the other 7 with probabilities from models
  that never saw i or that match, then decodes i from its own out-of-fold probabilities. Written to
  solution/reports/refiner_clean_speedfix.md.

Every model and probability file is saved under solution/cache/refiner_clean/, so each phase resumes
after a stop. Phases are separate because only predict() needs a lot of memory.

    .venv/Scripts/python solution/external/refiner_clean.py fit
    .venv/Scripts/python solution/external/refiner_clean.py predict
    .venv/Scripts/python solution/external/refiner_clean.py score
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
from common import CACHE_DIR, REPORT_DIR  # noqa: E402
from external import dropout_robustness, fold_cache, lomo, refiner  # noqa: E402

# Params
TRAIN_PROFILE = "r1"
TEST_PROFILE = "k"
CANDIDATE = "hgb_a0.75"             # declared from the screen, before this run
OUT_DIR = os.path.join(CACHE_DIR, "refiner_clean")
OUT_PATH = os.path.join(REPORT_DIR, "refiner_clean_speedfix.md")
DETAIL_PATH = os.path.join(CACHE_DIR, "refiner_clean_detail.json")
SCREEN = True          # compare with the leaky screen of refiner.py (False for runs without one)
SAVE_REFINED = False   # also save each match's refined out-of-fold probabilities in OUT_DIR


def _model_path(a, b):
    return os.path.join(OUT_DIR, "model_{}_{}.joblib".format(*sorted((a, b))))


def _probs_path(match, other):
    return os.path.join(OUT_DIR, "probs_{}_without_{}.npy".format(match, other))


def fit_pairs(names):
    """One stage-1 model per pair of matches."""
    import joblib
    started = time.time()
    for a, b in itertools.combinations(names, 2):
        target = _model_path(a, b)
        if os.path.isfile(target):
            continue
        rows = []
        for name in names:
            if name not in (a, b):
                with np.load(dropout_robustness._path(TRAIN_PROFILE, name + "_rows.npz")) as z:
                    rows.append((z["x"], z["y"]))
        joblib.dump(learned.fit(rows), target)
        print("  model without {} {} ({:.0f} s)".format(a, b, time.time() - started), flush=True)


def predict(sources, names):
    """Probabilities of each match under the models that excluded it."""
    import joblib
    started = time.time()
    dropout_robustness._use(TEST_PROFILE)
    for name, loader in sources:
        others = [o for o in names if o != name and not os.path.isfile(_probs_path(name, o))]
        if not others:
            continue
        _, _, _, _, x_home, x_swap = dropout_robustness._match_inputs(loader)
        for other in others:
            probs = learned.predict_proba(joblib.load(_model_path(name, other)), x_home, x_swap)
            np.save(_probs_path(name, other), probs.astype(np.float32))
        print("  predicted {} ({:.0f} s)".format(name, time.time() - started), flush=True)


def score():
    """Stage-2 without the leak; the same tables as the screen."""
    started = time.time()
    folds = fold_cache.load()
    names = [fold["name"] for fold in folds]
    dfl = [i for i, n in enumerate(names) if n.startswith("dfl_")]
    evidence = [run.evidence(fold["f"], fold["blocks"]) for fold in folds]
    details = {"P3": [refiner.result_row(fold, ev, fold["probs"], 0.5) for fold, ev in zip(folds, evidence)]}
    base = np.array([r["spell"] for r in details["P3"]])
    if (round(base.mean(), 4), round(base[dfl].mean(), 4)) != refiner.BASELINE:
        raise SystemExit("P3 does not reproduce {}".format(refiner.BASELINE))
    alphas = refiner.ALPHAS + refiner.INFO_ALPHAS
    for alpha in alphas:
        details["hgb_a{}".format(alpha)] = [None] * len(folds)
    shift = []
    for i, held in enumerate(folds):
        x, y = [], []
        for j, fold in enumerate(folds):
            if j == i:
                continue
            clean = np.load(_probs_path(fold["name"], held["name"])).astype(np.float64)
            shift.append(float(np.abs(clean - fold["probs"]).mean()))
            _, x_home, x_swap = refiner.match_rows(fold, clean, evidence[j])
            rows = learned.training_rows(x_home, x_swap, fold["labels"])
            x.append(rows[0])
            y.append(rows[1])
        model = refiner.make_model("hgb").fit(np.vstack(x), np.concatenate(y))
        del x, y
        _, x_home, x_swap = refiner.match_rows(held, held["probs"], evidence[i])
        probs = learned.predict_proba(model, x_home, x_swap)
        if SAVE_REFINED:
            np.save(os.path.join(OUT_DIR, "refined_{}.npy".format(held["name"])), probs.astype(np.float32))
        for alpha in alphas:
            details["hgb_a{}".format(alpha)][i] = refiner.result_row(held, evidence[i], probs, alpha)
        print("  {} {:.0f} s".format(held["name"], time.time() - started), flush=True)

    spell = {k: np.array([r["spell"] for r in v]) for k, v in details.items()}
    screen = np.full(len(names), np.nan)
    if SCREEN:
        with open(refiner.DETAIL_PATH, encoding="utf-8") as handle:
            screen = np.array([r["spell"] for r in json.load(handle)["details"][CANDIDATE]])
    head = "| variant | spellF1 (8) | change | spellF1 (DFL) | change | folds better | predicted | " + \
           " | ".join(refiner.CATEGORIES) + " | log-loss | team acc |"
    lines = ["# Stage-2 refiner without the stacking leak", "",
             "Stage-1 models excluded both the held-back match and the training match. Declared candidate: "
             "`{}`. Mean absolute change of the training probabilities against the screen: {:.4f}.".format(
                 CANDIDATE, np.mean(shift)), "", head, "|" + "---|" * (head.count("|") - 1)]
    for name in details:
        r, s = details[name], spell[name]
        lines.append("| {} | {:.4f} | {:+.4f} | {:.4f} | {:+.4f} | {}/8 | {}/{} | {} | {:.4f} | {:.4f} |".format(
            name, s.mean(), s.mean() - base.mean(), s[dfl].mean(), s[dfl].mean() - base[dfl].mean(),
            int(np.sum(s > base + 1e-9)), sum(x["n_pred"] for x in r), sum(x["n_gt"] for x in r),
            " | ".join(str(sum(x["cats"][c] for x in r)) for c in refiner.CATEGORIES),
            np.mean([x["logloss"] for x in r]), np.mean([x["team_acc"] for x in r])))
    lines += ["", "| match | P3 | screen | clean | change |", "|---|---|---|---|---|"]
    for i, name in enumerate(names):
        lines.append("| {} | {:.4f} | {:.4f} | {:.4f} | {:+.4f} |".format(
            name, base[i], screen[i], spell[CANDIDATE][i], spell[CANDIDATE][i] - base[i]))
    with open(OUT_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    with open(DETAIL_PATH, "w", encoding="utf-8") as handle:
        json.dump({"folds": names, "details": details}, handle, default=float)
    print("\n".join(lines))
    print("done in {:.0f} s".format(time.time() - started))


def main():
    """CLI entry point: fit, predict or score."""
    os.makedirs(OUT_DIR, exist_ok=True)
    sources = lomo.calibration_matches()
    names = [name for name, _ in sources]
    {"fit": lambda: fit_pairs(names), "predict": lambda: predict(sources, names), "score": score}[sys.argv[1]]()


if __name__ == "__main__":
    main()
