"""Stage-2 temporal refiner: a small model on P3's probability trajectory, tested on the fold cache.

File structure
--------------
- stage2_inputs(): per-frame features from P3's probabilities and the P1 evidence, for one team view.
- match_rows(): both team views of one fold, with the decoder evidence.
- make_model(): the two declared Stage-2 models.
- frame_scores(): log-loss and accuracy on graded frames.
- main(): leave-one-match-out over the 8 cached folds, every model at every alpha; tables in
  solution/reports/refiner_speedfix.md, detail in solution/cache/refiner_detail.json.
- train_final(): the declared Stage-2 model on all 8 folds, for the held-out matches.

Why: a single C_SWITCH cannot tell a 1.5 s deflection from a 2.5 s possession, while the shape of the
probabilities over 0.2-4 s might. Refinement stages like this are the idea of MS-TCN.

The cached probabilities are out-of-fold, but fold j's come from a model trained on fold i, so this is
a screen: a result that clears the bar is rerun on stage-1 models that excluded both matches before
any held-out run. No held-out match and no BAR-ATH data are used.

    .venv/Scripts/python solution/external/refiner.py
    .venv/Scripts/python solution/external/refiner.py final
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
from common import CACHE_DIR, D, REPORT_DIR, X  # noqa: E402
from external import diagnose_errors, fold_cache  # noqa: E402

# Params
from stage2 import LOG_FLOOR, TRENDS, WINDOWS  # noqa: E402,F401  (defined with stage2_inputs)
ALPHAS = (0.35, 0.5, 0.75)
INFO_ALPHAS = (1.0, 1.5)            # reported for the hgb model only, never selected
LR_C = 0.1
HGB_PARAMS = dict(max_iter=150, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=5.0,
                  early_stopping=False, random_state=0)
BASELINE = (0.6748, 0.6574)         # P3 on this cache, 8 matches and 6 DFL; the run stops if it differs
OUT_PATH = os.path.join(REPORT_DIR, "refiner_speedfix.md")
DETAIL_PATH = os.path.join(CACHE_DIR, "refiner_detail.json")
FINAL_PATH = os.path.join(CACHE_DIR, "refiner_final.joblib")
FINAL_ALPHA = 0.75
CATEGORIES = ("short", "split_dead", "split_team", "merged")


from stage2 import stage2_inputs  # noqa: E402  (one implementation, shared with the submission)


def match_rows(fold, probs=None, ev=None):
    """Both team views of one fold.

    Args:
        fold: one item of fold_cache.load().
        probs: probabilities to use instead of the cached ones (external/refiner_clean.py).
        ev: the fold's run.evidence(), when already built.

    Returns:
        tuple: decoder evidence, x_home, x_swap. Swapping H and A in the inputs of the same function
        makes the two views exact mirrors.
    """
    f = fold["f"]
    probs = fold["probs"] if probs is None else probs
    ev = run.evidence(f, fold["blocks"]) if ev is None else ev
    scores = np.column_stack((ev["h"], ev["a"], ev["d"]))
    inside = np.zeros(len(probs), dtype=bool)      # from the halves: labels must not reach the inputs
    for _, first, last in fold["blocks"]:
        inside[first:last + 1] = True
    swap = [1, 0, 2]
    x_home = stage2_inputs(probs, scores, f["visible"], f["outside"], inside)
    x_swap = stage2_inputs(probs[:, swap], scores[:, swap], f["visible"], f["outside"], inside)
    return ev, x_home, x_swap


def make_model(name):
    """A fresh Stage-2 model: 'logistic' or 'hgb'."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if name == "hgb":
        return HistGradientBoostingClassifier(**HGB_PARAMS)
    return make_pipeline(SimpleImputer(strategy="constant", fill_value=0.0), StandardScaler(),
                         LogisticRegression(C=LR_C, max_iter=500))


def frame_scores(labels, probs):
    """Log-loss, and team accuracy on live frames, over graded frames."""
    graded = labels != X
    y = labels[graded]
    p = np.clip(probs[graded], 1e-9, 1.0)
    order = {common.H: 0, common.A: 1, D: 2}
    picked = p[np.arange(len(y)), [order[v] for v in y]]
    live = y != D
    team = (p[live, 0] > p[live, 1]) == (y[live] == common.H)
    return float(-np.log(picked).mean()), float(team.mean())


def result_row(fold, ev, probs, alpha):
    """Scores of one fold under one set of probabilities and alpha."""
    labels = run.decode_evidence(ev, fold["blocks"], probs, alpha)
    r = diagnose_errors.breakdown(fold["labels"], labels)
    r["logloss"], r["team_acc"] = frame_scores(fold["labels"], probs)
    return r


def main():
    """CLI entry point."""
    started = time.time()
    folds = fold_cache.load()
    names = [fold["name"] for fold in folds]
    dfl = [i for i, n in enumerate(names) if n.startswith("dfl_")]
    prepared = [match_rows(fold) for fold in folds]
    print("  inputs ready, {} features, {:.0f} s".format(prepared[0][1].shape[1], time.time() - started),
          flush=True)
    rows = [learned.training_rows(x_home, x_swap, fold["labels"])
            for fold, (_, x_home, x_swap) in zip(folds, prepared)]

    details = {"P3": [result_row(fold, item[0], fold["probs"], 0.5) for fold, item in zip(folds, prepared)]}
    base = np.array([r["spell"] for r in details["P3"]])
    if (round(base.mean(), 4), round(base[dfl].mean(), 4)) != BASELINE:
        raise SystemExit("P3 does not reproduce {}: the cache is not r1 -> k".format(BASELINE))
    print("  baseline reproduced: {:.4f} / {:.4f}".format(base.mean(), base[dfl].mean()), flush=True)

    for model_name, alpha in itertools.product(("logistic", "hgb"), ALPHAS + INFO_ALPHAS):
        details["{}_a{}".format(model_name, alpha)] = [None] * len(folds)
    for model_name in ("logistic", "hgb"):
        for i, fold in enumerate(folds):
            x = np.vstack([rows[j][0] for j in range(len(folds)) if j != i])
            y = np.concatenate([rows[j][1] for j in range(len(folds)) if j != i])
            model = make_model(model_name).fit(x, y)
            del x, y
            ev, x_home, x_swap = prepared[i]
            probs = learned.predict_proba(model, x_home, x_swap)
            mirror = learned.predict_proba(model, x_swap, x_home)
            assert np.allclose(probs[:, [1, 0, 2]], mirror), "refined probabilities are not symmetric"
            for alpha in ALPHAS + INFO_ALPHAS:
                details["{}_a{}".format(model_name, alpha)][i] = result_row(fold, ev, probs, alpha)
            print("  {} {} {:.0f} s".format(model_name, fold["name"], time.time() - started), flush=True)

    spell = {k: np.array([r["spell"] for r in v]) for k, v in details.items()}
    variants = ["{}_a{}".format(m, a) for m, a in itertools.product(("logistic", "hgb"), ALPHAS)]
    head = "| variant | spellF1 (8) | change | spellF1 (DFL) | change | folds better | predicted | " + \
           " | ".join(CATEGORIES) + " | log-loss | team acc |"
    lines = ["# Stage-2 temporal refiner on the r1 -> k fold cache", "",
             "Trained on the other 7 folds; `C_SWITCH` and `W_NEAR` as submitted. A screen: stage-1 "
             "probabilities of the training folds saw the held-back match.", "",
             head, "|" + "---|" * (head.count("|") - 1)]
    for name in details:
        r, s = details[name], spell[name]
        lines.append("| {} | {:.4f} | {:+.4f} | {:.4f} | {:+.4f} | {}/8 | {}/{} | {} | {:.4f} | {:.4f} |".format(
            name, s.mean(), s.mean() - base.mean(), s[dfl].mean(), s[dfl].mean() - base[dfl].mean(),
            int(np.sum(s > base + 1e-9)), sum(x["n_pred"] for x in r), sum(x["n_gt"] for x in r),
            " | ".join(str(sum(x["cats"][c] for x in r)) for c in CATEGORIES),
            np.mean([x["logloss"] for x in r]), np.mean([x["team_acc"] for x in r])))

    best = max(variants, key=lambda n: spell[n].mean())
    gains, picks = [], []
    for i in range(len(folds)):
        others = np.arange(len(folds)) != i
        pick = max(variants, key=lambda n: spell[n][others].mean())
        picks.append(pick)
        gains.append(spell[pick][i] - base[i])
    gains = np.array(gains)
    flips = np.array(list(itertools.product((1, -1), repeat=len(base))))
    diff = spell[best] - base
    p_value = float(np.mean(np.abs((flips * diff).mean(1)) >= abs(diff.mean()) - 1e-12))
    lines += ["", "Best: `{}`, {:+.4f} (8), {:+.4f} (DFL); per-match SD {:.4f}, sign-flip p {:.3f}.".format(
        best, diff.mean(), diff[dfl].mean(), diff.std(ddof=1), p_value),
        "", "Nested estimate over {} variants: {:+.4f} (8), {:+.4f} (DFL).".format(
            len(variants), gains.mean(), gains[dfl].mean()),
        "", "| match | P3 | {} | change | nested pick | nested gain |".format(best), "|---|---|---|---|---|---|"]
    for i, name in enumerate(names):
        lines.append("| {} | {:.4f} | {:.4f} | {:+.4f} | {} | {:+.4f} |".format(
            name, base[i], spell[best][i], diff[i], picks[i], gains[i]))
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    with open(DETAIL_PATH, "w", encoding="utf-8") as handle:
        json.dump({"folds": names, "best": best, "nested": {"picks": picks, "gains": gains.tolist()},
                   "details": details}, handle, default=float)
    print("\n".join(lines))
    print("done in {:.0f} s".format(time.time() - started))


def train_final():
    """The hgb Stage-2 model on all 8 folds, saved in the cache."""
    import joblib
    folds = fold_cache.load()
    rows = [learned.training_rows(*match_rows(fold)[1:], fold["labels"]) for fold in folds]
    model = make_model("hgb").fit(np.vstack([r[0] for r in rows]), np.concatenate([r[1] for r in rows]))
    joblib.dump(model, FINAL_PATH)
    print("saved", FINAL_PATH)


if __name__ == "__main__":
    train_final() if sys.argv[1:] == ["final"] else main()
