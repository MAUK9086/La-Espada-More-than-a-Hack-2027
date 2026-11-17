"""Where is spellF1 lost? Error breakdown of the current model on leave-one-match-out folds.

File structure
--------------
- breakdown(): for one held-out calibration match -- oracle swaps, miss taxonomy, boundary offsets,
  spellF1 per IoU threshold, predicted vs true counts.
- main(): every requested fold, the per-fold rows and the mean, written to
  solution/reports/error_breakdown.md.

Uses calibration matches only (the held-out matches and the public window are never touched),
so it can be rerun after every change to see whether the intended weakness moved.

Miss taxonomy, for each true possession not matched at IoU 0.5 by a same-team prediction:
  short       shorter than SHORT_S
  split_dead  a predicted stoppage of at least 0.5 s inside it
  split_team  a predicted run of the other team of at least 0.5 s inside it
  merged      covered by one same-team prediction much longer than it (a missed turnover)
  boundary    best IoU in [0.3, 0.5): right possession, wrong edges
  other       anything else

    python solution/external/diagnose_errors.py                 # all calibration folds
    python solution/external/diagnose_errors.py --folds dfl_J03WMX dfl_J03WN1
    python solution/external/diagnose_errors.py --beta 2        # with the P5 turnover gate
"""

import argparse
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
import turnover  # noqa: E402
import possession_metrics as pm  # noqa: E402
from common import A, D, FPS, H, REPORT_DIR, X, runs_of  # noqa: E402
from external import lomo  # noqa: E402

# Params
ALPHA = 0.5
SHORT_S = 4.0
MERGE_RATIO = 1.6                # a same-team prediction this much longer than the truth = merged
MIN_RUN_S = 0.5                  # runs shorter than this inside a possession are ignored
OUT_PATH = os.path.join(REPORT_DIR, "error_breakdown.md")
CATEGORIES = ("short", "split_dead", "split_team", "merged", "boundary", "other")


def _segments(labels, keep):
    starts, stops, values = runs_of(labels)
    return [(a, b, v) for a, b, v in zip(starts, stops, values) if v in keep]


def _iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / float(max(a[1], b[1]) - min(a[0], b[0]))


def spell_per_iou(truth, pred):
    """spellF1 at each IoU threshold of the official metric."""
    runs = pm.valid_runs(truth)
    scores = pm.spell_scores(truth, pm.fill_abstentions(pred, runs), runs)
    return {float(k): v["f1"] for k, v in scores["per_iou"].items()}


def breakdown(truth, pred):
    """All diagnostics for one match.

    Args:
        truth: (N,) int8 true labels.
        pred: (N,) int8 predicted labels.

    Returns:
        dict of numbers (see the module docstring).
    """
    def spell(p):
        return pm.evaluate(truth, p)["spell_f1_mean"]

    out = {"spell": spell(pred)}
    oracle_dead = np.where(truth == D, D, np.where(pred == D, X, pred)).astype(np.int8)
    oracle_team = np.where(pred == D, D, np.where(truth == D, pred, truth)).astype(np.int8)
    out["oracle_dead"], out["oracle_team"] = spell(oracle_dead), spell(oracle_team)
    out["per_iou"] = spell_per_iou(truth, pred)

    gt, pr = _segments(truth, (H, A)), _segments(pred, (H, A))
    out["n_gt"], out["n_pred"] = len(gt), len(pr)
    min_run = int(MIN_RUN_S * FPS)
    cats = {c: 0 for c in CATEGORIES}
    offsets = []
    for g in gt:
        best = max(pr, key=lambda p: _iou(g, p)) if pr else None
        best_iou = _iou(g, best) if best else 0.0
        if best_iou >= 0.5 and best[2] == g[2]:
            offsets.append((best[0] - g[0], best[1] - g[1]))
            continue
        inside = pred[g[0]:g[1]]
        other = H if g[2] == A else A
        if (g[1] - g[0]) / FPS < SHORT_S:
            cats["short"] += 1
        elif any(b - a >= min_run for a, b, _ in _segments(inside, (D,))):
            cats["split_dead"] += 1
        elif any(b - a >= min_run for a, b, _ in _segments(inside, (other,))):
            cats["split_team"] += 1
        elif any(p[2] == g[2] and _iou(g, p) > 0 and (p[1] - p[0]) > MERGE_RATIO * (g[1] - g[0])
                 for p in pr):
            cats["merged"] += 1
        elif best_iou >= 0.3:
            cats["boundary"] += 1
        else:
            cats["other"] += 1
    out["cats"] = cats
    out["matched"] = len(offsets)
    o = np.abs(np.array(offsets, dtype=float)) / FPS if offsets else np.zeros((1, 2))
    out["start_abs_med"], out["end_abs_med"] = float(np.median(o[:, 0])), float(np.median(o[:, 1]))
    out["over_1s"] = float(np.mean(o > 1.0))
    return out


def predict_fold(matches, held, alpha=ALPHA, beta=None):
    """The current model trained without `held`, decoded on `held` (with the P5 gate if beta)."""
    others = [m for m in matches if m is not held]
    model = learned.fit([learned.training_rows(m["x_home"], m["x_swap"], m["labels"])
                         for m in others])
    probs = learned.predict_proba(model, held["x_home"], held["x_swap"])
    switch = None
    if beta:
        t_model, prior = turnover.fit([turnover.training_rows(m["t_inputs"], m["labels"])
                                       for m in others])
        switch = turnover.switch_costs(turnover.predict(t_model, held["t_inputs"]), prior, beta)
    return run.decode_evidence(held["ev"], held["blocks"], probs, alpha, switch)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--folds", nargs="*", default=None, help="match names (default: all)")
    parser.add_argument("--note", default="", help="what model state this breakdown describes")
    parser.add_argument("--beta", type=float, default=None, help="P5 turnover-gate weight")
    args = parser.parse_args()

    started = time.time()
    matches = lomo.prepare(lomo.calibration_matches(), with_turnover=bool(args.beta))
    held_out = [m for m in matches if not args.folds or m["name"] in args.folds]
    head = ("| fold | spellF1 | oracle dead | oracle team | IoU .3/.5/.7 | pred/true | "
            + " | ".join(CATEGORIES) + " | edge err s (start/end) | >1 s |")
    lines = ["# Error breakdown{}".format(": " + args.note if args.note else ""), "",
             "Leave-one-match-out on calibration matches; see the docstring of "
             "`external/diagnose_errors.py` for the categories.", "", head,
             "|" + "---|" * (head.count("|") - 1)]
    rows = []
    for held in held_out:
        r = breakdown(held["labels"], predict_fold(matches, held, beta=args.beta))
        rows.append(r)
        iou = r["per_iou"]
        lines.append("| {} | {:.4f} | {:.4f} | {:.4f} | {:.2f}/{:.2f}/{:.2f} | {}/{} | {} | {:.2f}/{:.2f} | {:.0%} |".format(
            held["name"], r["spell"], r["oracle_dead"], r["oracle_team"],
            iou.get(0.3, np.nan), iou.get(0.5, np.nan), iou.get(0.7, np.nan), r["n_pred"], r["n_gt"],
            " | ".join(str(r["cats"][c]) for c in CATEGORIES), r["start_abs_med"],
            r["end_abs_med"], r["over_1s"]))
        print(lines[-1], flush=True)
    mean = lambda key: np.mean([r[key] for r in rows])
    lines.append("| **mean** | **{:.4f}** | {:.4f} | {:.4f} | | {}/{} | {} | {:.2f}/{:.2f} | {:.0%} |".format(
        mean("spell"), mean("oracle_dead"), mean("oracle_team"),
        sum(r["n_pred"] for r in rows), sum(r["n_gt"] for r in rows),
        " | ".join(str(sum(r["cats"][c] for r in rows)) for c in CATEGORIES),
        mean("start_abs_med"), mean("end_abs_med"), mean("over_1s")))
    print(lines[-1])
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("wrote {} in {:.0f} s".format(OUT_PATH, time.time() - started))


if __name__ == "__main__":
    main()
