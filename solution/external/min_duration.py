"""R1 of the fourth review: the minimum-duration (duration-expanded) Viterbi on the p3-local fold cache.

File structure
--------------
- GRID: the declared sweep. L_min 30, 40, 50 frames x C_SWITCH 4, 6, 8, 10 x alpha 0.5, 0.75 x restart
  "steady" (after D into H_0 / A_0) and "count" (after D into H_1 / A_1), on two probability sets:
  the p3-local fold probabilities (cache/post_local/folds) and the leak-free Stage-2 refined ones
  (cache/post_local/refiner_clean/refined_*.npy). 96 decodes over the 8 calibration matches.
- BASES: before any grid result the standard 3-state decoder must reproduce p3-local (alpha 0.5,
  0.6945 / 0.6763) and p3-refined (alpha 0.75, 0.7077 / 0.6889), or the script stops.
- sweep(): every setting through review_ablation.decode_variant (which sets C_SWITCH and alpha) with
  possession.MIN_DURATION / MIN_RESTART set; diagnostics from diagnose_errors.breakdown. Results are
  saved after each setting, so a stopped run resumes.
- report(): reports/min_duration.md; the best setting on refined probabilities (highest 8-match
  mean, either alpha) is recorded for the TCN step (external/tcn.py uses its restart, L_min and
  C_SWITCH at alpha 0.5 and 0.75).

Declared before running. Baseline p3-refined 0.7077 / 0.6889. A candidate goes
to held-out (kinematic dfl and metrica3, once each, named in a commit first) only at >= 0.7177 over 8
matches and >= 0.6939 on DFL.

    .venv/Scripts/python solution/external/min_duration.py sweep
    .venv/Scripts/python solution/external/min_duration.py report
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

import possession  # noqa: E402
from common import CACHE_DIR, REPORT_DIR  # noqa: E402
from external import diagnose_errors, fold_cache, review_ablation  # noqa: E402

# Params
POST_DIR = os.path.join(CACHE_DIR, "post_local")
REFINED_DIR = os.path.join(POST_DIR, "refiner_clean")
OUT_DIR = os.path.join(CACHE_DIR, "min_duration")
RESULTS_PATH = os.path.join(OUT_DIR, "results.json")
REPORT_PATH = os.path.join(REPORT_DIR, "min_duration.md")
L_MINS = (30, 40, 50)
C_SWITCHES = (4.0, 6.0, 8.0, 10.0)
ALPHAS = (0.5, 0.75)
RESTARTS = ("steady", "count")
PROBS = ("p3-local", "refined")
BASES = {"p3-local": (0.5, (0.6945, 0.6763)), "refined": (0.75, (0.7077, 0.6889))}
BASELINE = (0.7077, 0.6889)             # p3-refined
BAR = (0.7177, 0.6939)                  # standing bar against p3-refined
CATS = ("short", "split_dead", "split_team", "merged")


def load_folds():
    """The 8 cached folds with both probability sets."""
    fold_cache.FOLD_DIR = os.path.join(POST_DIR, "folds")
    folds = fold_cache.load()
    for f in folds:
        f["probs_p3-local"] = f["probs"]
        f["probs_refined"] = np.load(os.path.join(REFINED_DIR, "refined_{}.npy".format(
            f["name"]))).astype(np.float64)
    return folds


def run_setting(folds, probs, alpha, c_switch, l_min, restart, cache):
    """breakdown() of every fold under one decoder setting (l_min None = the 3-state decoder)."""
    saved = possession.MIN_DURATION, possession.MIN_RESTART
    possession.MIN_DURATION, possession.MIN_RESTART = l_min, restart
    try:
        out = []
        for f in folds:
            fold = dict(f, probs=f["probs_" + probs])
            labels = review_ablation.decode_variant(fold, {"alpha": alpha, "c_switch": c_switch}, cache)
            r = diagnose_errors.breakdown(f["labels"], labels)
            out.append({"spell": float(r["spell"]), "n_pred": int(r["n_pred"]), "n_gt": int(r["n_gt"]),
                        "cats": {c: int(r["cats"][c]) for c in CATS},
                        "start": float(r["start_abs_med"]), "end": float(r["end_abs_med"])})
        return out
    finally:
        possession.MIN_DURATION, possession.MIN_RESTART = saved


def key(probs, alpha, c_switch, l_min, restart):
    return "{}|a{}|cs{}|L{}|{}".format(probs, alpha, c_switch, l_min, restart)


def sweep():
    os.makedirs(OUT_DIR, exist_ok=True)
    folds = load_folds()
    names = [f["name"] for f in folds]
    dfl = [i for i, n in enumerate(names) if n.startswith("dfl_")]
    results = {"names": names, "rows": {}}
    if os.path.isfile(RESULTS_PATH):
        with open(RESULTS_PATH, encoding="utf-8") as handle:
            results = json.load(handle)
    cache = {}
    started = time.time()
    for probs, (alpha, expected) in BASES.items():
        k = key(probs, alpha, possession.C_SWITCH, None, None)
        if k not in results["rows"]:
            results["rows"][k] = run_setting(folds, probs, alpha, possession.C_SWITCH, None, None, cache)
        s = np.array([r["spell"] for r in results["rows"][k]])
        got = (round(s.mean(), 4), round(s[dfl].mean(), 4))
        if got != expected:
            raise SystemExit("base {} gives {}, expected {}".format(probs, got, expected))
        print("  base {} reproduced {}".format(probs, got), flush=True)
    for probs, restart, l_min, c_switch, alpha in itertools.product(PROBS, RESTARTS, L_MINS, C_SWITCHES, ALPHAS):
        k = key(probs, alpha, c_switch, l_min, restart)
        if k in results["rows"]:
            continue
        results["rows"][k] = run_setting(folds, probs, alpha, c_switch, l_min, restart, cache)
        with open(RESULTS_PATH, "w", encoding="utf-8") as handle:
            json.dump(results, handle)
        s = np.array([r["spell"] for r in results["rows"][k]])
        print("  {} {:.4f} / {:.4f} ({:.0f} s)".format(k, s.mean(), s[dfl].mean(), time.time() - started),
              flush=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as handle:
        json.dump(results, handle)


def best_refined(results=None):
    """(restart, l_min, c_switch, alpha) with the highest 8-match mean on refined probabilities."""
    if results is None:
        with open(RESULTS_PATH, encoding="utf-8") as handle:
            results = json.load(handle)
    best, best_s = None, -1.0
    for probs, restart, l_min, c_switch, alpha in itertools.product(("refined",), RESTARTS, L_MINS, C_SWITCHES, ALPHAS):
        s = np.mean([r["spell"] for r in results["rows"][key(probs, alpha, c_switch, l_min, restart)]])
        if s > best_s + 1e-12:
            best, best_s = (restart, l_min, c_switch, alpha), s
    return best, best_s


def report():
    with open(RESULTS_PATH, encoding="utf-8") as handle:
        results = json.load(handle)
    names = results["names"]
    dfl = [i for i, n in enumerate(names) if n.startswith("dfl_")]
    base = np.array([r["spell"] for r in results["rows"][key("refined", 0.75, possession.C_SWITCH, None, None)]])
    head = ("| probabilities | restart | L_min | C_SWITCH | alpha | spellF1 (8) | vs p3-refined | spellF1 (DFL) | "
            "vs p3-refined | better | predicted | short | split_dead | split_team | merged | edge s/e (s) | per match |")
    lines = ["# R1: minimum-duration Viterbi on the p3-local fold cache", "",
             "Trained r1, tested kinematic, 8 calibration matches. Baseline p3-refined {:.4f} / {:.4f}; "
             "bar {} / {}. Rows with L_min - are the standard 3-state decoder.".format(
                 base.mean(), base[dfl].mean(), *BAR), "",
             head, "|" + "---|" * (head.count("|") - 1)]
    for k, rs in results["rows"].items():
        probs, a, cs, l_min, restart = k.split("|")
        s = np.array([r["spell"] for r in rs])
        lines.append("| {} | {} | {} | {} | {} | {:.4f} | {:+.4f} | {:.4f} | {:+.4f} | {}/8 | {}/{} | {} | {:.2f} / {:.2f} | {} |".format(
            probs, restart if restart != "None" else "-", l_min[1:] if l_min != "LNone" else "-", cs[2:], a[1:],
            s.mean(), s.mean() - base.mean(), s[dfl].mean(), s[dfl].mean() - base[dfl].mean(),
            int(np.sum(s > base + 1e-9)), sum(r["n_pred"] for r in rs), sum(r["n_gt"] for r in rs),
            " | ".join(str(sum(r["cats"][c] for r in rs)) for c in CATS),
            np.mean([r["start"] for r in rs]), np.mean([r["end"] for r in rs]),
            " ".join("{:.4f}".format(v) for v in s)))
    best, best_s = best_refined(results)
    rs = results["rows"][key("refined", best[3], best[2], best[1], best[0])]
    s = np.array([r["spell"] for r in rs])
    passes = s.mean() >= BAR[0] - 1e-12 and s[dfl].mean() >= BAR[1] - 1e-12
    lines += ["", "Best on refined probabilities: restart {}, L_min {}, C_SWITCH {}, alpha {}: {:.4f} / {:.4f} "
              "({}). Used for the TCN step.".format(*best, s.mean(), s[dfl].mean(),
                                                    "meets the bar" if passes else "below the bar"),
              "", "Matches: " + ", ".join(names)]
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT_DIR, "best.json"), "w", encoding="utf-8") as handle:
        json.dump({"restart": best[0], "l_min": best[1], "c_switch": best[2], "alpha": best[3],
                   "spell": float(s.mean()), "spell_dfl": float(s[dfl].mean())}, handle)
    print("\n".join(lines[-3:]))


if __name__ == "__main__":
    {"sweep": sweep, "report": report}[sys.argv[1]]()
