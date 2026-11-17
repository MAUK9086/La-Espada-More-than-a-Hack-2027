"""Where the spellF1 is lost: an anatomy of the errors of several systems, on external matches only.

File structure
--------------
- Params: duration buckets, the systems compared.
- _match(): one-to-one matching of true and predicted possessions at one IoU (the metric's greedy order).
- anatomy(): every count for one match (oracles, recall by duration and by how the possession starts
  and ends, precision by predicted duration, frame-level error shares, reasons for unmatched possessions).
- _decode(): labels of one system on one fold (the standard decoder).
- _table(): the aggregate tables for one set of matches.
- main(): `calib` (LOMO predictions on the 8 calibration matches) or `skc` (final models on the 20
  SkillCorner matches); prints Markdown, writes nothing unless a file name is given.

Diagnostic only: no model is fitted and nothing is chosen here. No BAR-ATH data is read.

Usage: .venv/Scripts/python solution/external/error_anatomy.py calib|skc [report-name.md]
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from external import tcn_gpu as tg  # noqa: E402  (sets up the import paths and common flags)
import possession_metrics as pm  # noqa: E402
from common import A, D, H, REPORT_DIR  # noqa: E402
from external import diagnose_errors, min_duration, review_ablation  # noqa: E402

# Params
FPS = 25
BUCKETS = ((0, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 1e9))      # seconds
BUCKET_NAMES = ("< 1 s", "1–2 s", "2–4 s", "4–8 s", "8–16 s", "≥ 16 s")
IOU = 0.5
CALIB_SYSTEMS = (("p3-refined", None, None), ("aug4_e10 ×3 (r1)", "aug4_e10", (0, 1, 2)),
                 ("k4_e10 ×3 (k)", "k4_e10", (0, 1, 2)))
REASONS = ("short", "split_dead", "split_team", "merged", "boundary", "other")


def _bucket(n_frames):
    s = n_frames / FPS
    return next(i for i, (lo, hi) in enumerate(BUCKETS) if lo <= s < hi)


def _match(gt, pr, threshold):
    """Matched flags of the true and of the predicted possessions at one IoU threshold."""
    got, used = np.zeros(len(gt), bool), np.zeros(len(pr), bool)
    for iou, i, j in pm.spell_overlaps(gt, pr):
        if iou < threshold:
            break
        if not got[i] and not used[j]:
            got[i], used[j] = True, True
    return got, used


def anatomy(true, pred):
    """All counts for one match.

    Args:
        true, pred: (N,) int8 label arrays.

    Returns:
        dict of counts and scores.
    """
    runs = pm.valid_runs(true)
    pred = pm.fill_abstentions(pred, runs)
    seg_t = pm.segments(true, runs)
    gt = pm.possession_spells(true, runs)
    pr = pm.possession_spells(pred, runs)
    got, used = _match(gt, pr, IOU)
    b = diagnose_errors.breakdown(true, pred)
    out = {"spell": b["spell"], "oracle_dead": b["oracle_dead"], "oracle_team": b["oracle_team"],
           "n_gt": len(gt), "n_pr": len(pr), "tp": int(got.sum()), "reasons": b["cats"]}
    out["gt_bucket"] = np.zeros((len(BUCKETS), 2), int)       # [n, matched]
    out["pr_bucket"] = np.zeros((len(BUCKETS), 2), int)
    for (a, e, _), m in zip(gt, got):
        out["gt_bucket"][_bucket(e - a)] += (1, int(m))
    for (a, e, _), m in zip(pr, used):
        out["pr_bucket"][_bucket(e - a)] += (1, int(m))
    # how a true possession starts and ends: after / before a stoppage, or the other team
    index = {(a, e): i for i, (a, e, _) in enumerate(seg_t)}
    out["start"] = {"after stoppage": [0, 0], "after the other team": [0, 0], "start of a half": [0, 0]}
    out["end"] = {"into stoppage": [0, 0], "to the other team": [0, 0], "end of a half": [0, 0]}
    for (a, e, k), m in zip(gt, got):
        i = index[(a, e)]
        before = seg_t[i - 1] if i > 0 and seg_t[i - 1][1] == a else None
        after = seg_t[i + 1] if i + 1 < len(seg_t) and seg_t[i + 1][0] == e else None
        ks = "start of a half" if before is None else ("after stoppage" if before[2] == D else "after the other team")
        ke = "end of a half" if after is None else ("into stoppage" if after[2] == D else "to the other team")
        for store, key in ((out["start"], ks), (out["end"], ke)):
            store[key][0] += 1
            store[key][1] += int(m)
    scored = np.zeros(len(true), bool)
    for a, e in runs:
        scored[a:e] = True
    t, p = true[scored], pred[scored]
    live_t, live_p = (t == H) | (t == A), (p == H) | (p == A)
    out["frames"] = {"n": int(scored.sum()), "true dead called live": int(((t == D) & live_p).sum()),
                     "true live called dead": int((live_t & (p == D)).sum()),
                     "wrong team": int((live_t & live_p & (t != p)).sum()),
                     "true dead": int((t == D).sum()), "true live": int(live_t.sum())}
    return out


def _decode(fold, probs, cache, alpha=0.75):
    return review_ablation.decode_variant(dict(fold, probs=probs), {"alpha": alpha, "c_switch": 8.0}, cache)


def _table(title, systems, folds):
    """Markdown for one set of matches.

    Args:
        title: heading.
        systems: [(label, {match: probs})].
        folds: fold dicts (name, labels, blocks, f).
    """
    cache = {}
    res = [(label, [anatomy(f["labels"], _decode(f, probs[f["name"]], cache)) for f in folds])
           for label, probs in systems]
    labels = [l for l, _ in res]
    head = "| | " + " | ".join(labels) + " |"
    rule = "|---|" + "---|" * len(labels)
    lines = ["# " + title, "", "{} matches; standard decoder, α 0.75; matching at IoU {} for the recall and "
             "precision tables.".format(len(folds), IOU), "", "## Scores and oracles (mean over matches)", "", head, rule]

    def row(name, fn, fmt="{:.4f}"):
        lines.append("| {} | ".format(name) + " | ".join(fmt.format(fn(rows)) for _, rows in res) + " |")

    row("spellF1", lambda r: np.mean([x["spell"] for x in r]))
    row("spellF1, true dead frames given (team as predicted)", lambda r: np.mean([x["oracle_dead"] for x in r]))
    row("spellF1, true team given (alive/dead as predicted)", lambda r: np.mean([x["oracle_team"] for x in r]))
    row("predicted / true possessions", lambda r: sum(x["n_pr"] for x in r) / sum(x["n_gt"] for x in r), "{:.3f}")
    row("recall at IoU 0.5", lambda r: sum(x["tp"] for x in r) / sum(x["n_gt"] for x in r), "{:.3f}")
    row("precision at IoU 0.5", lambda r: sum(x["tp"] for x in r) / sum(x["n_pr"] for x in r), "{:.3f}")
    lines += ["", "## Frame-level errors (share of graded frames)", "", head, rule]
    for key in ("true dead called live", "true live called dead", "wrong team"):
        row(key, lambda r, k=key: 100.0 * sum(x["frames"][k] for x in r) / sum(x["frames"]["n"] for x in r), "{:.2f} %")
    row("of the true dead frames, called live",
        lambda r: 100.0 * sum(x["frames"]["true dead called live"] for x in r) / sum(x["frames"]["true dead"] for x in r), "{:.1f} %")
    row("of the true live frames, called dead",
        lambda r: 100.0 * sum(x["frames"]["true live called dead"] for x in r) / sum(x["frames"]["true live"] for x in r), "{:.1f} %")
    lines += ["", "## True possessions by duration: share of all, recall, and share of all misses", "",
              "| duration | share of true possessions | " + " | ".join("recall, " + l + " | share of misses" for l in labels) + " |",
              "|---|---|" + "---|---|" * len(labels)]
    n_gt = sum(x["n_gt"] for x in res[0][1])
    for i, name in enumerate(BUCKET_NAMES):
        cells = []
        for _, rows in res:
            n = sum(x["gt_bucket"][i, 0] for x in rows)
            m = sum(x["gt_bucket"][i, 1] for x in rows)
            miss = sum(x["n_gt"] - x["tp"] for x in rows)
            cells.append("{:.3f} | {:.1f} %".format(m / max(n, 1), 100.0 * (n - m) / max(miss, 1)))
        lines.append("| {} | {:.1f} % | ".format(name, 100.0 * sum(x["gt_bucket"][i, 0] for x in res[0][1]) / n_gt)
                     + " | ".join(cells) + " |")
    lines += ["", "## Predicted possessions by duration: share of the system's predictions, precision", "",
              "| duration | " + " | ".join("share, " + l + " | precision" for l in labels) + " |",
              "|---|" + "---|---|" * len(labels)]
    for i, name in enumerate(BUCKET_NAMES):
        cells = []
        for _, rows in res:
            n = sum(x["pr_bucket"][i, 0] for x in rows)
            m = sum(x["pr_bucket"][i, 1] for x in rows)
            cells.append("{:.1f} % | {:.3f}".format(100.0 * n / sum(x["n_pr"] for x in rows), m / max(n, 1)))
        lines.append("| {} | ".format(name) + " | ".join(cells) + " |")
    for field, heading in (("start", "how the true possession starts"), ("end", "how the true possession ends")):
        lines += ["", "## Recall by " + heading, "", "| | share of true possessions | "
                  + " | ".join("recall, " + l for l in labels) + " |", "|---|---|" + "---|" * len(labels)]
        for key in res[0][1][0][field]:
            n = sum(x[field][key][0] for x in res[0][1])
            lines.append("| {} | {:.1f} % | ".format(key, 100.0 * n / n_gt) + " | ".join(
                "{:.3f}".format(sum(x[field][key][1] for x in rows) / max(n, 1)) for _, rows in res) + " |")
    lines += ["", "## Why a true possession has no prediction with IoU ≥ 0.5 of its team (share of true possessions)", "", head, rule]
    for key in REASONS:
        row(key, lambda r, k=key: 100.0 * sum(x["reasons"][k] for x in r) / sum(x["n_gt"] for x in r), "{:.1f} %")
    lines += ["", "short: the true possession is under {} s; split_dead / split_team: a predicted stoppage / other-team "
              "run of at least {} s lies inside it; merged: a prediction of its team covers it but is much longer; "
              "boundary: best IoU 0.3–0.5; other: the rest.".format(diagnose_errors.SHORT_S, diagnose_errors.MIN_RUN_S), "",
              "Matches: " + ", ".join(f["name"] for f in folds) + "."]
    return "\n".join(lines) + "\n"


def main():
    mode = sys.argv[1]
    if mode == "calib":
        folds = min_duration.load_folds()
        names = [f["name"] for f in folds]
        systems = [(label, {f["name"]: f["probs_refined"] for f in folds} if config is None
                    else tg.ensemble(config, seeds, names)) for label, config, seeds in CALIB_SYSTEMS]
        text = _table("Error anatomy, LOMO on the 8 calibration matches (tested on profile k)", systems, folds)
    else:
        data = tg.load_skc()
        folds = [{"name": n, "labels": d["labels"], "blocks": [tuple(int(v) for v in b) for b in d["blocks"]],
                  "f": dict({k[2:]: d[k] for k in d if k.startswith("f_")}, pitch_length=float(d["pitch_length"]))}
                 for n, d in data.items()]
        systems = [("p3-refined", {n: d["probs_refined"].astype(np.float64) for n, d in data.items()})]
        for label, config, _ in CALIB_SYSTEMS[1:]:
            systems.append((label, {n: np.load(os.path.join(tg.OUT_DIR, config, "skc", n + ".npy")).astype(np.float64)
                                    for n in data}))
        text = _table("Error anatomy, models trained on the calibration matches, on the 20 SkillCorner matches", systems, folds)
    if len(sys.argv) > 2:
        path = os.path.join(REPORT_DIR, sys.argv[2])
        if os.path.isfile(path):
            raise SystemExit("{} exists; reports are never overwritten".format(path))
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
    sys.stdout.reconfigure(encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
