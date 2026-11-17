"""Stage A of the review assessment: decoder-only variants on cached out-of-fold model outputs.

File structure
--------------
- VARIANTS: name -> settings; every setting left out keeps the current model (A0).
- decode_variant(): rebuild the P1 evidence (W_NEAR may change), assemble the switch costs
  (constant / P5 / contact gating), decode with alpha 0.5, then optional island absorption and
  stoppage-edge snapping.
- main(): every requested variant on every cached fold; spellF1 means (8 matches, 6 DFL) and the
  error breakdown, written to solution/reports/review_ablation.md.
- bridge_check(): how often the D2 rule would fill a real stoppage.
- main_sweep(): the declared D1 -> D2 -> D3 sweep and its nested estimate, written to
  solution/reports/decoder_sweep_speedfix.md.

Inputs come from external/fold_cache.py (P3 and P5 trained without the held-out match), so no
held-out match and no public-window data are used. Item numbers (R11 ...) label the decoder
variants of an external review of the method.

    python solution/external/review_ablation.py                 # all variants
    python solution/external/review_ablation.py --only A0 A2_45
    python solution/external/review_ablation.py --sweep
"""

import argparse
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
import run  # noqa: E402
import snap  # noqa: E402
import turnover  # noqa: E402
import possession_metrics as pm  # noqa: E402
from common import CACHE_DIR, REPORT_DIR  # noqa: E402
from external import diagnose_errors, fold_cache  # noqa: E402

# Params
ALPHA = 0.5
OUT_PATH = os.path.join(REPORT_DIR, "review_ablation_speedfix.md")
SWEEP_PATH = os.path.join(REPORT_DIR, "decoder_sweep_speedfix.md")
DETAIL_PATH = os.path.join(CACHE_DIR, "decoder_sweep_detail.json")
BASELINE = 0.6748           # 8-match spellF1 of A0 on the r1 -> k cache; the sweep stops if it differs
D1_ALPHA = (0.5, 0.65, 0.75)
D1_SWITCH = (10.0, 12.0, 14.0)
D2_FRAMES = (50, 75, 100)
D3_WINDOWS = (12, 20)
CANDIDATE = {"w_near": 0.0, "alpha": 0.65, "c_switch": 12.0, "island": (35, 15)}  # for held-out, once
VARIANTS = {
    "A0": {},
    # R11 contact-gated switch costs: (lock cost, gate radius m)
    "A1_lock12_r1.8": {"lock": (12.0, 1.8)}, "A1_lock12_r2.4": {"lock": (12.0, 2.4)},
    "A1_lock16_r1.8": {"lock": (16.0, 1.8)}, "A1_lock16_r2.4": {"lock": (16.0, 2.4)},
    "A1_lock24_r1.8": {"lock": (24.0, 1.8)}, "A1_lock24_r2.4": {"lock": (24.0, 2.4)},
    # R12 short-island absorption: (max island frames, clear-control frames to keep; 0 = never keep)
    "A2_25": {"island": (25, 0)}, "A2_35": {"island": (35, 0)},
    "A2_45": {"island": (45, 0)}, "A2_55": {"island": (55, 0)},
    "A2_25_keep": {"island": (25, 15)}, "A2_35_keep": {"island": (35, 15)},
    "A2_45_keep": {"island": (45, 15)}, "A2_55_keep": {"island": (55, 15)},
    # R10 base switch cost
    "A3_cs6": {"c_switch": 6.0}, "A3_cs10": {"c_switch": 10.0},
    "A3_cs12": {"c_switch": 12.0}, "A3_cs14": {"c_switch": 14.0},
    # R18 nearest-team tie-breaker weight
    "A4_wnear0": {"w_near": 0.0}, "A4_wnear0.05": {"w_near": 0.05},
    # R21 visibility-adaptive alpha: (visible scale, gap scale)
    "A5_0.8_1.3": {"alpha_scales": (0.8, 1.3)}, "A5_0.6_1.6": {"alpha_scales": (0.6, 1.6)},
    # R13 stoppage-edge snapping window (frames)
    "A6_dead8": {"snap_dead": 8}, "A6_dead15": {"snap_dead": 15},
    # R22 peak-pinned P5 gate; the original gate at beta 1 for reference
    "A7_peak0.5": {"p5": ("peak", 0.5)}, "A7_peak1": {"p5": ("peak", 1.0)},
    "A7_peak2": {"p5": ("peak", 2.0)}, "A7_orig1": {"p5": ("orig", 1.0)},
    # alpha, the weight of P3, rechecked after the DFL speed fix
    "AL_0.25": {"alpha": 0.25}, "AL_0.35": {"alpha": 0.35}, "AL_0.75": {"alpha": 0.75},
    "AL_1": {"alpha": 1.0}, "AL_1.5": {"alpha": 1.5}, "AL_2": {"alpha": 2.0}, "AL_4": {"alpha": 4.0},
    # A8: combinations of the small positives above
    "A8_wnear0.05_cs10": {"w_near": 0.05, "c_switch": 10.0},
    "A8_wnear0_cs10": {"w_near": 0.0, "c_switch": 10.0},
    "A8_wnear0_cs10_isl35": {"w_near": 0.0, "c_switch": 10.0, "island": (35, 15)},
    "A8_wnear0.05_cs10_isl25": {"w_near": 0.05, "c_switch": 10.0, "island": (25, 15)},
}
MBR_CANDIDATES = ("A0", "A1_lock12_r2.4", "A1_lock16_r2.4", "A2_25_keep", "A2_35_keep",
                  "A3_cs10", "A3_cs12", "A4_wnear0", "A4_wnear0.05",
                  "A8_wnear0.05_cs10", "A8_wnear0_cs10")


def decode_variant(fold, settings, evidence_cache):
    """Labels for one cached fold under one variant."""
    w_near = settings.get("w_near", possession.W_NEAR)
    key = (fold["name"], w_near)
    if key not in evidence_cache:
        saved = possession.W_NEAR
        possession.W_NEAR = w_near
        try:
            evidence_cache[key] = run.evidence(fold["f"], fold["blocks"])
        finally:
            possession.W_NEAR = saved
    ev = evidence_cache[key]
    f = fold["f"]

    saved_switch = possession.C_SWITCH
    possession.C_SWITCH = settings.get("c_switch", saved_switch)
    try:
        switch = None
        if "p5" in settings:
            form, beta = settings["p5"]
            costs = turnover.switch_costs_peak if form == "peak" else turnover.switch_costs
            switch = costs(fold["p_turn"], fold["prior"], beta)
        if "lock" in settings:
            lock, gate = settings["lock"]
            switch = possession.contact_gated_switch_costs(f, switch, lock=lock, gate=gate)
        labels = run.decode_evidence(ev, fold["blocks"], fold["probs"],
                                     settings.get("alpha", ALPHA), switch,
                                     settings.get("alpha_scales"), f)
    finally:
        possession.C_SWITCH = saved_switch
    if "bridge" in settings:
        frames, same_team = settings["bridge"]
        labels = possession.bridge_short_dead(labels, f, frames, same_team)
    if "island" in settings:
        frames, clear = settings["island"]
        labels = possession.absorb_short_islands(labels, f if clear else None, frames, clear)
    if "snap_dead" in settings:
        labels = snap.snap_dead(labels, f, ev["kicks"], settings["snap_dead"])
    if "snap_contact" in settings:
        labels = snap.snap(labels, snap.contact_nearest(f), settings["snap_contact"])
    return labels


def _spell(truth, pred):
    """spellF1 of pred against truth (the official spell matching, no other metrics)."""
    runs = pm.valid_runs(truth)
    return pm.spell_scores(truth, pm.fill_abstentions(pred, runs), runs)["mean_f1"]


def mbr(candidates):
    """Minimum Bayes risk: the candidate with the highest mean spellF1 against all the others
    (each in turn taken as the truth). No labels are involved.

    Returns:
        tuple: (index of the chosen candidate, its mean agreement).
    """
    k = len(candidates)
    agree = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i != j:
                agree[i, j] = _spell(candidates[j], candidates[i])
    score = agree.sum(1) / (k - 1)
    return int(np.argmax(score)), float(score.max())


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="*", default=None, help="variant names (default: all)")
    parser.add_argument("--mbr", action="store_true", help="A9: MBR choice among MBR_CANDIDATES")
    parser.add_argument("--sweep", action="store_true", help="the declared D1, D2, D3 sweep")
    args = parser.parse_args()
    if args.sweep:
        return main_sweep()
    names = args.only or list(VARIANTS)
    if args.mbr:
        return main_mbr()

    started = time.time()
    folds = fold_cache.load()
    dfl = [i for i, fold in enumerate(folds) if fold["name"].startswith("dfl_")]
    evidence_cache = {}
    head = ("| variant | spellF1 (8) | d vs A0 | spellF1 (6 DFL) | d DFL | folds better | pred/true | "
            "split_team | merged | short | edge err s |")
    lines = ["# Stage A: decoder-only variants (review assessment)", "",
             "Out-of-fold P3/P5 outputs from `external/fold_cache.py`; alpha {}; 2 s labels. "
             "`edge err` is the median start/end error of matched possessions.".format(ALPHA), "",
             head, "|" + "---|" * (head.count("|") - 1)]
    print("\n".join(lines[-2:]), flush=True)
    base = None
    for name in names:
        rows = [diagnose_errors.breakdown(fold["labels"], decode_variant(fold, VARIANTS[name],
                                                                         evidence_cache))
                for fold in folds]
        spell = np.array([r["spell"] for r in rows])
        if base is None:
            base = spell if name == "A0" else np.array(
                [diagnose_errors.breakdown(fold["labels"], decode_variant(fold, {}, evidence_cache))["spell"]
                 for fold in folds])
        cats = {c: sum(r["cats"][c] for r in rows) for c in ("split_team", "merged", "short")}
        lines.append("| {} | {:.4f} | {:+.4f} | {:.4f} | {:+.4f} | {}/8 | {}/{} | {} | {} | {} | {:.2f}/{:.2f} |".format(
            name, spell.mean(), spell.mean() - base.mean(), spell[dfl].mean(),
            spell[dfl].mean() - base[dfl].mean(), int(np.sum(spell > base + 1e-9)),
            sum(r["n_pred"] for r in rows), sum(r["n_gt"] for r in rows), cats["split_team"],
            cats["merged"], cats["short"], np.mean([r["start_abs_med"] for r in rows]),
            np.mean([r["end_abs_med"] for r in rows])))
        print(lines[-1], flush=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    mode = "a" if args.only is not None and os.path.isfile(OUT_PATH) else "w"
    with open(OUT_PATH, mode, encoding="utf-8") as handle:
        handle.write("\n".join(lines if mode == "w" else lines[4:]) + "\n")
    print("wrote {} in {:.0f} s".format(OUT_PATH, time.time() - started))


def main_mbr():
    """A9: per fold, decode every MBR candidate, keep the MBR choice, score it."""
    started = time.time()
    folds = fold_cache.load()
    evidence_cache = {}
    rows, base, chosen = [], [], []
    for fold in folds:
        cands = [decode_variant(fold, VARIANTS[n], evidence_cache) for n in MBR_CANDIDATES]
        pick, agreement = mbr(cands)
        rows.append(diagnose_errors.breakdown(fold["labels"], cands[pick])["spell"])
        base.append(diagnose_errors.breakdown(fold["labels"], cands[0])["spell"])
        chosen.append(MBR_CANDIDATES[pick])
        print("  {}: MBR picks {} (agreement {:.3f}) -> {:.4f} vs A0 {:.4f}".format(
            fold["name"], chosen[-1], agreement, rows[-1], base[-1]), flush=True)
    rows, base = np.array(rows), np.array(base)
    dfl = [i for i, fold in enumerate(folds) if fold["name"].startswith("dfl_")]
    line = "| A9_MBR | {:.4f} | {:+.4f} | {:.4f} | {:+.4f} | {}/8 | picks: {} |".format(
        rows.mean(), rows.mean() - base.mean(), rows[dfl].mean(), rows[dfl].mean() - base[dfl].mean(),
        int(np.sum(rows > base + 1e-9)), ", ".join(chosen))
    print(line)
    with open(OUT_PATH, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print("done in {:.0f} s".format(time.time() - started))


def bridge_check(folds, evidence_cache, settings=None):
    """How often the D2 rule would fill a real stoppage, on labels decoded with `settings`.

    Returns:
        list[str]: Markdown table lines, per L: stoppages filled, of which truly dead (at least
        half their frames are D in the labels), and true stoppages shorter than L.
    """
    lines = ["| L frames | filled | truly dead | share | true stoppages under L | of those no `outside` |",
             "|---|---|---|---|---|---|"]
    for frames in D2_FRAMES:
        filled = dead = short = hidden = 0
        for fold in folds:
            labels = decode_variant(fold, settings or {}, evidence_cache)
            bridged = possession.bridge_short_dead(labels, fold["f"], frames)
            changed = np.flatnonzero(np.diff(np.concatenate(([0], (bridged != labels).astype(np.int8), [0]))))
            for first, last in changed.reshape(-1, 2):
                filled += 1
                dead += int(np.mean(fold["labels"][first:last] == common.D) >= 0.5)
            truth = (fold["labels"] == common.D).astype(np.int8)
            edges = np.flatnonzero(np.diff(np.concatenate(([0], truth, [0]))))
            for first, last in edges.reshape(-1, 2):
                if last - first < frames:
                    short += 1
                    hidden += int(not fold["f"]["outside"][first:last].any())
        lines.append("| {} | {} | {} | {:.0%} | {} | {} |".format(frames, filled, dead, dead / max(filled, 1),
                                                                  short, hidden))
    return lines


def main_sweep():
    """D1 decoder settings, D2 stoppage bridging, D3 contact snapping, each on the best so far.

    The nested estimate picks the best variant on seven matches and scores it on the eighth, so it
    does not carry the optimism of taking the maximum over the sweep.
    """
    started = time.time()
    folds = fold_cache.load()
    dfl = [i for i, fold in enumerate(folds) if fold["name"].startswith("dfl_")]
    evidence_cache = {}
    spells, details = {}, {}
    head = "| variant | spellF1 (8) | change | spellF1 (6 DFL) | change | folds better | pred/true | split_dead |"
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    print("\n".join(lines), flush=True)

    def score(name, settings):
        rows = [diagnose_errors.breakdown(fold["labels"], decode_variant(fold, settings, evidence_cache))
                for fold in folds]
        spells[name] = np.array([r["spell"] for r in rows])
        details[name] = rows
        base = spells["A0"]
        lines.append("| {} | {:.4f} | {:+.4f} | {:.4f} | {:+.4f} | {}/8 | {}/{} | {} |".format(
            name, spells[name].mean(), spells[name].mean() - base.mean(), spells[name][dfl].mean(),
            spells[name][dfl].mean() - base[dfl].mean(), int(np.sum(spells[name] > base + 1e-9)),
            sum(r["n_pred"] for r in rows), sum(r["n_gt"] for r in rows),
            sum(r["cats"]["split_dead"] for r in rows)))
        print(lines[-1], flush=True)
        return spells[name].mean()

    if round(score("A0", {}), 4) != BASELINE:
        raise SystemExit("A0 does not reproduce {}: the cache is not r1 -> k".format(BASELINE))

    d1 = {"D1_a{}_cs{:.0f}".format(alpha, switch): {"w_near": 0.0, "alpha": alpha, "c_switch": switch,
                                                    "island": (35, 15)}
          for alpha, switch in itertools.product(D1_ALPHA, D1_SWITCH)}
    means = {name: score(name, settings) for name, settings in d1.items()}
    best_d1 = max(means, key=means.get)

    d2 = {}
    for frames, same_team in itertools.product(D2_FRAMES, (False, True)):
        tag = "D2_L{}_{}".format(frames, "same" if same_team else "prev")
        d2[tag] = {"bridge": (frames, same_team)}
        d2["D1+" + tag] = dict(d1[best_d1], bridge=(frames, same_team))
    means.update({name: score(name, settings) for name, settings in d2.items()})
    best = max(means, key=means.get)
    settings_best = dict(d1, **d2)[best]

    d3 = {"{}+D3_{}".format(best, window): dict(settings_best, snap_contact=window)
          for window in D3_WINDOWS}
    means.update({name: score(name, settings) for name, settings in d3.items()})
    best = max(means, key=means.get)

    # Each part of the candidate alone, and the candidate without it. Not part of the selection.
    defaults = {"w_near": possession.W_NEAR, "alpha": ALPHA, "c_switch": possession.C_SWITCH}
    for key, value in CANDIDATE.items():
        score("only_" + key, {key: value})
        rest = dict(CANDIDATE)
        del rest[key]
        score("without_" + key, dict(rest, **({key: defaults[key]} if key in defaults else {})))

    gains, picks = [], []
    for i in range(len(folds)):
        others = np.arange(len(folds)) != i
        pick = max(means, key=lambda n: spells[n][others].mean())
        gains.append(spells[pick][i] - spells["A0"][i])
        picks.append(pick)
        print("  nested {}: {} {:+.4f}".format(folds[i]["name"], pick, gains[-1]), flush=True)
    gains = np.array(gains)
    nested = "Nested estimate, selection on the other 7 matches: {:+.4f} (8), {:+.4f} (6 DFL).".format(
        gains.mean(), gains[dfl].mean())
    check = bridge_check(folds, evidence_cache)
    check_candidate = bridge_check(folds, evidence_cache, CANDIDATE)
    print("\n".join(["best: " + best, nested] + check + check_candidate))
    with open(DETAIL_PATH, "w", encoding="utf-8") as handle:
        json.dump({"folds": [fold["name"] for fold in folds], "selected": list(means), "best": best,
                   "nested": {"picks": picks, "gains": gains.tolist()}, "details": details,
                   "bridge": {"current": check, "candidate": check_candidate}}, handle, default=float)

    gain = spells[best].mean() - spells["A0"].mean()
    gain_dfl = spells[best][dfl].mean() - spells["A0"][dfl].mean()
    with open(SWEEP_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(["# Decoder sweep on the r1 -> k fold cache", "",
                                "D1 sets W_NEAR 0 in the decoder only, with islands (35, keep). P3 is not "
                                "retrained. {} variants, D1 best `{}`.".format(len(means), best_d1), ""] + lines
                               + ["", "Best: `{}`, {:+.4f} (8), {:+.4f} (6 DFL).".format(best, gain, gain_dfl),
                                  "", nested, "", "## D2 rule check, on the current model", ""] + check
                                  + ["", "## D2 rule check, on the candidate", ""] + check_candidate) + "\n")
    print("wrote {} in {:.0f} s".format(SWEEP_PATH, time.time() - started))


if __name__ == "__main__":
    main()
