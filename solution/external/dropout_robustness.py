"""Stage B0-B2 of the review assessment: which ball-dropout profile should P3 be trained with?

File structure
--------------
- PROFILES: name -> (dead/live dropout ratio or None for the original profile, Metrica labels
  controlled?). "old" is the current pipeline; the others use the label-free dropout of
  sportec_dfl.drop_ball_label_free (BAR-ATH's missing share and gap lengths, no labels).
- prepare_rows(): per profile and match, the P3 training rows, saved to solution/cache/dropout/.
- fit_folds(): per profile and held-out match, P3 trained on the other matches, saved.
- evaluate(): per test profile and match, rebuild the features once and score every training
  profile's fold model on them (spellF1 against that profile's labels).
- bar_ath_profile(): per training profile, a model on all matches applied to BAR-ATH; label-free
  stoppage and possession counts and how the ball-less frames are labelled.
- main(): runs the phases (each skipped when its outputs exist) and writes
  solution/reports/dropout_robustness.md.

Why a cross-test: the test inputs change with the dropout profile, so scores under different
profiles are not comparable on their own. A training profile that scores well whichever test
profile is closest to the real tracking is the robust choice. Only calibration matches and label-free BAR-ATH statistics are used.

    python solution/external/dropout_robustness.py
"""

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
from common import CACHE_DIR, D, FPS, REPORT_DIR, H, A, half_masks, runs_of  # noqa: E402
from external import diagnose_errors, lomo, metrica_sample, sportec_dfl  # noqa: E402

# Params
ALPHA = 0.5
PROFILES = {            # name -> (DROPOUT_RATIO, CONTROLLED_LABELS for Metrica)
    "old": (None, False),
    "r1": (1.0, True),
    "k": (sportec_dfl.KINEMATIC, True),
}
TEST_PROFILES = ("k", "r1")
OUT_DIR = os.path.join(CACHE_DIR, "dropout_speedfix")
OUT_PATH = os.path.join(REPORT_DIR, "dropout_robustness.md")


def _use(profile):
    """Set the loader switches for a profile."""
    sportec_dfl.DROPOUT_RATIO, metrica_sample.CONTROLLED_LABELS = PROFILES[profile]


def _path(profile, name):
    return os.path.join(OUT_DIR, profile, name)


def _match_inputs(loader):
    f, blocks, labels = loader()
    ev = run.evidence(f, blocks)
    _, x_home, x_swap = learned.match_inputs(f, blocks, ev)
    return f, blocks, labels, ev, x_home, x_swap


def prepare_rows():
    """Training rows of every match under every profile."""
    for profile in PROFILES:
        _use(profile)
        os.makedirs(_path(profile, ""), exist_ok=True)
        for name, loader in lomo.calibration_matches():
            target = _path(profile, name + "_rows.npz")
            if os.path.isfile(target):
                continue
            _, _, labels, _, x_home, x_swap = _match_inputs(loader)
            x, y = learned.training_rows(x_home, x_swap, labels)
            np.savez(target, x=x, y=y)
            print("  rows {} {}".format(profile, name), flush=True)


def fit_folds():
    """One P3 model per profile and held-out match."""
    import joblib
    names = [n for n, _ in lomo.calibration_matches()]
    for profile in PROFILES:
        for held in names:
            target = _path(profile, "model_{}.joblib".format(held))
            if os.path.isfile(target):
                continue
            rows = []
            for name in names:
                if name != held:
                    with np.load(_path(profile, name + "_rows.npz")) as z:
                        rows.append((z["x"], z["y"]))
            joblib.dump(learned.fit(rows), target)
            print("  model {} without {}".format(profile, held), flush=True)


def evaluate():
    """spellF1[train profile][test profile][match]; each (test, match) is saved as soon as it is
    scored (solution/cache/dropout/scores.json), so an interrupted run resumes."""
    import json
    import joblib
    path = os.path.join(OUT_DIR, "scores.json")
    done = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            done = json.load(handle)
    for test in TEST_PROFILES:
        _use(test)
        for name, loader in lomo.calibration_matches():
            key = "{}|{}".format(test, name)
            if key in done:
                continue
            _, blocks, labels, ev, x_home, x_swap = _match_inputs(loader)
            done[key] = {}
            for train in PROFILES:
                model = joblib.load(_path(train, "model_{}.joblib".format(name)))
                probs = learned.predict_proba(model, x_home, x_swap)
                result = diagnose_errors.breakdown(labels, run.decode_evidence(ev, blocks, probs, ALPHA))
                done[key][train] = {"spell": float(result["spell"]),
                                    "split_dead": int(result["cats"]["split_dead"])}
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(done, handle)
            print("  scored {} {}".format(test, name), flush=True)
    scores = {p: {q: {} for q in TEST_PROFILES} for p in PROFILES}
    for key, row in done.items():
        test, name = key.split("|")
        for train, value in row.items():
            scores[train][test][name] = value
    return scores


def bar_ath_profile():
    """Label-free BAR-ATH summary for a model trained on all matches of each profile."""
    import atd_cache
    import dead_ball
    import features
    from common import PITCH_LENGTH
    names = [n for n, _ in lomo.calibration_matches()]
    f = features.build()
    blocks, graded = half_masks()
    ev = run.evidence(f, blocks, dead_ball.tagged_kickoffs(f),
                      dead_ball.set_piece_windows(len(f["visible"])))
    f = dict(f, players=atd_cache.load()["players"], pitch_length=PITCH_LENGTH)
    _, x_home, x_swap = learned.match_inputs(f, blocks, ev)
    minutes = graded.sum() / FPS / 60
    out = {}
    for profile in PROFILES:
        rows = []
        for name in names:
            with np.load(_path(profile, name + "_rows.npz")) as z:
                rows.append((z["x"], z["y"]))
        model = learned.fit(rows)
        labels = run.decode_evidence(ev, blocks, learned.predict_proba(model, x_home, x_swap), ALPHA)
        starts, stops, values = runs_of(np.where(graded, labels, 3))
        dead = values == D
        lengths = (stops - starts)[dead] / FPS
        missing = graded & ~f["visible"]
        out[profile] = {
            "stoppages_90": dead.sum() / minutes * 90,
            "possessions_90": ((values == H) | (values == A)).sum() / minutes * 90,
            "short_stoppage_share": float(np.mean(lengths < 15)) if len(lengths) else 0.0,
            "dead_share": float(np.mean(labels[graded] == D)),
            "missing_called_dead": float(np.mean(labels[missing] == D)),
        }
        print("  BAR-ATH {}: {}".format(profile, out[profile]), flush=True)
    return out


def main():
    """CLI entry point."""
    started = time.time()
    prepare_rows()
    fit_folds()
    scores = evaluate()
    bar = bar_ath_profile()
    names = [n for n, _ in lomo.calibration_matches()]
    dfl = [n for n in names if n.startswith("dfl_")]
    lines = ["# Ball-dropout profile: cross-test, DFL speeds in m/s", "",
             "Rows: training profile. Columns: test profile. Cells: mean spellF1 over 8 matches / "
             "over the 6 DFL matches / split_dead total. Profiles: `old` = dropout rates read from "
             "the public window, reference only; `r1` = the BAR-ATH missing share and gap lengths, "
             "equal dead and live start rates; `k` = the BAR-ATH missing share per player-speed "
             "band, no labels of any match.", "",
             "| trained with | " + " | ".join("test {}".format(q) for q in TEST_PROFILES) +
             " | worst (DFL) |", "|---|" + "---|" * (len(TEST_PROFILES) + 1)]
    for train in PROFILES:
        cells, worst = [], []
        for test in TEST_PROFILES:
            s8 = np.mean([scores[train][test][n]["spell"] for n in names])
            s6 = np.mean([scores[train][test][n]["spell"] for n in dfl])
            split = sum(scores[train][test][n]["split_dead"] for n in names)
            cells.append("{:.4f} / {:.4f} / {}".format(s8, s6, split))
            worst.append(s6)
        lines.append("| {} | {} | {:.4f} |".format(train, " | ".join(cells), min(worst)))
    lines += ["", "## BAR-ATH, label-free, model trained on all 8 matches of each profile", "",
              "| trained with | stoppages / 90 min | possessions / 90 min | stoppages < 15 s | "
              "dead share | ball-less frames called D |", "|---|---|---|---|---|---|"]
    for profile, b in bar.items():
        lines.append("| {} | {:.0f} | {:.0f} | {:.0%} | {:.1%} | {:.1%} |".format(
            profile, b["stoppages_90"], b["possessions_90"], b["short_stoppage_share"],
            b["dead_share"], b["missing_called_dead"]))
    lines += ["", "Reference, true DFL matches: 90-107 stoppages and 209-254 possessions per 90 min "
              "(touch-level labels)."]
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("done in {:.0f} s".format(time.time() - started))


if __name__ == "__main__":
    main()
