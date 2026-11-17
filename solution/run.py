"""Data in, submission out: features -> events -> evidence -> Viterbi -> frame,label CSV.

File structure
--------------
- evidence(): P1 team and dead-ball evidence, kick/stop events and transition costs for one match.
- decode_evidence(): adds alpha * log p of a learned model and runs the Viterbi per half.
- predict_from_features(): P1 only, from a features dict (used by the calibration scripts).
- predict(): the full pipeline, returns label codes for every video frame.
- summarise(): possession count / length and label shares over the whole match (no labels used).
- write_csv(): the frame,label CSV.
- main(): writes evaluation/results/<name>.csv and runs the official validator on it.

    python solution/run.py                      # -> evaluation/results/La_Espada.csv (the submission, --model tcn)
    python solution/run.py --model p3           # -> evaluation/results/p3_learned.csv (the previous version)
    python solution/run.py --model p1           # -> evaluation/results/p1_viterbi.csv (rules only)
    python evaluation/run_evaluation.py         # score it (public window = sanity check only)

Deterministic: no randomness anywhere, so the same data gives the same file byte for byte.
"""

import argparse
import hashlib
import os
import subprocess
import sys
import time

import numpy as np

import dead_ball
import features
import possession
from common import (A, D, FPS, H, LETTERS, PITCH_LENGTH, RESULTS_DIR, ROOT, X, half_masks,
                    runs_of, write_flag)

# Params
FORBIDDEN = 1e6  # score that rules a state out on a frame
LEARNED_FLOOR = 1e-3  # probability floor before the log, so one confident frame cannot dominate


def evidence(f, blocks, tagged_kickoffs=None, set_pieces=None):
    """The rule-based (P1) evidence and event-gated costs, before decoding.

    Args:
        f: feature dict (features.build() or an external loader with the same keys).
        blocks: [(half, first, last)] spans to label.
        tagged_kickoffs: optional bool (N,) kick-off strikes from tagging.
        set_pieces: optional bool (N,) tagged set-piece windows.

    Returns:
        dict: h, a, d (N,) scores; to_dead, to_alive (N,) costs; kicks, stops,
        before_kickoff (N,) bool.
    """
    kicks = dead_ball.kick_events(f)
    if tagged_kickoffs is not None:
        kicks = kicks | tagged_kickoffs
    stops = dead_ball.stop_events(f)
    pending = dead_ball.pending_restart(f, kicks)
    score_h, score_a = possession.team_scores(f, pending)
    score_d = dead_ball.dead_scores(f, kicks, pending, set_pieces)
    before_kickoff = dead_ball.pre_kickoff(f, kicks, blocks, tagged_kickoffs)
    to_dead, to_alive = possession.transition_costs(stops, kicks)
    return {"h": score_h, "a": score_a, "d": score_d, "to_dead": to_dead, "to_alive": to_alive,
            "kicks": kicks, "stops": stops, "before_kickoff": before_kickoff}


def decode_evidence(ev, blocks, learned=None, alpha=0.0, switch=None, alpha_scales=None, f=None):
    """Viterbi over the P1 evidence, optionally plus alpha * learned log-probabilities.

    Args:
        ev: dict from evidence().
        blocks: [(half, first, last)].
        learned: optional (N, 3) probabilities for H, A, D (P3).
        alpha: weight of the learned log-probabilities, per frame.
        switch: optional (N,) or (N, 2) H <-> A switch costs (P5 or contact gating).
        alpha_scales: optional (visible, gap) multipliers of alpha on the H and A log-probabilities,
            applied where the ball is visible and in play vs elsewhere (review suggestion R21;
            needs f). None keeps alpha uniform.
        f: feature dict, only needed with alpha_scales.

    Returns:
        np.ndarray: (N,) int8 labels.
    """
    score_h, score_a, score_d = ev["h"], ev["a"], ev["d"]
    if learned is not None and alpha > 0:
        logp = np.log(np.clip(learned, LEARNED_FLOOR, 1.0))
        team_alpha = alpha
        if alpha_scales is not None:
            usable = f["visible"] & ~f["outside"]
            team_alpha = alpha * np.where(usable, alpha_scales[0], alpha_scales[1])
        score_h = score_h + team_alpha * logp[:, 0]
        score_a = score_a + team_alpha * logp[:, 1]
        score_d = score_d + alpha * logp[:, 2]
    score_h = np.where(ev["before_kickoff"], -FORBIDDEN, score_h)  # hard constraint: no play
    score_a = np.where(ev["before_kickoff"], -FORBIDDEN, score_a)  # before the half's kick-off
    return possession.decode(score_h, score_a, score_d, ev["to_dead"], ev["to_alive"], blocks,
                             switch)


def predict_from_features(f, blocks, tagged_kickoffs=None, set_pieces=None):
    """The P1 model on any match's features. Shared by this match and the calibration games.

    Returns:
        tuple: labels (N,) int8, kicks (N,) bool, stops (N,) bool.
    """
    ev = evidence(f, blocks, tagged_kickoffs, set_pieces)
    return decode_evidence(ev, blocks), ev["kicks"], ev["stops"]


def predict(rebuild=False, model="p1"):
    """Run the whole model on this match, with its Smart Tagging.

    Args:
        rebuild: recompute cached features.
        model: 'p1' (rules + Viterbi), 'p3' (plus learned team-shape evidence; refined by Stage 2
            when models/p3_stage2.joblib exists, stage2.py) or 'p5' (p3 plus the learned turnover
            gate on H <-> A switches) or 'tcn' (rules + the dilated TCN ensemble of tcn_infer.py, the
            submission). Each falls back to the previous one, with a flag, when its trained model does
            not exist ('tcn' falls back to 'p3').

    Returns:
        tuple: labels (N,) int8, kicks (N,) bool, stops (N,) bool.
    """
    f = features.build(rebuild=rebuild)
    blocks, _ = half_masks()
    ev = evidence(f, blocks, dead_ball.tagged_kickoffs(f),
                  dead_ball.set_piece_windows(len(f["visible"])))
    if model == "tcn":
        import atd_cache
        import tcn_infer
        f = dict(f, players=atd_cache.load()["players"], pitch_length=PITCH_LENGTH)
        probs, alpha = tcn_infer.match_probs(f, blocks, ev)
        if probs is None:
            write_flag("TCN requested but its model files are not in {}; wrote P3 instead".format(tcn_infer.MODEL_DIR))
            model = "p3"
        else:
            return decode_evidence(ev, blocks, probs, alpha), ev["kicks"], ev["stops"]
    if model in ("p3", "p5"):
        import atd_cache
        import learned
        bundle = learned.load()
        if bundle is None:
            write_flag("P3 requested but {} does not exist; wrote P1 instead. Train it with "
                       "solution/external/lomo.py --train-final".format(learned.MODEL_PATH))
        else:
            f = dict(f, players=atd_cache.load()["players"], pitch_length=PITCH_LENGTH)
            names, x_home, x_swap = learned.match_inputs(f, blocks, ev)
            if names != bundle["names"]:
                raise ValueError("feature set changed since the model was trained; retrain it")
            probs = learned.predict_proba(bundle["model"], x_home, x_swap)
            switch = None
            if model == "p5":
                import turnover
                gate = turnover.load()
                if gate is None:
                    write_flag("P5 requested but {} does not exist; wrote P3 instead. Train it with "
                               "solution/external/lomo.py --turnover --train-final".format(
                                   turnover.MODEL_PATH))
                else:
                    t_names, t_inputs = turnover.match_inputs(
                        f, blocks, ev, (names, x_home, x_swap))
                    if t_names != gate["names"]:
                        raise ValueError("turnover features changed since training; retrain it")
                    p_turn = turnover.predict(gate["model"], t_inputs)
                    switch = turnover.switch_costs(p_turn, gate["prior"], gate["meta"]["beta"])
            alpha = bundle["meta"]["alpha"]
            if model == "p3":
                import stage2
                refiner = stage2.load()
                if refiner is not None:
                    if refiner["stage1_names"] != bundle["names"]:
                        raise ValueError("Stage-2 model was trained on another P3 feature set; retrain it")
                    probs, alpha = stage2.refine(refiner, f, blocks, ev, probs), refiner["alpha"]
            labels = decode_evidence(ev, blocks, probs, alpha, switch)
            return labels, ev["kicks"], ev["stops"]
    return decode_evidence(ev, blocks), ev["kicks"], ev["stops"]


def summarise(labels, graded):
    """Possessions, durations and shares over the graded frames. Uses no ground truth."""
    starts, stops, values = runs_of(np.where(graded, labels, X))
    lengths = (stops - starts) / FPS
    alive = (values == H) | (values == A)
    dead = values == D
    shares = {str(LETTERS[c]): "{:.1%}".format(np.mean(labels[graded] == c)) for c in (H, A, D)}
    return ("possessions {}  mean {:.1f} s  median {:.1f} s  shortest {:.1f} s | "
            "stoppages {} mean {:.1f} s | shares {}").format(
        int(alive.sum()), lengths[alive].mean(), np.median(lengths[alive]),
        lengths[alive].min(), int(dead.sum()), lengths[dead].mean(), shares)


def write_csv(path, labels):
    """frame,label CSV with LF line endings."""
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, LETTERS[c]) for i, c in enumerate(labels))


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=("p1", "p3", "p5", "tcn"), default="tcn",
                        help="p1 = rules + Viterbi; p3 = plus learned team-shape evidence; "
                             "p5 = p3 plus the learned turnover gate; tcn = rules + the dilated TCN ensemble "
                             "(tcn_infer.py; the submission)")
    parser.add_argument("--name", default=None,
                        help="output file stem; default p1_viterbi / p3_learned / La_Espada (team La Espada)")
    parser.add_argument("--rebuild", action="store_true", help="recompute cached features")
    args = parser.parse_args()
    args.name = args.name or {"p1": "p1_viterbi", "p3": "p3_learned", "p5": "p5_turnover",
                              "tcn": "La_Espada"}[args.model]

    started = time.time()
    labels, kicks, stops = predict(args.rebuild, args.model)
    _, graded = half_masks()
    print("kick events {}  stop events {} (graded frames)".format(
        int((kicks & graded).sum()), int((stops & graded).sum())))
    print(summarise(labels, graded))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, args.name + ".csv")
    write_csv(path, labels)
    with open(path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()[:16]
    print("wrote {}  sha256 {}  ({:.1f} s)".format(os.path.relpath(path, ROOT), digest,
                                                   time.time() - started))
    validator = os.path.join(ROOT, "evaluation", "validate_submission.py")
    subprocess.call([sys.executable, validator, path])


if __name__ == "__main__":
    main()
