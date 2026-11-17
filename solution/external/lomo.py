"""Leave-one-match-out evaluation of P3 (learned team-shape evidence) against P1, on other matches.

File structure
--------------
- prepare(): per calibration match -- features, labels, P1 evidence, team-shape matrices.
- fold(): train on all matches but one, score the held-out one at every ALPHAS value.
- fold_turnover(): the same with the P5 turnover gate, at every BETAS value (alpha fixed).
- report(): the per-fold table and the mean over folds.
- main(): report(); optional --train-final writes the model trained on all calibration matches
  to solution/models/p3_hgb.joblib (--final-only skips the folds).

Held-out discipline: the held-out game (metrica_sample.HOLDOUT_GAME) is never a fold here. It is
scored once per decided change by holdout.py.

    python solution/external/lomo.py
    python solution/external/lomo.py --train-final --alpha 1.0
    python solution/external/lomo.py --train-final --final-only --alpha 0.5   # skip the folds
    python solution/external/lomo.py --turnover                 # P5 beta sweep at alpha 0.5
    python solution/external/lomo.py --turnover --train-final --final-only --alpha 0.5 --beta 2
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import common  # noqa: E402

common.FLAGS_ENABLED = False  # fallbacks on other matches are not flags about this submission

import learned  # noqa: E402
import run  # noqa: E402
import turnover  # noqa: E402
import possession_metrics as pm  # noqa: E402
from common import A, D, H, X  # noqa: E402
from external import metrica_sample  # noqa: E402

# Params
ALPHAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)   # weight of the learned log-probabilities (0 = P1)
BETAS = (0.0, 0.5, 1.0, 2.0, 4.0)          # weight of the turnover gate (0 = P3)
TURNOVER_ALPHA = 0.5                       # alpha the beta sweep runs at (chosen in P3)


def calibration_matches():
    """(name, loader) for every calibration match on disk, held-out matches excluded.

    Metrica sample games 1-2 always; the Sportec Open DFL matches whenever their files have been
    downloaded (external/sportec_dfl.py), except its held-out match.
    """
    from external import sportec_dfl
    sources = [("metrica{}".format(g), lambda g=g: metrica_sample.load_game(g))
               for g in metrica_sample.GAMES]
    sources += [("dfl_" + m, lambda m=m: sportec_dfl.load_match(m))
                for m in sportec_dfl.available() if m != sportec_dfl.HOLDOUT_MATCH]
    return sources


def prepare(sources, with_turnover=False):
    """Everything per match that does not depend on the fold.

    Args:
        sources: list of (name, loader) from calibration_matches().
        with_turnover: also build the P5 turnover inputs (t_names, t_inputs).

    Returns:
        list[dict]: name, f, blocks, labels, ev, x_home, x_swap (+ turnover matrices).
    """
    out = []
    for name, loader in sources:
        started = time.time()
        f, blocks, labels = loader()
        ev = run.evidence(f, blocks)
        names, x_home, x_swap = learned.match_inputs(f, blocks, ev)
        item = {"name": name, "f": f, "blocks": blocks, "labels": labels,
                "ev": ev, "names": names, "x_home": x_home, "x_swap": x_swap}
        if with_turnover:
            item["t_names"], item["t_inputs"] = turnover.match_inputs(
                f, blocks, ev, (names, x_home, x_swap))
        out.append(item)
        print("  prepared {} ({} features) in {:.0f} s".format(name, len(names),
                                                                 time.time() - started), flush=True)
    return out


def frame_scores(probs, labels):
    """Per-frame quality of the classifier alone: alive/dead and team accuracy on graded frames."""
    graded = labels != X
    pred = np.argmax(probs, 1)
    pred = np.array([H, A, D], dtype=np.int8)[pred]
    alive_true = labels[graded] != D
    alive_pred = pred[graded] != D
    both = graded & (labels != D) & (pred != D)
    return {"acc": float(np.mean(pred[graded] == labels[graded])),
            "alive_dead_acc": float(np.mean(alive_true == alive_pred)),
            "team_acc": float(np.mean(pred[both] == labels[both])) if both.any() else float("nan")}


def fold(matches, held):
    """Train on every match except `held`, score `held` for every alpha.

    Returns:
        dict: alpha -> (spellF1, macroF1, n_pred_spells); plus 'frame' classifier scores.
    """
    rows = [learned.training_rows(m["x_home"], m["x_swap"], m["labels"])
            for m in matches if m is not held]
    model = learned.fit(rows)
    probs = learned.predict_proba(model, held["x_home"], held["x_swap"])
    result = {"frame": frame_scores(probs, held["labels"])}
    for alpha in ALPHAS:
        labels = run.decode_evidence(held["ev"], held["blocks"], probs, alpha)
        r = pm.evaluate(held["labels"], labels)
        result[alpha] = (r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"])
    return result


def fold_turnover(matches, held, alpha=TURNOVER_ALPHA):
    """Train P3 and the turnover gate without `held`, score `held` for every beta.

    Returns:
        dict: beta -> (spellF1, macroF1, n_pred_spells, n_gt_spells); plus 'ap', the turnover
        classifier's average precision on the held-out match.
    """
    from sklearn.metrics import average_precision_score
    others = [m for m in matches if m is not held]
    model = learned.fit([learned.training_rows(m["x_home"], m["x_swap"], m["labels"])
                         for m in others])
    probs = learned.predict_proba(model, held["x_home"], held["x_swap"])
    t_model, prior = turnover.fit([turnover.training_rows(m["t_inputs"], m["labels"])
                                   for m in others])
    p_turn = turnover.predict(t_model, held["t_inputs"])
    y = turnover.targets(held["labels"])
    result = {"ap": float(average_precision_score(y[y >= 0], p_turn[y >= 0])),
              "base": float(np.mean(y[y >= 0]))}
    for beta in BETAS:
        switch = turnover.switch_costs(p_turn, prior, beta)
        labels = run.decode_evidence(held["ev"], held["blocks"], probs, alpha, switch)
        r = pm.evaluate(held["labels"], labels)
        result[beta] = (r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"])
    return result


def report_turnover(matches):
    """Per-fold beta sweep table and the mean over folds."""
    print("\n| held-out | turnover AP (base rate) | " +
          " | ".join("b={}".format(b) for b in BETAS) + " |")
    print("|---|---|" + "---|" * len(BETAS))
    table = {b: [] for b in BETAS}
    for held in matches:
        result = fold_turnover(matches, held)
        cells = []
        for b in BETAS:
            spell, macro, n_pred, n_gt = result[b]
            table[b].append((spell, macro))
            cells.append("{:.4f} / {:.3f} ({}/{})".format(spell, macro, n_pred, n_gt))
        print("| {} | {:.3f} ({:.3f}) | {} |".format(held["name"], result["ap"], result["base"],
                                                    " | ".join(cells)), flush=True)
    print("| **mean** | | " + " | ".join(
        "**{:.4f}** / {:.3f}".format(np.mean([s for s, _ in table[b]]),
                                      np.mean([m for _, m in table[b]])) for b in BETAS) + " |")
    print("cells: spellF1 / macroF1 (predicted/true possessions); b=0 is P3 at alpha={}".format(
        TURNOVER_ALPHA))


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-final", action="store_true",
                        help="fit on all calibration matches and save the model")
    parser.add_argument("--alpha", type=float, default=None,
                        help="alpha recorded with the saved model (required with --train-final)")
    parser.add_argument("--final-only", action="store_true",
                        help="with --train-final: skip the folds (alpha already chosen from them)")
    parser.add_argument("--turnover", action="store_true",
                        help="P5: sweep the turnover-gate weight beta instead of alpha")
    parser.add_argument("--beta", type=float, default=None,
                        help="with --turnover --train-final: beta recorded with the saved gate")
    args = parser.parse_args()

    started = time.time()
    if args.train_final and args.alpha is None:
        raise SystemExit("--train-final needs --alpha (the value chosen from this table)")
    if args.turnover and args.train_final and args.beta is None:
        raise SystemExit("--turnover --train-final needs --beta")
    if args.train_final and args.final_only and not args.turnover:
        train_final_lean(args.alpha)
        print("done in {:.0f} s".format(time.time() - started))
        return
    matches = prepare(calibration_matches(), with_turnover=args.turnover)
    if not args.final_only:
        report_turnover(matches) if args.turnover else report(matches)
    if args.train_final:
        rows = [learned.training_rows(m["x_home"], m["x_swap"], m["labels"]) for m in matches]
        model = learned.fit(rows)
        learned.save(model, matches[0]["names"], {
            "alpha": args.alpha, "trained_on": [m["name"] for m in matches],
            "hgb_params": learned.HGB_PARAMS, "subsample": learned.SUBSAMPLE})
        print("saved", learned.MODEL_PATH)
        if args.turnover:
            t_model, prior = turnover.fit([turnover.training_rows(m["t_inputs"], m["labels"])
                                           for m in matches])
            turnover.save(t_model, prior, matches[0]["t_names"], {
                "beta": args.beta, "alpha": args.alpha, "prior": prior, "tol": turnover.TOL,
                "trained_on": [m["name"] for m in matches]})
            print("saved", turnover.MODEL_PATH)
    print("done in {:.0f} s".format(time.time() - started))


def train_final_lean(alpha):
    """The final model, keeping only each match's training rows in memory."""
    rows, trained_on, names = [], [], None
    for name, loader in calibration_matches():
        f, blocks, labels = loader()
        ev = run.evidence(f, blocks)
        names, x_home, x_swap = learned.match_inputs(f, blocks, ev)
        rows.append(learned.training_rows(x_home, x_swap, labels))
        trained_on.append(name)
        print("  rows {}".format(name), flush=True)
    learned.save(learned.fit(rows), names, {
        "alpha": alpha, "trained_on": trained_on,
        "hgb_params": learned.HGB_PARAMS, "subsample": learned.SUBSAMPLE})
    print("saved", learned.MODEL_PATH)


def report(matches):
    """Print the per-fold table and the mean over folds."""
    print("\n| held-out | classifier acc | alive/dead | team | " +
          " | ".join("a={}".format(a) for a in ALPHAS) + " |")
    print("|---|---|---|---|" + "---|" * len(ALPHAS))
    table = {a: [] for a in ALPHAS}
    for held in matches:
        result = fold(matches, held)
        fr = result["frame"]
        cells = []
        for a in ALPHAS:
            spell, macro, n_pred, n_gt = result[a]
            table[a].append((spell, macro))
            cells.append("{:.4f} / {:.3f} ({})".format(spell, macro, n_pred))
        print("| {} | {:.3f} | {:.3f} | {:.3f} | {} |".format(
            held["name"], fr["acc"], fr["alive_dead_acc"], fr["team_acc"], " | ".join(cells)),
            flush=True)
    print("| **mean** | | | | " + " | ".join(
        "**{:.4f}** / {:.3f}".format(np.mean([s for s, _ in table[a]]),
                                      np.mean([m for _, m in table[a]])) for a in ALPHAS) + " |")
    print("cells: spellF1 / macroF1 (predicted possessions); a=0 is P1")


if __name__ == "__main__":
    main()
