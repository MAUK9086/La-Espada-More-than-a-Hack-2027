"""Score the current model on a held-out match (Metrica game 3 or DFL J03WR9), once per decided change.

File structure
--------------
- HOLDOUTS: the held-out matches and their loaders.
- evaluate_holdout(): the model on one held-out match, official metric.
- main(): print the result and append one line to solution/reports/holdout_log.md together with
  the git commit it ran on.

Discipline: a change is decided on calibration games 1-2 first; this script then confirms or
refutes it. The log is the audit trail that the held-out game was not used for tuning -- one
line per decision, never a loop.

    python solution/external/holdout.py --note "P1 baseline"
    python solution/external/holdout.py --holdout dfl --model p3 --note "P3, 8 matches"
"""

import argparse
import datetime
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import run  # noqa: E402
from common import REPORT_DIR, ROOT  # noqa: E402
import possession_metrics as pm  # noqa: E402
from external import metrica_sample, sportec_dfl  # noqa: E402
import common  # noqa: E402

common.FLAGS_ENABLED = False  # fallbacks on other matches are not flags about this submission

# Params
LOG_PATH = os.path.join(REPORT_DIR, "holdout_log.md")
HOLDOUTS = {   # name -> loader returning (features, blocks, labels); the log tags non-default ones
    "metrica3": lambda: metrica_sample.load_game(metrica_sample.HOLDOUT_GAME),
    "dfl": lambda: sportec_dfl.load_match(sportec_dfl.HOLDOUT_MATCH),
}


def evaluate_holdout(model="p1", holdout="metrica3", decoder=None, stage2=False):
    """The official metrics of the current model on a held-out match.

    Args:
        model: 'p1', 'p3' (the saved model from lomo.py --train-final) or 'p5' (p3 plus the saved
            turnover gate from lomo.py --turnover --train-final).
        holdout: a key of HOLDOUTS.
        decoder: optional review_ablation settings, applied to the p3 probabilities only.
        stage2: refine the p3 probabilities with the saved Stage-2 model (external/refiner.py).
    """
    f, blocks, truth = HOLDOUTS[holdout]()
    ev = run.evidence(f, blocks)
    if model in ("p3", "p5"):
        import learned
        bundle = learned.load()
        if bundle is None:
            raise SystemExit("no trained P3 model; run solution/external/lomo.py --train-final")
        names, x_home, x_swap = learned.match_inputs(f, blocks, ev)
        assert names == bundle["names"], "feature set changed since training"
        probs = learned.predict_proba(bundle["model"], x_home, x_swap)
        if stage2:
            import joblib
            from external import refiner
            _, s_home, s_swap = refiner.match_rows({"f": f, "blocks": blocks, "probs": probs}, ev=ev)
            refined = learned.predict_proba(joblib.load(refiner.FINAL_PATH), s_home, s_swap)
            return pm.evaluate(truth, run.decode_evidence(ev, blocks, refined, refiner.FINAL_ALPHA))
        if decoder is not None:
            from external import review_ablation
            fold = {"name": holdout, "f": f, "blocks": blocks, "probs": probs}
            return pm.evaluate(truth, review_ablation.decode_variant(fold, decoder, {}))
        switch = None
        if model == "p5":
            import turnover
            gate = turnover.load()
            if gate is None:
                raise SystemExit("no turnover gate; run lomo.py --turnover --train-final")
            t_names, t_inputs = turnover.match_inputs(f, blocks, ev, (names, x_home, x_swap))
            assert t_names == gate["names"], "turnover features changed since training"
            switch = turnover.switch_costs(turnover.predict(gate["model"], t_inputs),
                                           gate["prior"], gate["meta"]["beta"])
        labels = run.decode_evidence(ev, blocks, probs, bundle["meta"]["alpha"], switch)
    else:
        labels = run.decode_evidence(ev, blocks)
    return pm.evaluate(truth, labels)


def commit_id():
    """Short git commit of the working tree, marked '+dirty' when there are uncommitted edits."""
    try:
        head = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
        dirty = subprocess.call(["git", "-C", ROOT, "diff", "--quiet", "--", "solution"]) != 0
        return head + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--note", required=True, help="which decided change this run checks")
    parser.add_argument("--model", choices=("p1", "p3", "p5"), default="p1")
    parser.add_argument("--holdout", choices=sorted(HOLDOUTS), default="metrica3")
    parser.add_argument("--old-profile", action="store_true",
                        help="test inputs and labels as before the label-free dropout (B0)")
    parser.add_argument("--kinematic", action="store_true",
                        help="test inputs with the kinematic dropout, the closest to BAR-ATH")
    parser.add_argument("--decoder-candidate", action="store_true",
                        help="p3 with review_ablation.CANDIDATE, the declared decoder settings")
    parser.add_argument("--stage2", action="store_true", help="p3 with the saved Stage-2 refiner")
    parser.add_argument("--local", nargs="+", default=None,
                        help="team_shape.LOCAL_GROUPS, with the model from external/local_final.py")
    parser.add_argument("--weight", default=None,
                        help="with --local: the sample weights the model was trained with, e.g. short=1,edge=1")
    parser.add_argument("--stage2-model", default=None, help="with --stage2: a Stage-2 model file")
    parser.add_argument("--stage2-alpha", type=float, default=None, help="with --stage2: its alpha")
    parser.add_argument("--set", action="append", default=[], help="team_shape NAME=VALUE, as in post_local.py")
    args = parser.parse_args()
    if args.set:
        from external.post_local import apply_sets
        apply_sets(sum((["--set", s] for s in args.set), []))
    if args.local:
        import learned
        import team_shape
        from external import local_final
        team_shape.LOCAL_GROUPS = tuple(args.local)
        learned.MODEL_PATH = local_final.model_path(tuple(args.local), local_final.parse_weight(args.weight))
    if args.stage2_model or args.stage2_alpha is not None:
        from external import refiner
        refiner.FINAL_PATH = args.stage2_model or refiner.FINAL_PATH
        refiner.FINAL_ALPHA = args.stage2_alpha if args.stage2_alpha is not None else refiner.FINAL_ALPHA
    if args.kinematic:
        sportec_dfl.DROPOUT_RATIO = sportec_dfl.KINEMATIC
    if args.old_profile:
        sportec_dfl.DROPOUT_RATIO, metrica_sample.CONTROLLED_LABELS = None, False

    decoder = None
    if args.decoder_candidate:
        from external import review_ablation
        decoder = review_ablation.CANDIDATE
    result = evaluate_holdout(args.model, args.holdout, decoder, args.stage2)
    note = args.note if args.holdout == "metrica3" else "[{}] {}".format(args.holdout, args.note)
    line = ("| {} | {} | {} | {:.4f} | {:.4f} | {}/{} |".format(
        datetime.date.today().isoformat(), commit_id(), note, result["spell_f1_mean"],
        result["macro_f1"], result["n_pred_spells"], result["n_gt_spells"]))
    os.makedirs(REPORT_DIR, exist_ok=True)
    if not os.path.isfile(LOG_PATH):
        with open(LOG_PATH, "w", encoding="utf-8") as handle:
            handle.write("# Held-out game log (Metrica sample game 3)\n\n"
                         "| date | commit | change checked | spellF1 | macroF1 | poss pred/true |\n"
                         "|---|---|---|---|---|---|\n")
    with open(LOG_PATH, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line)


if __name__ == "__main__":
    main()
