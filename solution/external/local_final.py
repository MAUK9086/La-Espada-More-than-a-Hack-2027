"""A final P3 model for a feature-group or training-weight candidate, saved apart from the submitted model.

File structure
--------------
- model_path(): where a candidate's model lives, in the gitignored cache, so
  solution/models/p3_hgb.joblib stays the submission.
- parse_weight(): "short=1,edge=0.5" -> learned.SAMPLE_WEIGHT.
- main(): retrain on all 8 calibration matches with the groups (and weights) on the command line.

Used by external/holdout.py --local [--weight]. Same data and settings as the submitted model otherwise.
The groups ball and near3 without weights keep the name p3_local.joblib, the model run held-out
in an earlier round.

    .venv/Scripts/python solution/external/local_final.py ball near3
    .venv/Scripts/python solution/external/local_final.py ball near3 gap impulse bounds
    .venv/Scripts/python solution/external/local_final.py ball near3 --weight short=1,edge=1
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import common  # noqa: E402

common.FLAGS_ENABLED = False

import learned  # noqa: E402
import team_shape  # noqa: E402
from common import CACHE_DIR  # noqa: E402

# Params
MODEL_PATH = os.path.join(CACHE_DIR, "p3_local.joblib")   # ball + near3, no weights
ALPHA = 0.5


def parse_weight(text):
    """'short=1,edge=0.5' -> {'short': 1.0, 'edge': 0.5}; None or '' -> None."""
    if not text:
        return None
    pairs = (item.split("=") for item in text.split(","))
    return {key.strip(): float(value) for key, value in pairs}


def model_path(groups, weight=None):
    """Cache path of the final model for these groups and sample weights."""
    if tuple(groups) == ("ball", "near3") and not weight:
        return MODEL_PATH
    name = "p3_" + "_".join(groups)
    if weight:
        name += "_w" + "_".join("{}{:g}".format(k, v) for k, v in sorted(weight.items()))
    return os.path.join(CACHE_DIR, name + ".joblib")


def main():
    """CLI entry point."""
    from external import lomo
    from external.post_local import apply_sets
    args = apply_sets(sys.argv[1:])
    weight = None
    if "--weight" in args:
        i = args.index("--weight")
        weight = parse_weight(args[i + 1])
        args = args[:i] + args[i + 2:]
    team_shape.LOCAL_GROUPS = tuple(args)
    learned.SAMPLE_WEIGHT = weight
    learned.MODEL_PATH = model_path(tuple(args), weight)
    lomo.train_final_lean(ALPHA)


if __name__ == "__main__":
    main()
