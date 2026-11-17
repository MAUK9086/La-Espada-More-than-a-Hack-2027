"""Out-of-fold model outputs for every calibration match, cached once for decoder experiments.

File structure
--------------
- FEATURE_KEYS: the per-frame feature arrays the P1 evidence and the decoder need (no players).
- build(): for each calibration match under a test profile, apply the fold model saved by
  dropout_robustness.py for a training profile (trained on the OTHER matches), and save
  probabilities, features, labels and blocks to
  solution/cache/folds/<match>.npz. One match is in memory at a time. P5 stays rejected, so it is
  not rebuilt.
- load(): the cached folds, as a list of dicts with a feature dict `f` ready for run.evidence().

Decoder-only experiments (external/review_ablation.py) then never hold the large team-shape
matrices in memory and run in seconds per fold. Frame-level derived data stays in the gitignored
cache, never in the repository.

    python solution/external/fold_cache.py r1 k     # train profile, test profile
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
from common import CACHE_DIR  # noqa: E402
from external import dropout_robustness, lomo  # noqa: E402

# Params
FOLD_DIR = os.path.join(CACHE_DIR, "folds")
FEATURE_KEYS = ("ball", "visible", "outside", "near_home", "near_away", "ball_speed", "team_speed")


def build(train, test):
    """Cache each fold's held-out outputs: `train` fold models on `test` inputs."""
    import joblib
    started = time.time()
    dropout_robustness._use(test)
    os.makedirs(FOLD_DIR, exist_ok=True)
    for name, loader in lomo.calibration_matches():
        f, blocks, labels = loader()
        ev = run.evidence(f, blocks)
        _, x_home, x_swap = learned.match_inputs(f, blocks, ev)
        model = joblib.load(dropout_robustness._path(train, "model_{}.joblib".format(name)))
        probs = learned.predict_proba(model, x_home, x_swap)
        np.savez(os.path.join(FOLD_DIR, name + ".npz"), probs=probs.astype(np.float32),
                 labels=labels, blocks=np.array(blocks, dtype=np.int64),
                 pitch_length=np.float64(f["pitch_length"]),
                 **{"f_" + k: f[k] for k in FEATURE_KEYS})
        print("  cached {} ({:.0f} s)".format(name, time.time() - started), flush=True)


def load():
    """Cached folds in calibration order.

    Returns:
        list[dict]: name, f (feature dict), blocks [(half, first, last)], labels, probs.
    """
    out = []
    for name, _ in lomo.calibration_matches():
        path = os.path.join(FOLD_DIR, name + ".npz")
        if not os.path.isfile(path):
            raise SystemExit("no cached fold {}; run solution/external/fold_cache.py".format(name))
        with np.load(path) as z:
            f = {k: z["f_" + k] for k in FEATURE_KEYS}
            f["pitch_length"] = float(z["pitch_length"])
            out.append({"name": name, "f": f, "labels": z["labels"],
                        "blocks": [tuple(int(v) for v in b) for b in z["blocks"]],
                        "probs": z["probs"].astype(np.float64)})
    return out


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
