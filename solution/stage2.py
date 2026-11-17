"""Stage 2: a small classifier on P3's probability trajectory, refining it before the Viterbi.

File structure
--------------
- stage2_inputs(): the 33 features of one team view (log-probabilities, P1 evidence, rolling means
  and past-vs-future trends of the probabilities, entropy, ball seen, ball outside).
- match_inputs(): both team views of one match; swapping H and A in the inputs of the same
  function makes them exact mirrors.
- load(), refine(): the saved Stage-2 model in models/p3_stage2.joblib, applied to P3's probabilities.

Training uses out-of-fold P3 probabilities of the 8 calibration matches only (external/refiner.py,
external/post_local.py); nothing here reads a label. run.py applies Stage 2 only when the model file
exists and was trained on the same P3 feature set.
"""

import os

import numpy as np

import team_shape
from common import HERE

# Params
WINDOWS = (5, 12, 25, 50, 100)      # frames, half-windows of the rolling means (0.2-4 s)
TRENDS = (12, 25, 50)               # frames, past-vs-future windows
LOG_FLOOR = 1e-3
MODEL_PATH = os.path.join(HERE, "models", "p3_stage2.joblib")


def stage2_inputs(probs, scores, visible, outside, inside):
    """The 33 Stage-2 features of one team view.

    Args:
        probs: (N, 3) P3 probabilities, columns H, A, D.
        scores: (N, 3) P1 evidence, same columns.
        visible, outside: (N,) bool.
        inside: (N,) bool frames inside the halves; no window reaches across half-time.

    Returns:
        np.ndarray: (N, 33) float32.
    """
    p = np.where(inside[:, None], probs, np.nan).astype(np.float32)
    columns = [np.log(np.clip(p, LOG_FLOOR, 1.0)), scores.astype(np.float32)]
    for half in WINDOWS:
        columns.append(np.column_stack([team_shape.rolling(p[:, c], half) for c in range(3)]))
    for half in TRENDS:
        columns.append(np.column_stack([team_shape.trend(p[:, c], half) for c in range(3)]))
    entropy = -np.sum(p * np.log(np.clip(p, LOG_FLOOR, 1.0)), axis=1)
    columns.append(np.column_stack((entropy, visible, outside)).astype(np.float32))
    return np.hstack(columns)


def match_inputs(f, blocks, ev, probs):
    """Both team views for one match.

    Args:
        f: feature dict with 'visible' and 'outside'.
        blocks: [(half, first, last)].
        ev: run.evidence() output.
        probs: (N, 3) P3 probabilities.

    Returns:
        tuple: x_home, x_swap.
    """
    scores = np.column_stack((ev["h"], ev["a"], ev["d"]))
    inside = np.zeros(len(probs), dtype=bool)      # from the halves: labels must not reach the inputs
    for _, first, last in blocks:
        inside[first:last + 1] = True
    swap = [1, 0, 2]
    x_home = stage2_inputs(probs, scores, f["visible"], f["outside"], inside)
    x_swap = stage2_inputs(probs[:, swap], scores[:, swap], f["visible"], f["outside"], inside)
    return x_home, x_swap


def load():
    """The saved Stage-2 bundle {'model', 'alpha', 'stage1_names'}, or None when absent."""
    import joblib
    if not os.path.isfile(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def refine(bundle, f, blocks, ev, probs):
    """Stage-2 probabilities for one match, symmetric like P3's."""
    import learned
    x_home, x_swap = match_inputs(f, blocks, ev, probs)
    return learned.predict_proba(bundle["model"], x_home, x_swap)
