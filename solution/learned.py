"""P3: a learned per-frame H/A/D classifier on team-shape features, used as Viterbi evidence.

File structure
--------------
- match_inputs(): P1 evidence + team-shape feature matrices for one match.
- training_rows(): labelled, subsampled rows from one match in both team perspectives.
- fit(): a deterministic HistGradientBoostingClassifier on rows from OTHER matches.
- predict_proba(): symmetric H/A/D probabilities for every frame of a match.
- save() / load(): the fitted model, at solution/models/p3_hgb.joblib.

Why this shape:
- Borghesi et al. (2023) found that a learned dead/alive classifier beats rule-based ones by 33 points.
- PathCRF found that constrained Viterbi decoding beats unconstrained per-frame decoding by 10 F1 points.
So the classifier only supplies evidence (alpha * log p is added to the P1 scores), and the
existing event-gated Viterbi still decides the sequence.

Training uses labelled data from other matches only (RULES.md). The model never sees a BAR-ATH
label. A fixed random_state and single-pass training make the fitted model, and therefore the
submission, reproducible.
"""

import os

import numpy as np

import team_shape
from common import A, D, H, HERE, X

# Params
MODEL_PATH = os.path.join(HERE, "models", "p3_hgb.joblib")
SUBSAMPLE = 5          # train on every 5th labelled frame: neighbouring frames are near duplicates
HGB_PARAMS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
                  l2_regularization=1.0, early_stopping=False, random_state=0)
SWAP = np.array([A, H, D], dtype=np.int8)   # label map for the team-swapped perspective
SAMPLE_WEIGHT = None   # tested after p3-local, off: dict(short=, edge=) upweights training frames in
                       # short live spells and near label changes (sample_weights())
WEIGHT_SHORT_S = 6.0   # a live spell shorter than this gains up to 1 + short in weight
WEIGHT_EDGE_FRAMES = 12  # frames either side of a label change gain 1 + edge
SKLEARN_VERSION = "1.9.1"                   # pinned: another version gives different scores


def match_inputs(f, blocks, ev):
    """Feature matrices for one match.

    Args:
        f: feature dict with 'players', 'slot_team', 'pitch_length'.
        blocks: [(half, first, last)].
        ev: run.evidence() output for this match.

    Returns:
        tuple: (names, X_home, X_swapped).
    """
    p1 = {"h": ev["h"], "a": ev["a"], "d": ev["d"]}
    return team_shape.build(f, f["players"], f["slot_team"], blocks, p1, f["pitch_length"])


def training_rows(x_home, x_swap, labels):
    """Labelled rows in both perspectives, every SUBSAMPLE-th graded frame.

    Args:
        x_home, x_swap: (N, F) matrices.
        labels: (N,) int8 H/A/D/X.

    Returns:
        tuple: X (M, F), y (M,); with SAMPLE_WEIGHT set, also w (M,).
    """
    index = np.flatnonzero(labels != X)[::SUBSAMPLE]
    y = labels[index]
    out = (np.vstack((x_home[index], x_swap[index])), np.concatenate((y, SWAP[y])))
    if SAMPLE_WEIGHT is None:
        return out
    w = sample_weights(labels, **SAMPLE_WEIGHT)[index]
    return out + (np.concatenate((w, w)),)


def sample_weights(labels, short=0.0, edge=0.0):
    """Per-frame training weights that bring frame-level loss closer to a spell-level metric.

    spellF1 counts a 2 s possession as much as a 30 s one, while an unweighted frame loss gives the
    long one 15 times the weight. A frame of a live (H or A) spell of L seconds gains
    1 + short * max(0, (WEIGHT_SHORT_S - L) / WEIGHT_SHORT_S). A frame within WEIGHT_EDGE_FRAMES of a
    change between H, A and D is multiplied by 1 + edge. Uses only the training match's labels.

    Returns:
        np.ndarray: (N,) weights, 1 where neither applies.
    """
    from common import FPS, runs_of
    w = np.ones(len(labels))
    if short:
        starts, stops, values = runs_of(labels)
        for start, stop, value in zip(starts, stops, values):
            if value in (H, A):
                w[start:stop] += short * max(0.0, (WEIGHT_SHORT_S - (stop - start) / FPS) / WEIGHT_SHORT_S)
    if edge:
        change = np.flatnonzero((labels[1:] != labels[:-1]) & (labels[1:] != X) & (labels[:-1] != X)) + 1
        near = np.zeros(len(labels) + 1)
        np.add.at(near, np.maximum(change - WEIGHT_EDGE_FRAMES, 0), 1)
        np.add.at(near, np.minimum(change + WEIGHT_EDGE_FRAMES, len(labels)), -1)
        w *= np.where(np.cumsum(near)[:-1] > 0, 1.0 + edge, 1.0)
    return w


def fit(rows):
    """Fit the classifier on stacked rows from several matches.

    Args:
        rows: list of (X, y), or (X, y, w) with sample weights.

    Returns:
        sklearn estimator.
    """
    import warnings
    import sklearn
    from sklearn.ensemble import HistGradientBoostingClassifier
    if sklearn.__version__ != SKLEARN_VERSION:
        warnings.warn("scikit-learn {} is not the pinned {}: use the project .venv".format(
            sklearn.__version__, SKLEARN_VERSION))
    x = np.vstack([r[0] for r in rows])
    y = np.concatenate([r[1] for r in rows])
    weighted = all(len(r) == 3 for r in rows)
    if not weighted and any(len(r) == 3 for r in rows):
        raise ValueError("mixed weighted and unweighted rows")
    model = HistGradientBoostingClassifier(**HGB_PARAMS)
    model.fit(x, y, sample_weight=np.concatenate([r[2] for r in rows]) if weighted else None)
    return model


def predict_proba(model, x_home, x_swap):
    """Symmetric H/A/D probabilities: each frame seen from both teams' side and averaged.

    Returns:
        np.ndarray: (N, 3) columns H, A, D.
    """
    classes = list(model.classes_)
    order = [classes.index(c) for c in (H, A, D)]
    p_home = model.predict_proba(x_home)[:, order]
    p_swap = model.predict_proba(x_swap)[:, order]
    p = np.empty_like(p_home)
    p[:, 0] = 0.5 * (p_home[:, 0] + p_swap[:, 1])
    p[:, 1] = 0.5 * (p_home[:, 1] + p_swap[:, 0])
    p[:, 2] = 0.5 * (p_home[:, 2] + p_swap[:, 2])
    return p


def save(model, names, meta):
    """Persist the model with its feature names and training provenance."""
    import joblib
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": model, "names": names, "meta": meta}, MODEL_PATH)


def load():
    """The persisted model bundle, or None when it has not been trained yet."""
    if not os.path.isfile(MODEL_PATH):
        return None
    import joblib
    return joblib.load(MODEL_PATH)
