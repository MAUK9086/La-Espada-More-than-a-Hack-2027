"""P5: a learned "controlled turnover happens here" probability that gates H <-> A switches.

File structure
--------------
- window_features(): before/after summaries around every frame -- each team's control evidence,
  its distance to the ball, ball visibility, a kick spike -- in both team perspectives.
- match_inputs(): those plus the P3 team-shape matrix (which carries both teams' oriented
  centroid velocities and their trends: after a real turnover both teams reverse direction).
  The team-shape columns are borrowed by reference, not copied: rows are only joined for the
  frames being trained or predicted on, in chunks, so memory stays at the window columns.
- targets(): 1 within TOL frames of a direct H <-> A change in the labels, else 0.
- training_rows() / fit() / predict(): a deterministic HistGradientBoostingClassifier, symmetric
  over the two perspectives.
- switch_costs(): per-frame H <-> A cost for the Viterbi, C_SWITCH - beta * log(p / prior).
- switch_costs_peak(): review variant (R22): discount only at local peaks of p, penalise where
  p < prior, clipped to [2, 24].
- save() / load(): the fitted model at solution/models/turnover_hgb.joblib.

Why: ~85% of the lost spellF1 on the calibration
matches is in *who* has the ball and *when it changes*. With one constant switch cost a touch,
a deflection and a real turnover all pay the same price, so the decoder either splits true
possessions on touches or merges real turnovers. Boundary/transition branches that gate class
changes are the standard fix for that in action segmentation (ASRF, WACV 2021; Boundary-Aware
Cascade, ECCV 2020), and it mirrors how P1 already gates dead-ball changes on detected events.

With beta = 0 the switch cost is the constant C_SWITCH, i.e. exactly the previous model.
Trained on other matches only (RULES.md); fixed random_state, so the output is reproducible.
"""

import os

import numpy as np

import learned
from common import A, H, HERE, X
from possession import C_SWITCH

# Params
MODEL_PATH = os.path.join(HERE, "models", "turnover_hgb.joblib")
TOL = 6                        # frames either side of a labelled switch counted as positive
WINDOWS = (12, 25, 50, 100)    # before/after window lengths, frames (0.5, 1, 2, 4 s)
KICK_HALF = 5                  # frames either side for the ball-speed spike
PROB_FLOOR = 1e-3


def _past_future(values, half):
    """NaN-aware mean of the previous and of the next `half` frames (frame itself excluded)."""
    v = np.nan_to_num(values, nan=0.0).astype(np.float64)
    w = (~np.isnan(values)).astype(np.float64)
    cv = np.concatenate(([0.0], np.cumsum(v)))
    cw = np.concatenate(([0.0], np.cumsum(w)))
    n = len(values)
    idx = np.arange(n)
    a0, a1 = np.maximum(idx - half, 0), idx
    b0, b1 = np.minimum(idx + 1, n), np.minimum(idx + 1 + half, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        past = (cv[a1] - cv[a0]) / (cw[a1] - cw[a0])
        future = (cv[b1] - cv[b0]) / (cw[b1] - cw[b0])
    return past.astype(np.float32), future.astype(np.float32)


def _local_max(values, half):
    """Maximum over +/- half frames, NaN treated as 0."""
    v = np.nan_to_num(values, nan=0.0)
    padded = np.concatenate((np.zeros(half), v, np.zeros(half)))
    return np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1).max(1).astype(np.float32)


def window_features(f, ev):
    """Before/after summaries for every frame, in both team perspectives.

    Args:
        f: feature dict (near_home, near_away, visible, ball_speed).
        ev: run.evidence() output (h, a).

    Returns:
        tuple: (names, X_home (N, F) float32, X_swapped (N, F) float32).
    """
    per_team = {"ev": (ev["h"].astype(np.float64), ev["a"].astype(np.float64)),
                "near": (f["near_home"], f["near_away"])}
    home, swap = {}, {}
    for key, (h, a) in per_team.items():
        for w in WINDOWS:
            hp, hf = _past_future(h, w)
            ap, af = _past_future(a, w)
            for side, (own_p, own_f, opp_p, opp_f) in (("home", (hp, hf, ap, af)),
                                                       ("swap", (ap, af, hp, hf))):
                cols = home if side == "home" else swap
                cols["{}_own_past_{}".format(key, w)] = own_p
                cols["{}_own_future_{}".format(key, w)] = own_f
                cols["{}_opp_past_{}".format(key, w)] = opp_p
                cols["{}_opp_future_{}".format(key, w)] = opp_f
    shared = {}
    visible = f["visible"].astype(np.float32)
    for w in WINDOWS[:3]:
        shared["visible_past_{}".format(w)], shared["visible_future_{}".format(w)] = \
            _past_future(visible, w)
    speed = np.asarray(f["ball_speed"], dtype=np.float64)
    shared["ball_speed_max"] = _local_max(speed, KICK_HALF)
    sp, sf = _past_future(speed, KICK_HALF)
    shared["ball_speed_jump"] = sf - sp
    home.update(shared)
    swap.update(shared)
    names = sorted(home)
    return (names, np.column_stack([home[k] for k in names]),
            np.column_stack([swap[k] for k in names]))


class Inputs:
    """Team-shape matrices (shared with P3, not copied) side by side with the window features."""

    def __init__(self, ts_home, ts_swap, w_home, w_swap):
        self.parts_home, self.parts_swap = (ts_home, w_home), (ts_swap, w_swap)

    def rows(self, index, swapped=False):
        """Joined feature rows for the given frame indices (or a slice)."""
        parts = self.parts_swap if swapped else self.parts_home
        return np.hstack([part[index] for part in parts])

    def __len__(self):
        return len(self.parts_home[0])


def match_inputs(f, blocks, ev, team_shape_inputs=None):
    """Turnover feature inputs for one match.

    Args:
        f, blocks, ev: as for learned.match_inputs().
        team_shape_inputs: optional (names, X_home, X_swapped) already computed by
            learned.match_inputs(), to avoid building them twice.

    Returns:
        tuple: (names, Inputs).
    """
    if team_shape_inputs is None:
        team_shape_inputs = learned.match_inputs(f, blocks, ev)
    ts_names, ts_home, ts_swap = team_shape_inputs
    w_names, w_home, w_swap = window_features(f, ev)
    names = ["ts_" + n for n in ts_names] + w_names
    return names, Inputs(ts_home, ts_swap, w_home, w_swap)


def targets(labels, tol=TOL):
    """1 within `tol` frames of a direct H <-> A change, 0 elsewhere, -1 on X frames."""
    change = np.flatnonzero(((labels[1:] == H) & (labels[:-1] == A))
                            | ((labels[1:] == A) & (labels[:-1] == H))) + 1
    y = np.zeros(len(labels), dtype=np.int8)
    for t in change:
        y[max(t - tol, 0):t + tol + 1] = 1
    y[labels == X] = -1
    return y


def training_rows(inputs, labels):
    """All positive frames plus every SUBSAMPLE-th negative one, in both perspectives."""
    y = targets(labels)
    neg = np.flatnonzero(y == 0)[::learned.SUBSAMPLE]
    pos = np.flatnonzero(y == 1)
    index = np.sort(np.concatenate((neg, pos)))
    yy = y[index]
    return (np.vstack((inputs.rows(index), inputs.rows(index, swapped=True))),
            np.concatenate((yy, yy)))


def fit(rows):
    """Fit the turnover classifier on rows from several matches.

    Returns:
        tuple: (sklearn estimator, prior) where prior is the positive share of the training rows,
        the reference level of predict() (negatives are subsampled, so it is not the raw rate).
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    x = np.vstack([r[0] for r in rows])
    y = np.concatenate([r[1] for r in rows])
    model = HistGradientBoostingClassifier(**learned.HGB_PARAMS)
    model.fit(x, y)
    return model, float(np.mean(y))


def predict(model, inputs, chunk=20000):
    """Symmetric turnover probability per frame, (N,), predicted in chunks to bound memory."""
    col = list(model.classes_).index(1)
    out = np.empty(len(inputs))
    for start in range(0, len(inputs), chunk):
        part = slice(start, min(start + chunk, len(inputs)))
        out[part] = 0.5 * (model.predict_proba(inputs.rows(part))[:, col]
                           + model.predict_proba(inputs.rows(part, swapped=True))[:, col])
    return out


def switch_costs(p, prior, beta):
    """Per-frame H <-> A switch cost: C_SWITCH where p equals the prior, cheaper where a turnover
    is likely, dearer where it is not; never below 0. beta = 0 gives the constant C_SWITCH."""
    gain = np.log(np.clip(p, PROB_FLOOR, 1.0) / prior)
    return np.maximum(C_SWITCH - beta * gain, 0.0)


def switch_costs_peak(p, prior, beta, half=3):
    """Review variant (R22) of switch_costs(): the discount applies only at a local maximum of p
    (within +/- half frames), a penalty of 0.75 * beta * |gain| applies where p < prior, and the
    result is clipped to [2, 24]. beta = 0 gives the constant C_SWITCH."""
    p = np.asarray(p, dtype=float)
    if beta <= 0:
        return np.full(len(p), C_SWITCH)
    gain = np.log(np.clip(p, PROB_FLOOR, 1.0) / prior)
    padded = np.concatenate((np.zeros(half), p, np.zeros(half)))
    local_max = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1).max(1)
    peak = (p >= local_max - 1e-9) & (gain > 0)
    cost = C_SWITCH - np.where(peak, beta * gain, 0.0) + np.where(gain < 0, -0.75 * beta * gain, 0.0)
    return np.clip(cost, 2.0, 24.0)


def save(model, prior, names, meta):
    """Persist the model with its prior, feature names and training provenance."""
    import joblib
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": model, "prior": prior, "names": names, "meta": meta}, MODEL_PATH)


def load():
    """The persisted model bundle, or None when it has not been trained yet."""
    if not os.path.isfile(MODEL_PATH):
        return None
    import joblib
    return joblib.load(MODEL_PATH)
