"""Sportec Open DFL matches (Bassek et al., Sci. Data 2025, CC-BY 4.0) as calibration data.

File structure
--------------
- MATCHES: the seven match ids.
- paths(): the XML files of one match under DFL_DIR.
- available(): matches whose files are on disk.
- parse_meta(): home/away team ids, person -> team, pitch size.
- parse_tracking(): streaming lxml parse -> players (N, S, 3), ball (N, 3), BallStatus and
  BallPossession per frame, period of each frame.
- controlled(): touch-level possession -> controlled possession (CHALLENGE.md's definition).
- load_match(): features dict (same keys as features.build()) + blocks + labels, with the same
  ATD-like degradation as metrica_sample, cached per match to solution/cache/dfl_<id>.npz.
- drop_ball(), drop_ball_label_free(), drop_ball_kinematic(): simulated ball dropout (old profile, the
  label-free profile r1 used for training, and the kinematic test profile k).

Why this data matters: the DFL position feed carries frame-level BallStatus
(alive/dead) and BallPossession (home/away) from DFL's corrected tracking -- the same kind of
label the hidden ground truth is derived from. These are other matches, so using them is allowed.

BallStatus is used as is. BallPossession is touch-level: it flips on every touch, deflection and
duel (median spell ~3 s, most under 4 s), while CHALLENGE.md counts only *controlled* possession
("a simple touch of the ball without control" does not change it; in loose phases "possession
stays with the last team that had it"). controlled() therefore lets a team take over only once it
has held the ball for CONTROL_MIN_S -- the challenge's own first proxy, "a player keeps the ball
for a few seconds".

CONTROL_MIN_S was first fixed at 3 s from that wording. It is now 2 s, chosen for robustness
rather than score, in a label-rule robustness test on these matches. A model trained on
2 s labels scores as well as the 3 s model when 3 s is the true rule, and better when the true rule
is shorter. It has the best worst case over rules of 1-4 s, and loses only if the truth is 4 s or
longer. The held-out matches and the public window were not used for this choice.

Getting the files: figshare blocks automated downloads, so download them in a browser from the
article's data record (Scientific Data, doi:10.1038/s41597-025-04505-y), or use the direct links
https://springernature.figshare.com/ndownloader/files/<id> with the ids below, and save them as
  ../external/sportec-dfl/<match>_meta.xml  and  ../external/sportec-dfl/<match>_tracking.xml

Format (from kloppy's sportec deserializer): <FrameSet GameSection=firstHalf|secondHalf
TeamId=... PersonId=...> holding <Frame N=.. X=.. Y=.. S=..>; the ball FrameSet has
TeamId="BALL" and its frames carry BallStatus (1 alive) and BallPossession (1 home, 2 away).
Coordinates are metres from the centre spot.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import features  # noqa: E402
from common import A, CACHE_DIR, D, H, ROOT, X, runs_of, FPS  # noqa: E402
from external.metrica_sample import degrade_ball, degrade_players  # noqa: E402

# Params
DFL_DIR = os.environ.get("DFL_DIR") or os.path.join(os.path.dirname(ROOT), "external", "sportec-dfl")
MATCHES = {   # match id -> (meta file id, tracking file id) on figshare (kloppy's table)
    "J03WMX": (51643475, 51643514), "J03WN1": (51643472, 51643517),
    "J03WPY": (51643487, 51643526), "J03WOH": (51643478, 51643520),
    "J03WQQ": (51643484, 51643529), "J03WOY": (51643481, 51643523),
    "J03WR9": (51643490, 51643532),
}
HOLDOUT_MATCH = "J03WR9"   # second held-out match; never a calibration fold
SPEED_TO_MS = 1 / 3.6      # the S attribute is in km/h; every other source and threshold is m/s
SECTIONS = {"firstHalf": 1, "secondHalf": 2}
CONTROL_MIN_S = 2.0        # a team takes over possession once it has held the ball this long


def paths(match):
    """(meta, tracking) file paths for a match."""
    return (os.path.join(DFL_DIR, "{}_meta.xml".format(match)),
            os.path.join(DFL_DIR, "{}_tracking.xml".format(match)))


def available():
    """Match ids whose two files are present and non-empty."""
    return [m for m in MATCHES if all(os.path.isfile(p) and os.path.getsize(p) > 1000
                                      for p in paths(m))]


def parse_meta(path):
    """Teams, persons and pitch size from the DFL match-information XML.

    Returns:
        dict: home_id, away_id, team_of {person_id: 0/1}, length, width.
    """
    from lxml import etree
    root = etree.parse(path).getroot()
    home_id = away_id = None
    team_of = {}
    for team in root.iter("Team"):
        role, team_id = team.get("Role"), team.get("TeamId")
        code = 0 if role == "home" else 1
        if role == "home":
            home_id = team_id
        else:
            away_id = team_id
        for player in team.iter("Player"):
            team_of[player.get("PersonId")] = code
    environment = next(root.iter("Environment"), None)
    length = float(environment.get("PitchX", 105)) if environment is not None else 105.0
    width = float(environment.get("PitchY", 68)) if environment is not None else 68.0
    return {"home_id": home_id, "away_id": away_id, "team_of": team_of,
            "length": length, "width": width}


def parse_tracking(path, meta):
    """Stream the position XML into dense arrays (periods concatenated in order).

    Returns:
        dict: players (N, S, 3) in metres from the bottom-left corner, ball (N, 3), status (N,),
        possession (N,), periods (N,), slot_team (S,).
    """
    from lxml import etree
    persons = sorted(meta["team_of"], key=lambda p: (meta["team_of"][p], p))
    slot_of = {p: i for i, p in enumerate(persons)}
    frames = {1: {}, 2: {}}    # period -> {N: {"ball": (...), slot: (x, y, s)}}
    for _, elem in etree.iterparse(path, events=("end",), tag="FrameSet"):
        period = SECTIONS.get(elem.get("GameSection"))
        if period is None:
            elem.clear()
            continue
        is_ball = elem.get("TeamId") == "BALL"
        person = elem.get("PersonId")
        if not is_ball and person not in slot_of:
            elem.clear()   # officials and unknown persons
            continue
        store = frames[period]
        for frame in elem.iterchildren("Frame"):
            n = int(frame.get("N"))
            row = store.setdefault(n, {})
            x, y = float(frame.get("X")), float(frame.get("Y"))
            if is_ball:
                row["ball"] = (x, y, float(frame.get("S", "nan")),
                               int(frame.get("BallStatus", 0)), int(frame.get("BallPossession", 0)))
            else:
                row[slot_of[person]] = (x, y, float(frame.get("S", "nan")))
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

    order = [(period, n) for period in (1, 2) for n in sorted(frames[period])
             if "ball" in frames[period][n]]
    total = len(order)
    players = np.full((total, len(persons), 3), np.nan, dtype=np.float32)
    ball = np.full((total, 3), np.nan, dtype=np.float32)
    status = np.zeros(total, dtype=np.int8)
    possession = np.zeros(total, dtype=np.int8)
    periods = np.zeros(total, dtype=np.int8)
    half_l, half_w = meta["length"] / 2.0, meta["width"] / 2.0
    for i, (period, n) in enumerate(order):
        row = frames[period][n]
        bx, by, bs, st, po = row.pop("ball")
        ball[i] = (bx + half_l, by + half_w, bs)
        status[i], possession[i], periods[i] = st, po, period
        for slot, (x, y, s) in row.items():
            players[i, slot] = (x + half_l, y + half_w, s)
    slot_team = np.array([meta["team_of"][p] for p in persons], dtype=np.int8)
    return {"players": players, "ball": ball, "status": status, "possession": possession,
            "periods": periods, "slot_team": slot_team}


def load_match(match, degrade=True):
    """Features, blocks and labels for one DFL match, in the calibration format.

    Labels: D where BallStatus is 0, otherwise H (BallPossession 1) or A (2) after controlled().
    Frames with BallStatus 1 but no possession value are X (not graded).

    Returns:
        tuple: (features dict, blocks, labels).
    """
    cache = os.path.join(CACHE_DIR, "dfl_{}.npz".format(match))
    if os.path.isfile(cache):
        with np.load(cache) as cached:
            raw = {k: cached[k] for k in cached.files}
    else:
        meta_path, tracking_path = paths(match)
        meta = parse_meta(meta_path)
        raw = parse_tracking(tracking_path, meta)
        raw["pitch"] = np.array([meta["length"], meta["width"]])
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(cache, **raw)

    touch = np.where(raw["status"] == 0, D,
                     np.where(raw["possession"] == 1, H,
                              np.where(raw["possession"] == 2, A, X))).astype(np.int8)
    labels = controlled(touch)
    seed = sum(ord(c) for c in match)
    players, ball = raw["players"].copy(), raw["ball"]
    players[:, :, 2] *= SPEED_TO_MS
    if degrade:
        ball = degrade_ball(ball, seed=seed)
        players = degrade_players(players, seed=seed)
        # The DFL ball is always present; drop it in spells like the ATD's gaps.
        ball = drop_ball(ball, labels, seed, features.team_speed(players))
    ball = features.clean_ball(ball)
    near_home, near_away = features.nearest_by_team(players, ball, raw["slot_team"])
    f = {
        "ball": ball, "visible": ~np.isnan(ball[:, 0]), "ball_speed": features.ball_speed(ball),
        "near_home": near_home, "near_away": near_away,
        "team_speed": features.team_speed(players), "outside": features.outside_pitch(ball),
        "slot_team": raw["slot_team"], "players": players,
        "pitch_length": float(raw["pitch"][0]),
    }
    blocks = []
    for period in (1, 2):
        index = np.flatnonzero(raw["periods"] == period)
        if len(index):
            blocks.append((period, int(index[0]), int(index[-1])))
    return f, blocks, labels


def controlled(touch, min_s=None):
    """Touch-level H/A/D labels -> controlled possession.

    Within each live spell the restarting team owns the ball from the restart (CHALLENGE.md: a
    restart always has a clear possessor). After that, possession passes to the other team only
    at the start of a run of theirs lasting at least `min_s`; shorter runs (touches, deflections,
    lost duels) stay with the current owner.

    Args:
        touch: (N,) int8 labels H/A/D/X.
        min_s: minimum held time, seconds; CONTROL_MIN_S if None.

    Returns:
        np.ndarray: (N,) int8 labels, D and X frames unchanged.
    """
    min_s = CONTROL_MIN_S if min_s is None else min_s
    out = touch.copy()
    min_frames = int(round(min_s * FPS))
    starts, stops, values = runs_of(touch)
    owner = None
    for start, stop, value in zip(starts, stops, values):
        if value not in (H, A):
            if value == D:
                owner = None   # a stoppage: the next restart sets the owner afresh
            continue
        if owner is None or (value != owner and stop - start >= min_frames):
            owner = value
        out[start:stop] = owner
    return out


# Ball-dropout profile of the ATD on BAR-ATH, measured without labels (diagnostics.md):
# 59% visible on graded frames, gaps mostly >= 5 s. In the public window the ATD ball was
# missing on ~34% of dead frames and ~8% of live frames, so dropout is applied per state.
# Not used any more (DROPOUT_RATIO = None restores it): these rates came from public-window labels;
# kept only so the earlier results can be reproduced.
DROP_RATE = {D: 0.55, H: 0.10, A: 0.10}
DROP_MEAN_S = 12.0


def _drop_ball_like_atd(ball, labels, seed):
    """Remove the ball in long spells, more often during stoppages.

    Each second a gap starts with probability DROP_RATE[state] / DROP_MEAN_S and lasts an
    exponential DROP_MEAN_S on average, so the steady-state missing share is about
    rate / (1 + rate): ~35% of dead frames and ~9% of live ones.
    """
    rng = np.random.default_rng(2000 + seed)
    out = ball.copy()
    n = len(ball)
    t = 0
    while t < n:
        state = labels[t] if labels[t] in DROP_RATE else H
        if rng.random() < DROP_RATE[state] / DROP_MEAN_S:
            gap = max(FPS, int(rng.exponential(DROP_MEAN_S * FPS)))
            out[t:t + gap] = np.nan
            t += gap
        t += FPS
    return out


# Label-free dropout (review item R3 / B0). The rates above were read from the public-window
# labels, which RULES.md reserves for pipeline checks. The profile below uses only label-free
# statistics of the BAR-ATH tracking (solution/reports/diagnostics.md): 40.8% of graded frames without a
# ball, and the length distribution of its 165 gaps (5% quantiles, seconds; capped at GAP_CAP_S).
# Whether gaps start more often in dead play cannot be measured without labels, so the ratio of
# the dead to the live start rate is a free parameter, chosen by a robustness test
# (external/dropout_robustness.py).
DROPOUT_RATIO = 1.0           # B0: dead/live start-rate ratio; None: the original profile above
TARGET_MISSING = 0.408        # share of frames without a ball on BAR-ATH (graded, label-free)
GAP_QUANTILES_S = (0.04, 0.08, 0.34, 0.61, 5.10, 5.48, 5.98, 6.76, 7.32, 8.26, 9.56, 10.73,
                   11.61, 13.52, 16.69, 20.44, 22.95, 25.27, 31.38, 49.45, 221.48)
GAP_CAP_S = 90.0

# Kinematic dropout: BAR-ATH loses the ball on 71% of graded frames when the mean player speed is
# under 1.3 m/s, 37% at 1.3-1.8 and 12% above (no labels involved). Gaps are placed to reproduce
# these shares, so the dropout never looks at a label of the calibration match.
KINEMATIC = "kinematic"
BAND_EDGES = (1.3, 1.8)
BAND_TARGETS = (0.714, 0.368, 0.120)


def drop_ball(ball, labels, seed, team_speed=None):
    """Ball dropout for one calibration match: the original, ratio or kinematic profile."""
    if DROPOUT_RATIO is None:
        return _drop_ball_like_atd(ball, labels, seed)
    if DROPOUT_RATIO == KINEMATIC:
        return drop_ball_kinematic(ball, team_speed, seed)
    return drop_ball_label_free(ball, labels, seed, DROPOUT_RATIO)


def _sample_gaps(rng, size):
    """Gap lengths in frames, by inverse CDF of the BAR-ATH gap-length quantiles."""
    levels = np.linspace(0.0, 1.0, len(GAP_QUANTILES_S))
    seconds = np.interp(rng.random(size), levels, np.minimum(GAP_QUANTILES_S, GAP_CAP_S))
    return np.maximum(1, np.round(seconds * FPS)).astype(int)


def _simulate(labels, n, rate, ratio, seed):
    """Missing-ball mask for a per-second live start probability `rate` (dead: ratio * rate)."""
    rng = np.random.default_rng(3000 + seed)
    gaps = _sample_gaps(rng, 4 * n // FPS + 16)
    draws = rng.random(n // FPS + 2)
    missing = np.zeros(n, dtype=bool)
    t, k, i = 0, 0, 0
    while t < n:
        p = rate * (ratio if labels[t] == D else 1.0)
        if draws[i] < p:
            missing[t:t + gaps[k]] = True
            t += gaps[k]
            k += 1
        i = min(i + 1, len(draws) - 1)
        t += FPS
    return missing


def drop_ball_label_free(ball, labels, seed, ratio, target=TARGET_MISSING):
    """Remove the ball in gaps with BAR-ATH's length distribution until `target` of the frames
    have no ball; dead-play seconds start a gap `ratio` times as often as live ones. The live
    start rate is found by bisection; the result is deterministic for a given seed."""
    n = len(ball)
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if _simulate(labels, n, mid, ratio, seed).mean() < target:
            lo = mid
        else:
            hi = mid
    out = ball.copy()
    out[_simulate(labels, n, hi, ratio, seed)] = np.nan
    return out


def _speed_band(speed):
    return np.searchsorted(BAND_EDGES, speed, side="right")


def _place_gaps(speed, weights, gaps, draws):
    """Missing-ball mask: each second, a candidate gap is kept with the weight of the band of the
    mean speed over the gap. The mean over the gap, not the speed at its start, because gaps begin
    before the players slow down."""
    n = len(speed)
    total = np.concatenate(([0.0], np.cumsum(speed)))
    missing = np.zeros(n, dtype=bool)
    t, k, i = 0, 0, 0
    while t < n:
        stop = min(n, t + gaps[k])
        if draws[i] < weights[_speed_band((total[stop] - total[t]) / (stop - t))]:
            missing[t:stop] = True
            t = stop
            k += 1
        i = min(i + 1, len(draws) - 1)
        t += FPS
    return missing


def drop_ball_kinematic(ball, team_speed, seed, rounds=40):
    """Remove the ball so that each speed band has the BAR-ATH missing share, counting frames that
    already have no ball (Metrica). The band weights are found by repeated correction; the result
    is deterministic for a given seed."""
    n = len(ball)
    blank = np.isnan(ball[:, 0])
    speed = np.nan_to_num(team_speed.astype(np.float64), nan=float(np.nanmedian(team_speed)))
    bands = _speed_band(speed)
    rng = np.random.default_rng(4000 + seed)
    gaps = _sample_gaps(rng, 4 * n // FPS + 16)
    draws = rng.random(n // FPS + 2)
    targets = np.array(BAND_TARGETS)
    weights = targets / 5
    for _ in range(rounds):
        missing = _place_gaps(speed, weights, gaps, draws) | blank
        shares = np.array([missing[bands == k].mean() if (bands == k).any() else targets[k]
                           for k in range(len(targets))])
        weights = np.clip(weights * (targets / np.maximum(shares, 1e-3)) ** 0.7, 1e-4, 1.0)
    out = ball.copy()
    out[_place_gaps(speed, weights, gaps, draws)] = np.nan
    return out


if __name__ == "__main__":
    have = available()
    if not have:
        print("no DFL files in {} -- see the module docstring for how to get them".format(DFL_DIR))
    for match in have:
        f, blocks, labels = load_match(match)
        starts, stops, values = runs_of(labels)
        alive = (values == H) | (values == A)
        graded = labels != X
        print("{}: {} frames, blocks {}, ball visible {:.1%}, possessions {}, D share {:.1%}".format(
            match, len(labels), blocks, f["visible"][graded].mean(), int(alive.sum()),
            np.mean(labels[graded] == D)))
