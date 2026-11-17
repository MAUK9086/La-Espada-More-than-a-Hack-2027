"""Identity-free team-shape and space features for the learned possession evidence (P3).

File structure
--------------
- attack_signs(): per frame, +1 where the home team attacks towards x = pitch length, else -1.
- team_geometry(): centroid, centroid velocity towards goal, width, depth, line heights, per team.
- pitch_control_at(): simplified Spearman time-to-intercept control of one point per frame.
- ball_context(): ball position (last known during gaps), time since/until seen, pitch control there.
- ball_motion(), near_group(): signed ball velocity and the players nearest the ball (on since
  p3-local).
- gap_velocity(), ball_impulse(): ball velocity across a tracking gap; one-sided velocities and
  their change (off unless named in LOCAL_GROUPS). "bounds" adds distances to the lines.
- near1_group(): forward velocity, approach speed, speed relative to the ball and goal-side offset of
  each team's single nearest player (group "near1", off unless named).
- contact_chain(), control_rule(): who last and next touches the ball, the touch chain's length,
  and that chain under the 2 s control rule of the training labels (group "chain", off unless named).
- rolling(): centred rolling means and past-vs-future trends -- the cheap stand-in for a TCN window.
- build(): the full feature matrix in the HOME perspective, and its team-swapped mirror.

Nothing here uses player identities: every quantity is pooled over a team, so fragmenting and
swapping track slots (46 slots for 22 players in the ATD) do not matter. Everything is oriented so
"forward" means towards the opponent's goal, which makes features comparable across halves and
matches (as in BallRadar, TranSPORTmer, Bialkowski et al.'s ghosting work).

The swapped mirror exists for symmetry: the model sees each frame once from each team's side, with
the labels H and A exchanged, so it cannot learn "home is more often in possession".
"""

import numpy as np

from common import FPS
from features import SPEED_HALF_WINDOW

# Params
VELOCITY_HALF_WINDOW = 12   # frames, centred difference for centroid velocity (~0.5 s each side)
ROLL_WINDOWS = (25, 75)     # frames: +/- 1 s and +/- 3 s of context
PC_MAX_SPEED = 7.0          # m/s, Spearman-style player max speed
PC_REACTION_S = 0.7         # s, reaction time
PC_SIGMA_S = 0.45           # s, uncertainty of arrival-time difference
KICKOFF_PROBE_S = 60        # s at the start of each block used to read the attack direction (median)
GAP_CAP_S = 30.0            # s, cap on time-since/until-ball features
# Optional switches tested on external data; the defaults reproduce the submitted model.
USE_P1 = True               # R6: feed the rule-based P1 scores to P3 as inputs
OUTFIELD_ONLY = False       # R4: leave a detected goalkeeper out of the team geometry
GK_MAX_DEPTH_M = 18.0       # a team's deepest player this close to its own goal line...
GK_GAP_M = 8.0              # ...and this far behind the next one is taken as the goalkeeper
PC_DECAY_S = None           # R5: pull pitch control at the last-seen ball towards 0.5 with this
                            # time constant (s) while the ball is unseen; None = no decay
EXTRA_TRENDS = ()           # R7: extra past-vs-future trend windows (frames) for shape keys
PC_PROJECT_S = None         # project players this far ahead along their recent motion before
                            # measuring distance in pitch control; None = current positions
PC_VELOCITY_FRAMES = 5      # frames of the backward difference used for that motion
EXTRA_TREND_KEYS = ("h_cx", "a_cx", "h_width", "a_width", "h_depth", "a_depth")
LOCAL_GROUPS = ("ball", "near3")  # extra feature groups: "ball" (signed velocity), "near3"
                                  # (nearest players); adopted at tag p3-local, () = 82 features.
                                  # Tested after p3-local, off unless named: "gap" (ball velocity
                                  # across a tracking gap), "impulse" (one-sided ball velocity and
                                  # its change), "bounds" (distance to the lines), "chain"
                                  # (contact chain and the 2 s rule, contact_chain()), "near1"
                                  # (the single nearest player's kinematics, near1_group(); R3 of
                                  # the fourth review)
GAP_MAX_S = 10.0            # "gap": longer gaps are stoppages or lost tracking, value 0
GAP_DECAY_S = 4.0           # "gap": the chord velocity is weighted by exp(-gap / GAP_DECAY_S)
IMPULSE_FRAMES = 10         # "impulse": one-sided window, 0.4 s
BOUNDS_WIDTH = 68.0         # "bounds": pitch width of every match used (BAR-ATH, DFL, Metrica)
CHAIN_CAP_S = 15.0          # "chain": times and durations are capped here
CHAIN_MIN_S = 2.0           # "chain": the control rule of the training labels (sportec_dfl.CONTROL_MIN_S)
CHAIN_EXACT = False         # "chain" as the third review wrote it: no ball-speed limit on a contact, and
                            # raw_touch = 2 only on outside frames (False: the first variant, with the 15 m/s
                            # limit and the chain reset until the next contact after an outside frame)
NEAR3_DEDUP_M = None        # review fix: drop a teammate within this distance of a nearer chosen one
                            # before taking the 3 nearest (duplicate tracks); None = off
LOCAL_PLAYERS = 3           # players nearest the ball, per team
LOCAL_MAX_SPEED = 12.0      # m/s, clip of per-slot velocity: slot reassignments create false speeds
LOCAL_CHUNK = 20000         # frames per chunk of the nearest-player search


def attack_signs(players, slot_team, blocks, n_frames):
    """Which way the home team attacks, read from the kick-off formation of each block.

    At a kick-off every player stands in his own half, so the team whose centroid is nearer x = 0
    defends the left goal and attacks right. Read from the first seconds of each block (half).

    Args:
        players: (N, S, 3) metres.
        slot_team: (S,) 0 home / 1 away.
        blocks: [(half, first, last)].
        n_frames: N.

    Returns:
        np.ndarray: (N,) +1 / -1 (0 outside blocks).
    """
    signs = np.zeros(n_frames, dtype=np.int8)
    for _, first, last in blocks:
        probe = slice(first, min(last + 1, first + KICKOFF_PROBE_S * FPS))
        with np.errstate(invalid="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                home_x = np.nanmedian(np.nanmean(players[probe][:, slot_team == 0, 0], 1))
                away_x = np.nanmedian(np.nanmean(players[probe][:, slot_team == 1, 0], 1))
        signs[first:last + 1] = 1 if home_x < away_x else -1
    return signs


def _nan_stat(values, func):
    """Row-wise NaN-aware statistic without RuntimeWarnings on all-NaN rows."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return func(values, axis=1)


def team_geometry(players, slot_team, sign, length):
    """Per-team shape, each team oriented so that it attacks towards +x.

    Args:
        players: (N, S, 3) metres.
        slot_team: (S,).
        sign: (N,) attack sign of the home team.
        length: pitch length in metres.

    Returns:
        dict[str, np.ndarray]: keys like 'h_cx', 'a_width', ... each (N,) float32.
    """
    out = {}
    x_all, y_all, s_all = players[:, :, 0], players[:, :, 1], players[:, :, 2]
    for code, prefix, team_sign in ((0, "h", sign), (1, "a", -sign)):
        members = slot_team == code
        x = x_all[:, members].astype(np.float64)
        y = y_all[:, members].astype(np.float64)
        # Orient: forward = towards the opponent goal for this team.
        x = np.where(team_sign[:, None] > 0, x, length - x)
        speed = s_all[:, members].astype(np.float64)
        if OUTFIELD_ONLY:
            x, y, speed = _drop_goalkeeper(x, y, speed)
        cx = _nan_stat(x, np.nanmean)
        out[prefix + "_cx"] = cx
        out[prefix + "_cy"] = _nan_stat(y, np.nanmean)
        out[prefix + "_width"] = _nan_stat(y, np.nanstd)
        out[prefix + "_depth"] = _nan_stat(x, np.nanstd)
        out[prefix + "_front"] = _nan_stat(x, lambda v, axis: np.nanpercentile(v, 90, axis=axis))
        out[prefix + "_back"] = _nan_stat(x, lambda v, axis: np.nanpercentile(v, 15, axis=axis))
        out[prefix + "_speed"] = _nan_stat(speed, np.nanmean)
        k = VELOCITY_HALF_WINDOW
        vx = np.full(len(cx), np.nan)
        vx[k:-k] = (cx[2 * k:] - cx[:-2 * k]) * FPS / (2 * k)
        out[prefix + "_vx"] = vx
    out["gap_cx"] = out["h_cx"] - (length - out["a_cx"])   # home centroid minus away centroid, home frame
    return {key: value.astype(np.float32) for key, value in out.items()}


def _drop_goalkeeper(x, y, speed):
    """Blank each frame's deepest player when he is clearly a goalkeeper (oriented x: own goal at 0).

    The test needs no positional metadata, so it behaves the same on every data source.
    """
    filled = np.where(np.isnan(x), np.inf, x)
    order = np.argsort(filled, axis=1)
    rows = np.arange(len(x))
    deepest, second = filled[rows, order[:, 0]], filled[rows, order[:, 1]]
    keeper = np.isfinite(second) & (deepest < GK_MAX_DEPTH_M) & (second - deepest > GK_GAP_M)
    x, y, speed = x.copy(), y.copy(), speed.copy()
    for array in (x, y, speed):
        array[rows[keeper], order[keeper, 0]] = np.nan
    return x, y, speed


def pitch_control_at(players, slot_team, point):
    """Spearman-style control of one point per frame: P(home reaches it first).

    Arrival time of each player = reaction time + distance / max speed (straight line, no current
    velocity -- a deliberate simplification that needs no identities); a team's time is its
    fastest player's. Control = logistic of the arrival-time difference.

    Args:
        players: (N, S, 3).
        slot_team: (S,).
        point: (N, 2) location per frame (NaN allowed).

    Returns:
        np.ndarray: (N,) in [0, 1], NaN where undefined.
    """
    px, py = players[:, :, 0], players[:, :, 1]
    if PC_PROJECT_S:
        k, gain = PC_VELOCITY_FRAMES, PC_PROJECT_S * FPS / PC_VELOCITY_FRAMES
        dx, dy = np.zeros_like(px), np.zeros_like(py)
        dx[k:] = np.nan_to_num(px[k:] - px[:-k])
        dy[k:] = np.nan_to_num(py[k:] - py[:-k])
        px, py = px + gain * dx, py + gain * dy
    distance = np.hypot(px - point[:, None, 0], py - point[:, None, 1])
    arrival = PC_REACTION_S + distance / PC_MAX_SPEED
    arrival = np.where(np.isnan(arrival), np.inf, arrival)
    t_home = arrival[:, slot_team == 0].min(1)
    t_away = arrival[:, slot_team == 1].min(1)
    with np.errstate(invalid="ignore", over="ignore"):
        control = 1.0 / (1.0 + np.exp((t_home - t_away) / PC_SIGMA_S))
    control[~(np.isfinite(t_home) & np.isfinite(t_away))] = np.nan
    return control.astype(np.float32)


def _ffill_index(valid):
    """Index of the last valid frame at or before each frame (-1 if none)."""
    index = np.where(valid, np.arange(len(valid)), -1)
    return np.maximum.accumulate(index)


def ball_context(f, players, slot_team, sign, length):
    """Ball features, using the last known ball position during gaps.

    Returns:
        dict[str, np.ndarray]: (N,) float32 arrays.
    """
    visible = f["visible"]
    n = len(visible)
    last = _ffill_index(visible)
    nxt = n - 1 - _ffill_index(visible[::-1])[::-1]
    nxt = np.where(_ffill_index(visible[::-1])[::-1] < 0, n, nxt)
    frames = np.arange(n)
    since = np.where(last >= 0, (frames - last) / FPS, GAP_CAP_S)
    until = np.where(nxt < n, (nxt - frames) / FPS, GAP_CAP_S)

    ball = f["ball"][:, :2].astype(np.float64)
    known = np.where(last[:, None] >= 0, ball[np.maximum(last, 0)], np.nan)
    bx = np.where(sign > 0, known[:, 0], length - known[:, 0])     # home-attack frame
    out = {
        "ball_x": bx,
        "ball_y": known[:, 1],
        "ball_since": np.minimum(since, GAP_CAP_S),
        "ball_until": np.minimum(until, GAP_CAP_S),
        "ball_visible": visible.astype(np.float64),
        "ball_speed": np.nan_to_num(f["ball_speed"], nan=-1.0),
        "ball_outside": f["outside"].astype(np.float64),
        "near_h": np.minimum(np.nan_to_num(f["near_home"], nan=50.0), 50.0),
        "near_a": np.minimum(np.nan_to_num(f["near_away"], nan=50.0), 50.0),
        "pc_ball": pitch_control_at(players, slot_team, known),
    }
    if PC_DECAY_S:
        weight = np.exp(-np.where(visible, 0.0, since) / PC_DECAY_S)
        out["pc_ball"] = 0.5 + (out["pc_ball"] - 0.5) * weight
    out["near_diff"] = out["near_h"] - out["near_a"]
    if "ball" in LOCAL_GROUPS:
        out.update(ball_motion(f, sign))
    if "near3" in LOCAL_GROUPS:
        out.update(near_group(players, slot_team, known, sign))
    if "near1" in LOCAL_GROUPS:
        out.update(near1_group(players, slot_team, known, sign, f))
    if "gap" in LOCAL_GROUPS:
        out["ball_vx_gap"] = gap_velocity(ball, visible, last, nxt, sign)
    if "impulse" in LOCAL_GROUPS:
        out.update(ball_impulse(f, sign))
    if "bounds" in LOCAL_GROUPS:
        out["ball_dist_sideline"] = np.minimum(known[:, 1], BOUNDS_WIDTH - known[:, 1])
        out["ball_dist_goalline"] = np.minimum(known[:, 0], length - known[:, 0])
    return {key: np.asarray(value, dtype=np.float32) for key, value in out.items()}


def gap_velocity(ball, visible, last, nxt, sign):
    """Mean ball velocity across a tracking gap, from the sightings on either side.

    On an unseen frame inside a gap of at most GAP_MAX_S, the chord from the last to the next
    sighting gives the direction the ball travelled. It is weighted down by exp(-gap / GAP_DECAY_S),
    so long gaps, which are mostly stoppages, tend to 0. Seen frames, longer gaps and gaps open at
    either end are 0.

    Args:
        ball: (N, 2) metres, NaN where unseen.
        visible: (N,) bool.
        last, nxt: (N,) index of the last and next sighting (-1 / N when none).
        sign: (N,) attack sign of the home team.

    Returns:
        np.ndarray: (N,) m/s towards the home team's attacking goal.
    """
    n = len(visible)
    out = np.zeros(n)
    inside = ~visible & (last >= 0) & (nxt < n)
    lo, hi = last[inside], nxt[inside]
    dt = (hi - lo) / FPS
    chord = (ball[hi, 0] - ball[lo, 0]) / np.maximum(dt, 1.0 / FPS)
    value = sign[inside] * chord * np.exp(-dt / GAP_DECAY_S)
    out[inside] = np.where(dt <= GAP_MAX_S, np.nan_to_num(value, nan=0.0), 0.0)
    return out


def ball_impulse(f, sign):
    """One-sided ball velocity before and after each frame, and the size of the change.

    A centred mean averages an incoming pass and the outgoing ball across a touch. The separate past
    and future velocities over IMPULSE_FRAMES keep both signs, and their vector difference is the
    impulse of a touch, deflection or interception. 0 where the ball is unseen at either end.

    Returns:
        dict: ball_vx_past10, ball_vx_fut10 (m/s towards the home team's attacking goal), ball_dv10 (m/s).
    """
    k = IMPULSE_FRAMES
    ball = f["ball"]
    n = len(ball)
    past, fut = np.full((n, 2), np.nan), np.full((n, 2), np.nan)
    past[k:] = (ball[k:, :2] - ball[:-k, :2]) * FPS / k
    fut[:-k] = (ball[k:, :2] - ball[:-k, :2]) * FPS / k
    dv = np.hypot(fut[:, 0] - past[:, 0], fut[:, 1] - past[:, 1])
    return {"ball_vx_past10": np.nan_to_num(sign * past[:, 0], nan=0.0),
            "ball_vx_fut10": np.nan_to_num(sign * fut[:, 0], nan=0.0),
            "ball_dv10": np.nan_to_num(dv, nan=0.0)}


def ball_motion(f, sign):
    """Signed ball velocity, 0 where the ball is unseen at either end of the difference.

    Args:
        f: feature dict with 'ball'.
        sign: (N,) attack sign of the home team.

    Returns:
        dict: ball_vx (m/s towards the home team's attacking goal), ball_vy_abs (m/s).
    """
    k = SPEED_HALF_WINDOW
    ball = f["ball"]
    vx, vy = np.zeros(len(ball)), np.zeros(len(ball))
    vx[k:-k] = (ball[2 * k:, 0] - ball[:-2 * k, 0]) * FPS / (2 * k)
    vy[k:-k] = (ball[2 * k:, 1] - ball[:-2 * k, 1]) * FPS / (2 * k)
    return {"ball_vx": np.nan_to_num(sign * vx, nan=0.0), "ball_vy_abs": np.nan_to_num(np.abs(vy), nan=0.0)}


def near_group(players, slot_team, point, sign):
    """Distance and forward velocity of each team's LOCAL_PLAYERS players nearest a point.

    Velocity is per slot, so a fragmented track simply has no value at its ends. With fewer players
    tracked the mean uses those present; with none it is NaN.

    Args:
        players: (N, S, 3) metres.
        slot_team: (S,).
        point: (N, 2) location per frame (NaN allowed).
        sign: (N,) attack sign of the home team.

    Returns:
        dict: h_near3, a_near3 (m), h_near3_vx, a_near3_vx (m/s towards each team's attacking goal).
    """
    n = len(players)
    k = VELOCITY_HALF_WINDOW
    out = {}
    for code, prefix, team_sign in ((0, "h", sign), (1, "a", -sign)):
        members = np.flatnonzero(slot_team == code)
        count = min(LOCAL_PLAYERS, len(members))
        distance, velocity = np.full(n, np.nan), np.full(n, np.nan)
        for start in range(0, n, LOCAL_CHUNK):
            frames = np.arange(start, min(start + LOCAL_CHUNK, n))
            p = players[frames][:, members]
            d = np.hypot(p[:, :, 0] - point[frames, None, 0], p[:, :, 1] - point[frames, None, 1])
            d = np.where(np.isnan(d), np.inf, d)
            if NEAR3_DEDUP_M:
                nearest = _dedup_nearest(p[:, :, :2], d, count, NEAR3_DEDUP_M)
                valid = nearest >= 0
                nearest = np.where(valid, nearest, 0)
                d = np.where(valid, np.take_along_axis(d, nearest, axis=1), np.inf)
            else:
                nearest = np.argpartition(d, count - 1, axis=1)[:, :count]
                d = np.take_along_axis(d, nearest, axis=1)
            seen = np.isfinite(d)
            ahead = players[np.minimum(frames + k, n - 1)][:, members, 0]
            behind = players[np.maximum(frames - k, 0)][:, members, 0]
            v = np.take_along_axis((ahead - behind) * FPS / (2 * k), nearest, axis=1)
            v = np.clip(v * team_sign[frames, None], -LOCAL_MAX_SPEED, LOCAL_MAX_SPEED)
            v[~seen | (frames[:, None] < k) | (frames[:, None] >= n - k)] = np.nan
            distance[frames] = _nan_stat(np.where(seen, d, np.nan), np.nanmean)
            velocity[frames] = _nan_stat(v, np.nanmean)
        out[prefix + "_near3"], out[prefix + "_near3_vx"] = distance, velocity
    return out


def near1_group(players, slot_team, point, sign, f):
    """Kinematics of each team's single player nearest a point (k = 1; R3 of the fourth review).

    Player velocity is the per-slot centred difference over VELOCITY_HALF_WINDOW frames, each
    component clipped to LOCAL_MAX_SPEED (as near_group()); ball velocity is the centred difference
    over SPEED_HALF_WINDOW frames (as ball_motion()).

    Args:
        players: (N, S, 3) metres.
        slot_team: (S,).
        point: (N, 2) last-known ball position (NaN before the first sighting).
        sign: (N,) attack sign of the home team.
        f: feature dict with 'ball' (NaN where unseen) and 'visible'.

    Returns:
        dict: per team prefix h_ / a_ (each oriented to that team's attack):
        near1_vx (m/s forward), near1_approach (m/s towards the point; NaN at zero distance),
        near1_rel_speed (|v_player - v_ball| m/s; NaN where the ball is unseen or its velocity is
        undefined), near1_goal_side (m, team_sign * (point_x - player_x): > 0 when the player is
        between the ball and his own goal). NaN where the team has no tracked player.
    """
    n = len(players)
    k = VELOCITY_HALF_WINDOW
    j = SPEED_HALF_WINDOW
    ball = f["ball"][:, :2].astype(np.float64)
    ball_v = np.full((n, 2), np.nan)
    ball_v[j:-j] = (ball[2 * j:] - ball[:-2 * j]) * FPS / (2 * j)
    ball_v[~f["visible"]] = np.nan
    out = {}
    for code, prefix, team_sign in ((0, "h", sign), (1, "a", -sign)):
        members = np.flatnonzero(slot_team == code)
        cols = {name: np.full(n, np.nan) for name in ("vx", "approach", "rel_speed", "goal_side")}
        if len(members):
            for start in range(0, n, LOCAL_CHUNK):
                frames = np.arange(start, min(start + LOCAL_CHUNK, n))
                p = players[frames][:, members, :2].astype(np.float64)
                d = np.hypot(p[:, :, 0] - point[frames, None, 0], p[:, :, 1] - point[frames, None, 1])
                d = np.where(np.isnan(d), np.inf, d)
                nearest = np.argmin(d, axis=1)
                rows = np.arange(len(frames))
                dist = d[rows, nearest]
                seen = np.isfinite(dist)
                pos = p[rows, nearest]
                ahead = players[np.minimum(frames + k, n - 1)][:, members, :2]
                behind = players[np.maximum(frames - k, 0)][:, members, :2]
                v = (ahead[rows, nearest] - behind[rows, nearest]) * FPS / (2 * k)
                v = np.clip(v, -LOCAL_MAX_SPEED, LOCAL_MAX_SPEED)
                v[~seen | (frames < k) | (frames >= n - k)] = np.nan
                rel = point[frames] - pos                      # player -> ball
                with np.errstate(invalid="ignore", divide="ignore"):
                    approach = (rel[:, 0] * v[:, 0] + rel[:, 1] * v[:, 1]) / dist
                approach[~seen | (dist <= 0)] = np.nan
                cols["vx"][frames] = team_sign[frames] * v[:, 0]
                cols["approach"][frames] = approach
                cols["rel_speed"][frames] = np.hypot(v[:, 0] - ball_v[frames, 0], v[:, 1] - ball_v[frames, 1])
                cols["goal_side"][frames] = np.where(seen, team_sign[frames] * rel[:, 0], np.nan)
        for name, values in cols.items():
            out["{}_near1_{}".format(prefix, name)] = values
    return out


def control_rule(touch, min_frames):
    """The 2 s control rule on a touch stream of H/A/D codes: a team takes over only at the start
    of a run of its own lasting at least min_frames; shorter runs stay with the owner; D resets.

    The same rule as external/sportec_dfl.controlled(), which builds the training labels; kept
    here so the submission pipeline does not import external/. No automated test checks that the two agree.
    """
    out = touch.copy()
    change = np.flatnonzero(touch[1:] != touch[:-1]) + 1
    starts = np.concatenate(([0], change))
    stops = np.concatenate((change, [len(touch)]))
    owner = None
    for start, stop in zip(starts, stops):
        value = touch[start]
        if value not in (0, 1):
            owner = None
            continue
        if owner is None or (value != owner and stop - start >= min_frames):
            owner = value
        out[start:stop] = owner
    return out


def _run_bounds(values):
    """Start and stop index of the constant run each element belongs to."""
    n = len(values)
    change = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.concatenate(([0], change))
    stops = np.concatenate((change, [n]))
    run = np.repeat(np.arange(len(starts)), stops - starts)
    return starts[run], stops[run]


def contact_chain(f, blocks):
    """Who last and next touches the ball, how long a team's chain of touches lasts, and that chain
    under the 2 s control rule. Label-free, per block, so nothing crosses half-time.

    A contact is a visible, in-pitch ball with the team's nearest player within R_CONTROL and nearer
    than any opponent, at no more than V_PASSING (a ball flying past is not a contact), as in
    possession.team_scores. The touch chain carries the last contacting team forward until the
    other team's contact; an outside-the-pitch frame resets it.

    Returns:
        dict of (N,) arrays: h/a_since_touch, h/a_until_touch, last_touch_diff, next_touch_diff
        (seconds); h/a_chain_dur (seconds); chain_ctrl (+1 H, -1 A, 0 none); h/a_ctrl_age,
        h/a_ctrl_rem (seconds).
    """
    import possession
    n = len(f["visible"])
    near_h = np.nan_to_num(f["near_home"], nan=np.inf)
    near_a = np.nan_to_num(f["near_away"], nan=np.inf)
    speed = np.nan_to_num(f["ball_speed"], nan=0.0)
    usable = f["visible"].astype(bool) & ~f["outside"].astype(bool)
    if not CHAIN_EXACT:
        usable = usable & (speed <= possession.V_PASSING)
    contact = {0: usable & (near_h <= possession.R_CONTROL) & (near_h < near_a),
               1: usable & (near_a <= possession.R_CONTROL) & (near_a < near_h)}
    cap, fps = CHAIN_CAP_S, float(FPS)
    out = {k: np.full(n, cap) for k in ("h_since_touch", "a_since_touch", "h_until_touch", "a_until_touch")}
    for k in ("h_chain_dur", "a_chain_dur", "chain_ctrl", "h_ctrl_age", "a_ctrl_age", "h_ctrl_rem", "a_ctrl_rem"):
        out[k] = np.zeros(n)
    if "bridge" in LOCAL_GROUPS:
        out["h_pass_bridge"], out["a_pass_bridge"] = np.zeros(n), np.zeros(n)
    min_frames = int(round(CHAIN_MIN_S * FPS))
    for _, first, last in blocks:
        s = slice(first, last + 1)
        m = last + 1 - first
        idx = np.arange(m)
        last_idx = {}
        for code, prefix in ((0, "h"), (1, "a")):
            c = contact[code][s]
            prev = np.maximum.accumulate(np.where(c, idx, -1))
            nxt = np.minimum.accumulate(np.where(c, idx, m)[::-1])[::-1]
            out[prefix + "_since_touch"][s] = np.where(prev >= 0, np.minimum((idx - prev) / fps, cap), cap)
            out[prefix + "_until_touch"][s] = np.where(nxt < m, np.minimum((nxt - idx) / fps, cap), cap)
            last_idx[code] = prev
        outside = f["outside"][s].astype(bool)
        newest = np.maximum(last_idx[0], last_idx[1])
        raw = np.where(last_idx[0] > last_idx[1], 0, 1).astype(np.int8)
        if CHAIN_EXACT:
            raw = np.where((newest < 0) | outside, 2, raw).astype(np.int8)
        else:
            reset = np.maximum.accumulate(np.where(outside, idx, -1))
            raw = np.where((newest < 0) | (newest <= reset), 2, raw).astype(np.int8)
        if "bridge" in LOCAL_GROUPS:
            # between two contacts of the same team: the last contact before and the next after are both theirs
            any_contact = contact[0][s] | contact[1][s]
            nxt_any = np.minimum.accumulate(np.where(any_contact, idx, m)[::-1])[::-1]
            nxt_team = np.where(nxt_any < m, np.where(contact[0][s][np.minimum(nxt_any, m - 1)], 0, 1), 2)
            last_team = np.where(newest < 0, 2, np.where(last_idx[0] > last_idx[1], 0, 1))
            between = ~any_contact & (last_team == nxt_team)
            out["h_pass_bridge"][s] = (between & (last_team == 0)).astype(float)
            out["a_pass_bridge"][s] = (between & (last_team == 1)).astype(float)
        lo, hi = _run_bounds(raw)
        dur = np.minimum((hi - lo) / fps, cap)
        out["h_chain_dur"][s] = np.where(raw == 0, dur, 0.0)
        out["a_chain_dur"][s] = np.where(raw == 1, dur, 0.0)
        ctrl = control_rule(raw, min_frames)
        out["chain_ctrl"][s] = np.where(ctrl == 0, 1.0, np.where(ctrl == 1, -1.0, 0.0))
        lo, hi = _run_bounds(ctrl)
        age, rem = np.minimum((idx - lo) / fps, cap), np.minimum((hi - idx) / fps, cap)
        for code, prefix in ((0, "h"), (1, "a")):
            out[prefix + "_ctrl_age"][s] = np.where(ctrl == code, age, 0.0)
            out[prefix + "_ctrl_rem"][s] = np.where(ctrl == code, rem, 0.0)
    out["last_touch_diff"] = out["a_since_touch"] - out["h_since_touch"]
    out["next_touch_diff"] = out["a_until_touch"] - out["h_until_touch"]
    return {k: v.astype(np.float32) for k, v in out.items()}


def _dedup_nearest(pos, d, count, min_sep):
    """Indices of the `count` nearest players, skipping any within `min_sep` of a nearer one already
    chosen (a duplicate track of the same player). -1 where fewer remain.

    Args:
        pos: (F, M, 2) positions of one team's slots.
        d: (F, M) distance to the point, inf where untracked.
        count: players to choose.
        min_sep: metres.
    """
    frames, members = d.shape
    order = np.argsort(d, axis=1, kind="stable")
    chosen = np.full((frames, count), -1)
    filled = np.zeros(frames, dtype=int)
    rows = np.arange(frames)
    for j in range(members):
        cand = order[:, j]
        ok = np.isfinite(d[rows, cand]) & (filled < count)
        cxy = pos[rows, cand]
        for c in range(count):
            prev = chosen[:, c]
            has = prev >= 0
            pxy = pos[rows, np.maximum(prev, 0)]
            close = has & (np.hypot(cxy[:, 0] - pxy[:, 0], cxy[:, 1] - pxy[:, 1]) < min_sep)
            ok &= ~close
        chosen[rows[ok], filled[ok]] = cand[ok]
        filled += ok
    return chosen


def rolling(values, half):
    """Centred rolling mean over +/- half frames, NaN-aware."""
    v = np.nan_to_num(values, nan=0.0).astype(np.float64)
    w = (~np.isnan(values)).astype(np.float64)
    cv = np.concatenate(([0.0], np.cumsum(v)))
    cw = np.concatenate(([0.0], np.cumsum(w)))
    idx = np.arange(len(values))
    lo, hi = np.maximum(idx - half, 0), np.minimum(idx + half + 1, len(values))
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((cv[hi] - cv[lo]) / (cw[hi] - cw[lo])).astype(np.float32)


def trend(values, half):
    """Mean of the next `half` frames minus mean of the previous `half` frames."""
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
    return (future - past).astype(np.float32)


CONTEXT_KEYS = ("h_cx", "a_cx", "h_vx", "a_vx", "h_width", "a_width", "h_depth", "a_depth",
                "gap_cx", "h_speed", "a_speed", "pc_ball", "near_diff", "ball_visible",
                "p1_h", "p1_a", "p1_d", "ball_vx", "h_near3", "a_near3", "h_near3_vx", "a_near3_vx",
                "ball_vx_gap", "h_near1_vx", "a_near1_vx", "h_near1_approach", "a_near1_approach",
                "h_near1_rel_speed", "a_near1_rel_speed", "h_near1_goal_side", "a_near1_goal_side")


def _swap(columns):
    """The same frame seen from the other team's side: h_* <-> a_*, home-frame x mirrored."""
    swapped = {}
    for key, value in columns.items():
        if key.startswith("h_"):
            swapped["a_" + key[2:]] = value
        elif key.startswith("a_"):
            swapped["h_" + key[2:]] = value
        else:
            swapped[key] = value
    swapped["gap_cx"] = -columns["gap_cx"]
    swapped["near_h"], swapped["near_a"] = columns["near_a"], columns["near_h"]
    swapped["near_diff"] = -columns["near_diff"]
    swapped["pc_ball"] = 1.0 - columns["pc_ball"]
    swapped["ball_x"] = None  # recomputed below: x in the (new) home-attack frame
    if "p1_h" in columns:
        swapped["p1_h"], swapped["p1_a"] = columns["p1_a"], columns["p1_h"]
    for key in ("ball_vx", "ball_vx_gap", "ball_vx_past10", "ball_vx_fut10",
                "last_touch_diff", "next_touch_diff", "chain_ctrl"):
        if key in columns:
            swapped[key] = -columns[key]
    return swapped


def build(f, players, slot_team, blocks, p1_scores, length):
    """Feature matrices for every frame, in both team perspectives.

    Args:
        f: per-frame feature dict (features.build() or an external loader).
        players: (N, S, 3) metres.
        slot_team: (S,).
        blocks: [(half, first, last)].
        p1_scores: dict with 'h', 'a', 'd' per-frame P1 evidence (rule-based scores).
        length: pitch length.

    Returns:
        tuple: (names, X_home (N, F) float32, X_swapped (N, F) float32).
    """
    n = len(f["visible"])
    sign = attack_signs(players, slot_team, blocks, n)
    sign = np.where(sign == 0, 1, sign)
    columns = {}
    columns.update(team_geometry(players, slot_team, sign, length))
    columns.update(ball_context(f, players, slot_team, sign, length))
    if "chain" in LOCAL_GROUPS:
        columns.update(contact_chain(f, blocks))
    if USE_P1:
        columns["p1_h"] = p1_scores["h"].astype(np.float32)
        columns["p1_a"] = p1_scores["a"].astype(np.float32)
        columns["p1_d"] = p1_scores["d"].astype(np.float32)

    def assemble(cols, ball_x):
        cols = dict(cols)
        cols["ball_x"] = ball_x
        for key in CONTEXT_KEYS:
            if key not in cols:
                continue
            for half in ROLL_WINDOWS:
                cols["{}_m{}".format(key, half)] = rolling(cols[key], half)
            cols["{}_t{}".format(key, ROLL_WINDOWS[0])] = trend(cols[key], ROLL_WINDOWS[0])
        for key in EXTRA_TREND_KEYS:
            for half in EXTRA_TRENDS:
                cols["{}_t{}".format(key, half)] = trend(cols[key], half)
        names = sorted(cols)
        return names, np.column_stack([np.asarray(cols[k], dtype=np.float32) for k in names])

    names, x_home = assemble(columns, columns["ball_x"])
    swapped = _swap(columns)
    names_s, x_swap = assemble(swapped, (length - columns["ball_x"]).astype(np.float32))
    assert names == names_s
    return names, x_home, x_swap
