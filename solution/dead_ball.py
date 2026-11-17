"""Dead-ball evidence and the two events that bound a stoppage: the stop and the restart kick.

File structure
--------------
- slowness(): 0..1, how far the players' collective speed is below normal play.
- stop_events(): frames where play plausibly stops (ball crosses a line, ball tracking drops).
- kick_events(): frames where a still or re-appearing ball is struck back into play.
- set_piece_windows(): Smart Tagging SET PIECES intervals as video-frame windows.
- dead_scores(): per-frame evidence for the D state, relative to the two alive states.

The model in possession.py lets alive->D and D->alive happen cheaply only at these events, which
is what pins stoppage boundaries to the frame instead of wherever a smoother drifts to. A missing
ball is weak evidence on its own: the ball is also lost in live play (aerial balls, crowded boxes).

Parameters are set from football and tracking reasoning (see docs/report/La_Espada.pdf), never fitted to the
public ground-truth window.
"""

import numpy as np

from common import FPS, runs_of

# Params
SPEED_DEAD = 1.0          # m/s mean player speed typical of a stoppage (players walking)
SPEED_ALIVE = 2.0         # m/s mean player speed typical of open play
STILL_BALL = 1.0          # m/s, the ball is placed / at rest
KICK_PRE = (-20, -5)      # frames [start, stop) looked back for a still ball before a restart; the
                          # 5 frames next to the strike are skipped because the tracker smooths the
                          # ball (~8-frame Gaussian), which smears the strike over them
KICK_POST = (5, 15)       # frames looked ahead for the ball moving after the strike (past the smear)
KICK_PRE_MAX = 2.5        # m/s, median ball speed before a kick (placed, or rolled into place)
KICK_POST_MIN = 4.0       # m/s, min ball speed after a kick (a short pass is ~5-8 m/s)
KICK_POST_RATIO = 2.0     # the strike at least doubles the ball speed
PENDING_MAX_S = 90        # s, longest wait for a restart (injuries, VAR) before giving up on it
W_PENDING = 0.5           # per frame, ball went out and has not been restarted yet
KICKOFF_SPOT_M = 3.0      # m, the kick-off is struck from the centre spot (tracking noise ~1 m)
OUT_TOLERANCE_M = 0.5     # m, same tolerance as features.OUT_MARGIN_M for a ball on a goal line
KICKOFF_SEARCH_S = 300    # s, how far into a graded half to look for its kick-off
KICK_PLAYER_M = 2.5       # someone must be at the ball to kick it
W_OUT = 1.5               # per frame, visible ball beyond the lines
W_MISSING_SLOW = 0.4      # per frame, ball unseen while players are slow
W_STILL_SLOW = 0.6        # per frame, ball at rest while players are slow (set piece being set up)
W_FAST = 0.25             # per frame, against D when the players are running (weak: players
                          # keep jogging for seconds after the ball goes out)
W_TAG = 0.3               # per frame, inside a Smart Tagging SET PIECES window before its restart
W_SPOT = 1.0              # per frame, ball at rest on a restart spot (line band, goal area, centre)
LINE_BAND_M = 1.5         # a ball held for a throw-in or placed for a corner sits ON the line, and the
                          # line itself is only known to ~1 m (still balls cluster at y 68-69 m)
GOAL_AREA = (5.5, 9.16)   # m: depth from the goal line, half-width beyond the goalposts
CENTRE_SPOT_M = 2.0       # m around the centre spot
# Optional switches tested on external data (R16, R17); the defaults reproduce the submission.
STOP_MIN_GAP_S = 0.0      # a ball dropout is a stop candidate, and adds missing-ball D evidence,
                          # only in gaps at least this long (0 = every gap, the original rule)
ADAPTIVE_SLOWNESS = False # blend SPEED_ALIVE / SPEED_DEAD with the match's own 65th / 15th
                          # percentile of mean player speed (label-free)


def slowness(team_speed):
    """0 at normal play speed, 1 at stoppage speed, linear in between.

    Args:
        team_speed: (N,) mean player speed.

    Returns:
        np.ndarray: (N,) float in [0, 1]; 0 where unknown.
    """
    alive, dead = SPEED_ALIVE, SPEED_DEAD
    if ADAPTIVE_SLOWNESS:
        valid = team_speed[~np.isnan(team_speed)]
        if len(valid) > 1000:
            a = 0.5 * SPEED_ALIVE + 0.5 * float(np.percentile(valid, 65))
            d = 0.5 * SPEED_DEAD + 0.5 * float(np.percentile(valid, 15))
            if a - d >= 0.6:
                alive, dead = a, d
    value = (alive - team_speed) / (alive - dead)
    return np.clip(np.nan_to_num(value, nan=0.0), 0.0, 1.0)


def long_gaps(visible, min_s=None):
    """Missing-ball frames that belong to gaps of at least min_s seconds (all gaps when 0)."""
    min_s = STOP_MIN_GAP_S if min_s is None else min_s
    missing = ~np.asarray(visible, dtype=bool)
    if min_s <= 0:
        return missing
    starts, stops, values = runs_of(missing)
    out = np.zeros_like(missing)
    for a, b, v in zip(starts, stops, values):
        if v and b - a >= min_s * FPS:
            out[a:b] = True
    return out


def _rolling(values, start, stop, reducer):
    """reducer(values[t+start : t+stop]) for every t, NaN-aware, NaN where the window is empty."""
    n = len(values)
    width = stop - start
    padded = np.full(n + 2 * width + abs(start) + abs(stop), np.nan, dtype=np.float64)
    offset = width + abs(start)
    padded[offset:offset + n] = values
    windows = np.lib.stride_tricks.sliding_window_view(padded, width)
    index = np.arange(n) + offset + start
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return reducer(windows[index], axis=1)


def restart_spot(ball):
    """Ball where restarts are taken: on a touchline or goal line, in a goal area, at the centre.

    Args:
        ball: (N, 3) array in metres.

    Returns:
        np.ndarray: bool (N,), False where the ball is unseen.
    """
    from common import PITCH_LENGTH, PITCH_WIDTH
    x, y = ball[:, 0], ball[:, 1]
    depth, half_width = GOAL_AREA
    goal_half = 7.32 / 2 + half_width
    with np.errstate(invalid="ignore"):
        line_band = ((y < LINE_BAND_M) | (y > PITCH_WIDTH - LINE_BAND_M)
                     | (x < LINE_BAND_M) | (x > PITCH_LENGTH - LINE_BAND_M))
        goal_area = (((x < depth) | (x > PITCH_LENGTH - depth))
                     & (np.abs(y - PITCH_WIDTH / 2) < goal_half))
        centre = np.hypot(x - PITCH_LENGTH / 2, y - PITCH_WIDTH / 2) < CENTRE_SPOT_M
    return line_band | goal_area | centre


def first_of_runs(mask):
    """Keep only the first frame of each run of True."""
    out = mask.copy()
    out[1:] &= ~mask[:-1]
    return out


def stop_events(f):
    """Frames where play plausibly stops.

    Two kinds: the ball crossing a pitch line (first frame outside), and the ball tracking
    dropping out (first missing frame of a gap). The second is only a candidate -- the model
    decides from the evidence that follows whether it was a stoppage.

    Args:
        f: feature dict from features.build().

    Returns:
        np.ndarray: bool (N,).
    """
    crossing = first_of_runs(f["outside"])
    gap_start = first_of_runs(long_gaps(f["visible"]))
    return crossing | gap_start


def kick_events(f):
    """Frames where the ball is struck back into play.

    A restart looks the same for every set piece: the ball is still (or has just re-appeared),
    a player is at it, and a fraction of a second later it is moving inside the pitch.

    Args:
        f: feature dict from features.build().

    Returns:
        np.ndarray: bool (N,), first frame of each candidate strike.
    """
    speed = f["ball_speed"].astype(np.float64)
    before_med = _rolling(speed, KICK_PRE[0], KICK_PRE[1], np.nanmedian)
    before_seen = _rolling(f["visible"].astype(np.float64), KICK_PRE[0], KICK_PRE[1], np.nansum)
    after_med = _rolling(speed, KICK_POST[0], KICK_POST[1], np.nanmedian)
    nearest = np.fmin(f["near_home"], f["near_away"])

    # Median, not max: the tracked ball jitters by a metre or so frame to frame, which reads as
    # brief 2-3 m/s spikes on a ball that is actually at rest.
    still_before = (before_med < KICK_PRE_MAX) | (before_seen == 0)
    moving_after = (after_med > KICK_POST_MIN) & (
        (before_seen == 0) | (after_med > KICK_POST_RATIO * np.nan_to_num(before_med, nan=0.0)))
    # A throw-in is taken from outside a TOUCHLINE (Law 15), so the strike may start up to a line
    # band beyond the touchlines -- as long as the ball is inside the pitch a moment later. Never
    # beyond a goal line: those restarts (goal kick, corner) are taken from inside the pitch, and a
    # ball carried back in over the goal line after going out is not a restart.
    from common import PITCH_LENGTH, PITCH_WIDTH
    x, y = f["ball"][:, 0], f["ball"][:, 1]
    with np.errstate(invalid="ignore"):
        near_pitch = (f["visible"] & (x >= -OUT_TOLERANCE_M) & (x <= PITCH_LENGTH + OUT_TOLERANCE_M)
                      & (y >= -LINE_BAND_M) & (y <= PITCH_WIDTH + LINE_BAND_M))
    inside_later = np.zeros_like(near_pitch)
    inside_later[:-KICK_POST[1]] = (f["visible"] & ~f["outside"])[KICK_POST[1]:]
    in_play = near_pitch & inside_later
    someone_there = nearest <= KICK_PLAYER_M
    return first_of_runs(still_before & moving_after & in_play & someone_there)


def pending_restart(f, kicks):
    """Frames after the ball has crossed a line and before the next restart strike.

    Laws of the Game: once the whole ball crosses a line it is out of play until a restart. The
    tracking keeps showing the ball being fetched, carried and placed in that time, which looks
    exactly like live dribbling -- so the context, not the frame, has to say it is dead.

    Args:
        f: feature dict from features.build().
        kicks: bool (N,) from kick_events().

    Returns:
        np.ndarray: bool (N,).
    """
    pending = np.zeros(len(kicks), dtype=bool)
    kick_frames = np.flatnonzero(kicks)
    limit = PENDING_MAX_S * FPS
    for start in np.flatnonzero(first_of_runs(f["outside"])):
        later = kick_frames[kick_frames > start]
        stop = later[0] if len(later) and later[0] - start <= limit else start + limit
        pending[start:stop] = True
    return pending


def tagged_kickoffs(f):
    """Kick-off strikes located inside the Smart Tagging kick-off windows.

    The tracked ball is unreliable around kick-offs (it can be carried to and through the centre
    spot without ever being seen at rest), so the stillness-based detector misses some. The
    automatic tagging marks a ~5 s window around every kick-off (start of each half and after each
    goal); inside it the strike is taken where the ball is closest to the centre spot.

    Args:
        f: feature dict from features.build().

    Returns:
        np.ndarray: bool (N,), one frame per tagged kick-off. Empty, with a flag, if the tagging
        cannot be read.
    """
    from common import PITCH_LENGTH, PITCH_WIDTH, write_flag
    n = len(f["visible"])
    mask = np.zeros(n, dtype=bool)
    try:
        from load_smart_tagging import load as load_tags
        events = load_tags()
    except Exception as error:
        write_flag("Smart Tagging kick-offs not used: {}".format(error))
        return mask
    ball = f["ball"]
    centre = np.hypot(ball[:, 0] - PITCH_LENGTH / 2, ball[:, 1] - PITCH_WIDTH / 2)
    for event in events:
        if not (event["tags"].get("Type") or "").strip().lower().startswith("kick off"):
            continue
        first, stop = max(event["start_frame"], 0), min(event["end_frame"], n)
        if stop <= first:
            continue
        window = centre[first:stop]
        if np.all(np.isnan(window)):
            mask[(first + stop) // 2] = True
        else:
            mask[first + int(np.nanargmin(window))] = True
    return mask


def pre_kickoff(f, kicks, blocks, tagged=None):
    """Frames of each graded half before its kick-off strike.

    Laws of the Game: every half starts with a kick-off from the centre spot, so the ball is not
    in play before it. The kick-off is detected, not looked up: the first strike of the half with
    the ball on the centre spot. If none is found the half gets no constraint and a flag.

    Args:
        f: feature dict from features.build().
        kicks: bool (N,) from kick_events().
        blocks: [(half, first, last)] graded halves.
        tagged: optional bool (N,) from tagged_kickoffs(), accepted as kick-offs directly.

    Returns:
        np.ndarray: bool (N,).
    """
    from common import PITCH_LENGTH, PITCH_WIDTH, write_flag
    ball = f["ball"]
    centre = np.hypot(ball[:, 0] - PITCH_LENGTH / 2, ball[:, 1] - PITCH_WIDTH / 2) < KICKOFF_SPOT_M
    kickoff = kicks & centre
    if tagged is not None:
        kickoff = kickoff | tagged
    mask = np.zeros(len(kicks), dtype=bool)
    for half, first, last in blocks:
        window = slice(first, min(last + 1, first + KICKOFF_SEARCH_S * FPS))
        candidates = np.flatnonzero(kickoff[window])
        if len(candidates) == 0:
            write_flag("half {}: no kick-off strike found in the first {} s; no pre-kickoff "
                       "constraint applied".format(half, KICKOFF_SEARCH_S))
            continue
        mask[first:first + candidates[0]] = True
    return mask


def set_piece_windows(n_frames):
    """Smart Tagging SET PIECES intervals, as a bool mask over video frames.

    The tags are automatic (DATA.md) -- evidence, not truth -- so they only add a small prior.
    Falls back to an empty mask, with a flag, if the tagging file cannot be read.
    """
    from common import write_flag
    mask = np.zeros(n_frames, dtype=bool)
    try:
        from load_smart_tagging import load as load_tags
        for event in load_tags():
            if (event["code"] or "").strip().upper() == "SET PIECES":
                mask[max(event["start_frame"], 0):min(event["end_frame"], n_frames)] = True
    except Exception as error:  # the tagging is optional evidence; never block on it
        write_flag("Smart Tagging not used: {}".format(error))
    return mask


def dead_scores(f, kicks, pending, tagged=None):
    """Per-frame score for D relative to the alive states (positive favours D).

    Args:
        f: feature dict from features.build().
        kicks: bool (N,) from kick_events().
        pending: bool (N,) from pending_restart().
        tagged: optional bool (N,) set-piece windows; None means no tagging (other matches).

    Returns:
        np.ndarray: (N,) float64.
    """
    slow = slowness(f["team_speed"])
    fast = np.clip(np.nan_to_num((f["team_speed"] - SPEED_ALIVE) / SPEED_DEAD, nan=0.0), 0, 1)
    speed = np.nan_to_num(f["ball_speed"], nan=np.inf)
    visible, outside = f["visible"], f["outside"]

    score = np.zeros(len(slow))
    score += W_OUT * outside
    score += W_MISSING_SLOW * slow * long_gaps(visible)
    still = visible & ~outside & (speed < STILL_BALL)
    score += W_STILL_SLOW * slow * still
    # A ball at rest on a restart spot is a set piece being taken, even before the players slow.
    score += W_SPOT * (0.5 + 0.5 * slow) * (still & restart_spot(f["ball"]))
    score -= W_FAST * fast
    score += W_PENDING * pending

    # Inside a tagged set piece, favour D up to the first strike in it -- the tag window opens
    # before the restart and runs on well after it, so only its head is a stoppage.
    if tagged is None:
        return score
    starts = np.flatnonzero(first_of_runs(tagged))
    kick_frames = np.flatnonzero(kicks)
    for start in starts:
        later = kick_frames[kick_frames >= start]
        stop = later[0] if len(later) else start
        if stop - start <= 40 * FPS:
            score[start:stop] += W_TAG
    return score
