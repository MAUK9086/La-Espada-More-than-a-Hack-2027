"""Per-frame features derived from the cached tracking: ball state, proximity, collective motion.

File structure
--------------
- clean_ball(): drop single-frame ball "teleports" that no kicked ball could make.
- ball_speed(): centred finite-difference ball speed, NaN where the ball is not seen.
- nearest_by_team(): distance from the ball to the closest home / away player (chunked, low memory).
- team_speed(): mean tracker speed of all tracked players.
- outside_pitch(): visible ball beyond the lines by more than a tolerance.
- build(): every feature in one dict, cached to solution/cache/features.npz.

All distances are metres, speeds m/s, one value per video frame. Nothing here reads a label.
"""

import os

import numpy as np

import atd_cache
from common import CACHE_DIR, FPS, PITCH_LENGTH, PITCH_WIDTH

# Params
FEATURES_PATH = os.path.join(CACHE_DIR, "features.npz")
FUSED_FEATURES_PATH = os.path.join(CACHE_DIR, "features_fused.npz")   # P4: ATD + video ball
FUSED_BALL_PATH = os.path.join(CACHE_DIR, "ball_fused.npz")         # written by video/fuse.py
TELEPORT_MPS = 40.0      # a hard shot is ~30 m/s; a faster one-frame jump is a false detection
SPEED_HALF_WINDOW = 2    # centred difference over +/- 2 frames (0.16 s) for ball speed
OUT_MARGIN_M = 0.5       # tracking noise on the ball; the whole ball must cross the line anyway
CHUNK = 20000            # frames per block when computing distances, keeps memory flat
TOUCH_HALF_WINDOW = 3    # frames each side for the before/after ball velocity (0.12 s)
TOUCH_MIN_DV = 4.0       # m/s change of ball velocity that counts as being played
TOUCH_RADIUS_M = 2.0     # m, a player must be at the ball


def clean_ball(ball):
    """Remove isolated ball detections that jump faster than any kicked ball.

    A detection is dropped when it is reached from the previous visible frame AND left to the
    next visible frame at more than TELEPORT_MPS; a genuine fast ball only produces one such
    edge, a spurious blob produces two.

    Args:
        ball: (N, 3) array, metres, NaN where missing.

    Returns:
        np.ndarray: copy of ball with spurious detections set to NaN.
    """
    out = ball.copy()
    step = np.hypot(np.diff(ball[:, 0]), np.diff(ball[:, 1])) * FPS
    fast = np.nan_to_num(step, nan=0.0) > TELEPORT_MPS
    spurious = np.zeros(len(ball), dtype=bool)
    spurious[1:-1] = fast[:-1] & fast[1:]
    out[spurious] = np.nan
    return out


def ball_speed(ball):
    """Ball speed by centred finite difference, NaN unless both ends are seen.

    Args:
        ball: (N, 3) cleaned ball array.

    Returns:
        np.ndarray: (N,) float32 speed in m/s.
    """
    k = SPEED_HALF_WINDOW
    speed = np.full(len(ball), np.nan, dtype=np.float32)
    dx = ball[2 * k:, 0] - ball[:-2 * k, 0]
    dy = ball[2 * k:, 1] - ball[:-2 * k, 1]
    speed[k:-k] = np.hypot(dx, dy) * FPS / (2 * k)
    return speed


def nearest_by_team(players, ball, slot_team):
    """Distance from the ball to the closest home and the closest away player.

    Args:
        players: (N, S, 3) array.
        ball: (N, 3) array.
        slot_team: (S,) 0 home / 1 away.

    Returns:
        tuple[np.ndarray, np.ndarray]: (N,) float32 each, NaN where the ball or the team is unseen.
    """
    n = len(ball)
    near = [np.full(n, np.nan, dtype=np.float32), np.full(n, np.nan, dtype=np.float32)]
    for start in range(0, n, CHUNK):
        p = players[start:start + CHUNK]
        b = ball[start:start + CHUNK]
        distance = np.hypot(p[:, :, 0] - b[:, None, 0], p[:, :, 1] - b[:, None, 1])
        distance = np.where(np.isnan(distance), np.inf, distance)
        for team in (0, 1):
            closest = distance[:, slot_team == team].min(1)
            closest[~np.isfinite(closest)] = np.nan
            near[team][start:start + CHUNK] = closest
    return near[0], near[1]


def team_speed(players):
    """Mean tracker speed over every tracked player, NaN when nobody is tracked."""
    speed = players[:, :, 2]
    seen = ~np.isnan(speed)
    count = seen.sum(1)
    total = np.where(seen, speed, 0.0).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan).astype(np.float32)


def outside_pitch(ball, margin=OUT_MARGIN_M):
    """Visible ball beyond the pitch lines by more than `margin` metres."""
    x, y = ball[:, 0], ball[:, 1]
    with np.errstate(invalid="ignore"):
        return ((x < -margin) | (x > PITCH_LENGTH + margin)
                | (y < -margin) | (y > PITCH_WIDTH + margin))


def ball_touches(ball, near_home, near_away, half_window=None, min_dv=None, radius=None):
    """Frames where a player plays the ball: an abrupt change of ball velocity next to a player.

    The velocity before and after t is taken over `half_window` frames each side, which averages
    out frame-to-frame position jitter; a touch is a local maximum of |v_after - v_before| above
    `min_dv` with a player within `radius`. The team is the one whose nearest player is closer.

    Args:
        ball: (N, 3) cleaned ball, metres.
        near_home, near_away: (N,) nearest distances.
        half_window: frames each side for the velocity estimate (default TOUCH_HALF_WINDOW).
        min_dv: m/s, minimum velocity change (default TOUCH_MIN_DV).
        radius: m, a player must be this close (default TOUCH_RADIUS_M).

    Returns:
        tuple[np.ndarray, np.ndarray]: touch strength dv (N,) float32 (0 where no touch), and
        touching team (N,) int8 (0 home, 1 away, -1 none).
    """
    # Module constants are read at call time so calibration can vary them.
    k = TOUCH_HALF_WINDOW if half_window is None else half_window
    min_dv = TOUCH_MIN_DV if min_dv is None else min_dv
    radius = TOUCH_RADIUS_M if radius is None else radius
    n = len(ball)
    dv = np.zeros(n, dtype=np.float32)
    position = ball[:, :2].astype(np.float64)
    before = (position[k:n - k] - position[:n - 2 * k]) * FPS / k       # v over [t-k, t]
    after = (position[2 * k:] - position[k:n - k]) * FPS / k            # v over [t, t+k]
    change = np.hypot(*(after - before).T)
    dv[k:n - k] = np.nan_to_num(change, nan=0.0)

    peak = np.zeros(n, dtype=bool)
    peak[1:-1] = (dv[1:-1] >= dv[:-2]) & (dv[1:-1] > dv[2:])
    near_h = np.nan_to_num(near_home, nan=np.inf)
    near_a = np.nan_to_num(near_away, nan=np.inf)
    touched = peak & (dv > min_dv) & (np.fmin(near_h, near_a) <= radius)
    team = np.where(touched, np.where(near_h <= near_a, 0, 1), -1).astype(np.int8)
    return np.where(touched, dv, 0.0).astype(np.float32), team


def build(rebuild=False, fused=False):
    """Every per-frame feature, cached.

    Args:
        rebuild: ignore an existing cache.
        fused: use the ATD ball with P4 video detections filled in (video/fuse.py).

    Returns:
        dict[str, np.ndarray]: ball (cleaned), visible, ball_speed, near_home, near_away,
        team_speed, outside, plus slot_team and the metadata flags from atd_cache.
    """
    path = FUSED_FEATURES_PATH if fused else FEATURES_PATH
    if os.path.isfile(path) and not rebuild:
        with np.load(path, allow_pickle=False) as cached:
            return {key: cached[key] for key in cached.files}

    atd = atd_cache.load()
    raw_ball = atd["ball"]
    if fused:
        with np.load(FUSED_BALL_PATH) as cached:
            raw_ball = cached["ball"]
    ball = clean_ball(raw_ball)
    near_home, near_away = nearest_by_team(atd["players"], ball, atd["slot_team"])
    out = {
        "ball": ball,
        "visible": ~np.isnan(ball[:, 0]),
        "ball_speed": ball_speed(ball),
        "near_home": near_home,
        "near_away": near_away,
        "team_speed": team_speed(atd["players"]),
        "outside": outside_pitch(ball),
        "slot_team": atd["slot_team"],
        "home_attacks_right_first_half": np.array(atd["home_attacks_right_first_half"]),
    }
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(path, **out)
    return out
