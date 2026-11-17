"""Per-player input columns ("tokens") for a model that reads every player, not only team aggregates.

File structure
--------------
- Params: token layout and the scales that bring every channel to order 1.
- _velocity(): per-slot velocity from positions (central difference, clipped).
- _team_tokens(): the up-to-11 players of one team that are tracked in each frame, in slot order.
- tokens(): the two team views (home first; teams swapped and the pitch rotated by 180 degrees) as flat
  (N, N_COLS) matrices with their column names.

Layout of one view: 2 teams x PER_TEAM players x CHANNELS (x, y, vx, vy, present), then the ball
(x, y, visible, vx, vy). Coordinates are centred on the pitch centre and oriented so that the first team
attacks towards +x. A missing player or ball is all zeros with its flag at 0.

Not part of the submitted pipeline (run.py does not import this file); used by experiment E65.
"""

import numpy as np

import team_shape

# Params
PER_TEAM = 11
CHANNELS = ("x", "y", "vx", "vy", "on")
BALL_CHANNELS = ("x", "y", "on", "vx", "vy")
N_COLS = 2 * PER_TEAM * len(CHANNELS) + len(BALL_CHANNELS)
HALF_WIDTH = 34.0        # m; y is centred with the standard width (the pitch length comes with the match)
X_SCALE, Y_SCALE = 30.0, 20.0
V_SCALE, BALL_V_SCALE = 5.0, 10.0
MAX_SPEED, MAX_BALL_SPEED = 12.0, 40.0   # m/s clips: slot reassignments and ball jumps create false speeds
FPS = 25


def _velocity(xy, limit):
    """Central-difference velocity (N, ..., 2) in m/s, NaN where a neighbour is missing, clipped in norm."""
    v = np.full_like(xy, np.nan)
    v[1:-1] = (xy[2:] - xy[:-2]) * (FPS / 2.0)
    norm = np.sqrt((v ** 2).sum(-1, keepdims=True))
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.where(norm > limit, v * (limit / norm), v)
    return v


def _team_tokens(xy, vel):
    """(N, PER_TEAM, 5) tokens of one team from its slots' centred positions and velocities.

    Args:
        xy, vel: (N, S, 2) centred, oriented positions (m) and velocities (m/s); NaN where not tracked.

    Returns:
        np.ndarray: float32, players in slot order; unused places are zero.
    """
    n = len(xy)
    present = ~np.isnan(xy[:, :, 0])
    rank = np.cumsum(present, axis=1) - 1
    out = np.zeros((n, PER_TEAM, len(CHANNELS)), np.float32)
    rows = np.arange(n)
    for k in range(PER_TEAM):
        hit = present & (rank == k)
        has = hit.any(1)
        slot = hit.argmax(1)
        p, v = xy[rows, slot], np.nan_to_num(vel[rows, slot], nan=0.0)
        out[:, k, 0] = np.where(has, p[:, 0] / X_SCALE, 0.0)
        out[:, k, 1] = np.where(has, p[:, 1] / Y_SCALE, 0.0)
        out[:, k, 2] = np.where(has, v[:, 0] / V_SCALE, 0.0)
        out[:, k, 3] = np.where(has, v[:, 1] / V_SCALE, 0.0)
        out[:, k, 4] = has
    return out


def tokens(f, blocks):
    """Token matrices of one match in the two team views.

    Args:
        f: feature dict with players (N, S, 3), slot_team (S,), ball (N, >=2; NaN where not seen),
            pitch_length.
        blocks: [(half, first, last)].

    Returns:
        tuple: (names, t_home, t_swap); the matrices are (N, N_COLS) float32.
    """
    players, slot_team, ball = f["players"], np.asarray(f["slot_team"]), f["ball"]
    n = len(players)
    sign = team_shape.attack_signs(players, slot_team, blocks, n).astype(np.float64)
    sign = np.where(sign == 0, 1.0, sign)                    # frames outside the blocks are never used
    centre = np.array([float(f["pitch_length"]) / 2.0, HALF_WIDTH])
    xy = (players[:, :, :2] - centre) * sign[:, None, None]  # home attacks towards +x
    vel = _velocity(players[:, :, :2], MAX_SPEED) * sign[:, None, None]
    home = _team_tokens(xy[:, slot_team == 0], vel[:, slot_team == 0])
    away = _team_tokens(xy[:, slot_team == 1], vel[:, slot_team == 1])
    bxy = (ball[:, :2] - centre) * sign[:, None]
    seen = ~np.isnan(bxy[:, 0])
    bvel = np.nan_to_num(_velocity(ball[:, :2], MAX_BALL_SPEED) * sign[:, None], nan=0.0)
    b = np.zeros((n, len(BALL_CHANNELS)), np.float32)
    b[:, 0] = np.where(seen, bxy[:, 0] / X_SCALE, 0.0)
    b[:, 1] = np.where(seen, bxy[:, 1] / Y_SCALE, 0.0)
    b[:, 2] = seen
    b[:, 3] = np.where(seen, bvel[:, 0] / BALL_V_SCALE, 0.0)
    b[:, 4] = np.where(seen, bvel[:, 1] / BALL_V_SCALE, 0.0)

    def rotate(t):
        """The same tokens seen from the other end: x, y, vx, vy change sign, the flag stays."""
        r = t.copy()
        r[..., :4] *= -1.0
        return r

    b_rot = b.copy()
    b_rot[:, [0, 1, 3, 4]] *= -1.0
    t_home = np.concatenate((home.reshape(n, -1), away.reshape(n, -1), b), axis=1)
    t_swap = np.concatenate((rotate(away).reshape(n, -1), rotate(home).reshape(n, -1), b_rot), axis=1)
    names = ["pl_{}{}_{}".format(team, k, c) for team in "ha" for k in range(PER_TEAM) for c in CHANNELS]
    names += ["pl_ball_" + c for c in BALL_CHANNELS]
    return names, t_home, t_swap
