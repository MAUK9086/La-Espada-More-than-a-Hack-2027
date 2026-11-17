"""Team-control evidence and an event-gated Viterbi decoder over {H, A, D}.

File structure
--------------
- team_scores(): per-frame evidence for H and for A from ball-player proximity.
- transition_costs(): per-frame switching costs, cheap only at stop / kick events.
  (The H <-> A cost is C_SWITCH, or per frame from turnover.switch_costs() when given.)
- contact_gated_switch_costs(): direction-specific H <-> A costs that lock a switch towards a team
  while the ball is visible and none of its players is near it (review suggestion R11; off unless
  passed to the decoder).
- viterbi(): best H/A/D path for one half under scores minus switching costs.
- viterbi_min_duration(): the same with each live state expanded into duration-counting sub-states,
  so an open-play takeover must last MIN_DURATION frames unless D ends it (fourth review, R1; off
  unless MIN_DURATION is set).
- decode(): the whole match, half by half, X outside the graded halves.
- absorb_short_islands(): post-decoding, give a short run of one team sandwiched between two runs
  of the other back to the surrounding possession (review suggestion R12; opt-in).
- bridge_short_dead(): post-decoding, fill a short stoppage with no line crossing with the team
  before it (decoder sweep D2; opt-in).

Why a decoder rather than a smoother
------------------------------------
The ranking metric matches whole possessions, so every spurious flip costs a possession and every
missed one costs another. A centred majority vote both flickers and shifts boundaries. A Viterbi
path with a cost per switch only changes state when the evidence accumulated since the last change
pays for it, so possession is sticky through loose balls and ball-tracking gaps (CHALLENGE.md:
"possession stays with the last team that had it"), and the change lands where the evidence
changes rather than half a window away.

Parameters are set from football and tracking reasoning (docs/report/La_Espada.pdf), never fitted to the
public ground-truth window.
"""

import numpy as np

from common import A, D, H, X, half_masks

# Params
R_CONTROL = 1.5       # m, a player this close to the ball can play it (possession-zone radius)
V_PASSING = 15.0      # m/s, a ball this fast going past a player is not under his control
STILL_BALL = 0.7      # m/s, a ball at rest near a player is weak evidence (set pieces look like it)
NEAR_RADIUS = 10.0    # m, the nearest-team tie-breaker only applies within this distance
W_CONTROL = 1.0       # per frame of clear control
W_CONTESTED = 0.4     # per frame when an opponent is also within R_CONTROL
W_STILL = 0.3         # per frame of control of a ball at rest
W_NEAR = 0.1          # per frame, nearest team; places the switch inside a ball flight
PENDING_DISCOUNT = 0.5  # control while a restart is pending counts for half
W_TOUCH = 0.0         # total evidence per detected touch (0 = off, the P1 behaviour)
TOUCH_SPREAD = 5      # frames after a touch over which its evidence is spread
C_SWITCH = 8.0        # H <-> A: about 8 frames (0.3 s) of net control by the other team
C_STOP = 15.0         # alive -> D anywhere
C_STOP_EVENT = 2.0    # alive -> D at a stop event
C_RESTART = 60.0      # D -> alive without a detected strike: play only restarts with a kick or
                      # throw (Laws of the Game), so this is a fallback for an unseen restart and
                      # needs ~2.4 s of clear control to pay for it
C_RESTART_EVENT = 2.0  # D -> alive at a kick event
EVENT_TOLERANCE = 3   # frames either side of an event where the cheap cost applies
C_LOCK = 24.0         # H <-> A cost towards a team with no player near a visible ball (R11)
R_GATE = 2.4          # m, a team "can touch" the ball within this distance (R11)
GATE_TOLERANCE = 4    # frames either side of a possible touch where the normal cost applies
ISLAND_FRAMES = 45    # runs shorter than this between two runs of the other team are absorbed (R12)
ISLAND_CLEAR = 15     # ...unless the island team had this many frames of clear control
MIN_DURATION = None   # R1 (fourth review): frames a team must keep the ball after an open-play
                      # takeover unless D ends it; None = the 3-state viterbi() (the submission)
MIN_RESTART = "steady"  # R1: after D, play restarts in the steady state ("steady", no minimum) or in
                        # the first counting state ("count", the minimum also applies after a restart)


def team_scores(f, pending=None):
    """Per-frame evidence for each team.

    Args:
        f: feature dict from features.build().
        pending: optional bool (N,), ball out and not yet restarted; control there is a player
            fetching or placing the ball, so it counts for half.

    Returns:
        tuple[np.ndarray, np.ndarray]: (N,) scores for H and for A.
    """
    near_h = np.nan_to_num(f["near_home"], nan=np.inf)
    near_a = np.nan_to_num(f["near_away"], nan=np.inf)
    speed = np.nan_to_num(f["ball_speed"], nan=0.0)
    usable = f["visible"] & ~f["outside"]

    weight = np.where(speed < STILL_BALL, W_STILL, W_CONTROL)
    weight = np.where(speed > V_PASSING, 0.0, weight)
    h_close, a_close = near_h <= R_CONTROL, near_a <= R_CONTROL
    contested = h_close & a_close
    weight = np.where(contested, np.minimum(weight, W_CONTESTED), weight)
    if pending is not None:
        weight = np.where(pending, weight * PENDING_DISCOUNT, weight)

    h_wins, a_wins = near_h < near_a, near_a < near_h
    score_h = usable * (h_close & h_wins) * weight
    score_a = usable * (a_close & a_wins) * weight

    in_range = usable & (np.fmin(near_h, near_a) <= NEAR_RADIUS)
    score_h = score_h + W_NEAR * (in_range & h_wins)
    score_a = score_a + W_NEAR * (in_range & a_wins)

    if W_TOUCH > 0:
        # A touch -- the ball's velocity changing abruptly at a player -- is the moment a team
        # plays the ball. Spread over TOUCH_SPREAD frames after it, so the switch lands on the
        # touch and the pulse survives ball-position jitter.
        import features
        _, team = features.ball_touches(f["ball"], f["near_home"], f["near_away"])
        team = np.where(usable & ~(pending if pending is not None else False), team, -1)
        for code, score in ((0, score_h), (1, score_a)):
            pulse = (team == code).astype(np.float64)
            spread = np.convolve(pulse, np.ones(TOUCH_SPREAD), mode="full")[:len(pulse)]
            score += (W_TOUCH / TOUCH_SPREAD) * spread
    return score_h, score_a


def _near_event(events):
    """Dilate an event mask by EVENT_TOLERANCE frames each way."""
    k = EVENT_TOLERANCE
    padded = np.concatenate((np.zeros(k, bool), events, np.zeros(k, bool)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * k + 1)
    return windows.any(1)


def transition_costs(stops, kicks):
    """Per-frame alive->D and D->alive costs.

    Args:
        stops: bool (N,) stop events.
        kicks: bool (N,) kick events.

    Returns:
        tuple[np.ndarray, np.ndarray]: cost_to_dead, cost_to_alive, each (N,).
    """
    to_dead = np.where(_near_event(stops), C_STOP_EVENT, C_STOP)
    to_alive = np.where(_near_event(kicks), C_RESTART_EVENT, C_RESTART)
    return to_dead, to_alive


def contact_gated_switch_costs(f, base=None, lock=C_LOCK, gate=R_GATE, tolerance=GATE_TOLERANCE):
    """Direction-specific H <-> A switch costs: a team cannot take a visible ball it is not near.

    Args:
        f: feature dict (near_home, near_away, visible, outside).
        base: optional (N,) symmetric costs to start from; C_SWITCH if None.
        lock: cost towards a team none of whose players is within `gate` of a visible ball.
        gate: metres.
        tolerance: frames either side of a possible touch where `base` still applies.

    Returns:
        np.ndarray: (N, 2) costs, column 0 for A -> H, column 1 for H -> A.
    """
    n = len(f["visible"])
    base = np.full(n, C_SWITCH) if base is None else np.asarray(base, dtype=float)
    usable = f["visible"] & ~f["outside"]
    out = np.empty((n, 2))
    for col, near in ((0, f["near_home"]), (1, f["near_away"])):
        can_touch = usable & (np.nan_to_num(near, nan=np.inf) <= gate)
        k = tolerance
        padded = np.concatenate((np.zeros(k, bool), can_touch, np.zeros(k, bool)))
        can_touch = np.lib.stride_tricks.sliding_window_view(padded, 2 * k + 1).any(1)
        out[:, col] = np.where(can_touch | ~usable, base, np.maximum(base, lock))
    return out


def viterbi(scores, to_dead, to_alive, switch=None):
    """Maximum-score state path for one half.

    Args:
        scores: (T, 3) per-frame scores for H, A, D.
        to_dead: (T,) cost of entering D at frame t from H or A.
        to_alive: (T,) cost of entering H or A at frame t from D.
        switch: optional H <-> A switch costs: (T,) for both directions, or (T, 2) with column 0
            for A -> H and column 1 for H -> A; C_SWITCH everywhere if None.

    Returns:
        np.ndarray: (T,) int8 state codes.
    """
    n = len(scores)
    back = np.zeros((n, 3), dtype=np.int8)
    rows = scores.tolist()
    dead_costs, alive_costs = to_dead.tolist(), to_alive.tolist()
    if switch is None:
        to_h_costs = to_a_costs = [C_SWITCH] * n
    else:
        switch = np.asarray(switch, dtype=float)
        if switch.ndim == 2:
            to_h_costs, to_a_costs = switch[:, 0].tolist(), switch[:, 1].tolist()
        else:
            to_h_costs = to_a_costs = switch.tolist()
    v_h, v_a, v_d = rows[0]
    for t in range(1, n):
        s_h, s_a, s_d = rows[t]
        c_dead, c_alive = dead_costs[t], alive_costs[t]
        c_to_h, c_to_a = to_h_costs[t], to_a_costs[t]
        # into H
        options = (v_h, v_a - c_to_h, v_d - c_alive)
        best = 0 if options[0] >= options[1] and options[0] >= options[2] else (
            1 if options[1] >= options[2] else 2)
        n_h = options[best] + s_h
        back[t, 0] = best
        # into A
        options = (v_h - c_to_a, v_a, v_d - c_alive)
        best = 1 if options[1] >= options[0] and options[1] >= options[2] else (
            0 if options[0] >= options[2] else 2)
        n_a = options[best] + s_a
        back[t, 1] = best
        # into D
        options = (v_h - c_dead, v_a - c_dead, v_d)
        best = 2 if options[2] >= options[0] and options[2] >= options[1] else (
            0 if options[0] >= options[1] else 1)
        n_d = options[best] + s_d
        back[t, 2] = best
        v_h, v_a, v_d = n_h, n_a, n_d

    path = np.zeros(n, dtype=np.int8)
    final = (v_h, v_a, v_d)
    path[-1] = int(np.argmax(final))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def viterbi_min_duration(scores, to_dead, to_alive, switch=None, l_min=None, restart=None):
    """Maximum-score H/A/D path for one half with a minimum takeover duration (R1).

    States, as the fourth review specified: H_0 (steady, held >= l_min frames), H_1 .. H_{l_min-1}
    (frames 1 .. l_min-1 since an A -> H takeover), the same for A, and D.
      - H_k, 1 <= k < l_min-1: only to H_{k+1} (score s_h) or to D (cost to_dead).
      - H_{l_min-1}: to H_0 or D.
      - H_0: stays (cost 0), to A_1 (cost of the H -> A switch) or to D (cost to_dead).
      - D: stays, or restarts (cost to_alive) into H_0 / A_0 ("steady") or H_1 / A_1 ("count").
      - The same for A. The first frame may be H_0, A_0 or D.
    The counting states advance by a one-step array shift, v_h[1:] = v_h[:-1] + s_h; back-pointers
    are kept only for the states with a choice (H_0, H_1, A_0, A_1, D).

    Args:
        scores: (T, 3) per-frame scores for H, A, D.
        to_dead, to_alive: (T,) costs, as in viterbi().
        switch: optional (T,) or (T, 2) H <-> A costs, as in viterbi(); C_SWITCH if None.
        l_min: frames (>= 2); MIN_DURATION if None.
        restart: "steady" or "count"; MIN_RESTART if None.

    Returns:
        np.ndarray: (T,) int8 state codes 0 = H, 1 = A, 2 = D.
    """
    l_min = MIN_DURATION if l_min is None else int(l_min)
    restart = MIN_RESTART if restart is None else restart
    if l_min < 2 or restart not in ("steady", "count"):
        raise ValueError("l_min must be >= 2 and restart 'steady' or 'count'")
    n = len(scores)
    m = l_min - 1                      # counting states per team: H_1 .. H_{l_min-1}
    if switch is None:
        to_h_costs = to_a_costs = np.full(n, C_SWITCH)
    else:
        switch = np.asarray(switch, dtype=float)
        to_h_costs, to_a_costs = (switch[:, 0], switch[:, 1]) if switch.ndim == 2 else (switch, switch)
    steady = restart == "steady"
    neg = -np.inf
    # global state index: 0 = H_0, 1..m = H_k, m+1 = A_0, m+2..2m+1 = A_k, 2m+2 = D
    idx_a0, idx_d = m + 1, 2 * m + 2
    back_h0 = np.zeros(n, dtype=np.int8)   # 0 from H_0, 1 from H_{l_min-1}, 2 from D
    back_a0 = np.zeros(n, dtype=np.int8)
    back_h1 = np.zeros(n, dtype=np.int8)   # 0 from A_0 (H_1) / H_0 (A_1), 1 from D
    back_a1 = np.zeros(n, dtype=np.int8)
    back_d = np.zeros(n, dtype=np.int32)   # global index of the predecessor of D
    s = np.asarray(scores, dtype=float)
    v_h0, v_a0, v_d = s[0, 0], s[0, 1], s[0, 2]
    v_h = np.full(m, neg)                  # v_h[k-1] = value of H_k
    v_a = np.full(m, neg)
    for t in range(1, n):
        s_h, s_a, s_d = s[t, 0], s[t, 1], s[t, 2]
        c_dead, c_alive = to_dead[t], to_alive[t]
        # into H_0: stay, H_{l_min-1} -> H_0, or (steady) restart from D
        opts = (v_h0, v_h[-1], v_d - c_alive if steady else neg)
        b = 0 if opts[0] >= opts[1] and opts[0] >= opts[2] else (1 if opts[1] >= opts[2] else 2)
        n_h0, back_h0[t] = opts[b] + s_h, b
        opts = (v_a0, v_a[-1], v_d - c_alive if steady else neg)
        b = 0 if opts[0] >= opts[1] and opts[0] >= opts[2] else (1 if opts[1] >= opts[2] else 2)
        n_a0, back_a0[t] = opts[b] + s_a, b
        # into H_1 / A_1: a takeover from the other steady state, or (count) a restart from D
        opts = (v_a0 - to_h_costs[t], neg if steady else v_d - c_alive)
        b = 0 if opts[0] >= opts[1] else 1
        n_h1, back_h1[t] = opts[b] + s_h, b
        opts = (v_h0 - to_a_costs[t], neg if steady else v_d - c_alive)
        b = 0 if opts[0] >= opts[1] else 1
        n_a1, back_a1[t] = opts[b] + s_a, b
        # into D: stay, or from any live state
        k_h, k_a = int(np.argmax(v_h)), int(np.argmax(v_a))
        opts = (v_d, v_h0 - c_dead, v_h[k_h] - c_dead, v_a0 - c_dead, v_a[k_a] - c_dead)
        b = int(np.argmax(opts))
        n_d = opts[b] + s_d
        back_d[t] = (idx_d, 0, 1 + k_h, idx_a0, idx_a0 + 1 + k_a)[b]
        # counting states: the one-step shift
        v_h[1:] = v_h[:-1] + s_h
        v_a[1:] = v_a[:-1] + s_a
        v_h[0], v_a[0] = n_h1, n_a1
        v_h0, v_a0, v_d = n_h0, n_a0, n_d

    final = np.concatenate(([v_h0], v_h, [v_a0], v_a, [v_d]))
    state = int(np.argmax(final))
    path = np.zeros(n, dtype=np.int8)
    for t in range(n - 1, -1, -1):
        path[t] = 2 if state == idx_d else (0 if state <= m else 1)
        if t == 0:
            break
        if state == idx_d:
            state = int(back_d[t])
        elif state == 0:
            state = (0, m, idx_d)[back_h0[t]]
        elif state == idx_a0:
            state = (idx_a0, idx_a0 + m, idx_d)[back_a0[t]]
        elif state == 1:
            state = (idx_a0, idx_d)[back_h1[t]]
        elif state == idx_a0 + 1:
            state = (0, idx_d)[back_a1[t]]
        else:
            state -= 1                     # H_k <- H_{k-1}, A_k <- A_{k-1}
    return path


def decode(score_h, score_a, score_d, to_dead, to_alive, blocks=None, switch=None):
    """Labels for the whole video: Viterbi inside each graded half, X elsewhere.

    Args:
        score_h, score_a, score_d: (N,) per-frame scores.
        to_dead, to_alive: (N,) transition costs.
        blocks: [(half, first, last)] spans to decode; defaults to this match's graded halves.
        switch: optional (N,) or (N, 2) H <-> A switch costs (see viterbi()).

    Returns:
        np.ndarray: (N,) int8 codes H/A/D/X.
    """
    if blocks is None:
        blocks, _ = half_masks()
    labels = np.full(len(score_h), X, dtype=np.int8)
    scores = np.stack((score_h, score_a, score_d), axis=1)
    for _, first, last in blocks:
        span = slice(first, last + 1)
        decoder = viterbi if MIN_DURATION is None else viterbi_min_duration
        path = decoder(scores[span], to_dead[span], to_alive[span],
                       None if switch is None else switch[span])
        labels[span] = np.array((H, A, D), dtype=np.int8)[path]
    return labels


def absorb_short_islands(labels, f=None, min_frames=ISLAND_FRAMES, clear_frames=ISLAND_CLEAR):
    """Give short sandwiched possessions back to the surrounding team.

    A run of one team shorter than `min_frames`, directly between two runs of the other team (no
    stoppage in between), is relabelled as the other team: CHALLENGE.md, "a touch, a deflection or a
    lost aerial duel does not change possession". With `f`, an island is kept if its team had at
    least `clear_frames` frames of clear control (nearest player within R_CONTROL, opponents beyond
    R_CONTROL + 0.5 m). D runs are never changed.

    Args:
        labels: (N,) int8 H/A/D/X labels.
        f: optional feature dict.
        min_frames, clear_frames: thresholds in frames (clear_frames ignored without f).

    Returns:
        np.ndarray: (N,) int8 labels.
    """
    out = labels.copy()
    clear = None
    if f is not None and clear_frames:
        near_h = np.nan_to_num(f["near_home"], nan=np.inf)
        near_a = np.nan_to_num(f["near_away"], nan=np.inf)
        usable = f["visible"] & ~f["outside"]
        clear = {H: usable & (near_h <= R_CONTROL) & (near_a > R_CONTROL + 0.5),
                 A: usable & (near_a <= R_CONTROL) & (near_h > R_CONTROL + 0.5)}
    change = np.flatnonzero(out[1:] != out[:-1]) + 1
    starts = np.concatenate(([0], change))
    stops = np.concatenate((change, [len(out)]))
    values = out[starts]
    for i in range(1, len(values) - 1):
        team, before, after = values[i], values[i - 1], values[i + 1]
        if team not in (H, A) or before != after or before not in (H, A) or before == team:
            continue
        if stops[i] - starts[i] >= min_frames:
            continue
        if clear is not None and clear[team][starts[i]:stops[i]].sum() >= clear_frames:
            continue
        out[starts[i]:stops[i]] = before
        values[i] = before          # a later island sees the merged run
    return out


def bridge_short_dead(labels, f, max_frames, same_team=False):
    """Fill short stoppages between two live runs where the ball never left the pitch.

    Hidden ball frames cannot show a crossing, so `outside` says less the more dropout a stoppage
    has. Stoppages at the start or end of a half are never changed.

    Args:
        labels: (N,) int8 H/A/D/X labels.
        f: feature dict (outside).
        max_frames: stoppages of this many frames or more are kept.
        same_team: only fill when the same team holds the ball on both sides.

    Returns:
        np.ndarray: (N,) int8 labels.
    """
    out = labels.copy()
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], change))
    stops = np.concatenate((change, [len(labels)]))
    values = labels[starts]
    for i in range(1, len(values) - 1):
        before, after = values[i - 1], values[i + 1]
        if values[i] != D or before not in (H, A) or after not in (H, A):
            continue
        if stops[i] - starts[i] >= max_frames or f["outside"][starts[i]:stops[i]].any():
            continue
        if same_team and before != after:
            continue
        out[starts[i]:stops[i]] = before
    return out
