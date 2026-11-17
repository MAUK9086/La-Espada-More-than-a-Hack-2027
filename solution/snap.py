"""Step 3: move each decoded H <-> A switch onto a sharper local signal, within a small window.

File structure
--------------
- switches(): frames where the decoded label changes directly between H and A.
- contact(): per frame and team, "this team's nearest player is at the ball and nearer than any
  opponent" (the P1 control test without its weights).
- snap(): relocate every switch to a target frame inside +/- window, never past the neighbouring
  change points, so possessions are moved, not added or removed.
- targets for snap(): contact_first / contact_nearest (the new team's control begins) and
  peak (argmax of a per-frame score such as the turnover probability).
- snap_dead(): move alive -> D changes onto a line crossing and D -> alive changes onto a kick
  event, within a window (review suggestion R13; opt-in).

Why: on matched possessions the decoded switch is a median 0.5 s from the true
one and more than 1 s off on 30% of them, while the metric already loses a third of a
possession's value at 1 s. P3's evidence uses +/- 1-3 s rolling windows, which blurs *when* a team
takes over; a local signal can place the switch inside that blur. Boundary refinement after
segmentation is standard in action segmentation (ASRF, WACV 2021).
"""

import numpy as np

from common import A, D, H
from possession import R_CONTROL


def switches(labels):
    """Indices s with labels[s-1], labels[s] both in {H, A} and different (s is the first new frame)."""
    prev, cur = labels[:-1], labels[1:]
    live = ((prev == H) | (prev == A)) & ((cur == H) | (cur == A))
    return np.flatnonzero(live & (prev != cur)) + 1


def contact(f):
    """(N, 2) bool: home / away has the ball at its feet (visible, inside, nearest within R_CONTROL)."""
    near_h = np.nan_to_num(f["near_home"], nan=np.inf)
    near_a = np.nan_to_num(f["near_away"], nan=np.inf)
    usable = f["visible"] & ~f["outside"]
    home = usable & (near_h <= R_CONTROL) & (near_h < near_a)
    away = usable & (near_a <= R_CONTROL) & (near_a < near_h)
    return np.stack((home, away), axis=1)


def _bounds(labels, s, window):
    """Search range for switch s: +/- window, strictly inside the neighbouring change points."""
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    k = np.searchsorted(change, s)
    left = change[k - 1] + 1 if k > 0 else 1
    right = change[k + 1] - 1 if k + 1 < len(change) else len(labels) - 1
    return max(s - window, left), min(s + window, right)


def snap(labels, target, window):
    """Relocate every H <-> A switch.

    Args:
        labels: (N,) int8 decoded labels.
        target: function (new_team_code, lo, hi, s) -> chosen frame in [lo, hi], or None to keep s.
        window: max frames a switch may move.

    Returns:
        np.ndarray: (N,) int8 labels with moved switches; same number and order of possessions.
    """
    out = labels.copy()
    for s in switches(labels):
        # Bounds from the labels as already moved: switches are processed left to right, so a
        # moved neighbour can never be crossed and no possession can vanish.
        lo, hi = _bounds(out, s, window)
        if lo > hi:
            continue
        new, old = labels[s], labels[s - 1]
        t = target(new, lo, hi, s)
        if t is None or t == s:
            continue
        if t < s:
            out[t:s] = new
        else:
            out[s:t] = old
    return out


def contact_first(f):
    """Target: the first frame in the window where the new team has the ball."""
    has = contact(f)

    def target(new, lo, hi, s):
        hits = np.flatnonzero(has[lo:hi + 1, 0 if new == H else 1])
        return lo + int(hits[0]) if len(hits) else None
    return target


def contact_nearest(f):
    """Target: the start of the new team's contact run nearest to the decoded switch."""
    has = contact(f)

    def target(new, lo, hi, s):
        col = has[lo:hi + 1, 0 if new == H else 1]
        starts = np.flatnonzero(col & ~np.concatenate(([False], col[:-1])))
        if not len(starts):
            return None
        starts = lo + starts
        return int(starts[np.argmin(np.abs(starts - s))])
    return target


def peak(score):
    """Target: the frame of the highest score in the window (e.g. the turnover probability)."""
    def target(new, lo, hi, s):
        return lo + int(np.argmax(score[lo:hi + 1]))
    return target


def snap_dead(labels, f, kicks, window):
    """Move stoppage edges onto events: alive -> D to the first line crossing, D -> alive to the
    last kick event, each within +/- window and never past a neighbouring change.

    Args:
        labels: (N,) int8 labels.
        f: feature dict (outside).
        kicks: (N,) bool kick events.
        window: frames.

    Returns:
        np.ndarray: (N,) int8 labels with the same sequence of runs.
    """
    out = labels.copy()
    outside = np.asarray(f["outside"], dtype=bool)
    crossing = outside.copy()
    crossing[1:] &= ~outside[:-1]
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    for s in change:
        old, new = out[s - 1], out[s]
        if old in (H, A) and new == D:
            events, pick = crossing, 0
        elif old == D and new in (H, A):
            events, pick = kicks, -1
        else:
            continue
        lo, hi = _bounds(out, s, window)
        if lo > hi:
            continue
        hits = np.flatnonzero(events[lo:hi + 1])
        if not len(hits):
            continue
        t = lo + int(hits[pick])
        if t < s:
            out[t:s] = new
        elif t > s:
            out[s:t] = old
    return out
