"""SkillCorner open data (20 A-League 2024/25 matches, broadcast tracking, MIT licence) as external data.

File structure
--------------
- SKC_DIR, matches(): where the clone of github.com/SkillCorner/opendata lives and which matches it holds.
- parse(): one match -> arrays at the native 10 fps, cached to solution/cache/skc_<id>.npz: player
  positions per roster slot, detected flags, ball, the per-frame possession group, the team in possession
  of the phase of play covering each frame (0 where no phase: the ball is out of play), periods.
- labels10(): H/A/D/X at 10 fps. D where no phase of play covers the frame; otherwise the team named by
  the per-frame possession group, carried forward inside the phase where the group is empty, and the
  phase's team before the first group of the phase.
- load_match(): features dict, blocks and labels at 25 fps in the calibration format (as
  sportec_dfl.load_match): positions interpolated linearly from 10 fps, labels by the nearest earlier
  frame and then the control rule (sportec_dfl.controlled), speeds from positions. The ball is used only
  where SkillCorner detected it (is_detected), so its gaps are the real gaps of an automatic tracker;
  no simulated degradation is added. Players include the extrapolated off-screen ones.
- seq(): the TCN inputs, decoder features, labels and the submission's p3-refined probabilities per match.
- stats(): label statistics per match (possessions per hour under each control rule, dead share, ball
  visibility), for comparing the labels with DFL's before any use.

Licence: MIT (LICENSE in the repository, "Copyright (c) 2020 SkillCorner"), checked on 11 Nov 2026 (the
challenge allows any dataset whose licence permits the use). The labels are SkillCorner's own (automatic tracking
and their event model), not hand labels of ours and not the provided game.

Data format (README of the repository): <id>_tracking_extrapolated.jsonl, one JSON object per frame at
10 fps with frame, period, ball_data {x, y, z, is_detected}, possession {player_id, group}, player_data
[{x, y, player_id, is_detected}]; coordinates in metres from the centre spot. <id>_phases_of_play.csv has
frame_start, frame_end and team_in_possession_id per phase; phases exist only while the ball is in play.

    .venv/Scripts/python solution/external/skillcorner.py            # parse all matches, print statistics
    .venv/Scripts/python solution/external/skillcorner.py seq        # model inputs -> cache/tcn_skc/
"""

import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import features  # noqa: E402
from common import A, CACHE_DIR, D, FPS, H, ROOT, X, runs_of  # noqa: E402

# Params
SKC_DIR = os.environ.get("SKC_DIR") or os.path.join(os.path.dirname(ROOT), "external", "skillcorner-opendata")
SKC_FPS = 10
SPEED_HALF_WINDOW = 2      # frames at 25 fps, centred difference for player speed (as metrica_sample)
MAX_INTERP_GAP = 1         # 10 fps frames: a position is interpolated only between consecutive frames


def matches():
    """Match ids (strings) present on disk, sorted."""
    folder = os.path.join(SKC_DIR, "data", "matches")
    if not os.path.isdir(folder):
        return []
    return sorted(m for m in os.listdir(folder)
                  if os.path.isfile(os.path.join(folder, m, m + "_tracking_extrapolated.jsonl")))


def parse(match):
    """One match at 10 fps, cached.

    Returns:
        dict: players (N, S, 2) metres from the bottom-left corner (NaN where absent), detected (N, S) bool,
        slot_team (S,) 0 home / 1 away, ball (N, 3) x, y, z (NaN where absent), ball_detected (N,) bool,
        group (N,) 0 none / 1 home / 2 away, phase_team (N,) 0 no phase / 1 home / 2 away, periods (N,),
        pitch (2,) length, width.
    """
    cache = os.path.join(CACHE_DIR, "skc_{}.npz".format(match))
    if os.path.isfile(cache):
        with np.load(cache) as z:
            return {k: z[k] for k in z.files}
    folder = os.path.join(SKC_DIR, "data", "matches", match)
    with open(os.path.join(folder, match + "_match.json"), encoding="utf-8") as handle:
        meta = json.load(handle)
    home_id, away_id = meta["home_team"]["id"], meta["away_team"]["id"]
    length, width = float(meta["pitch_length"]), float(meta["pitch_width"])
    roster = [p for p in meta["players"] if p["team_id"] in (home_id, away_id)]
    roster.sort(key=lambda p: (p["team_id"] != home_id, p["id"]))
    slot_of = {p["id"]: i for i, p in enumerate(roster)}
    slot_team = np.array([0 if p["team_id"] == home_id else 1 for p in roster], dtype=np.int8)
    rows = []
    with open(os.path.join(folder, match + "_tracking_extrapolated.jsonl"), encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    rows = [r for r in rows if r.get("period") in (1, 2)]
    n = len(rows)
    players = np.full((n, len(roster), 2), np.nan, dtype=np.float32)
    detected = np.zeros((n, len(roster)), dtype=bool)
    ball = np.full((n, 3), np.nan, dtype=np.float32)
    ball_detected = np.zeros(n, dtype=bool)
    group = np.zeros(n, dtype=np.int8)
    periods = np.zeros(n, dtype=np.int8)
    frames = np.zeros(n, dtype=np.int64)
    for i, r in enumerate(rows):
        periods[i], frames[i] = r["period"], r["frame"]
        b = r.get("ball_data") or {}
        if b.get("x") is not None and b.get("y") is not None:
            ball[i] = (b["x"] + length / 2, b["y"] + width / 2, b["z"] if b.get("z") is not None else np.nan)
            ball_detected[i] = bool(b.get("is_detected"))
        g = (r.get("possession") or {}).get("group")
        group[i] = 1 if g == "home team" else (2 if g == "away team" else 0)
        for p in r.get("player_data") or []:
            slot = slot_of.get(p["player_id"])
            if slot is not None and p.get("x") is not None:
                players[i, slot] = (p["x"] + length / 2, p["y"] + width / 2)
                detected[i, slot] = bool(p.get("is_detected"))
    phase_team = np.zeros(n, dtype=np.int8)
    index_of = {int(f): i for i, f in enumerate(frames)}
    with open(os.path.join(folder, match + "_phases_of_play.csv"), encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            team = int(float(row["team_in_possession_id"]))
            code = 1 if team == home_id else (2 if team == away_id else 0)
            for frame in range(int(row["frame_start"]), int(row["frame_end"]) + 1):
                i = index_of.get(frame)
                if i is not None:
                    phase_team[i] = code
    out = {"players": players, "detected": detected, "slot_team": slot_team, "ball": ball,
           "ball_detected": ball_detected, "group": group, "phase_team": phase_team, "periods": periods,
           "frames": frames, "pitch": np.array([length, width])}
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(cache, **out)
    return out


def labels10(raw):
    """Touch-level H/A/D labels at 10 fps (see the module docstring).

    Args:
        raw: parse() output.

    Returns:
        np.ndarray: (N,) int8.
    """
    group, phase = raw["group"], raw["phase_team"]
    out = np.full(len(group), D, dtype=np.int8)
    starts, stops, values = runs_of(phase != 0)
    for start, stop, alive in zip(starts, stops, values):
        if not alive:
            continue
        owner = phase[start]
        for t in range(start, stop):
            if phase[t] != phase[t - 1] and t > start:
                owner = phase[t] if group[t] == 0 else owner   # a new phase names its team
            if group[t] != 0:
                owner = group[t]
            out[t] = H if owner == 1 else A
    return out


def _upsample_index(n10):
    """For n10 frames at 10 fps: the 25 fps sample times as (left index, weight of the right neighbour)."""
    t = np.arange(int(round((n10 - 1) * FPS / SKC_FPS)) + 1) * SKC_FPS / FPS
    left = np.minimum(np.floor(t).astype(int), n10 - 1)
    return left, (t - left).astype(np.float32)


def load_match(match, control_s=None):
    """Features, blocks and labels of one match at 25 fps, in the calibration format.

    Args:
        match: match id.
        control_s: control rule in seconds (sportec_dfl.CONTROL_MIN_S if None).

    Returns:
        tuple: (features dict, blocks, labels).
    """
    from external import sportec_dfl
    from external.metrica_sample import player_speed
    raw = parse(match)
    lab10 = labels10(raw)
    ball10 = raw["ball"][:, :2].copy()
    ball10[~raw["ball_detected"]] = np.nan
    players, ball, labels, blocks = [], [], [], []
    offset = 0
    for period in (1, 2):
        idx = np.flatnonzero(raw["periods"] == period)
        if not len(idx):
            continue
        left, w = _upsample_index(len(idx))
        right = np.minimum(left + 1, len(idx) - 1)
        p10, b10 = raw["players"][idx], ball10[idx]
        players.append(p10[left] * (1 - w)[:, None, None] + p10[right] * w[:, None, None])
        ball.append(b10[left] * (1 - w)[:, None] + b10[right] * w[:, None])
        labels.append(lab10[idx][left])
        blocks.append((period, offset, offset + len(left) - 1))
        offset += len(left)
    xy = np.concatenate(players)
    ball_xy = np.concatenate(ball)
    labels = sportec_dfl.controlled(np.concatenate(labels).astype(np.int8), control_s)
    players = np.concatenate((xy, player_speed(xy)[..., None]), axis=2).astype(np.float32)
    ball = np.concatenate((ball_xy, np.full((len(ball_xy), 1), np.nan)), axis=1).astype(np.float32)
    ball = features.clean_ball(ball)
    near_home, near_away = features.nearest_by_team(players, ball, raw["slot_team"])
    f = {"ball": ball, "visible": ~np.isnan(ball[:, 0]), "ball_speed": features.ball_speed(ball),
         "near_home": near_home, "near_away": near_away, "team_speed": features.team_speed(players),
         "outside": features.outside_pitch(ball), "slot_team": raw["slot_team"], "players": players,
         "pitch_length": float(raw["pitch"][0])}
    return f, blocks, labels


def stats():
    """Print label statistics per match, at 10 fps, for the comparison with DFL."""
    from external import sportec_dfl
    print("match | graded min | dead % | ball detected % (all / live / dead) | players per frame (detected) | "
          "possessions per h: touch, 1 s, 2 s, 3 s | % under 2 s: touch, 2 s | group empty inside phases %")
    for match in matches():
        raw = parse(match)
        lab = labels10(raw)
        hours = len(lab) / SKC_FPS / 3600
        live = lab != D
        cells = []
        short = []
        for rule in (None, 1.0, 2.0, 3.0):
            lab_r = lab if rule is None else _controlled10(lab, rule)
            starts, stops, values = runs_of(lab_r)
            alive = (values == H) | (values == A)
            # runs across the half-time break are split by the D of the phases, so no special case
            cells.append("{:.0f}".format(alive.sum() / hours))
            if rule in (None, 2.0):
                short.append("{:.1f}".format(100 * np.mean((stops - starts)[alive] / SKC_FPS < 2.0)))
        print("{} | {:.0f} | {:.1f} | {:.1f} / {:.1f} / {:.1f} | {:.1f} ({:.1f}) | {} | {} | {:.1f}".format(
            match, hours * 60, 100 * np.mean(~live), 100 * raw["ball_detected"].mean(),
            100 * raw["ball_detected"][live].mean(), 100 * raw["ball_detected"][~live].mean(),
            np.mean(np.isfinite(raw["players"][:, :, 0]).sum(1)), np.mean(raw["detected"].sum(1)),
            ", ".join(cells), ", ".join(short), 100 * np.mean(raw["group"][live] == 0)))


def _controlled10(labels, min_s):
    """sportec_dfl.controlled at 10 fps (its frame count comes from common.FPS, so it is rescaled here)."""
    from external import sportec_dfl
    return sportec_dfl.controlled(labels, min_s * SKC_FPS / FPS)




SEQ_DIR = os.path.join(CACHE_DIR, "tcn_skc")
TCN_GROUPS = ("ball", "near3", "near1")
FOLD_KEYS = ("ball", "visible", "outside", "near_home", "near_away", "ball_speed", "team_speed")


def seq():
    """Per match: the TCN's 75 input columns (as tcn.seq), the decoder's features, the labels (2 s rule) and
    the touch-level labels at 25 fps, and the submission's p3-refined probabilities (models/), saved to
    cache/tcn_skc/skc_<id>.npz. Resumable."""
    import time
    import learned
    import run
    import stage2
    import team_shape
    import common
    common.FLAGS_ENABLED = False
    os.makedirs(SEQ_DIR, exist_ok=True)
    bundle, refiner = learned.load(), stage2.load()
    base_groups = team_shape.LOCAL_GROUPS
    for match in matches():
        target = os.path.join(SEQ_DIR, "skc_{}.npz".format(match))
        if os.path.isfile(target):
            continue
        started = time.time()
        f, blocks, labels = load_match(match)
        _, _, touch = load_match(match, control_s=0.0)
        ev = run.evidence(f, blocks)
        names, x_home, x_swap = learned.match_inputs(f, blocks, ev)
        if names != bundle["names"]:
            raise SystemExit("P3 feature set differs from the saved model")
        refined = stage2.refine(refiner, f, blocks, ev, learned.predict_proba(bundle["model"], x_home, x_swap))
        team_shape.LOCAL_GROUPS = TCN_GROUPS
        try:
            names, x_home, x_swap = learned.match_inputs(f, blocks, ev)
        finally:
            team_shape.LOCAL_GROUPS = base_groups
        keep = [i for i, c in enumerate(names) if not c.endswith(("_m25", "_m75"))]
        np.savez(target, x_home=x_home[:, keep], x_swap=x_swap[:, keep], labels=labels, touch=touch,
                 blocks=np.array(blocks, dtype=np.int64), names=np.array([names[i] for i in keep]),
                 probs_refined=refined.astype(np.float32), pitch_length=np.float64(f["pitch_length"]),
                 **{"f_" + k: f[k] for k in FOLD_KEYS})
        print("  seq skc_{} ({:.0f} s)".format(match, time.time() - started), flush=True)


if __name__ == "__main__":
    seq() if len(sys.argv) > 1 and sys.argv[1] == "seq" else stats()
