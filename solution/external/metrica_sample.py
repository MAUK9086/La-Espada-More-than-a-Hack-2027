"""Metrica open sample games 1 and 2 as calibration data: features in our format, labels from events.

File structure
--------------
- fetch(): shallow-clone github.com/metrica-sports/sample-data if it is not present.
- load_tracking(): the two per-team CSVs -> players (N, S, 3) and ball (N, 3) in metres.
- player_speed(): speed from positions (this export has no speed channel).
- events_csv(), events_json(): the event data of games 1-2 (CSV) and game 3 (JSON).
- labels_from_events(): H/A/D per frame from the human-coded event data.
- load_tracking_epts(): game 3 tracking from the EPTS export.
- degrade_ball(), degrade_players(): ATD-like smoothing, noise and player dropout.
- load_game(): features dict with the same keys as features.build(), blocks, labels.

These are OTHER matches, and RULES.md allows labelled data about other games in any amount. The
data is Metrica's public sample ("for the broader football analytics community"; acknowledged in
docs/report/La_Espada.pdf). It lives outside this repository and is never committed.

Label derivation, written once from Metrica's event definitions and not adjusted afterwards:
- D starts at the END frame of a BALL OUT event (the ball crossing the line), at the end of a
  SHOT whose subtype says OUT or GOAL, and at a FAULT RECEIVED (the whistle).
- D ends at the next restart: a SET PIECE event, or an event whose subtype contains GOAL KICK.
- While alive, the owner is the team of the latest controlling event (PASS, RECOVERY, SET PIECE,
  SHOT, BALL LOST, BALL OUT). CHALLENGE is not control, so a duel does not move possession; a
  BALL LOST keeps the ball with the losing team until the other team RECOVERS it -- the "loose
  ball stays with the last team" rule of CHALLENGE.md.
- Any span longer than EVENT_GAP_S between two consecutive events is X (not graded). Event data
  only codes on-ball actions, so injuries, substitutions and referee stoppages leave silent gaps
  that it would otherwise call live play (seen as 20-60 s spans with no ball and players walking).
  The rule uses only the event timestamps, never the tracking, so it cannot favour the model.
"""

import io
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import features  # noqa: E402
from common import A, D, FPS, H, X, runs_of  # noqa: E402

# Params
REPO_URL = "https://github.com/metrica-sports/sample-data.git"
SAMPLE_DIR = os.environ.get("METRICA_SAMPLE_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "external", "metrica-sample-data")
PITCH = (105.0, 68.0)    # the sample README: both games are 105 x 68 m
SPEED_HALF_WINDOW = 2    # frames, centred difference for player speed
CONTROL_TYPES = {"PASS", "CARRY", "RECOVERY", "SET PIECE", "SHOT", "BALL LOST", "BALL OUT"}
GAMES = (1, 2)         # calibration games
HOLDOUT_GAME = 3        # held out: evaluated only after a change is decided on games 1-2
CONTROLLED_LABELS = True  # review item R2: apply sportec_dfl.controlled() to the event labels
EVENT_GAP_S = 15         # s without any coded event -> the event data cannot say what happened
DEGRADE_SIGMA = 8.0      # frames, Gaussian smoothing that matches the ATD ball's acceleration tail
DEGRADE_NOISE_M = 0.0025 # m, position noise that matches the ATD ball's median acceleration
PLAYER_VISIBLE_MEAN_S = 66.0  # s, mean visible spell per player  } ~93% of players seen, as in
PLAYER_MISSING_MEAN_S = 5.0   # s, mean missing spell per player  } the ATD (10.2-10.3 of 11)


def fetch():
    """Clone the sample-data repository if it is not already on disk.

    Returns:
        str: path to its data/ folder.
    """
    data = os.path.join(SAMPLE_DIR, "data")
    if not os.path.isdir(data):
        subprocess.check_call(["git", "clone", "--depth", "1", REPO_URL, SAMPLE_DIR])
    return data


def _read_team(path):
    """One team's tracking CSV -> (frames, periods, positions (N, S, 2), ball (N, 2))."""
    frame = pd.read_csv(path, skiprows=2, low_memory=False)
    periods = frame["Period"].values.astype(int)
    frames = frame["Frame"].values.astype(int)
    columns = list(frame.columns)
    player_cols = [c for c in columns if c.startswith("Player")]
    xy = []
    for name in player_cols:
        index = columns.index(name)
        xy.append(frame.iloc[:, [index, index + 1]].values.astype(np.float64))
    ball_index = columns.index("Ball")
    ball = frame.iloc[:, [ball_index, ball_index + 1]].values.astype(np.float64)
    return frames, periods, np.stack(xy, axis=1), ball


def player_speed(xy):
    """Speed from positions by centred difference, NaN unless both ends are seen.

    Args:
        xy: (N, S, 2) positions in metres.

    Returns:
        np.ndarray: (N, S) speed in m/s.
    """
    k = SPEED_HALF_WINDOW
    speed = np.full(xy.shape[:2], np.nan)
    delta = xy[2 * k:] - xy[:-2 * k]
    speed[k:-k] = np.hypot(delta[..., 0], delta[..., 1]) * FPS / (2 * k)
    return speed


def load_tracking(game):
    """Both teams' tracking merged into our array layout.

    Args:
        game: 1 or 2.

    Returns:
        dict: players (N, S, 3), ball (N, 3), slot_team (S,), periods (N,).
    """
    data = fetch()
    stem = os.path.join(data, "Sample_Game_{0}".format(game), "Sample_Game_{0}_RawTrackingData_".format(game))
    frames_h, periods, home, ball = _read_team(stem + "Home_Team.csv")
    frames_a, _, away, _ = _read_team(stem + "Away_Team.csv")
    if not np.array_equal(frames_h, frames_a) or frames_h[0] != 1 or np.any(np.diff(frames_h) != 1):
        raise ValueError("game {}: home and away tracking are not one row per frame from 1".format(game))
    scale = np.array(PITCH)
    xy = np.concatenate((home, away), axis=1) * scale
    players = np.concatenate((xy, player_speed(xy)[..., None]), axis=2).astype(np.float32)
    ball3 = np.concatenate((ball * scale, np.full((len(ball), 1), np.nan)), axis=1).astype(np.float32)
    slot_team = np.array([0] * home.shape[1] + [1] * away.shape[1], dtype=np.int8)
    return {"players": players, "ball": ball3, "slot_team": slot_team, "periods": periods}


def events_csv(game):
    """Games 1-2 event CSV as the normalised table: team (H/A), kind, subtype, start, end (0-based)."""
    raw = pd.read_csv(os.path.join(fetch(), "Sample_Game_{0}".format(game),
                                   "Sample_Game_{0}_RawEventsData.csv".format(game)))
    return pd.DataFrame({
        "team": np.where(raw["Team"].astype(str).str.strip().str.lower() == "home", H, A),
        "kind": raw["Type"].astype(str).str.strip().str.upper(),
        "subtype": raw["Subtype"].fillna("").astype(str).str.strip().str.upper(),
        "start": raw["Start Frame"].astype(int) - 1,
        "end": raw["End Frame"].astype(int) - 1,
    })


def events_json(home_team_id):
    """Game 3 event JSON as the normalised table. Subtypes may be one dict or a list of dicts."""
    with open(os.path.join(fetch(), "Sample_Game_3", "Sample_Game_3_events.json")) as handle:
        raw = json.load(handle)["data"]
    rows = []
    for event in raw:
        subtypes = event.get("subtypes") or []
        if isinstance(subtypes, dict):
            subtypes = [subtypes]
        rows.append({
            "team": H if (event.get("team") or {}).get("id") == home_team_id else A,
            "kind": event["type"]["name"].strip().upper(),
            "subtype": "-".join(s["name"].strip().upper() for s in subtypes),
            "start": int(event["start"]["frame"]) - 1,
            "end": int(event["end"]["frame"]) - 1,
        })
    return pd.DataFrame(rows)


def labels_from_events(events, n_frames, periods):
    """H/A/D per frame from the event data (see the module docstring for the rules).

    Args:
        events: normalised table from events_csv() / events_json().
        n_frames: tracking length.
        periods: (N,) period of each frame.

    Returns:
        np.ndarray: (N,) int8 codes; X outside the two periods.
    """
    events = events.sort_values(["start", "end"], kind="stable")
    owner = np.full(n_frames, -1, dtype=np.int8)
    dead = np.zeros(n_frames, dtype=bool)

    stop_at = None
    for event in events.itertuples(index=False):
        kind, subtype, team = event.kind, event.subtype, event.team
        start, end = int(event.start), int(event.end)

        restart = kind == "SET PIECE" or "GOAL KICK" in subtype
        if restart and stop_at is not None and start > stop_at:
            dead[stop_at:start] = True
            stop_at = None
        if kind in CONTROL_TYPES and stop_at is None:
            owner[max(start, 0):] = team
        if kind == "BALL OUT":
            stop_at = max(end, start)
        elif kind == "SHOT" and ("OUT" in subtype or "GOAL" in subtype):
            stop_at = max(end, start)
        elif kind == "FAULT RECEIVED":
            stop_at = start

    labels = np.where(dead, D, np.where(owner == 1, A, H)).astype(np.int8)
    labels[owner < 0] = D  # before the first event: the kick-off has not happened

    # Silent spans: no event starts or ends for more than EVENT_GAP_S. Dead-ball spans are
    # already explained by their bounding events, so only alive frames are masked.
    marks = np.unique(np.concatenate((events["start"].values, events["end"].values)))
    marks = marks[(marks >= 0) & (marks < n_frames)]
    for left, right in zip(marks[:-1], marks[1:]):
        if right - left > EVENT_GAP_S * FPS:
            span = slice(left + 1, right)
            labels[span] = np.where(labels[span] == D, D, X)
    labels[(periods != 1) & (periods != 2)] = X
    return labels


def load_tracking_epts():
    """Game 3 (FIFA-EPTS) tracking in our array layout.

    The export is split into blocks (one per substitution window); each block lists its own 22
    player channels in its own order. Every player gets one fixed slot, so a slot is a person here.

    Returns:
        dict: players (N, S, 3), ball (N, 3), slot_team (S,), periods (N,), home_team_id.
    """
    import xml.etree.ElementTree as ET
    folder = os.path.join(fetch(), "Sample_Game_3")
    root = ET.parse(os.path.join(folder, "Sample_Game_3_metadata.xml")).getroot()
    teams = [team.get("id") for team in root.iter("Team")]
    home_id = teams[0]  # FIFA/EPTS: first team is home
    team_of_player = {p.get("id"): p.get("teamId") for p in root.iter("Player")}
    player_of_channel = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    players = sorted(team_of_player, key=lambda pid: (team_of_player[pid] != home_id, pid))
    slot_of = {pid: i for i, pid in enumerate(players)}

    def param(name):
        for parameter in root.iter("ProviderParameter"):
            if parameter.findtext("Name") == name:
                return int(parameter.findtext("Value"))
        raise KeyError(name)

    with open(os.path.join(folder, "Sample_Game_3_tracking.txt")) as handle:
        text = handle.read().replace(":", ",").replace(";", ",")
    table = pd.read_csv(io.StringIO(text), header=None, dtype=np.float64, na_values=["NaN"]).values
    frame_ids = table[:, 0].astype(int)
    n = len(table)
    if frame_ids[0] != 1 or np.any(np.diff(frame_ids) != 1):
        raise ValueError("game 3 tracking is not one row per frame from 1")

    xy = np.full((n, len(players), 2), np.nan)
    for spec in root.iter("DataFormatSpecification"):
        first, last = int(spec.get("startFrame")) - 1, int(spec.get("endFrame"))
        channels = [ref.get("playerChannelId") for ref in spec.iter("PlayerChannelRef")]
        for column, channel in enumerate(channels):
            axis = 0 if channel.endswith("_x") else 1
            xy[first:last, slot_of[player_of_channel[channel]], axis] = table[first:last, 1 + column]
    ball = table[:, -2:]

    scale = np.array(PITCH)
    xy = xy * scale
    periods = np.zeros(n, dtype=int)
    periods[param("first_half_start") - 1:param("first_half_end")] = 1
    periods[param("second_half_start") - 1:param("second_half_end")] = 2
    out = {
        "players": np.concatenate((xy, player_speed(xy)[..., None]), axis=2).astype(np.float32),
        "ball": np.concatenate((ball * scale, np.full((n, 1), np.nan)), axis=1).astype(np.float32),
        "slot_team": np.array([0 if team_of_player[p] == home_id else 1 for p in players],
                              dtype=np.int8),
        "periods": periods,
        "home_team_id": home_id,
    }
    return out


def degrade_ball(ball, sigma=DEGRADE_SIGMA, noise=DEGRADE_NOISE_M, seed=0):
    """Make a clean sample-game ball track look like the automatic BAR-ATH one.

    The automatic tracker smooths the ball (its acceleration tail is ~10x shorter than the sample
    data's) and leaves a little jitter (its median acceleration is ~7x higher). A NaN-aware Gaussian
    smoothing plus white position noise reproduces both; sigma and noise were chosen by matching
    the acceleration quantiles of the two datasets on live play, which uses no labels of either.

    Args:
        ball: (N, 3) ball in metres.
        sigma: Gaussian smoothing, frames.
        noise: m, standard deviation of added position noise.
        seed: RNG seed, so calibration is deterministic.

    Returns:
        np.ndarray: degraded copy.
    """
    out = ball.copy()
    if sigma > 0:
        k = int(4 * sigma)
        x = np.arange(-k, k + 1)
        weights = np.exp(-x ** 2 / (2.0 * sigma ** 2))
        weights /= weights.sum()
        for c in (0, 1):
            values = ball[:, c].astype(np.float64)
            seen = np.isfinite(values)
            num = np.convolve(np.where(seen, values, 0.0), weights, "same")
            den = np.convolve(seen.astype(np.float64), weights, "same")
            out[:, c] = np.where(seen, num / np.maximum(den, 1e-9), np.nan)
    if noise > 0:
        rng = np.random.default_rng(seed)
        out[:, :2] += rng.normal(0.0, noise, size=(len(ball), 2)).astype(out.dtype)
    return out


def degrade_players(players, seed=0, visible_mean_s=PLAYER_VISIBLE_MEAN_S,
                    missing_mean_s=PLAYER_MISSING_MEAN_S):
    """Drop player observations in spells, so complete sample tracking looks like the ATD's.

    The automatic tracker sees about 10.2-10.3 of 11 players per team on a graded frame (~93%).
    Each player alternates exponential visible spells and missing spells whose means give that
    share. Players are only ever removed, never moved: the ATD's duplicates are not simulated.

    Args:
        players: (N, S, 3).
        seed: RNG seed (deterministic).
        visible_mean_s, missing_mean_s: mean spell lengths in seconds.

    Returns:
        np.ndarray: degraded copy.
    """
    rng = np.random.default_rng(1000 + seed)
    out = players.copy()
    n = len(players)
    for slot in range(players.shape[1]):
        t = int(rng.exponential(visible_mean_s * FPS))
        while t < n:
            gap = max(1, int(rng.exponential(missing_mean_s * FPS)))
            out[t:t + gap, slot, :] = np.nan
            t += gap + max(1, int(rng.exponential(visible_mean_s * FPS)))
    return out


def load_game(game, degrade=True):
    """Everything calibration needs for one sample game.

    Args:
        game: 1, 2 (CSV format) or 3 (FIFA-EPTS + JSON, the held-out game).
        degrade: make the ball track look like the automatic ATD (degrade_ball). On by default:
            the model is calibrated for the data it will actually be run on.

    Returns:
        tuple: (features dict, blocks [(period, first, last)], labels (N,) int8).
    """
    if game == HOLDOUT_GAME:
        tracking = load_tracking_epts()
        events = events_json(tracking["home_team_id"])
    else:
        tracking = load_tracking(game)
        events = events_csv(game)
    periods = tracking["periods"]
    labels = labels_from_events(events, len(periods), periods)
    if CONTROLLED_LABELS:
        # Review item R2: the same controlled-possession rule as the DFL labels.
        from external.sportec_dfl import controlled
        labels = controlled(labels)
    ball = tracking["ball"]
    if degrade:
        ball = degrade_ball(ball, seed=game)
        tracking["players"] = degrade_players(tracking["players"], seed=game)
        # Review item R1: these games had no ball dropout at all. With the label-free profile
        # (sportec_dfl.DROPOUT_RATIO set) they get the same dropout as the DFL matches.
        from external import sportec_dfl
        if sportec_dfl.DROPOUT_RATIO is not None:
            ball = sportec_dfl.drop_ball(ball, labels, game, features.team_speed(tracking["players"]))
    ball = features.clean_ball(ball)
    near_home, near_away = features.nearest_by_team(tracking["players"], ball, tracking["slot_team"])
    f = {
        "ball": ball,
        "visible": ~np.isnan(ball[:, 0]),
        "ball_speed": features.ball_speed(ball),
        "near_home": near_home,
        "near_away": near_away,
        "team_speed": features.team_speed(tracking["players"]),
        "outside": features.outside_pitch(ball),
        "slot_team": tracking["slot_team"],
        "players": tracking["players"],
        "pitch_length": float(PITCH[0]),
    }
    blocks = []
    for period in (1, 2):
        index = np.flatnonzero(periods == period)
        if len(index):
            blocks.append((period, int(index[0]), int(index[-1])))
    return f, blocks, labels


if __name__ == "__main__":
    for game in GAMES:
        f, blocks, labels = load_game(game)
        starts, stops, values = runs_of(labels)
        alive = (values == H) | (values == A)
        graded = labels != X
        print("game {}: {} frames, blocks {}, ball visible {:.1%}, possessions {}, mean {:.1f} s, "
              "D share {:.1%}".format(game, len(labels), blocks, f["visible"][graded].mean(),
                                      int(alive.sum()), ((stops - starts)[alive] / FPS).mean(),
                                      np.mean(labels[graded] == D)))
