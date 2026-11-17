"""Parse the tactical ATD once into numpy arrays and cache them.

File structure
--------------
- parse_meta(): slot -> team, goalkeeper slots, attack directions, from the metadata XML.
- parse_raw(): the raw text file -> players (N, 46, 3) and ball (N, 3) float32 arrays.
- load(): cached arrays, building the cache on first call.
- __main__: build the cache and print the sanity checks.

Arrays are in METRES (x along the 105.3 m length, y along the 68 m width) with NaN where the
tracker saw nothing; channel 2 is the tracker's own speed in m/s. Row i is video frame i
(ATD frame_id - 1), so every array lines up with a submission row.

The cache lives in solution/cache/, which is gitignored: it is a frame-level export of the
match data and must never be committed (RULES.md, "do not publish derived data").

    python solution/atd_cache.py
"""

import io
import os
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from common import CACHE_DIR, GAME, N_FRAMES, PITCH_LENGTH, PITCH_WIDTH, require

# Params
CACHE_PATH = os.path.join(CACHE_DIR, "atd_tactical.npz")
N_CHANNELS = 3  # x, y, speed
HOME, AWAY = 0, 1


def parse_meta(meta_path):
    """Read the ATD metadata XML.

    Args:
        meta_path: path to BAR-ATH_tactical_atd.xml.

    Returns:
        dict: slot_team (int8 per slot, 0 home / 1 away), slot_is_gk (bool per slot),
        home_attacks_right_first_half (bool), team_names (home, away).
    """
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError("expected 2 teams in {}, found {}".format(meta_path, len(teams)))
    home_id = teams[0].get("id")  # FIFA/EPTS convention: first team is home (DATA.md agrees)

    def team_param(team, name):
        for parameter in team.iter("ProviderParameter"):
            if parameter.findtext("Name") == name:
                return parameter.findtext("Value")
        return None

    slot_team, slot_is_gk = [], []
    for player in root.iter("Player"):
        slot_team.append(HOME if player.get("teamId") == home_id else AWAY)
        position = None
        for parameter in player.iter("ProviderParameter"):
            if parameter.findtext("Name") == "position_type":
                position = parameter.findtext("Value")
        slot_is_gk.append(position == "Goalkeeper")

    direction = team_param(teams[0], "attack_direction_first_half")
    return {
        "slot_team": np.array(slot_team, dtype=np.int8),
        "slot_is_gk": np.array(slot_is_gk, dtype=bool),
        "home_attacks_right_first_half": direction == "left_to_right",
        "team_names": (teams[0].findtext("Name"), teams[1].findtext("Name")),
    }


def parse_raw(raw_path, n_slots):
    """Parse the raw ATD text into dense arrays in metres.

    Each line is "frame_id:p1;p2;...;pN:ball" with every group "x,y,speed". Flattening both
    separators to commas turns the file into a plain numeric CSV that pandas' C parser reads in
    seconds, instead of the minutes a per-line Python loop takes.

    Args:
        raw_path: path to BAR-ATH_tactical_atd.txt.
        n_slots: number of track slots declared in the metadata.

    Returns:
        tuple[np.ndarray, np.ndarray]: players (N_FRAMES, n_slots, 3), ball (N_FRAMES, 3), float32.
    """
    with open(raw_path, "r") as handle:
        text = handle.read()
    text = text.replace(":", ",").replace(";", ",")
    n_columns = 1 + N_CHANNELS * (n_slots + 1)
    table = pd.read_csv(io.StringIO(text), header=None, dtype=np.float64,
                        na_values=["NaN"], engine="c").values
    del text
    if table.shape[1] != n_columns:
        raise ValueError("raw ATD has {} columns, expected {}".format(table.shape[1], n_columns))

    frame_ids = table[:, 0].astype(np.int64)
    if not (len(frame_ids) == N_FRAMES and frame_ids[0] == 1
            and np.all(np.diff(frame_ids) == 1)):
        raise ValueError("ATD is not one record per video frame numbered 1..{}".format(N_FRAMES))

    players = table[:, 1:1 + N_CHANNELS * n_slots].reshape(N_FRAMES, n_slots, N_CHANNELS)
    ball = table[:, 1 + N_CHANNELS * n_slots:]
    scale = np.array([PITCH_LENGTH, PITCH_WIDTH, 1.0])
    return (players * scale).astype(np.float32), (ball * scale).astype(np.float32)


def load(rebuild=False):
    """Cached ATD arrays, building them on first use.

    Args:
        rebuild: ignore an existing cache.

    Returns:
        dict: players, ball, slot_team, slot_is_gk, home_attacks_right_first_half, team_names.
    """
    if os.path.isfile(CACHE_PATH) and not rebuild:
        with np.load(CACHE_PATH, allow_pickle=False) as cached:
            out = {key: cached[key] for key in cached.files}
        out["home_attacks_right_first_half"] = bool(out["home_attacks_right_first_half"])
        out["team_names"] = tuple(str(name) for name in out["team_names"])
        return out

    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME)
    meta = parse_meta(meta_path)
    players, ball = parse_raw(raw_path, len(meta["slot_team"]))
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(CACHE_PATH, players=players, ball=ball, slot_team=meta["slot_team"],
             slot_is_gk=meta["slot_is_gk"],
             home_attacks_right_first_half=np.array(meta["home_attacks_right_first_half"]),
             team_names=np.array(meta["team_names"]))
    out = dict(meta, players=players, ball=ball)
    return out


if __name__ == "__main__":
    started = time.time()
    atd = load(rebuild=True)
    players, ball = atd["players"], atd["ball"]
    visible = ~np.isnan(ball[:, 0])
    print("parsed in {:.1f} s -> {}".format(time.time() - started, CACHE_PATH))
    print("teams: H = {}, A = {}".format(*atd["team_names"]))
    print("frames {}, slots {} (home {}, away {}), goalkeeper slots {}".format(
        players.shape[0], players.shape[1], int(np.sum(atd["slot_team"] == HOME)),
        int(np.sum(atd["slot_team"] == AWAY)), np.flatnonzero(atd["slot_is_gk"]).tolist()))
    print("home attacks right in 1st half: {}".format(atd["home_attacks_right_first_half"]))
    print("ball visible on {:.1f}% of all frames (DATA.md: 56.9%)".format(100 * visible.mean()))
