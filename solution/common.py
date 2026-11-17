"""Shared paths, constants and small helpers for the possession solution.

File structure
--------------
- Data location: resolves the gated data bundle (MTAH_DATA_DIR, ./data, or ../data).
- Repository imports: puts the repo root on sys.path so games.py, starter/ and evaluation/ import.
- Constants: frame rate, pitch size, label codes, output folders.
- Helpers: half_masks(), runs_of(), write_flag().

Every other module in solution/ imports from here, so the data location and the label encoding
are decided in exactly one place.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _resolve_data_dir():
    """Pick the data bundle: MTAH_DATA_DIR, else ./data, else ../data (sibling of the repo).

    Returns:
        str: the directory that contains tactical/ and broadcast/.
    """
    env = os.environ.get("MTAH_DATA_DIR")
    if env:
        return env
    for candidate in (os.path.join(ROOT, "data"), os.path.join(os.path.dirname(ROOT), "data")):
        if os.path.isdir(os.path.join(candidate, "tactical")):
            return candidate
    return os.path.join(ROOT, "data")


# games.py reads MTAH_DATA_DIR at import time, so this has to be set before it is imported.
os.environ.setdefault("MTAH_DATA_DIR", _resolve_data_dir())
DATA_DIR = os.environ["MTAH_DATA_DIR"]

for _path in (ROOT, os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "starter")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402

from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402

# Params
FPS = GAMES[GAME]["fps"]
PITCH_LENGTH, PITCH_WIDTH = GAMES[GAME]["pitch"]
N_FRAMES = source_frames("tactical", GAME)

# Label codes. Same order as evaluation/possession_metrics.py (H=0, A=1, D=2, X=3).
H, A, D, X = 0, 1, 2, 3
LETTERS = np.array(["H", "A", "D", "X"])

CACHE_DIR = os.path.join(HERE, "cache")      # derived frame-level arrays: gitignored, never published
REPORT_DIR = os.path.join(HERE, "reports")   # aggregate diagnostics and plots
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
FLAGS_PATH = os.path.join(REPORT_DIR, "FLAGS.md")


def half_masks(n_frames=N_FRAMES):
    """Boolean masks of the two graded halves, from the published half boundaries.

    These boundaries are published in DATA.md and games.py as match structure, not labels.

    Args:
        n_frames: length of the video.

    Returns:
        list[tuple[int, int, int]]: (half, first_frame, last_frame_inclusive).
        np.ndarray: bool mask, True on graded frames.
    """
    blocks = half_video_blocks(GAME)
    graded = np.zeros(n_frames, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    return blocks, graded


def runs_of(values):
    """Constant-value runs of a 1-D array.

    Args:
        values: 1-D array.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: starts, stops (exclusive), and the run values.
    """
    values = np.asarray(values)
    if len(values) == 0:
        empty = np.array([], dtype=int)
        return empty, empty, values[:0]
    change = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.concatenate(([0], change))
    stops = np.concatenate((change, [len(values)]))
    return starts, stops, values[starts]


FLAGS_ENABLED = True  # calibration on other matches turns this off: its fallbacks are not ours


def write_flag(message):
    """Append a blocker or caveat to reports/FLAGS.md instead of stopping the pipeline.

    Args:
        message: one line describing what was assumed or skipped, and why.
    """
    if not FLAGS_ENABLED:
        return
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(FLAGS_PATH, "a", encoding="utf-8") as handle:
        handle.write("- {}\n".format(message))
    print("FLAG: {}".format(message))
