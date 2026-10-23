"""Read the tactical ATD once and cache it as numpy arrays.

The raw file is 187 MB of text and takes a minute or two to parse; the cache loads in under a
second. Everything is indexed by video frame (ATD frame_id - 1), so array row i is submission row i.

    ball      (N, 3)      x, y in metres, speed in m/s. NaN where the ball is not tracked.
    players   (N, S, 3)   the same per track slot. NaN where the slot is empty.
    team      (S,)        0 home, 1 away, per track slot.

The cache lives in cache/, which is gitignored: it is a copy of the gated data.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames  # noqa: E402

# Params
CACHE_DIR = os.path.join(ROOT, "cache")
CACHE_PATH = os.path.join(CACHE_DIR, "atd_tactical.npz")
BROADCAST_CACHE_PATH = os.path.join(CACHE_DIR, "atd_broadcast.npz")


def team_of_each_track(meta_path):
    """0 for a home slot, 1 for an away slot. The first team in the metadata is home."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))
    return np.array([0 if player.get("teamId") == team_ids[0] else 1
                     for player in root.iter("Player")], dtype=np.int8)


def _group(text):
    """'x,y,s' -> three floats; 'NaN,...' parses to NaN without special-casing."""
    a, b, c = text.split(",")
    return float(a), float(b), float(c)


def parse(meta_path, raw_path, n_frames, pitch):
    team = team_of_each_track(meta_path)
    n_slots = len(team)
    length, width = pitch
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            row = int(parts[0]) - 1
            groups = parts[1].split(";")
            players[row, :len(groups)] = [_group(g) for g in groups[:n_slots]]
            ball[row] = _group(parts[2])

    scale = np.array([length, width, 1.0], dtype=np.float32)
    return ball * scale, players * scale, team


# Set to (ball, players, team) to run the whole pipeline on modified tracking -- used by the
# gap simulation in gapsim.py to delete the ball where it was actually seen.
OVERRIDE = None
# Fill tactical ball gaps from the time-aligned broadcast tracking (see broadcast.py).
FUSE_BROADCAST = True
FUSED_PATH = os.path.join(CACHE_DIR, "atd_fused_ball.npz")


def load(rebuild=False):
    """(ball, players, team), from the cache if it exists."""
    if OVERRIDE is not None and not rebuild:
        return OVERRIDE
    ball, players, team = load_tactical(rebuild)
    if FUSE_BROADCAST and not rebuild:
        ball = fused(ball)
    return ball, players, team


def fused(ball):
    """The tactical ball with broadcast fills, cached."""
    if os.path.isfile(FUSED_PATH):
        return np.load(FUSED_PATH)["ball"]
    import broadcast                       # deferred: broadcast imports this module
    from features import rolling_mean
    from games import half_video_blocks
    blocks = [(first, last) for _, first, last in half_video_blocks(GAME)]
    out, _ = broadcast.fused_ball(ball, blocks, rolling_mean)
    np.savez(FUSED_PATH, ball=out)
    return out


def load_tactical(rebuild=False):
    """(ball, players, team) exactly as the tactical ATD has them."""
    if not rebuild and os.path.isfile(CACHE_PATH):
        data = np.load(CACHE_PATH)
        return data["ball"], data["players"], data["team"]

    meta, raw = require(["atd_meta", "atd_raw"])
    ball, players, team = parse(meta, raw, source_frames(), GAMES[GAME]["pitch"])
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(CACHE_PATH, ball=ball, players=players, team=team)
    return ball, players, team


def load_broadcast(rebuild=False):
    """(ball, players, team) for the broadcast recording, on ITS OWN frame index.

    Not aligned to the tactical video -- see broadcast.py for the alignment.
    """
    if not rebuild and os.path.isfile(BROADCAST_CACHE_PATH):
        data = np.load(BROADCAST_CACHE_PATH)
        return data["ball"], data["players"], data["team"]

    meta, raw = require(["atd_meta", "atd_raw"], source="broadcast")
    ball, players, team = parse(meta, raw, source_frames("broadcast"), GAMES[GAME]["pitch"])
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(BROADCAST_CACHE_PATH, ball=ball, players=players, team=team)
    return ball, players, team


if __name__ == "__main__":
    ball, players, team = load(rebuild=True)
    # Shapes and coverage only -- never print the tracking itself.
    print("ball {}  players {}  slots home {} away {}".format(
        ball.shape, players.shape, int((team == 0).sum()), int((team == 1).sum())))
    print("ball present on {:.1%} of frames".format(np.isfinite(ball[:, 0]).mean()))
