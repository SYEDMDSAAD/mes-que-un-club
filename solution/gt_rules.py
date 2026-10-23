"""What the ground truth's boundaries line up with in the tracking -- the public window only.

The ground truth is mechanical: corrected tracking says "dead" or names an owning team, and that
is the label. So every boundary in it should coincide with something physical -- a touch by the
new team, a ball crossing a line, a kick that restarts play. For each of the public window's
boundaries this prints *offsets in frames* between the boundary and the nearest such event
(positive = the event comes after the boundary), never positions.

    python solution/gt_rules.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
from features import compute  # noqa: E402

# Params
GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
PITCH = (105.3, 68.0)
SEARCH = 75              # frames either side of a boundary
TOUCH = 1.5              # m
KICK = 8.0               # m/s of ball speed between consecutive frames


def nearest_offset(mask, t):
    """Signed offset from t to the closest True in mask[t-SEARCH : t+SEARCH], or None."""
    lo = max(t - SEARCH, 0)
    hits = np.flatnonzero(mask[lo:t + SEARCH + 1]) + lo - t
    return int(hits[np.argmin(np.abs(hits))]) if len(hits) else None


def fmt(v):
    return "  -  " if v is None else "{:+5d}".format(v)


if __name__ == "__main__":
    f = compute()
    gt = pd.read_csv(GT).label.values
    with np.errstate(invalid="ignore"):
        home_touch = (f["d_home"] < TOUCH) & (f["d_home"] < f["d_away"])
        away_touch = (f["d_away"] < TOUCH) & (f["d_away"] < f["d_home"])
        out = ((f["ball_x"] < 0) | (f["ball_x"] > PITCH[0]) |
               (f["ball_y"] < 0) | (f["ball_y"] > PITCH[1]))
        v = np.hypot(np.diff(f["ball_x"], prepend=np.nan), np.diff(f["ball_y"], prepend=np.nan)) * 25
        kick = v > KICK
    touch = {"H": home_touch, "A": away_touch}

    idx = np.flatnonzero(gt != "X")
    lo, hi = idx[0], idx[-1] + 1
    w = gt[lo:hi]
    bounds = np.flatnonzero(w[1:] != w[:-1]) + 1 + lo

    print("frame   change  seen±10  newTeamTouch  oldTeamTouch(last)  ballOut   kick")
    rows = []
    for t in bounds:
        a, b = gt[t - 1], gt[t]
        seen = f["ball_seen"][t - 10:t + 10].mean()
        new_touch = nearest_offset(touch[b], t) if b in "HA" else None
        old_last = None
        if a in "HA":
            before = np.flatnonzero(touch[a][t - SEARCH:t + 1])
            old_last = int(before[-1] - SEARCH) if len(before) else None
        rows.append((t, a + "->" + b, seen, new_touch, old_last,
                     nearest_offset(out, t), nearest_offset(kick, t)))
        print("{:>6}   {}     {:.2f}     {}         {}             {}    {}".format(
            t, a + "->" + b, seen, fmt(new_touch), fmt(old_last),
            fmt(nearest_offset(out, t)), fmt(nearest_offset(kick, t))))
