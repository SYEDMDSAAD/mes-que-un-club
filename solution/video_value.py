"""Does the video ball detector (solution/video/ball_detector.py) help? Aggregates only.

Reads cache/ball_video_tactical.csv (the detector's output, derived from the gated video), puts
its top detection on the pitch with the tactical homography, fills the frames where our ball is
missing with detections that are confident and consistent with the next or previous frame, and
compares the pipeline with and without that fill on the BAR-ATH checks.

    python solution/video_value.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "starter")]
import possession_metrics as pm  # noqa: E402
import experiment  # noqa: E402
import load as loader  # noqa: E402
import model  # noqa: E402
import tags  # noqa: E402
from features import compute  # noqa: E402
from load_homography import load as load_homography  # noqa: E402

# Params
CSV = os.path.join(ROOT, "cache", "ball_video_tactical.csv")
CONF = 0.1
LINK_PX = 15
STILL_PX = 2              # a detection that moved less than this since the previous frame is still


def detections():
    """(N, 2) pitch metres of the top detection, (N,) its confidence, (N, 2) its pixel."""
    hom, _, _ = load_homography()
    n = len(hom)
    top = pd.read_csv(CSV)
    top = top[top["rank"] == 1]
    f = top["frame"].values
    px, py = top["px"].values.astype(float), top["py"].values.astype(float)
    inv = np.linalg.inv(hom[f])
    d = inv[:, 2, 0] * px + inv[:, 2, 1] * py + inv[:, 2, 2]
    pitch = np.full((n, 2), np.nan)
    pitch[f, 0] = (inv[:, 0, 0] * px + inv[:, 0, 1] * py + inv[:, 0, 2]) / d * model.PITCH[0]
    pitch[f, 1] = (inv[:, 1, 0] * px + inv[:, 1, 1] * py + inv[:, 1, 2]) / d * model.PITCH[1]
    conf = np.zeros(n)
    conf[f] = top["conf"].values
    pix = np.full((n, 2), np.nan)
    pix[f] = np.c_[px, py]
    return pitch, conf, pix


def linked(conf, pix):
    """Confident detections with a confident neighbour frame within LINK_PX pixels."""
    ok = conf >= CONF
    step = np.hypot(*(pix[1:] - pix[:-1]).T)
    with np.errstate(invalid="ignore"):
        prev = np.r_[False, ok[:-1] & (step <= LINK_PX)]
        nxt = np.r_[ok[1:] & (step <= LINK_PX), False]
    return ok & (prev | nxt)


def main():
    pitch, conf, pix = detections()
    ball, players, team = loader.load()
    gap = linked(conf, pix) & ~np.isfinite(ball[:, 0]) & model.graded_mask(len(ball))
    with np.errstate(invalid="ignore"):
        moving = np.r_[False, np.hypot(*(pix[1:] - pix[:-1]).T) >= STILL_PX]
    fills = {"v10": None, "v10 + video fill": gap, "v10 + video fill, moving only": gap & moving}
    events = tags.load()
    gt = pm.label_codes(pm.read_label_csv(experiment.GT))
    print("{:<30} spellF1  macroF1  tagTeam  restart  rLag    poss   dur  Dshare".format(""))
    for name, fill in fills.items():
        b = ball.copy()
        if fill is not None:
            b[fill, 0], b[fill, 1] = pitch[fill, 0], pitch[fill, 1]
        loader.OVERRIDE = (b, players, team)
        try:
            row = experiment.score(model.predict({}, compute(), events), gt, events)
        finally:
            loader.OVERRIDE = None
        print("{:<30} {}  filled {}".format(name, row, 0 if fill is None else int(fill.sum())),
              flush=True)


if __name__ == "__main__":
    main()
