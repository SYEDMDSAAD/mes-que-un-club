"""How close is the damaged DFL tracking to ours? Aggregates only, never a frame or a position.

Every learned model is trained on DFL tracking damaged to look like BAR-ATH (dfl_eval.degrade),
plus Metrica's sample matches as they are. A model is only as good as that imitation, feature by
feature. This prints the statistics the features are built on, side by side, for BAR-ATH (fused
ball), clean DFL, damaged DFL and Metrica:

    seen        share of graded frames with a ball, and by state where labels exist: DFL and
                Metrica truth, BAR-ATH's public window, and the full match from --labels (your
                own submission file: predicted states, not ground truth)
    gap live    mean / median / p90 length, in frames, of missing-ball runs inside live play
    gap dead    the same inside stoppages (mean and share missing are what dfl_eval's
                MISS_* and GAP_* constants want)
    v live      per-frame ball speed on live frames, |step| * 25, p10 / p50 / p90. Independent
                position noise of s metres per axis adds about s * 44 m/s on its own
    still dead  1-s ball displacement on dead frames with a ball, p50 (TRACAB freezes the ball)
    near live   ball to the nearest player on live frames, p10 / p50 / p90
    dv live     change of ball velocity between frames on live frames, p50 / p90: a smoothed
                ball has a small one, a noisy ball a large one

If damaged DFL and BAR-ATH disagree on a row, every feature built on that row means something
different in training and in the submission.

    python solution/domain_check.py [--labels "Mes Que Un Club.csv"]
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import dfl_eval  # noqa: E402
import external_metrica  # noqa: E402
from external_dfl import load_match as load_dfl  # noqa: E402
from games import GAME, half_video_blocks  # noqa: E402
from load import load  # noqa: E402

# Params
FPS = 25
GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def pct(x, qs):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, q)) if len(x) else np.nan for q in qs]


def runs(mask):
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def stats(ball, players, labels, blocks):
    n = len(ball)
    graded = np.zeros(n, bool)
    for first, last in blocks:
        graded[first:last + 1] = True
    lab = np.asarray(labels) if labels is not None else np.full(n, "X")
    live = graded & np.isin(lab, ["H", "A"])
    dead = graded & (lab == "D")
    x, y = ball[:, 0].astype(float), ball[:, 1].astype(float)
    seen = np.isfinite(x)

    vx = np.r_[np.nan, np.diff(x)] * FPS
    vy = np.r_[np.nan, np.diff(y)] * FPS
    v = np.hypot(vx, vy)
    dv = np.hypot(np.r_[np.nan, np.diff(vx)], np.r_[np.nan, np.diff(vy)])
    still = np.full(n, np.nan)
    still[FPS:] = np.hypot(x[FPS:] - x[:-FPS], y[FPS:] - y[:-FPS])
    d = np.hypot(players[..., 0] - x[:, None], players[..., 1] - y[:, None])
    near = np.nanmin(np.where(np.isfinite(d), d, np.nan), axis=1)

    gaps = {"live": [], "dead": []}
    for lo, hi in zip(*runs(graded & ~seen)):
        state = lab[(lo + hi) // 2]
        if state in ("H", "A"):
            gaps["live"].append(hi - lo)
        elif state == "D":
            gaps["dead"].append(hi - lo)

    def share(m):
        return seen[m].mean() if m.any() else np.nan

    return {
        "seen": share(graded), "seen live": share(live), "seen dead": share(dead),
        "gap live": [np.mean(gaps["live"]) if gaps["live"] else np.nan] + pct(gaps["live"], (50, 90)),
        "gap dead": [np.mean(gaps["dead"]) if gaps["dead"] else np.nan] + pct(gaps["dead"], (50, 90)),
        "v live": pct(v[live], (10, 50, 90)), "still dead": pct(still[dead], (50,)),
        "near live": pct(near[live], (10, 50, 90)), "dv live": pct(dv[live], (50, 90)),
    }


def show(name, s):
    def f(v):
        if isinstance(v, list):
            return "/".join("{:.1f}".format(a) for a in v)
        return "{:.2f}".format(v)
    print("{:<26} seen {} (live {}, dead {})  gap live {}  gap dead {}  v live {}  "
          "still dead {}  near live {}  dv live {}".format(
              name, f(s["seen"]), f(s["seen live"]), f(s["seen dead"]), f(s["gap live"]),
              f(s["gap dead"]), f(s["v live"]), f(s["still dead"]), f(s["near live"]),
              f(s["dv live"])), flush=True)


def main(labels_path=None):
    ball, players, _ = load()
    blocks = [(first, last) for _, first, last in half_video_blocks(GAME)]
    public = pd.read_csv(GT)["label"].values
    idx = np.flatnonzero(public != "X")
    show("BAR-ATH public window", stats(ball, players, public, [(idx[0], idx[-1])]))
    if labels_path:
        mine = pd.read_csv(labels_path)["label"].astype(str).values
        show("BAR-ATH full, our labels", stats(ball, players, mine, blocks))

    for m in dfl_eval.cached_matches():
        b, p, _, truth, bl = load_dfl(m)
        show("DFL {} clean".format(m), stats(b, p, truth, bl))
        d_b, d_p = dfl_eval.degrade(b, p, bl, truth=truth)
        show("DFL {} damaged".format(m), stats(d_b, d_p, truth, bl))
    for m in external_metrica.MATCHES:
        b, p, _, truth, bl = external_metrica.load_match(m)
        show("Metrica {}".format(m), stats(b, p, truth, bl))


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[args.index("--labels") + 1] if "--labels" in args else None)
