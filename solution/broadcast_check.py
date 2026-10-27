"""Can the broadcast tracking overrule the tactical one where the tactical one is wrong?

The tactical tracking sometimes puts the other team nearest the ball when the truth says
otherwise -- a ball in the wrong place, a player missing or swapped. The broadcast camera follows
the ball zoomed in, and Metrica tracked that recording separately, so near the ball its players
and ball are an independent second measurement. broadcast.py already lines the two up (one
offset per half) and borrows the broadcast ball for tactical gaps; this asks whether the
broadcast *players* help too.

On the public window, over live frames where a tracking puts someone within REACH m of the ball:

    tactical   the nearest team by the tactical tracking (fused ball), vs the ground truth
    broadcast  the same by the broadcast tracking, on frames where the broadcast is live footage
    rescue     of the frames where tactical is wrong and broadcast has an opinion, how often
               broadcast is right -- the number that decides whether it is worth fusing

and over the full match (no ground truth): how often the two disagree. Aggregates only.

    python solution/broadcast_check.py
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
import broadcast  # noqa: E402
from games import GAME, half_video_blocks  # noqa: E402
from load import load, load_broadcast  # noqa: E402

# Params
REACH = 2.0               # m
GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def nearest_team(ball_xy, players, team):
    """(team of the nearest player or -1, its distance) per frame."""
    d = np.hypot(players[..., 0] - ball_xy[:, None, 0], players[..., 1] - ball_xy[:, None, 1])
    d = np.where(np.isfinite(d), d, np.inf)
    k = np.argmin(d, axis=1)
    dist = d[np.arange(len(d)), k]
    return np.where(np.isfinite(dist), team[k], -1), dist


def main():
    t_ball, t_players, t_team = load()
    b_ball, b_players, b_team = load_broadcast()
    b_index, cost, _ = broadcast.aligned()
    live_b = (b_index >= 0) & (cost < broadcast.LIVE_COST)

    t_near, t_dist = nearest_team(t_ball[:, :2], t_players, t_team)
    n = len(t_ball)
    b_near, b_dist = np.full(n, -1), np.full(n, np.inf)
    rows = np.flatnonzero(live_b)
    bn, bd = nearest_team(b_ball[b_index[rows], :2], b_players[b_index[rows]], b_team)
    b_near[rows], b_dist[rows] = bn, bd
    t_ok = (t_near >= 0) & (t_dist < REACH)
    b_ok = (b_near >= 0) & (b_dist < REACH)

    gt = pd.read_csv(GT)["label"].values
    live = np.isin(gt, ["H", "A"])
    truth = np.where(gt == "H", 0, 1)
    print("public window, live frames: {}".format(int(live.sum())))
    for name, near, ok in (("tactical", t_near, t_ok), ("broadcast", b_near, b_ok)):
        m = live & ok
        print("  {:<10} has an opinion on {:.1%}, right on {:.1%} of those".format(
            name, m.sum() / live.sum(), (near[m] == truth[m]).mean() if m.any() else np.nan))
    wrong = live & t_ok & (t_near != truth)
    both = wrong & b_ok
    print("  tactical wrong on {} frames; broadcast has an opinion on {} of them and is right on "
          "{:.1%}".format(int(wrong.sum()), int(both.sum()),
                          (b_near[both] == truth[both]).mean() if both.any() else np.nan))
    gap = live & ~t_ok & b_ok
    print("  tactical silent on {} live frames where broadcast has an opinion; right on "
          "{:.1%}".format(int(gap.sum()), (b_near[gap] == truth[gap]).mean() if gap.any()
                          else np.nan))

    graded = np.zeros(n, bool)
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
    both_ok = graded & t_ok & b_ok
    print("full match: both have an opinion on {:.1%} of graded frames; they disagree on {:.1%} "
          "of those".format(both_ok.mean() / graded.mean(),
                            (t_near[both_ok] != b_near[both_ok]).mean()))


if __name__ == "__main__":
    main()
