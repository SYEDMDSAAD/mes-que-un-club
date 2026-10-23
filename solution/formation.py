"""Who has the ball, from the shape of the two teams alone -- for the frames where the ball is lost.

The ball is missing on 41% of graded frames, mostly in stretches longer than ten seconds, so
"nearest player to the ball" has nothing to say for a third of the match. The players are still
there, and the team in possession looks different from the team without it: it spreads wide and
long, pushes up, and moves towards the opponent's goal, while the other side stays compact and
drops back.

A logistic regression on formation features learns that difference. It is trained without any
ground truth: the targets are the labels the ball-based model already gives on frames where the
ball *is* tracked and close to a player, which is where that model is reliable. It is then
applied everywhere, and used as evidence where the ball is missing.

Every feature is expressed in the home team's attacking direction, worked out per half from where
each team's players stand, so the second half needs no special handling.
"""

import numpy as np

from features import rolling_mean
from load import load

# Params
FPS = 25
VEL_LAG = 5             # frames over which a player's velocity is measured
MAX_STEP = 1.0          # m per frame; anything faster is a tracker identity jump, not a run
L2 = 1e-2
ITERS = 500
LR = 0.5


def attack_sign(players, team, first, last):
    """+1 if the home team attacks towards +x in this block of frames, else -1.

    A team defends the end its players stand closer to on average, goalkeeper included.
    """
    x = players[first:last + 1, :, 0]
    home_x = np.nanmean(x[:, team == 0])
    away_x = np.nanmean(x[:, team == 1])
    return 1.0 if home_x < away_x else -1.0


def team_stats(x, y, vx, members):
    """Per-frame shape of one team: centroid, depth, width, push, front and back lines."""
    xs, ys, vs = x[:, members], y[:, members], vx[:, members]
    with np.errstate(invalid="ignore"):
        return np.stack([
            np.nanmean(xs, axis=1),
            np.nanstd(xs, axis=1),
            np.nanstd(ys, axis=1),
            np.nanmedian(vs, axis=1),
            np.nanpercentile(xs, 90, axis=1),
            np.nanpercentile(xs, 10, axis=1),
        ], axis=1)


def features(blocks, pitch):
    """(N, F) formation features, NaN outside the given (first, last) blocks."""
    _, players, team = load()
    n = players.shape[0]
    out = np.full((n, 15), np.nan)
    for first, last in blocks:
        sign = attack_sign(players, team, first, last)
        x = players[first:last + 1, :, 0]
        y = players[first:last + 1, :, 1]
        # Home always attacks +x after this flip.
        x = x if sign > 0 else pitch[0] - x
        dx = np.full_like(x, np.nan)
        dx[VEL_LAG:] = (x[VEL_LAG:] - x[:-VEL_LAG]) / VEL_LAG
        dx[np.abs(dx) > MAX_STEP] = np.nan
        vx = dx * FPS

        home = team_stats(x, y, vx, team == 0)
        away = team_stats(x, y, vx, team == 1)
        # Away's push and lines measured in its own direction, so "+" means attacking for both.
        away_own = away.copy()
        away_own[:, 0] = pitch[0] - away[:, 0]
        away_own[:, 3] = -away[:, 3]
        away_own[:, 4] = pitch[0] - away[:, 5]
        away_own[:, 5] = pitch[0] - away[:, 4]
        diff = home - away_own
        gap = (home[:, 0] - away[:, 0])[:, None]       # home centroid ahead of away's
        block = np.hstack([home, away_own, gap, diff[:, [1, 2]]])
        out[first:last + 1] = block[:, :15]

    # One second of smoothing: a shape is a slow thing, and single-frame noise is not.
    for k in range(out.shape[1]):
        out[:, k] = rolling_mean(out[:, k], FPS // 2)
    return out


def fit_logistic(x, y, weights=None):
    """L2-regularised logistic regression by gradient descent. Returns (mean, std, w, b)."""
    mean, std = x.mean(axis=0), x.std(axis=0) + 1e-9
    z = (x - mean) / std
    w, b = np.zeros(z.shape[1]), 0.0
    sw = np.ones(len(y)) if weights is None else weights
    sw = sw / sw.sum()
    for _ in range(ITERS):
        p = 1.0 / (1.0 + np.exp(-(z @ w + b)))
        g = (p - y) * sw
        w -= LR * (z.T @ g + L2 * w)
        b -= LR * g.sum()
    return mean, std, w, b


def log_odds(model, x):
    """log P(home)/P(away) per row; 0 where any feature is missing."""
    mean, std, w, b = model
    ok = np.isfinite(x).all(axis=1)
    out = np.zeros(len(x))
    out[ok] = ((x[ok] - mean) / std) @ w + b
    return out
