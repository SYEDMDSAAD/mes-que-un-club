"""Ball touches from the ball's path: where it changes direction or speed next to a player.

The ground truth changes team at the new team's first touch (gt_rules.py), and real possessions
can be under a second long. A single switch penalty cannot serve both needs: low enough to let
short possessions through, it lets noise through everywhere else. Touches let the decoder make
the penalty depend on the frame -- cheap right where the other team plays the ball, dear where
the ball is visible and nobody from the other team touches it.

A touch candidate is a frame where the smoothed ball path bends by more than MIN_ANGLE or changes
speed by more than MIN_DSPEED, measured over LAG frames either side, with a player within REACH.
Only the strongest candidate within SUPPRESS frames is kept, and it is given to the nearest
player's team.
"""

import numpy as np

# Params
FPS = 25
SMOOTH = 2               # frames either side for the position average (tracker jitter)
LAG = 4                  # frames before / after over which the in / out velocity is measured
MIN_ANGLE = 35.0         # degrees
MIN_DSPEED = 4.0         # m/s
MIN_SPEED = 2.0          # m/s: a ball barely moving bends at random
REACH = 2.0              # m
SUPPRESS = 6             # frames


def smooth_positions(x):
    """Centred mean over +/- SMOOTH frames; NaN unless every frame in the window is seen."""
    n = len(x)
    out = np.full(n, np.nan)
    finite = np.isfinite(x)
    csum = np.concatenate(([0.0], np.cumsum(np.where(finite, x, 0.0))))
    ccount = np.concatenate(([0], np.cumsum(finite)))
    i = np.arange(SMOOTH, n - SMOOTH)
    count = ccount[i + SMOOTH + 1] - ccount[i - SMOOTH]
    full = count == 2 * SMOOTH + 1
    out[i[full]] = (csum[i + SMOOTH + 1] - csum[i - SMOOTH])[full] / count[full]
    return out


def detect(ball, players, team):
    """Per frame: team of the player who touched the ball (0 home, 1 away), or -1."""
    n = len(ball)
    x, y = smooth_positions(ball[:, 0]), smooth_positions(ball[:, 1])
    vin_x = np.full(n, np.nan)
    vin_y = np.full(n, np.nan)
    vout_x = np.full(n, np.nan)
    vout_y = np.full(n, np.nan)
    vin_x[LAG:] = (x[LAG:] - x[:-LAG]) / LAG * FPS
    vin_y[LAG:] = (y[LAG:] - y[:-LAG]) / LAG * FPS
    vout_x[:-LAG] = (x[LAG:] - x[:-LAG]) / LAG * FPS
    vout_y[:-LAG] = (y[LAG:] - y[:-LAG]) / LAG * FPS
    s_in, s_out = np.hypot(vin_x, vin_y), np.hypot(vout_x, vout_y)

    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (vin_x * vout_x + vin_y * vout_y) / (s_in * s_out)
        angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        moving = np.maximum(s_in, s_out) > MIN_SPEED
        bend = np.where(moving & (np.minimum(s_in, s_out) > 0.5), angle / MIN_ANGLE, 0.0)
        kick = np.abs(s_out - s_in) / MIN_DSPEED
        strength = np.nan_to_num(np.maximum(bend, kick), nan=0.0)

    # keep local maxima above threshold
    candidate = strength >= 1.0
    keep = np.zeros(n, bool)
    for t in np.flatnonzero(candidate):
        lo, hi = max(t - SUPPRESS, 0), min(t + SUPPRESS + 1, n)
        keep[t] = strength[t] >= strength[lo:hi].max()

    out = np.full(n, -1, dtype=np.int8)
    frames = np.flatnonzero(keep)
    if len(frames):
        d = np.hypot(players[frames, :, 0] - x[frames, None], players[frames, :, 1] - y[frames, None])
        d = np.where(np.isfinite(d), d, np.inf)
        nearest = np.argmin(d, axis=1)
        close = d[np.arange(len(frames)), nearest] < REACH
        out[frames[close]] = team[nearest[close]]
    return out


def near(touch, side, window):
    """True on frames within `window` frames of a touch by `side`."""
    hit = (touch == side).astype(np.int32)
    c = np.concatenate(([0], np.cumsum(hit)))
    i = np.arange(len(touch))
    lo, hi = np.maximum(0, i - window), np.minimum(len(touch), i + window + 1)
    return (c[hi] - c[lo]) > 0
