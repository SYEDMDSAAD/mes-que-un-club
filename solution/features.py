"""Per-frame features from the cached tracking. Every array is length N, indexed by video frame.

Distances are in metres, speeds in m/s. NaN means "not observable on this frame" and is left for
the model to handle -- nothing here invents a ball position.
"""

import numpy as np

from load import load

# Params
FPS = 25


def nanmin_or_nan(values, axis):
    """np.nanmin without the all-NaN warning; NaN where nothing is observed."""
    out = np.full(values.shape[:axis] + values.shape[axis + 1:], np.nan, dtype=values.dtype)
    seen = np.isfinite(values).any(axis=axis)
    out[seen] = np.nanmin(values[seen], axis=axis)
    return out


def rolling_mean(values, half_window):
    """Centred mean over +/- half_window frames, ignoring NaN. NaN where the window is empty."""
    finite = np.isfinite(values)
    total = np.concatenate(([0.0], np.cumsum(np.where(finite, values, 0.0))))
    count = np.concatenate(([0], np.cumsum(finite)))
    index = np.arange(len(values))
    low = np.maximum(0, index - half_window)
    high = np.minimum(len(values), index + half_window + 1)
    n = count[high] - count[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (total[high] - total[low]) / n, np.nan)


def compute():
    ball, players, team = load()
    bx, by, bs = ball[:, 0], ball[:, 1], ball[:, 2]
    px, py, ps = players[..., 0], players[..., 1], players[..., 2]

    dist = np.hypot(px - bx[:, None], py - by[:, None])
    home, away = team == 0, team == 1

    n_players = np.isfinite(px).sum(axis=1)
    with np.errstate(invalid="ignore"):
        mean_speed = np.where(n_players > 0, np.nanmean(np.where(np.isfinite(ps), ps, np.nan),
                                                         axis=1), np.nan)

    # Ball speed from consecutive positions: the speed channel in the file is empty. Clipped at
    # 40 m/s, because a tracker jump between two objects is not a shot.
    ball_v = np.clip(np.hypot(np.diff(bx, prepend=np.nan), np.diff(by, prepend=np.nan)) * FPS,
                     0.0, 40.0)

    return {
        "ball_seen": np.isfinite(bx),
        "ball_v_1s": rolling_mean(ball_v, FPS // 2),
        "ball_x": bx,
        "ball_y": by,
        "ball_speed": bs,
        "d_home": nanmin_or_nan(dist[:, home], axis=1),
        "d_away": nanmin_or_nan(dist[:, away], axis=1),
        "n_players": n_players,
        "mean_speed": mean_speed,
        "mean_speed_2s": rolling_mean(mean_speed, 2 * FPS),
    }
