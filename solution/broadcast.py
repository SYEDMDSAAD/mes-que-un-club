"""Line the broadcast tracking up with the tactical timeline, then borrow its ball.

The broadcast camera follows the ball, so its tracking sees the ball on frames where the wide
tactical camera has lost it. The two recordings are not synchronised and no mapping is provided:
the broadcast has its own frame count, cuts and replays. But both trackings are in the same pitch
coordinates, so a moment of the match has the same player positions in both. That is the
fingerprint used here:

    match cost(b, t) = median distance from each broadcast player to the nearest tactical player
                       of the same team, broadcast frame b against tactical frame t

1. coarse   every SAMPLE_STEP broadcast frames, try every offset t - b on a coarse grid
2. fine     around the coarse winner, every offset to the frame
3. segments consecutive samples that agree on an offset form a live segment; replays and
            close-ups match nothing and are dropped

Only aggregates are printed.
"""

import numpy as np

from load import load_broadcast, load_tactical as load

# Params
SAMPLE_STEP = 100        # broadcast frames between alignment samples (4 s)
COARSE_STEP = 25         # offset grid, frames
MIN_PLAYERS = 6          # a broadcast frame needs this many players to be a fingerprint
GOOD_COST = 1.5          # m: median player distance under which a match is believed
CHUNK = 256


def frame_cost(b_xy, b_team, t_xy, t_team):
    """Median over broadcast players of distance to nearest same-team tactical player.

    b_xy (P, 2) one broadcast frame; t_xy (K, S, 2) K candidate tactical frames. Returns (K,).
    """
    d = np.hypot(b_xy[None, :, None, 0] - t_xy[:, None, :, 0],
                 b_xy[None, :, None, 1] - t_xy[:, None, :, 1])            # (K, P, S)
    same = b_team[:, None] == t_team[None, :]
    d = np.where(same[None] & np.isfinite(d), d, np.inf)
    return np.median(d.min(axis=2), axis=1)


def best_offsets(b_frames, offsets, bp, bt, tp, tt):
    """For each broadcast frame, the offset with the lowest cost and that cost."""
    n_t = tp.shape[0]
    best, cost = np.zeros(len(b_frames), int), np.full(len(b_frames), np.inf)
    for i, b in enumerate(b_frames):
        present = np.isfinite(bp[b, :, 0])
        if present.sum() < MIN_PLAYERS:
            continue
        b_xy, b_team = bp[b, present, :2], bt[present]
        cand = offsets if offsets.ndim == 1 else offsets[i]
        t = b + cand
        ok = (t >= 0) & (t < n_t)
        t, cand = t[ok], cand[ok]
        c = np.concatenate([frame_cost(b_xy, b_team, tp[t[k:k + CHUNK], :, :2], tt)
                            for k in range(0, len(t), CHUNK)]) if len(t) else np.zeros(0)
        if len(c):
            j = int(np.argmin(c))
            best[i], cost[i] = cand[j], c[j]
    return best, cost


def coarse(max_offset=None):
    _, tp, tt = load()
    _, bp, bt = load_broadcast()
    max_offset = max_offset or tp.shape[0] - bp.shape[0] + 10000
    b_frames = np.arange(0, bp.shape[0], SAMPLE_STEP)
    offsets = np.arange(-10000, max_offset, COARSE_STEP)
    return b_frames, *best_offsets(b_frames, offsets, bp, bt, tp, tt)


# Fixed offsets, tactical = broadcast + OFFSET, one per half. Found by coarse() and refined to the
# frame with best_offsets(): stable to +/-1 frame across each half, median fit 0.26 m.
OFFSETS = (4014, 26920)
LIVE_COST = 1.0          # m: per-frame fit below which a broadcast frame counts as live footage


def aligned():
    """Per tactical frame: broadcast frame index (-1 if none), fit cost, broadcast ball (x, y).

    A broadcast frame is used for the half whose offset fits it better, and only where that
    fit is good -- which drops replays, close-ups and crowd shots automatically.
    """
    tb, tp, tt = load()
    bb, bp, bt = load_broadcast()
    n_t, n_b = tp.shape[0], bp.shape[0]
    b_index = np.full(n_t, -1)
    cost = np.full(n_t, np.inf)
    for offset in OFFSETS:
        b = np.arange(n_b)
        t = b + offset
        ok = (t >= 0) & (t < n_t)
        b, t = b[ok], t[ok]
        c = np.full(len(b), np.inf)
        for k in range(0, len(b), 2048):
            bs, ts = b[k:k + 2048], t[k:k + 2048]
            bxy, txy = bp[bs, :, :2], tp[ts, :, :2]
            d = np.hypot(bxy[:, :, None, 0] - txy[:, None, :, 0],
                         bxy[:, :, None, 1] - txy[:, None, :, 1])
            same = bt[:, None] == tt[None, :]
            d = np.where(same[None] & np.isfinite(d), d, np.inf).min(axis=2)  # (F, Pb)
            n_seen = np.isfinite(d).sum(axis=1)
            d = np.where(np.isfinite(d), d, np.nan)
            with np.errstate(all="ignore"):
                med = np.nanmedian(d, axis=1)
            c[k:k + 2048] = np.where(n_seen >= MIN_PLAYERS, med, np.inf)
        better = c < cost[t]
        cost[t[better]] = c[better]
        b_index[t[better]] = b[better]
    live = cost < LIVE_COST
    ball = np.full((n_t, 2), np.nan, dtype=np.float32)
    ball[live] = bb[b_index[live], :2]
    return b_index, cost, ball


RELAX_WINDOW = 50        # frames: footage counts as live if >80% of tested frames within it are
FUSED_CACHE = "atd_fused_ball.npz"


def fused_ball(tactical_ball, blocks, rolling_mean):
    """Tactical ball, with gaps filled from the aligned broadcast ball on live footage.

    Speed (third column) is left NaN for filled frames, as it is everywhere in the source.
    """
    bb, _, _ = load_broadcast()
    _, cost, _ = aligned()
    n = len(tactical_ball)
    b_index = np.full(n, -1)
    for (first, last), offset in zip(blocks, OFFSETS):
        t = np.arange(first, last + 1)
        b = t - offset
        ok = (b >= 0) & (b < len(bb))
        b_index[t[ok]] = b[ok]
    tested = np.isfinite(cost)
    live_share = rolling_mean(np.where(tested, (cost < LIVE_COST).astype(float), np.nan),
                              RELAX_WINDOW)
    live = (b_index >= 0) & ((cost < LIVE_COST) | (live_share > 0.8))
    fill = live & ~np.isfinite(tactical_ball[:, 0]) & np.isfinite(bb[np.clip(b_index, 0, None), 0])
    out = tactical_ball.copy()
    out[fill, :2] = bb[b_index[fill], :2]
    return out, fill
