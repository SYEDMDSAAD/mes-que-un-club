"""Who is closest to the ball, from the players alone -- for the 41% of frames without a ball.

Asking "where is the ball?" is harder than we need. Asking "which player is closest to it?" is a
ranking over ~22 players, and its answer turns straight into team evidence: P(home) is the
summed probability of the home players.

Each player on each frame is described by things that are visible without the ball:

    crowding     opponents and team-mates within 3 / 6 / 10 m, nearest opponents
    pressing     players within 20 m running straight at this one
    role         position along the player's own attacking direction, distance from the
                 touchline, goalkeeper or not, distance from both team centroids
    memory       distance to where the ball was last seen and to where it is seen next, and
                 how long ago / how far ahead that is -- the model runs offline, so the
                 future is available, which no live tracker has

A gradient-boosted classifier learns "is this the player nearest the ball" on frames where the
ball *is* tracked and within reach of someone. On those frames the ball memory is computed as if
the ball had just disappeared for a gap drawn from the real gap lengths of this match, so that
training looks like the frames it will be applied to.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from formation import attack_sign
from load import load

# Params
PITCH = (105.3, 68.0)
FPS = 25
VEL_LAG = 5
MAX_STEP = 1.0           # m per frame; faster is an identity jump
TOUCH_RADIUS = 2.5       # m: the nearest player counts as "at the ball" only this close
PRESS_RADIUS = 20.0
PRESS_COS = 0.8
PRESS_SPEED = 2.0        # m/s
CHUNK = 5000
SEED = 0
FEATURES = ["x_att", "touchline", "speed", "opp3", "opp6", "opp10", "mate6", "mate10",
            "d_opp1", "d_opp2", "d_mate1", "pressers", "d_centroid", "d_own_centroid",
            "is_gk", "d_last", "dt_last", "d_next", "dt_next", "d_interp"]


def seen_index(seen):
    """(last, next): index of the most recent / next frame with a ball, -1 / n if none."""
    n = len(seen)
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(seen, idx, -1))
    nxt = np.minimum.accumulate(np.where(seen, idx, n)[::-1])[::-1]
    return last, nxt


def gap_lengths(seen, blocks):
    """Lengths of every run of missing ball inside the graded blocks."""
    out = []
    for first, last in blocks:
        miss = ~seen[first:last + 1]
        edges = np.diff(np.concatenate(([0], miss.astype(np.int8), [0])))
        out.extend((np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)).tolist())
    return np.array(out)


class Context:
    """Arrays shared by every chunk: players, velocities, attack direction, ball memory."""

    def __init__(self, blocks):
        ball, players, team = load()
        self.team = team
        self.px, self.py, self.ps = players[..., 0], players[..., 1], players[..., 2]
        self.bx, self.by = ball[:, 0], ball[:, 1]
        self.seen = np.isfinite(self.bx)
        self.last, self.next = seen_index(self.seen)

        n = len(self.bx)
        self.home_sign = np.zeros(n)
        for first, last in blocks:
            self.home_sign[first:last + 1] = attack_sign(players, team, first, last)

        vx = np.full_like(self.px, np.nan)
        vy = np.full_like(self.py, np.nan)
        vx[VEL_LAG:] = (self.px[VEL_LAG:] - self.px[:-VEL_LAG]) / VEL_LAG
        vy[VEL_LAG:] = (self.py[VEL_LAG:] - self.py[:-VEL_LAG]) / VEL_LAG
        jump = np.hypot(vx, vy) > MAX_STEP
        vx[jump] = np.nan
        vy[jump] = np.nan
        self.vx, self.vy = vx * FPS, vy * FPS


def frame_features(ctx, frames, anchor_last, anchor_next):
    """(rows, features), (frame of each row), (slot of each row) for the given frames.

    anchor_last / anchor_next are the frames whose ball position counts as "last seen" and
    "next seen" for each requested frame (-1 / n when there is none).
    """
    n_total = len(ctx.bx)
    px, py = ctx.px[frames], ctx.py[frames]                  # (T, S)
    vx, vy, sp = ctx.vx[frames], ctx.vy[frames], ctx.ps[frames]
    present = np.isfinite(px) & np.isfinite(py)
    is_home = (ctx.team == 0)[None, :]
    same = (ctx.team[:, None] == ctx.team[None, :])[None]    # (1, S, S)

    dx = px[:, None, :] - px[:, :, None]                      # [t, i, j] = x_j - x_i
    dy = py[:, None, :] - py[:, :, None]
    dist = np.hypot(dx, dy)
    valid = present[:, None, :] & present[:, :, None]
    np.einsum("tii->ti", valid)[:] = False
    dist = np.where(valid, dist, np.inf)
    opp_d = np.where(~same, dist, np.inf)
    mate_d = np.where(same, dist, np.inf)

    def count(d, r):
        return (d < r).sum(axis=2).astype(np.float32)

    opp_sorted = np.sort(opp_d, axis=2)[:, :, :2]
    mate_min = mate_d.min(axis=2)

    # Pressers: j within PRESS_RADIUS of i, running faster than PRESS_SPEED, heading at i.
    speed_j = np.hypot(vx, vy)[:, None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = -(dx * vx[:, None, :] + dy * vy[:, None, :]) / (dist * speed_j)
        pressers = ((dist < PRESS_RADIUS) & (speed_j > PRESS_SPEED) & (cos > PRESS_COS)
                    ).sum(axis=2).astype(np.float32)

    sign = ctx.home_sign[frames][:, None] * np.where(is_home, 1.0, -1.0)   # player's direction
    x_att = np.where(sign > 0, px, PITCH[0] - px)
    touchline = np.minimum(py, PITCH[1] - py)

    with np.errstate(invalid="ignore"):
        cx, cy = np.nanmean(px, axis=1)[:, None], np.nanmean(py, axis=1)[:, None]
        own_cx = np.where(is_home, np.nanmean(np.where(is_home, px, np.nan), axis=1)[:, None],
                          np.nanmean(np.where(~is_home, px, np.nan), axis=1)[:, None])
        own_cy = np.where(is_home, np.nanmean(np.where(is_home, py, np.nan), axis=1)[:, None],
                          np.nanmean(np.where(~is_home, py, np.nan), axis=1)[:, None])
    is_gk = np.zeros_like(px)
    for side in (is_home[0], ~is_home[0]):
        xa = np.where(present[:, side], x_att[:, side], np.inf)
        gk = np.zeros_like(xa)
        rows = np.isfinite(xa).any(axis=1)
        gk[rows, np.argmin(xa[rows], axis=1)] = 1.0
        is_gk[:, side] = gk

    def ball_at(anchor):
        ok = (anchor >= 0) & (anchor < n_total)
        a = np.clip(anchor, 0, n_total - 1)
        return np.where(ok, ctx.bx[a], np.nan), np.where(ok, ctx.by[a], np.nan), ok

    lx, ly, lok = ball_at(anchor_last)
    nx, ny, nok = ball_at(anchor_next)
    dt_last = np.where(lok, frames - anchor_last, 10 ** 5).astype(np.float32)
    dt_next = np.where(nok, anchor_next - frames, 10 ** 5).astype(np.float32)
    frac = np.clip(dt_last / np.maximum(dt_last + dt_next, 1), 0, 1)
    ix = np.where(lok & nok, lx + frac * (nx - lx), np.where(lok, lx, nx))
    iy = np.where(lok & nok, ly + frac * (ny - ly), np.where(lok, ly, ny))

    cols = [
        x_att, touchline, sp,
        count(opp_d, 3), count(opp_d, 6), count(opp_d, 10), count(mate_d, 6), count(mate_d, 10),
        opp_sorted[:, :, 0], opp_sorted[:, :, 1], mate_min, pressers,
        np.hypot(px - cx, py - cy), np.hypot(px - own_cx, py - own_cy), is_gk,
        np.hypot(px - lx[:, None], py - ly[:, None]), np.log1p(dt_last)[:, None] + 0 * px,
        np.hypot(px - nx[:, None], py - ny[:, None]), np.log1p(dt_next)[:, None] + 0 * px,
        np.hypot(px - ix[:, None], py - iy[:, None]),
    ]
    feats = np.stack(cols, axis=2).astype(np.float32)        # (T, S, F)
    feats[~np.isfinite(feats)] = np.nan                      # inf distances -> missing

    t_idx, s_idx = np.nonzero(present)
    return feats[t_idx, s_idx], frames[t_idx], s_idx


def build(ctx, frames, anchor_last, anchor_next):
    """frame_features over many frames, in chunks to bound memory."""
    parts = [frame_features(ctx, frames[i:i + CHUNK], anchor_last[i:i + CHUNK],
                            anchor_next[i:i + CHUNK]) for i in range(0, len(frames), CHUNK)]
    if not parts:
        return np.zeros((0, len(FEATURES)), np.float32), np.zeros(0, int), np.zeros(0, int)
    return tuple(np.concatenate([p[k] for p in parts]) for k in range(3))


def training_frames(ctx, frames, gaps, rng):
    """Frames with a ball at someone's feet, and simulated ball-memory anchors for each."""
    px, py = ctx.px[frames], ctx.py[frames]
    d = np.hypot(px - ctx.bx[frames][:, None], py - ctx.by[frames][:, None])
    d = np.where(np.isfinite(d), d, np.inf)
    nearest = np.argmin(d, axis=1)
    keep = ctx.seen[frames] & (d.min(axis=1) < TOUCH_RADIUS)
    frames, nearest = frames[keep], nearest[keep]

    # Pretend the ball vanished for a real-length gap that contains this frame.
    length = rng.choice(gaps, size=len(frames))
    before = (rng.random(len(frames)) * length).astype(int) + 1
    after = np.maximum(length - before, 0) + 1
    n = len(ctx.bx)
    anchor_last = ctx.last[np.clip(frames - before, 0, n - 1)]
    anchor_next = ctx.next[np.clip(frames + after, 0, n - 1)]
    return frames, nearest, anchor_last, anchor_next


def fit(ctx, frames, gaps, seed=SEED):
    rng = np.random.default_rng(seed)
    if not len(gaps):
        gaps = np.array([10 * FPS])           # a match with no gaps at all: assume 10 s ones
    frames, nearest, a_last, a_next = training_frames(ctx, frames, gaps, rng)
    x, f_of_row, s_of_row = build(ctx, frames, a_last, a_next)
    target = dict(zip(frames.tolist(), nearest.tolist()))
    y = np.array([target[f] == s for f, s in zip(f_of_row.tolist(), s_of_row.tolist())])
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
                                           l2_regularization=1.0, random_state=seed)
    model.fit(x, y)
    return model


def team_log_odds(ctx, model, frames, anchor_last, anchor_next):
    """log P(home nearest)/P(away nearest) for each frame, and the most likely slot."""
    x, f_of_row, s_of_row = build(ctx, frames, anchor_last, anchor_next)
    p = model.predict_proba(x)[:, 1] if len(x) else np.zeros(0)
    pos = np.searchsorted(frames, f_of_row)
    home = ctx.team[s_of_row] == 0
    p_home = np.bincount(pos, weights=p * home, minlength=len(frames))
    p_all = np.bincount(pos, weights=p, minlength=len(frames))
    best = np.full(len(frames), -1)
    order = np.lexsort((p, pos))                             # last per frame = highest p
    best[pos[order]] = s_of_row[order]
    with np.errstate(divide="ignore", invalid="ignore"):
        odds = np.log(np.clip(p_home, 1e-6, None)) - np.log(np.clip(p_all - p_home, 1e-6, None))
    odds[p_all == 0] = 0.0
    return np.clip(odds, -6, 6), best
