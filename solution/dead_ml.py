"""Dead-ball evidence learned from the DFL matches' official BallStatus labels.

The hand-built dead evidence (player speed, ball out, ball moving) calls play dead far too often
on full matches: slow live phases -- a keeper holding the ball, a slow build-up -- look like
stoppages, and a 2 s smoothing window stretches every stoppage at both ends. A classifier trained
on ~1M labelled frames learns the timing and the context instead:

    speed     mean and top player speed in 1 s windows from 4 s before to 4 s after the frame,
              scaled by the match's own typical speed (TRACAB and Metrica speeds differ)
    ball      seen or not, moving or still, out of bounds, distance to the touchline, goal line
              and corner, time since last seen and until next seen
    crowd     nearest player to the ball, players within 10 m of it, players tracked

Nothing in it depends on which team has the ball, so the DFL and Metrica possession conventions
cannot leak in -- only the referee's dead/alive call, which both follow.

The classifier is trained on DFL tracking damaged to look like ours (see dfl_eval.degrade). Its
output is a log-likelihood ratio: the classifier's log-odds minus the training prior.
"""

import os
import pickle

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

# Params
PITCH = (105.0, 68.0)
FPS = 25
OFFSETS = (-100, -50, -25, 0, 25, 50, 100)
TRAIN_STEP = 5
SEED = 0
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "cache", "dead_ml_model.pkl")


def window_mean(x, half=FPS // 2):
    finite = np.isfinite(x)
    total = np.concatenate(([0.0], np.cumsum(np.where(finite, x, 0.0))))
    count = np.concatenate(([0], np.cumsum(finite)))
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - half), np.minimum(len(x), i + half + 1)
    n = count[hi] - count[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (total[hi] - total[lo]) / n, np.nan)


def shift(x, k):
    """x[t + k], NaN past the ends."""
    out = np.full_like(x, np.nan, dtype=float)
    if k >= 0:
        out[:len(x) - k] = x[k:]
    else:
        out[-k:] = x[:k]
    return out


def features(ball, players, blocks, pitch=PITCH):
    """(N, F) features; rows outside the blocks are NaN."""
    n = len(ball)
    graded = np.zeros(n, bool)
    for first, last in blocks:
        graded[first:last + 1] = True

    ps = players[..., 2]
    with np.errstate(invalid="ignore"):
        mean_s = np.nanmean(ps, axis=1)
        top_s = np.nanpercentile(ps, 90, axis=1)
    scale = np.nanmedian(mean_s[graded])
    mean_s, top_s = mean_s / scale, top_s / scale
    mean_1s, top_1s = window_mean(mean_s), window_mean(top_s)

    bx, by = ball[:, 0], ball[:, 1]
    seen = np.isfinite(bx).astype(float)
    v = np.clip(np.hypot(np.diff(bx, prepend=np.nan), np.diff(by, prepend=np.nan)) * FPS, 0, 40)
    v_1s = window_mean(v)

    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(seen > 0, idx, -1))
    nxt = np.minimum.accumulate(np.where(seen > 0, idx, n)[::-1])[::-1]
    lx = np.where(last >= 0, bx[np.clip(last, 0, n - 1)], np.nan)
    ly = np.where(last >= 0, by[np.clip(last, 0, n - 1)], np.nan)
    since = np.log1p(np.where(last >= 0, idx - last, 10 ** 5))
    until = np.log1p(np.where(nxt < n, nxt - idx, 10 ** 5))
    out_x = np.maximum(-lx, lx - pitch[0])
    out_y = np.maximum(-ly, ly - pitch[1])
    to_goal_line = np.minimum(lx, pitch[0] - lx)
    to_touch = np.minimum(ly, pitch[1] - ly)
    to_corner = np.hypot(to_goal_line, to_touch)
    # displacement of the ball over the last second, where it was seen both times
    still = np.hypot(bx - shift(bx, -FPS), by - shift(by, -FPS))

    d = np.hypot(players[..., 0] - lx[:, None], players[..., 1] - ly[:, None])
    with np.errstate(invalid="ignore"):
        d_min = np.nanmin(np.where(np.isfinite(d), d, np.nan), axis=1)
    crowd = (d < 10).sum(axis=1).astype(float)
    n_tracked = np.isfinite(players[..., 0]).sum(axis=1).astype(float)

    cols = [shift(mean_1s, k) for k in OFFSETS] + [shift(top_1s, k) for k in (-25, 0, 25)]
    cols += [shift(window_mean(seen), k) for k in (-25, 0, 25)]
    cols += [shift(v_1s, k) for k in (-25, 0, 25)]
    cols += [shift(mean_1s, 25) - shift(mean_1s, -25)]
    cols += [since, until, out_x, out_y, to_goal_line, to_touch, to_corner, still,
             window_mean(d_min), window_mean(crowd), n_tracked]
    x = np.stack(cols, axis=1).astype(np.float32)
    x[~graded] = np.nan
    return x


def fit(xs, ys):
    """Classifier on several matches' features and dead labels (every TRAIN_STEP-th frame)."""
    x = np.concatenate([a[::TRAIN_STEP] for a in xs])
    y = np.concatenate([b[::TRAIN_STEP] for b in ys])
    ok = np.isfinite(x).any(axis=1)
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                           l2_regularization=1.0, random_state=SEED)
    model.fit(x[ok], y[ok])
    prior = np.log(y[ok].mean() / (1 - y[ok].mean()))
    return model, prior


def log_lr(model_prior, x):
    """Dead log-likelihood ratio per frame (0 where there are no features)."""
    model, prior = model_prior
    out = np.zeros(len(x))
    ok = np.isfinite(x).any(axis=1)
    if ok.any():
        p = np.clip(model.predict_proba(x[ok])[:, 1], 1e-4, 1 - 1e-4)
        out[ok] = np.log(p / (1 - p)) - prior
    return out


def dfl_training_set():
    """Features and labels for every DFL match, damaged like ours. Downloads what is missing."""
    import dfl_eval
    from external_dfl import convert, load_match, match_ids
    ids, files = match_ids()
    for m in ids:
        convert(m, files)
    xs, ys = [], []
    for m in sorted(ids):
        ball, players, _, truth, blocks = load_match(m)
        for d_ball, d_players in dfl_eval.training_copies(ball, players, blocks):
            xs.append(features(d_ball, d_players, blocks))
            ys.append(truth == "D")
    return xs, ys


def trained():
    """The model trained on all DFL matches, cached on disk."""
    if os.path.isfile(MODEL_PATH):
        with open(MODEL_PATH, "rb") as handle:
            return pickle.load(handle)
    model_prior = fit(*dfl_training_set())
    with open(MODEL_PATH, "wb") as handle:
        pickle.dump(model_prior, handle)
    return model_prior
