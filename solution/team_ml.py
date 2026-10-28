"""Team-in-possession evidence learned from the DFL matches' official BallPossession labels.

The hand-built team evidence adds up three separately weighted guesses (nearest player within a
radius, formation, ball-finder). A classifier trained on ~1M labelled live frames can weigh the
same information jointly, and learn what we did not think to encode:

    ball      nearest and second-nearest player of each team, players of each team within 3 / 6 m,
              raw and smoothed over 0.5 s and 1 s
    memory    the same measured at where the ball was last seen and where it is next seen, and
              how long ago / ahead that is -- for the frames without a ball
    shape     the formation features of formation.py

Possession conventions differ between DFL and Metrica mostly at the *moment* of a change, much
less in the middle of a possession, which is what a per-frame classifier mostly learns. The
decoder (switch penalty) still decides when a change is believed.

Output: log P(home)/P(away) minus the training prior, so neither league's home share leaks in.
"""

import os
import pickle

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import formation
import variant
from dead_ml import window_mean

# Params
PITCH = (105.0, 68.0)
FPS = 25
TRAIN_STEP = 5
SEED = 0
MODEL_PATH = variant.cache_path(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "team_ml_model.pkl"))


def ball_side_stats(bx, by, players, team):
    """Per frame: nearest and 2nd nearest distance, count within 3 m and 6 m, for each team."""
    d = np.hypot(players[..., 0] - bx[:, None], players[..., 1] - by[:, None])
    d = np.where(np.isfinite(d), d, np.inf)
    cols = []
    for side in (0, 1):
        ds = np.sort(d[:, team == side], axis=1)
        first, second = ds[:, 0], ds[:, 1]
        cols += [first, second, (ds < 3).sum(axis=1), (ds < 6).sum(axis=1)]
    out = np.stack(cols, axis=1).astype(np.float32)
    out[~np.isfinite(bx)] = np.nan
    out[np.isinf(out)] = np.nan
    return out


def features(ball, players, team, blocks, pitch=PITCH):
    """(N, F) features; rows outside the blocks are NaN. Uses load.OVERRIDE for formation."""
    import load as loader
    n = len(ball)
    bx, by = ball[:, 0].astype(float), ball[:, 1].astype(float)
    seen = np.isfinite(bx)
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(seen, idx, -1))
    nxt = np.minimum.accumulate(np.where(seen, idx, n)[::-1])[::-1]

    now = ball_side_stats(bx, by, players, team)
    li = np.clip(last, 0, n - 1)
    ni = np.clip(nxt, 0, n - 1)
    at_last = ball_side_stats(np.where(last >= 0, bx[li], np.nan),
                              np.where(last >= 0, by[li], np.nan), players, team)
    at_next = ball_side_stats(np.where(nxt < n, bx[ni], np.nan),
                              np.where(nxt < n, by[ni], np.nan), players, team)
    gap = now[:, 0] - now[:, 4]                       # home nearest minus away nearest
    smooth = [window_mean(gap, 6), window_mean(gap, 12), window_mean(gap, 25)]
    since = np.log1p(np.where(last >= 0, idx - last, 10 ** 5))
    until = np.log1p(np.where(nxt < n, nxt - idx, 10 ** 5))

    saved = loader.OVERRIDE
    loader.OVERRIDE = (ball, players, team)
    try:
        shape = formation.features(blocks, pitch)
    finally:
        loader.OVERRIDE = saved

    x = np.column_stack([now, at_last, at_next] + smooth + [since, until, shape]).astype(np.float32)
    graded = np.zeros(n, bool)
    for first, last_ in blocks:
        graded[first:last_ + 1] = True
    x[~graded] = np.nan
    return x


def fit(xs, labels):
    """Classifier on live frames of several matches: target 1 = home in possession."""
    x = np.concatenate([a[::TRAIN_STEP] for a in xs])
    lab = np.concatenate([b[::TRAIN_STEP] for b in labels])
    live = np.isin(lab, ["H", "A"]) & np.isfinite(x).any(axis=1)
    y = lab[live] == "H"
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                           l2_regularization=1.0, random_state=SEED)
    model.fit(x[live], y)
    prior = np.log(y.mean() / (1 - y.mean()))
    return model, prior


def log_odds(model_prior, x):
    model, prior = model_prior
    out = np.zeros(len(x))
    ok = np.isfinite(x).any(axis=1)
    if ok.any():
        p = np.clip(model.predict_proba(x[ok])[:, 1], 1e-4, 1 - 1e-4)
        out[ok] = np.log(p / (1 - p)) - prior
    return out


def dfl_training_set():
    import dfl_eval
    from external_dfl import convert, load_match, match_ids
    ids, files = match_ids()
    for m in ids:
        convert(m, files)
    xs, labels = [], []
    for m in sorted(ids):
        ball, players, team, truth, blocks = load_match(m)
        for d_ball, d_players in dfl_eval.training_copies(ball, players, blocks, truth, team):
            xs.append(features(d_ball, d_players, team, blocks))
            labels.append(truth)
    return xs, labels


def trained():
    if os.path.isfile(MODEL_PATH):
        with open(MODEL_PATH, "rb") as handle:
            return pickle.load(handle)
    model_prior = fit(*dfl_training_set())
    with open(MODEL_PATH, "wb") as handle:
        pickle.dump(model_prior, handle)
    return model_prior
