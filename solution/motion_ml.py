"""Who has the ball, from the players alone -- trained on the true labels, blind to the ball.

Every other team model reads the ball: nearest player, team_ml, the second stage, and even the
ball-finder, which learns "the player nearest the *tracked* ball" and so learns the tracker's
mistakes too (on a Metrica replica with a ball like ours it was right on only 33-36% of the frames
where the nearest player was on the wrong team). Where our tracking puts the ball on the wrong
team, none of them can see through it. The players can: they react to the real ball. The team on
the ball spreads and pushes, the other one closes in on the carrier.

This model sees only the players: the formation features of formation.py (centroids, depth, width,
push and lines of both teams, in the home team's attacking direction), how they change over the
second before and after, and pressing -- for each team, the most-pressed of its players (opponents
within PRESS_RADIUS running straight at him), which marks the likely carrier. A gradient-boosted
classifier learns home-in-possession from the DFL matches' BallPossession labels, damaged like
ours (dfl_eval.degrade: players only matter here). Its log-odds, minus the training prior, enter
the second stage as one more signal (seq.py, variant "motion"), so the second stage can learn when
to believe the players over the ball -- provided the training ball is sometimes wrong as well
(variant "wrongball").

On a Metrica replica with a ball made to look like ours (3 matches, leave-one-match-out; formation
features and their changes only, no pressing) it was right on 83-88% of in-play frames, and on
78-85% of the frames where the nearest player to the ball was on the wrong team. As a second-stage
signal it raised spellF1 0.594 -> 0.634 (all 3 up), and with touch features 0.656. There, perfect
team evidence on the frames whose nearest player is on the wrong team was worth +0.19
(ceiling_bench.py), on frames without a ball +0.03.
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
LAGS = (-50, -25, 25, 50)  # frames: the features again this far before / after
PRESS_RADIUS = 10.0       # m
PRESS_SPEED = 2.0         # m/s
PRESS_COS = 0.8
VEL_LAG = 5               # frames over which a player's velocity is measured
MAX_STEP = 1.0            # m per frame; faster is an identity jump
CHUNK = 4000
TRAIN_STEP = 5
SEED = 0
MODEL_PATH = variant.cache_path(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "motion_ml_model.pkl"))


def shift(x, k):
    """x[t + k] along axis 0, NaN past the ends."""
    out = np.full(x.shape, np.nan, dtype=float)
    if k > 0:
        out[:-k] = x[k:]
    elif k < 0:
        out[-k:] = x[:k]
    else:
        out[:] = x
    return out


def pressing(players, team):
    """(N, 3): per frame, the most-pressed home player's pressers, the most-pressed away
    player's, and their difference -- smoothed over half a second."""
    px, py = players[..., 0].astype(float), players[..., 1].astype(float)
    n = len(px)
    vx = np.full_like(px, np.nan)
    vy = np.full_like(py, np.nan)
    vx[VEL_LAG:] = (px[VEL_LAG:] - px[:-VEL_LAG]) / VEL_LAG
    vy[VEL_LAG:] = (py[VEL_LAG:] - py[:-VEL_LAG]) / VEL_LAG
    jump = np.hypot(vx, vy) > MAX_STEP
    vx[jump], vy[jump] = np.nan, np.nan
    vx, vy = vx * FPS, vy * FPS
    home = (np.asarray(team) == 0)
    opp = home[:, None] != home[None, :]                      # (S, S): i and j on other teams
    out = np.full((n, 3), np.nan)
    for a in range(0, n, CHUNK):
        b = min(n, a + CHUNK)
        dx = px[a:b, None, :] - px[a:b, :, None]              # [t, i, j] = x_j - x_i
        dy = py[a:b, None, :] - py[a:b, :, None]
        dist = np.hypot(dx, dy)
        sj = np.hypot(vx[a:b], vy[a:b])[:, None, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            cos = -(dx * vx[a:b, None, :] + dy * vy[a:b, None, :]) / (dist * sj)
            press = ((dist < PRESS_RADIUS) & (sj > PRESS_SPEED) & (cos > PRESS_COS)
                     & opp[None]).sum(axis=2).astype(float)    # pressers on player i
        present = np.isfinite(px[a:b])
        press[~present] = np.nan
        with np.errstate(invalid="ignore"):
            h = np.where(np.isfinite(press[:, home]).any(axis=1),
                         np.nanmax(np.where(np.isfinite(press[:, home]), press[:, home], -1), axis=1),
                         np.nan)
            w = np.where(np.isfinite(press[:, ~home]).any(axis=1),
                         np.nanmax(np.where(np.isfinite(press[:, ~home]), press[:, ~home], -1),
                                   axis=1), np.nan)
        out[a:b, 0], out[a:b, 1], out[a:b, 2] = h, w, h - w
    for k in range(3):
        out[:, k] = window_mean(out[:, k], FPS // 4)
    return out


def features(ball, players, team, blocks, pitch=PITCH):
    """(N, F) player-only features; rows outside the blocks are NaN. `ball` is not read: it is
    there so the pipeline can call every first-stage model the same way."""
    import load as loader
    saved = loader.OVERRIDE
    loader.OVERRIDE = (ball, players, team)
    try:
        shape = formation.features(blocks, pitch)
    finally:
        loader.OVERRIDE = saved
    base = np.hstack([shape, pressing(players, team)])
    cols = [base] + [shift(base, k) for k in LAGS] + [base - shift(base, -25), shift(base, 25) - base]
    x = np.hstack(cols).astype(np.float32)
    graded = np.zeros(len(x), bool)
    for first, last in blocks:
        graded[first:last + 1] = True
    x[~graded] = np.nan
    return x


def fit(xs, labels):
    """Classifier on live frames of several matches: target 1 = home in possession."""
    x = np.concatenate([a[::TRAIN_STEP] for a in xs])
    lab = np.concatenate([b[::TRAIN_STEP] for b in labels])
    live = np.isin(lab, ["H", "A"]) & np.isfinite(x).any(axis=1)
    y = lab[live] == "H"
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                           l2_regularization=1.0, random_state=SEED)
    model.fit(x[live], y)
    return model, float(np.log(y.mean() / (1 - y.mean())))


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
        for d_ball, d_players in dfl_eval.training_copies(ball, players, blocks, truth, team,
                                                          drop_len=dfl_eval.MOTION_DROP or None):
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
