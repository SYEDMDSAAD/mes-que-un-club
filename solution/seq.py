"""H / A / D per frame from a wide window of time -- a second-stage model over the first-stage evidence.

The per-frame evidence (nearest player, the learned team and dead models, the ball-finder) looks at
one moment, or a short smoothed stretch. Possessions are decided by order: who touched the ball
last, who touches it next, whether the nearest team keeps alternating (a scramble), how the team
evidence trends over the next few seconds. This model sees all of that at once:

    signals    nearest-team distance gap, learned team and dead log-odds, ball-finder, ball seen,
               ball speed, player speed -- each at offsets from -2 s to +2 s and as window means
               over +-0.5 / 1 / 2 / 4 s
    touches    frames since the last and until the next contact by each team (a player within
               REACH m of the ball and nearer than anyone on the other side), and which team
               made the last and makes the next one
    scramble   how often the nearest team flips within +-1 s and +-2 s

A gradient-boosted classifier learns H / A / D from the DFL matches' official labels. Its output
enters the decoder like any other evidence: team log-odds and dead log-odds, minus the training
priors.

The first-stage inputs of the training matches must be out-of-sample, as they will be on BAR-ATH,
so they are the leave-one-match-out outputs of dfl_eval.prepare.
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

# Params
FPS = 25
REACH = 1.5               # m
OFFSETS = (-50, -25, -12, -6, 6, 12, 25, 50)
WINDOWS = (12, 25, 50, 100)
CAP = 250                 # frames: "no touch for 10 s" is as far as the touch clock counts
TRAIN_STEP = 5
SEED = 0
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "cache", "seq_model.pkl")


def shift(x, k):
    """x[t + k], NaN past the ends."""
    out = np.full(len(x), np.nan)
    if k > 0:
        out[:-k] = x[k:]
    elif k < 0:
        out[-k:] = x[:k]
    else:
        out[:] = x
    return out


def centred_mean(x, half):
    return pd.Series(x).rolling(2 * half + 1, center=True, min_periods=1).mean().to_numpy()


def clocks(event):
    """Frames since the last and until the next True, capped at CAP."""
    n = len(event)
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(event, idx, -10 ** 9))
    nxt = np.minimum.accumulate(np.where(event, idx, 10 ** 9)[::-1])[::-1]
    return np.minimum(idx - last, CAP), np.minimum(nxt - idx, CAP)


def features(feat):
    """(N, F) from a features.compute() dict that also holds team_ml, dead_ml and ballfinder."""
    d_home = np.asarray(feat["d_home"], float)
    d_away = np.asarray(feat["d_away"], float)
    with np.errstate(invalid="ignore"):
        gap = np.clip(d_away - d_home, -10, 10)                 # > 0: home nearer
        home_touch = (d_home < REACH) & (d_home < d_away)
        away_touch = (d_away < REACH) & (d_away < d_home)
        nearest = np.where(np.isfinite(gap), np.sign(gap), 0.0)
    signals = {
        "gap": gap,
        "team_ml": feat["team_ml"],
        "dead_ml": feat["dead_ml"],
        "ballfinder": feat["ballfinder"],
        "seen": feat["ball_seen"].astype(float),
        "ball_v": np.clip(np.nan_to_num(feat["ball_v_1s"], nan=-1.0), -1, 40),
        "speed": feat["mean_speed"],
    }
    cols = []
    for x in signals.values():
        x = np.asarray(x, float)
        cols.append(x)
        cols += [shift(x, k) for k in OFFSETS]
        cols += [centred_mean(x, w) for w in WINDOWS]

    h_since, h_until = clocks(home_touch)
    a_since, a_until = clocks(away_touch)
    cols += [h_since, h_until, a_since, a_until, h_since - a_since, h_until - a_until]
    flips = np.r_[0.0, (nearest[1:] * nearest[:-1] < 0).astype(float)]
    cols += [centred_mean(flips, 25) * 51, centred_mean(flips, 50) * 101]
    return np.column_stack(cols).astype(np.float32)


def fit(xs, labels):
    """Classifier on graded frames: classes D, H, A."""
    x = np.concatenate([a[::TRAIN_STEP] for a in xs])
    lab = np.concatenate([b[::TRAIN_STEP] for b in labels])
    keep = np.isin(lab, ["H", "A", "D"])
    y = np.searchsorted(np.array(["A", "D", "H"]), lab[keep])
    model = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                                           l2_regularization=1.0, random_state=SEED)
    model.fit(x[keep], y)
    counts = np.bincount(y, minlength=3) / len(y)
    return model, np.log(counts)


def log_odds(model_prior, x):
    """(team log-odds H vs A, dead log-odds D vs alive), both minus the training priors."""
    model, log_prior = model_prior
    p = np.clip(model.predict_proba(x), 1e-4, 1.0)
    lp = np.log(p) - log_prior                                    # classes A, D, H
    team = lp[:, 2] - lp[:, 0]
    alive = np.logaddexp(np.log(p[:, 0]), np.log(p[:, 2]))
    prior_alive = np.logaddexp(log_prior[0], log_prior[2])
    dead = (np.log(p[:, 1]) - log_prior[1]) - (alive - prior_alive)
    return team, dead


def trained(build_training_set):
    """The model trained on every DFL match, cached. build_training_set() -> (xs, labels)."""
    if os.path.isfile(MODEL_PATH):
        with open(MODEL_PATH, "rb") as handle:
            return pickle.load(handle)
    model_prior = fit(*build_training_set())
    with open(MODEL_PATH, "wb") as handle:
        pickle.dump(model_prior, handle)
    return model_prior
