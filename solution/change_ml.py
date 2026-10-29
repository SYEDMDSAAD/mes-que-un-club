"""When does possession change hands? A detector for "an H <-> A change within +-TOL frames".

The second stage (seq.py) says which team has the ball; this says when it changes. Its log-odds
(minus the training prior) make the decoder's H <-> A cost cheaper where a change is likely
(model.PARAMS k_change > 0, never below change_min): timing evidence that per-frame team log-odds
blur over the second either side of a change.

On a Metrica replica it did not help: -0.005 spellF1 on the v13 second stage, +0.003 / 0.000 on
top of touch features (k_change 1 / 2), and even perfect change times are worth only ~+0.04
there (ceiling_bench.py). Kept so the DFL bench can confirm it (touch_bench.py).

Trained on the same rows as the second stage -- model.dfl_eval_training_set(), the DFL matches and
Metrica's (touch features blanked there, see seq.py) -- on live frames only, every TRAIN_STEP-th.
The DFL labels change team on any touch (30% of possessions under 2 s); MIN_KEEP > 0 keeps as
targets only the changes into a possession that lasts at least MIN_KEEP frames, closer to the
challenge's "controlled possession".
"""

import os
import pickle

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import variant

# Params
TOL = 3                   # frames either side of a true change that count as "change here"
MIN_KEEP = 50             # frames the new possession must last to be a target (0 = every change; v15: 0)
TRAIN_STEP = 5
SEED = 0
MODEL_PATH = variant.cache_path(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "change_model.pkl"))


def targets(labels, tol=TOL, min_keep=MIN_KEEP):
    """Boolean per frame: within +-tol of a direct H <-> A change (into a run >= min_keep)."""
    labels = np.asarray(labels)
    n = len(labels)
    y = np.zeros(n, dtype=bool)
    b = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    stops = np.r_[b[1:], n]
    for i, stop in zip(b, stops):
        if labels[i - 1] in ("H", "A") and labels[i] in ("H", "A") and stop - i >= min_keep:
            y[max(0, i - tol):i + tol + 1] = True
    return y


def fit(xs, labels):
    """Binary classifier on live frames of several matches; returns (model, log prior odds)."""
    keep = [np.isin(np.asarray(lab), ["H", "A"]) for lab in labels]
    x = np.concatenate([a[::TRAIN_STEP][k[::TRAIN_STEP]] for a, k in zip(xs, keep)])
    y = np.concatenate([targets(lab)[::TRAIN_STEP][k[::TRAIN_STEP]]
                        for lab, k in zip(labels, keep)])
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                           l2_regularization=1.0, random_state=SEED)
    model.fit(x, y)
    return model, float(np.log(y.mean() / (1 - y.mean())))


def log_odds(model_prior, x):
    """Change log-odds per frame minus the training prior: > 0 where a change is likelier than
    on an average live frame."""
    model, prior = model_prior
    p = np.clip(model.predict_proba(x)[:, 1], 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p)) - prior


def trained(build_training_set):
    """The detector trained on every training match, cached. build_training_set() -> (xs, labels)."""
    if os.path.isfile(MODEL_PATH):
        with open(MODEL_PATH, "rb") as handle:
            return pickle.load(handle)
    model_prior = fit(*build_training_set())
    with open(MODEL_PATH, "wb") as handle:
        pickle.dump(model_prior, handle)
    return model_prior
