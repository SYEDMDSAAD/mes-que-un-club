"""Score the pipeline on a whole DFL match, against its official labels, with the official metric.

The DFL tracking is clean, so it is damaged to look like ours first:

    ball      deleted wherever OUR tactical ball is missing at the same point of the same half --
              the real gap pattern of BAR-ATH, not a synthetic one -- plus position noise.
              The noise is NOT realism (our ATD ball is as smooth as TRACAB's), it is protection:
              TRACAB freezes the ball during stoppages, and without noise the learned models take
              "ball exactly still" as the sign of a dead ball -- a shortcut our ATD never offers.
              Measured: without it, BAR-ATH dead share fell from 0.48 to 0.34 and public spellF1
              from 0.51 to 0.31, while the DFL bench itself rose -- a pure overfit to TRACAB.
    players   each one drops out for ~2 s stretches about 7% of the time (ours averages 20.5 of
              22 tracked), plus position noise

Smart Tagging does not exist for these matches, so the tag evidence is simply absent: the scores
here are, if anything, a little pessimistic for BAR-ATH.

    python solution/dfl_eval.py [match_id ...]        # one match in detail
    python solution/dfl_eval.py --bench                # every cached match, averaged

The bench is how variants are compared from here on: seven full matches and thousands of
possessions, instead of seventeen.
"""

import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
import load as loader  # noqa: E402
import model  # noqa: E402
from external_dfl import load_match  # noqa: E402

# Params
BALL_NOISE = 0.3          # m, per axis -- see the docstring: it hides a TRACAB artefact
PLAYER_NOISE = 0.1
PLAYER_DROP = 0.07
DROP_LEN = 50             # frames
SEED = 0
# Training copies (see degrade(randomize=True)): how many per match, and the ranges drawn from.
TRAIN_COPIES = 1
RANDOM_BALL_NOISE = (0.15, 0.5)
RANDOM_PLAYER_NOISE = (0.05, 0.3)
RANDOM_PLAYER_DROP = (0.03, 0.12)


def training_copies(ball, players, blocks):
    """The damaged versions of one match a learned model is trained on."""
    if TRAIN_COPIES <= 1:
        return [degrade(ball, players, blocks)]
    return [degrade(ball, players, blocks, seed=100 + k, randomize=True)
            for k in range(TRAIN_COPIES)]


def degrade(ball, players, blocks, seed=SEED, randomize=False):
    """Damage clean DFL tracking to look like ours.

    randomize=True draws the noise levels, the dropout rate and where in our match the ball-gap
    pattern is taken from, per call -- used for training copies, so a learned model sees many
    versions of the damage and cannot lean on any single one of them.
    """
    rng = np.random.default_rng(seed)
    ball_noise, player_noise, player_drop = BALL_NOISE, PLAYER_NOISE, PLAYER_DROP
    shift = 0
    if randomize:
        ball_noise = rng.uniform(*RANDOM_BALL_NOISE)
        player_noise = rng.uniform(*RANDOM_PLAYER_NOISE)
        player_drop = rng.uniform(*RANDOM_PLAYER_DROP)
        shift = int(rng.integers(0, 60000))
    ours, _, _ = loader.load()
    our_blocks = [(first, last) for _, first, last in model.half_video_blocks(model.GAME)]
    ball = ball.copy()
    for (first, last), (o_first, o_last) in zip(blocks, our_blocks):
        n = last - first + 1
        seen = np.isfinite(ours[o_first:o_last + 1, 0])
        seen = np.roll(seen, shift)
        seen = np.resize(seen, n)                         # tile if the DFL half is longer
        ball[first:last + 1][~seen] = np.nan
    if ball_noise:
        ball[:, :2] += rng.normal(0, ball_noise, ball[:, :2].shape)

    players = players.copy()
    players[..., :2] += rng.normal(0, player_noise, players[..., :2].shape)
    n, s = players.shape[:2]
    starts = rng.random((n, s)) < player_drop / DROP_LEN
    for t, k in zip(*np.nonzero(starts)):
        players[t:t + DROP_LEN, k] = np.nan
    return ball, players


def baseline(ball, players, team, blocks, window=50):
    """The organisers' reference: nearest player's team, majority vote over +/- window, D if no ball."""
    d = np.hypot(players[..., 0] - ball[:, None, 0], players[..., 1] - ball[:, None, 1])
    d = np.where(np.isfinite(d), d, np.inf)
    nearest = team[np.argmin(d, axis=1)]
    code = np.where(np.isfinite(ball[:, 0]) & np.isfinite(d.min(axis=1)), nearest, -1)
    home = np.concatenate(([0], np.cumsum(code == 0)))
    away = np.concatenate(([0], np.cumsum(code == 1)))
    nobody = np.concatenate(([0], np.cumsum(code == -1)))
    i = np.arange(len(code))
    lo, hi = np.maximum(0, i - window), np.minimum(len(code), i + window + 1)
    h, a, x = home[hi] - home[lo], away[hi] - away[lo], nobody[hi] - nobody[lo]
    out = np.where((x > h) & (x > a), "D", np.where(h >= a, "H", "A"))
    labels = np.full(len(code), "X", dtype="<U1")
    for first, last in blocks:
        labels[first:last + 1] = out[first:last + 1]
    return labels


def run_pipeline(ball, players, team, blocks, params=None, feat=None):
    """The full pipeline on another match. Pass the same `feat` dict to reuse features across
    variants: everything cached in it depends on the tracking only, never on the parameters."""
    loader.OVERRIDE = (ball, players, team)
    model.BLOCKS = blocks
    try:
        if feat is not None and "ball_x" not in feat:
            feat.update(model.compute())
        return model.predict(params, feat=feat, events=[])
    finally:
        loader.OVERRIDE = None
        model.BLOCKS = None


def score(truth, labels):
    r = pm.evaluate(pm.label_codes(truth), pm.label_codes(labels), bootstrap_n=0)
    return "spellF1 {:.4f}  macroF1 {:.4f}  poss {:4d} (truth {:4d})  aliveDeadF1 {:.3f}  " \
           "teamAcc {:.3f}".format(r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"],
                                   r["n_gt_spells"], r["alive_dead_f1"], r["team_acc_both_alive"])


def metrics(truth, labels):
    r = pm.evaluate(pm.label_codes(truth), pm.label_codes(labels), bootstrap_n=0)
    return np.array([r["spell_f1_mean"], r["macro_f1"], r["alive_dead_f1"],
                     r["team_acc_both_alive"], r["n_pred_spells"] / max(r["n_gt_spells"], 1)])


def cached_matches():
    from external_dfl import CACHE_DIR
    return sorted(f[:-4] for f in os.listdir(CACHE_DIR) if f.endswith(".npz"))


def prepare(matches=None, dead_ml=False, team_ml=False):
    """{match: (ball, players, team, truth, blocks, feat)} on damaged data, learned evidence
    filled leave-one-match-out: the classifiers scoring a match never saw it."""
    matches = matches or cached_matches()
    data, copies = {}, {}
    for m in matches:
        ball, players, team, truth, blocks = load_match(m)
        d_ball, d_players = degrade(ball, players, blocks)
        data[m] = (d_ball, d_players, team, truth, blocks, {})
        if dead_ml or team_ml:
            copies[m] = (training_copies(ball, players, blocks)
                         if TRAIN_COPIES > 1 else [(d_ball, d_players)])
    if dead_ml:
        import dead_ml as dm
        x = {m: [dm.features(b, p, data[m][4]) for b, p in copies[m]] for m in matches}
        test = {m: dm.features(data[m][0], data[m][1], data[m][4]) for m in matches}
        for m in matches:
            others = [k for k in matches if k != m]
            fitted = dm.fit([a for k in others for a in x[k]],
                            [data[k][3] == "D" for k in others for _ in x[k]])
            data[m][5]["dead_ml"] = dm.log_lr(fitted, test[m])
    if team_ml:
        import team_ml as tm_
        x = {m: [tm_.features(b, p, data[m][2], data[m][4]) for b, p in copies[m]]
             for m in matches}
        test = {m: tm_.features(data[m][0], data[m][1], data[m][2], data[m][4]) for m in matches}
        for m in matches:
            others = [k for k in matches if k != m]
            fitted = tm_.fit([a for k in others for a in x[k]],
                             [data[k][3] for k in others for _ in x[k]])
            data[m][5]["team_ml"] = tm_.log_odds(fitted, test[m])
    return data


def seq_training_set():
    """(features, labels) per DFL match for the second-stage model (seq.py).

    Its inputs on each match are the first-stage outputs that match would get as a test match:
    the learned models leave-one-match-out, the ball-finder cross-fitted within the match --
    exactly the situation of BAR-ATH, where every first-stage model is out of sample.
    """
    import seq
    data = prepare(dead_ml=True, team_ml=True)
    xs, labels = [], []
    for ball, players, team, truth, blocks, feat in data.values():
        loader.OVERRIDE = (ball, players, team)
        model.BLOCKS = blocks
        try:
            feat.update(model.compute())
            model.ballfinder_log_odds(feat)
        finally:
            loader.OVERRIDE = None
            model.BLOCKS = None
        xs.append(seq.features(feat))
        labels.append(truth)
    return xs, labels


def bench(variants, matches=None, with_baseline=True, dead_ml=False, team_ml=False):
    """Mean (and per-match spellF1) of each variant over the cached DFL matches, damaged data.

    dead_ml / team_ml: also fill the learned evidence, leave-one-match-out (see prepare).
    """
    data = prepare(matches, dead_ml, team_ml)
    matches = list(data)
    rows = {}
    if with_baseline:
        rows["baseline smooth50"] = [metrics(t, baseline(b, p, tm, bl))
                                     for b, p, tm, t, bl, _ in data.values()]
    for name, params in variants.items():
        rows[name] = [metrics(t, run_pipeline(b, p, tm, bl, params, feat))
                      for b, p, tm, t, bl, feat in data.values()]
    print("{:<30} spellF1  (sd)     macroF1  aliveDead  teamAcc  poss/truth   per-match spellF1"
          .format("{} matches".format(len(matches))))
    for name, per in rows.items():
        per = np.array(per)
        mean = per.mean(axis=0)
        print("{:<30} {:.4f} ({:.3f})  {:.4f}   {:.3f}      {:.3f}    {:.2f}        {}".format(
            name, mean[0], per[:, 0].std(), mean[1], mean[2], mean[3], mean[4],
            " ".join("{:.3f}".format(v) for v in per[:, 0])))
    return rows


def evaluate(match_id, variants):
    ball, players, team, truth, blocks = load_match(match_id)
    d_ball, d_players = degrade(ball, players, blocks)
    print(match_id)
    print("  {:<34} {}".format("baseline smooth50 (damaged)",
                               score(truth, baseline(d_ball, d_players, team, blocks))))
    for name, (params, clean) in variants.items():
        b, p = (ball, players) if clean else (d_ball, d_players)
        print("  {:<34} {}".format(name, score(truth, run_pipeline(b, p, team, blocks, params))))


if __name__ == "__main__":
    if sys.argv[1:] == ["--bench"]:
        bench({"ours (current defaults)": {}})
        sys.exit(0)
    for match_id in sys.argv[1:] or ["J03WOH"]:
        evaluate(match_id, {
            "ours v6 (clean data)": ({}, True),
            "ours v6 (damaged)": ({}, False),
        })
