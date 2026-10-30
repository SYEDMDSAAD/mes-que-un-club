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
import variant  # noqa: E402
from external_dfl import load_match  # noqa: E402

# Params
BALL_NOISE = 0.3          # m, per axis -- see the docstring: it hides a TRACAB artefact
PLAYER_NOISE = 0.1
PLAYER_DROP = 0.07
# Variant "droplenS": players drop out for S frames at a time. 50 (v13) at 7% of the time is ~46
# track births a minute on a DFL match; ours has 6-11 (domain_check.py --more, "players"), in
# fewer, longer gaps. droplen250 keeps the 7% and gives ~9 a minute.
DROP_LEN = int(variant.value("droplen", 50))   # frames
# Variant "motiondropS": the same longer dropouts, but only in the players the motion model
# (motion_ml.py) trains on and, on a bench, reads -- every other model keeps v13's 50-frame
# dropouts. droplen250 for all models cost BAR-ATH macroF1 0.905 -> 0.871 on its own.
MOTION_DROP = int(variant.value("motiondrop", 250))   # frames (v15: 250); 0: as DROP_LEN
SEED = 0
# Training copies (see degrade(randomize=True)): how many per match, and the ranges drawn from.
TRAIN_COPIES = 1
RANDOM_BALL_NOISE = (0.15, 0.5)
RANDOM_PLAYER_NOISE = (0.05, 0.3)
RANDOM_PLAYER_DROP = (0.03, 0.12)
# How the ball is damaged (MQUC_VARIANT selects another mode; unset = the submission):
#
#   "drift" (v13, the submission): BAR-ATH's gap pattern laid over the DFL half by time, plus slow
#       noise (smoothed over ~0.5 s), plus a slowly wandering ball in stoppages. Independent noise
#       per frame (v12) was a median 12.5 m/s of fake ball speed, so no model could learn real ball
#       speed. But it also hid a TRACAB artefact: a dead ball frozen to the millimetre, where ours is
#       carried to the restart (1-s displacement on dead frames: 1.5 m over the full match, 2.5 m in
#       the public window, 0.0 clean DFL). Slow noise alone brought back "a still ball is dead" and
#       collapsed BAR-ATH (public macroF1 0.67); the drift in stoppages removes that shortcut.
#   "v12": independent 0.3 m noise per frame. "noise": slow noise without the drift.
#   "state": the ball goes missing mostly in stoppages, at MISS_* rates and GAP_* means
#       (domain_check.py measures them; "state40" uses the public window's rates).
DEGRADE_MODE = ("state" if any(part.startswith("state") for part in variant.PARTS)
                else "v12" if variant.has("v12") else "noise" if variant.has("noise") else "drift")
DEAD_DRIFT = 1.5          # m, slow noise added on dead frames only ("drift")
# Variant "taper": the drift fades in and out with a Gaussian of this many frames around each
# stoppage edge. 0 (v13) switches it on and off in one frame: a median 44 m/s jump of the training
# ball at every edge, a fake kick no BAR-ATH stoppage has.
DRIFT_TAPER = 12 if variant.has("taper") else 0
# Variant "smoothS": the clean ball blurred by a Gaussian of S frames before it is damaged. Our ATD
# ball changes velocity by a median 0.3 m/s per frame, clean TRACAB by 0.6-0.8: its kinks are
# sharper than ours. Match "dv live" and the "turn" rows of domain_check.py --more.
BALL_SMOOTH = variant.value("smooth", 0.0)
# Variant "wrongballR": R stretches a minute of play where the training ball sits on a player of
# the team WITHOUT the ball (the tracker on the wrong object), WRONG_LEN frames long, locked onto
# the opponent nearest the true ball when the stretch starts. Our tracking does this ("the other
# team nearest the ball when the truth says otherwise"); clean TRACAB almost never does, so the
# learned models never learn to doubt the ball. Set R so the "near right" row of
# domain_check.py --more on damaged DFL matches BAR-ATH's public window. Its own random stream:
# switching it on changes nothing else in the damage.
WRONG_BALL = variant.value("wrongball", 0.0)   # stretches per minute of play
WRONG_LEN = (12, 50)      # frames, uniform
WRONG_OFFSET = 0.3        # m from that opponent towards the true ball
MISS_LIVE = 0.074         # share of live frames without a ball
MISS_DEAD = 0.68          # share of dead frames without a ball
GAP_LIVE = 30             # mean missing run inside live play, frames
GAP_DEAD = 200            # mean missing run inside a stoppage, frames
if variant.has("state40"):
    MISS_LIVE, MISS_DEAD, GAP_LIVE, GAP_DEAD = 0.06, 0.40, 70, 260
BALL_NOISE_SMOOTH = 12    # frames: Gaussian smoothing of the slow ball noise (state mode)
# Variant "testbar": a second yardstick. Every model is still TRAINED on the damage above, but the
# bench matches are TESTED on a ball that looks like BAR-ATH's: missing by state at the full
# match's rates (MISS_LIVE / MISS_DEAD and GAP_*: 7.4% of in-play frames, not the ~30-37% that the
# clock-time gap pattern leaves on DFL's play -- domain_check "seen") and smoothed and offset like
# our ATD (the metatd transform: in-play speed, velocity change and ball-to-player distance of
# BAR-ATH), drifting in stoppages as in "drift". That is the submission's own situation -- trained
# on the drift damage, run on a ball missing by state -- so decisions that hinge on how often and
# how well the ball is seen (alive_level, seq_gate, motion, touch, video) can be checked on it.
TEST_BAR = variant.has("testbar")
# The defaults are the metatd transform, made for Metrica's event-drawn ball. On TRACAB they leave
# the ball rougher than ours and farther from the players (domain_check: impulse4 p90 4.4 against
# 2.8, turn8 p90 0.30 against 0.12, near live p50 2.2 against 1.7 m, near right 0.72 against 0.78)
# -- harsh on exactly what the touch features read. Parts "testblurS" / "testslowS" set them (the
# touch blur's calibration, blur 6 with the usual 0.3 m noise, matched BAR-ATH's kinks); the parts
# join the cache tag, so a new setting never reads an old cache.
TEST_BLUR = variant.value("testblur", 3.0)    # frames (external_metrica.ATD_BLUR)
TEST_SLOW = variant.value("testslow", 1.15)   # m, smoothed over TEST_SLOW_SMOOTH frames (ATD_SLOW)
TEST_SLOW_SMOOTH = 12
TEST_JITTER = 0.01        # m, smoothed over 1.5 frames


def training_copies(ball, players, blocks, truth=None, team=None, drop_len=None):
    """The damaged versions of one match a learned model is trained on."""
    if TRAIN_COPIES <= 1:
        return [degrade(ball, players, blocks, truth=truth, team=team, drop_len=drop_len)]
    return [degrade(ball, players, blocks, seed=100 + k, randomize=True, truth=truth, team=team,
                    drop_len=drop_len) for k in range(TRAIN_COPIES)]


def state_seen(labels, rng):
    """Ball-seen mask for one half from its labels: inside every live run and every stoppage,
    alternating seen / missing stretches of exponential length, at MISS_* rates and GAP_* means."""
    n = len(labels)
    seen = np.ones(n, dtype=bool)
    dead = labels == "D"
    breaks = np.flatnonzero(dead[1:] != dead[:-1]) + 1
    for lo, hi in zip(np.r_[0, breaks], np.r_[breaks, n]):
        share, gap = (MISS_DEAD, GAP_DEAD) if dead[lo] else (MISS_LIVE, GAP_LIVE)
        mean = {True: gap, False: gap * (1 - share) / share}
        missing = bool(rng.random() < share)
        t = lo
        while t < hi:
            length = 1 + int(rng.exponential(max(mean[missing] - 1, 1e-9)))
            if missing:
                seen[t:min(t + length, hi)] = False
            t += length
            missing = not missing
    return seen


def blur_runs(ball, sigma):
    """ball with x, y blurred by a Gaussian of sigma frames inside each run of finite values."""
    from scipy.ndimage import gaussian_filter1d
    out = ball.copy()
    seen = np.isfinite(ball[:, 0])
    den = gaussian_filter1d(seen.astype(float), sigma, mode="nearest")
    for k in (0, 1):
        num = gaussian_filter1d(np.where(seen, ball[:, k], 0.0).astype(float), sigma, mode="nearest")
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, k] = np.where(seen, num / den, np.nan)
    return out


def slow_noise(rng, n, sigma, smooth):
    """(n, 2) noise per axis: white noise smoothed by a Gaussian of `smooth` frames, rescaled to
    standard deviation sigma -- a slowly wandering offset with smooth velocity."""
    from scipy.ndimage import gaussian_filter1d
    z = gaussian_filter1d(rng.normal(0, 1, (n, 2)), smooth, axis=0, mode="nearest")
    return z * (sigma / z.std(axis=0))


def degrade_state(ball, players, blocks, seed, randomize, truth):
    """DEGRADE_MODE "state": see the Params comment. Players are damaged as in "v12"."""
    if truth is None:
        raise ValueError("degrade in 'state' mode needs the match's labels (truth=...)")
    rng = np.random.default_rng(seed)
    ball_noise, player_noise, player_drop = BALL_NOISE, PLAYER_NOISE, PLAYER_DROP
    if randomize:
        ball_noise = rng.uniform(*RANDOM_BALL_NOISE)
        player_noise = rng.uniform(*RANDOM_PLAYER_NOISE)
        player_drop = rng.uniform(*RANDOM_PLAYER_DROP)
    ball = ball.copy()
    for first, last in blocks:
        seen = state_seen(np.asarray(truth[first:last + 1]), rng)
        ball[first:last + 1][~seen] = np.nan
    if ball_noise:
        ball[:, :2] += slow_noise(rng, len(ball), ball_noise, BALL_NOISE_SMOOTH).astype(ball.dtype)
    players = players.copy()
    players[..., :2] += rng.normal(0, player_noise, players[..., :2].shape)
    n, s = players.shape[:2]
    starts = rng.random((n, s)) < player_drop / DROP_LEN
    for t, k in zip(*np.nonzero(starts)):
        players[t:t + DROP_LEN, k] = np.nan
    return ball, players


def wrong_ball(ball, players, team, truth, rate, rng):
    """Copy of ball with `rate` stretches a minute of play moved onto a player of the team without
    the ball (variant "wrongball"); only frames where the ball is seen and play is live."""
    out = ball.copy()
    truth = np.asarray(truth)
    live = np.isin(truth, ["H", "A"]) & np.isfinite(ball[:, 0])
    idx = np.flatnonzero(live)
    if not len(idx) or rate <= 0:
        return out
    k = rng.poisson(rate * len(idx) / 25.0 / 60.0)
    for s in np.sort(rng.choice(idx, size=k)):
        length = int(rng.integers(WRONG_LEN[0], WRONG_LEN[1] + 1))
        frames = np.arange(s, min(s + length, len(ball)))
        frames = frames[live[frames]]
        if not len(frames):
            continue
        holder = 0 if truth[frames[0]] == "H" else 1
        opp = np.flatnonzero(team != holder)
        d = np.hypot(players[frames[0], opp, 0] - ball[frames[0], 0],
                     players[frames[0], opp, 1] - ball[frames[0], 1])
        if not np.isfinite(d).any():
            continue
        j = opp[int(np.nanargmin(d))]
        for t in frames:
            if truth[t] not in ("H", "A") or not np.isfinite(players[t, j, 0]):
                continue
            vec = ball[t, :2] - players[t, j, :2]
            norm = np.hypot(*vec)
            step = vec / norm * min(WRONG_OFFSET, norm) if norm > 0 else 0.0
            out[t, :2] = players[t, j, :2] + step
    return out


def degrade(ball, players, blocks, seed=SEED, randomize=False, truth=None, team=None,
            drop_len=None):
    """Damage clean DFL tracking to look like ours.

    randomize=True draws the noise levels, the dropout rate and where in our match the ball-gap
    pattern is taken from, per call -- used for training copies, so a learned model sees many
    versions of the damage and cannot lean on any single one of them.
    truth (the match's labels) is needed by the "drift" and "state" modes; team by "wrongball".
    drop_len overrides DROP_LEN (variant "motiondrop").
    """
    if DEGRADE_MODE == "state":
        return degrade_state(ball, players, blocks, seed, randomize, truth)
    rng = np.random.default_rng(seed)
    ball_noise, player_noise, player_drop = BALL_NOISE, PLAYER_NOISE, PLAYER_DROP
    shift = 0
    if randomize:
        ball_noise = rng.uniform(*RANDOM_BALL_NOISE)
        player_noise = rng.uniform(*RANDOM_PLAYER_NOISE)
        player_drop = rng.uniform(*RANDOM_PLAYER_DROP)
        shift = int(rng.integers(0, 60000))
    # Our ball even when the pipeline is pointed at another match (loader.OVERRIDE): a model first
    # trained from inside such a run would otherwise learn that match's gap pattern.
    saved, loader.OVERRIDE = loader.OVERRIDE, None
    try:
        ours, _, _ = loader.load()
    finally:
        loader.OVERRIDE = saved
    our_blocks = [(first, last) for _, first, last in model.half_video_blocks(model.GAME)]
    ball = blur_runs(ball, BALL_SMOOTH) if BALL_SMOOTH > 0 else ball.copy()
    for (first, last), (o_first, o_last) in zip(blocks, our_blocks):
        n = last - first + 1
        seen = np.isfinite(ours[o_first:o_last + 1, 0])
        seen = np.roll(seen, shift)
        seen = np.resize(seen, n)                         # tile if the DFL half is longer
        ball[first:last + 1][~seen] = np.nan
    if ball_noise and DEGRADE_MODE in ("noise", "drift"):
        ball[:, :2] += slow_noise(rng, len(ball), ball_noise, BALL_NOISE_SMOOTH).astype(ball.dtype)
        if DEGRADE_MODE == "drift":
            if truth is None:
                raise ValueError("degrade in 'drift' mode needs the match's labels (truth=...)")
            drift = slow_noise(rng, len(ball), DEAD_DRIFT, BALL_NOISE_SMOOTH)
            dead = np.asarray(truth) == "D"
            if DRIFT_TAPER > 0:
                from scipy.ndimage import gaussian_filter1d
                dead = gaussian_filter1d(dead.astype(float), DRIFT_TAPER, mode="nearest")
            ball[:, :2] += (drift * dead[:, None]).astype(ball.dtype)
    elif ball_noise:
        ball[:, :2] += rng.normal(0, ball_noise, ball[:, :2].shape)
    if WRONG_BALL > 0:
        if team is None or truth is None:
            raise ValueError("variant 'wrongball' needs the match's labels and teams")
        ball = wrong_ball(ball, players, np.asarray(team), truth, WRONG_BALL,
                          np.random.default_rng(seed + 7919))

    players = players.copy()
    players[..., :2] += rng.normal(0, player_noise, players[..., :2].shape)
    n, s = players.shape[:2]
    drop_len = drop_len or DROP_LEN
    starts = rng.random((n, s)) < player_drop / drop_len
    for t, k in zip(*np.nonzero(starts)):
        players[t:t + drop_len, k] = np.nan
    return ball, players


def degrade_test(ball, players, blocks, truth, team=None, seed=SEED):
    """Variant "testbar": the bench's test copy of a DFL match with a BAR-like ball (see the
    TEST_BAR comment); the players are damaged exactly as degrade() damages them."""
    from scipy.ndimage import gaussian_filter1d
    _, d_players = degrade(ball, players, blocks, seed=seed, truth=truth, team=team)
    rng = np.random.default_rng(seed + 4049)
    out = blur_runs(ball, TEST_BLUR)
    seen = np.isfinite(out[:, 0])
    for sd, smooth in ((TEST_SLOW, TEST_SLOW_SMOOTH), (TEST_JITTER, 1.5)):
        z = gaussian_filter1d(rng.normal(0, 1, (len(out), 2)), smooth, axis=0, mode="nearest")
        out[:, :2] += (z * (sd / z.std(axis=0))).astype(out.dtype)
    drift = slow_noise(rng, len(out), DEAD_DRIFT, BALL_NOISE_SMOOTH)
    out[:, :2] += (drift * (np.asarray(truth) == "D")[:, None]).astype(out.dtype)
    out[~seen, :2] = np.nan
    for first, last in blocks:
        kept = state_seen(np.asarray(truth[first:last + 1]), rng)
        out[first:last + 1][~kept] = np.nan
    return out, d_players


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


def prepare(matches=None, dead_ml=False, team_ml=False, motion_ml=False, test_bar=None):
    """{match: (ball, players, team, truth, blocks, feat)} on damaged data, learned evidence
    filled leave-one-match-out: the classifiers scoring a match never saw it. test_bar (default:
    variant "testbar"): the arrays returned -- the ones scored -- carry the BAR-like test damage,
    while the classifiers still learn from the usual damage."""
    matches = matches or cached_matches()
    test_bar = TEST_BAR if test_bar is None else test_bar
    data, copies = {}, {}
    for m in matches:
        ball, players, team, truth, blocks = load_match(m)
        d_ball, d_players = degrade(ball, players, blocks, truth=truth, team=team)
        t_ball, t_players = (degrade_test(ball, players, blocks, truth, team) if test_bar
                             else (d_ball, d_players))
        data[m] = (t_ball, t_players, team, truth, blocks, {})
        if dead_ml or team_ml or motion_ml:
            copies[m] = (training_copies(ball, players, blocks, truth, team)
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
    if motion_ml:
        import motion_ml as mm
        if MOTION_DROP:                    # the motion model's own players, train and test alike
            own = {}
            for m in matches:
                ball, players, team, truth, blocks = load_match(m)
                test_players = degrade(ball, players, blocks, truth=truth, team=team,
                                       drop_len=MOTION_DROP)[1]
                own[m] = (test_players, [p for _, p in mm.training_copies(
                    ball, players, blocks, truth, team)]
                    if TRAIN_COPIES > 1 or mm.COPIES > 0 else [test_players])
        else:
            own = {m: (data[m][1], [p for _, p in copies[m]]) for m in matches}
            if mm.COPIES > 0:
                for m in matches:
                    ball, players, team, truth, blocks = load_match(m)
                    own[m] = (own[m][0], [p for _, p in mm.training_copies(
                        ball, players, blocks, truth, team)])
        x = {m: [mm.features(data[m][0], p, data[m][2], data[m][4]) for p in own[m][1]]
             for m in matches}
        test = {m: mm.features(data[m][0], own[m][0], data[m][2], data[m][4]) for m in matches}
        for m in matches:
            others = [k for k in matches if k != m]
            fitted = mm.fit([a for k in others for a in x[k]],
                            [data[k][3] for k in others for _ in x[k]])
            data[m][5]["motion_ml"] = mm.log_odds(fitted, test[m])
    return data


def seq_training_set():
    """(features, labels) per DFL match for the second-stage model (seq.py).

    Its inputs on each match are the first-stage outputs that match would get as a test match:
    the learned models leave-one-match-out, the ball-finder cross-fitted within the match --
    exactly the situation of BAR-ATH, where every first-stage model is out of sample.
    """
    import seq
    data = prepare(dead_ml=True, team_ml=True, motion_ml=seq.MOTION, test_bar=False)
    xs, labels = [], []
    for ball, players, team, truth, blocks, feat in data.values():
        loader.OVERRIDE = (ball, players, team)
        model.BLOCKS = blocks
        try:
            feat.update(model.compute())
            model.ballfinder_log_odds(feat)
            if seq.TOUCH:
                model.touch_features(feat)
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
    d_ball, d_players = degrade(ball, players, blocks, truth=truth, team=team)
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
