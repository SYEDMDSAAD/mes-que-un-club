"""Ball-touch features for the second stage: where the ball's motion changes next to a player.

seq.py sees who is *near* the ball. A possession changes at a *touch* by the new team: the ball's
velocity changes (a kick speeds it up, a reception slows it, a deflection turns it) while a player
of that team is at it. On our ATD the ball is smoothed (frame-to-frame velocity change, median
0.3 m/s, against 0.6-0.8 for TRACAB), so a touch is a gradual bend over a few frames rather than a
corner. Everything here is therefore measured at several time scales, and attributed to a team by
distance rather than decided by one threshold.

Per frame, from the ball (x, y; NaN where not seen) and the players:

    impulse     |v_out - v_in|, v_in / v_out the ball velocity over the K frames before / after,
                for K in SCALES; the signed speed change |v_out| - |v_in| (a kick > 0, a
                reception < 0); the turn (1 - cos of the angle) where the ball is moving
    relative    the ball's speed relative to each team's nearest player: small when the ball
                moves with him (control, dribble), large when it flies past
    touch       impulse at the middle scale x a proximity weight for each team (only the nearer
                team gets it), and its window maxima and sums over +-0.25 / 0.5 / 1 / 2 s
    events      local maxima of the impulse above EVENT_MIN, given to the team of the nearest
                player within EVENT_REACH: frames since / until the last / next one by each team,
                how many by each team and how often the touching team alternates within +-1 / 2 s
    control     share of the frames within +-0.5 / 1 s on which each team's nearest player is
                within CONTROL_REACH and the ball moves with him

NaN where the ball is not seen (HistGradientBoosting handles it).

On a Metrica replica (3 matches, second stage leave-one-match-out) they added +0.103 spellF1 on
Metrica's own ball and +0.042 on that ball blurred and offset to look like ours (all 3 matches up
both times). Most of the first number is the events showing through the ball; the second is the
honest estimate, and still optimistic -- that ball turns only at real touches, with no bounces or
tracker jumps.

Two conditions for these to mean the same thing in training and on BAR-ATH:
  * the training ball must be as smooth as ours (dfl_eval BALL_SMOOTH; domain_check.py measures
    it): a sharp TRACAB corner and a smoothed ATD bend are different features;
  * they must not be learned from Metrica's sample matches. That ball is drawn from the events --
    99% of its sharp turns in play sit within 2 frames of an event's start or end -- and our
    labels for those matches are rebuilt from the same events, so a touch feature there reads the
    label. seq.features() blanks them on any match whose feat dict says event_ball.
"""

import numpy as np
import pandas as pd

# Params
FPS = 25
SCALES = (2, 4, 8)        # frames each side for the in / out velocity
MID = 4                   # the scale touches and events are built on
NEAR, FAR = 1.0, 3.5      # m: proximity weight 1 within NEAR, falling to 0 at FAR
EVENT_MIN = 4.0           # m/s of impulse at the middle scale to count as a touch event
EVENT_SUPPRESS = 6        # frames: one event per +- this
EVENT_REACH = 3.0         # m: an event needs a player this close to be given to a team
CONTROL_REACH = 2.0       # m
CONTROL_REL_SPEED = 2.5   # m/s: ball moving with the player
PLAYER_MAX_SPEED = 12.0   # m/s: faster is a tracker jump, not a run
WINDOWS = (6, 12, 25, 50)
CAP = 250                 # frames: the event clocks stop counting at 10 s
NAMES = None              # filled by compute()


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


def rolling(x, half, how):
    """Centred rolling max / sum / mean over +-half frames, ignoring NaN (NaN if all NaN)."""
    s = pd.Series(np.asarray(x, dtype=float)).rolling(2 * half + 1, center=True, min_periods=1)
    return getattr(s, how)().to_numpy()


def clocks(event):
    """Frames since the last and until the next True, capped at CAP."""
    n = len(event)
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(event, idx, -10 ** 9))
    nxt = np.minimum.accumulate(np.where(event, idx, 10 ** 9)[::-1])[::-1]
    return np.minimum(idx - last, CAP).astype(float), np.minimum(nxt - idx, CAP).astype(float)


def proximity(d):
    return np.clip((FAR - d) / (FAR - NEAR), 0.0, 1.0)


def compute(ball, players, team):
    """(N, F) float32 touch features; NAMES holds the column names."""
    global NAMES
    b = np.asarray(ball[:, :2], dtype=float)
    p = np.asarray(players[..., :2], dtype=float)
    team = np.asarray(team)
    seen = np.isfinite(b).all(axis=1)
    cols, names = [], []

    def add(name, x):
        cols.append(np.where(seen, x, np.nan))
        names.append(name)

    # ball kinematics at several scales
    v_c = (shift(b, 2) - shift(b, -2)) / 4 * FPS                     # centred ball velocity
    impulse = {}
    dspeed = {}
    for k in SCALES:
        v_in = (b - shift(b, -k)) / k * FPS
        v_out = (shift(b, k) - b) / k * FPS
        s_in, s_out = np.hypot(*v_in.T), np.hypot(*v_out.T)
        impulse[k] = np.hypot(*(v_out - v_in).T)
        dspeed[k] = s_out - s_in
        with np.errstate(invalid="ignore", divide="ignore"):
            cos = (v_in * v_out).sum(axis=1) / (s_in * s_out)
        turn = np.where(np.minimum(s_in, s_out) > 1.5, 1.0 - np.clip(cos, -1, 1), 0.0)
        add("impulse_{}".format(k), impulse[k])
        add("dspeed_{}".format(k), dspeed[k])
        add("turn_{}".format(k), turn)

    # each team's nearest player, and the ball's speed relative to him
    d = np.hypot(p[..., 0] - b[:, None, 0], p[..., 1] - b[:, None, 1])
    d = np.where(np.isfinite(d), d, np.inf)
    vp = (shift(p, 2) - shift(p, -2)) / 4 * FPS
    vp[np.hypot(vp[..., 0], vp[..., 1]) > PLAYER_MAX_SPEED] = np.nan
    rows = np.arange(len(b))
    near, rel = {}, {}
    for side, name in ((0, "home"), (1, "away")):
        ds = np.where((team == side)[None, :], d, np.inf)
        j = np.argmin(ds, axis=1)
        near[side] = ds[rows, j]
        near[side][~np.isfinite(near[side])] = np.nan
        rel[side] = np.hypot(*(v_c - vp[rows, j]).T)
        add("rel_{}".format(name), rel[side])
    with np.errstate(invalid="ignore"):
        nearer = {0: near[0] <= near[1], 1: near[1] < near[0]}

    # touch strength per team, and its window maxima / sums
    touch = {}
    for side in (0, 1):
        w = np.where(nearer[side], proximity(near[side]), 0.0)
        touch[side] = np.nan_to_num(impulse[MID], nan=0.0) * np.nan_to_num(w, nan=0.0)
    for side, name in ((0, "home"), (1, "away")):
        add("touch_{}".format(name), touch[side])
        for h in WINDOWS:
            add("touch_{}_max{}".format(name, h), rolling(touch[side], h, "max"))
        for h in (25, 50):
            add("touch_{}_sum{}".format(name, h), rolling(touch[side], h, "sum"))
    for h in WINDOWS:
        add("touch_diff_max{}".format(h), cols[names.index("touch_home_max{}".format(h))]
            - cols[names.index("touch_away_max{}".format(h))])

    # touch events: local maxima of the impulse, given to the nearest player's team
    imp = np.nan_to_num(impulse[MID], nan=0.0)
    peak = (imp >= EVENT_MIN) & (imp >= rolling(imp, EVENT_SUPPRESS, "max"))
    nearest_d = np.fmin(near[0], near[1])
    with np.errstate(invalid="ignore"):
        event_team = np.where(peak & (nearest_d < EVENT_REACH),
                              np.where(nearer[0], 0, 1), -1)
    ev = {side: event_team == side for side in (0, 1)}
    for side, name in ((0, "home"), (1, "away")):
        since, until = clocks(ev[side])
        cols += [since, until]
        names += ["ev_{}_since".format(name), "ev_{}_until".format(name)]
    cols += [cols[-4] - cols[-2], cols[-3] - cols[-1]]
    names += ["ev_since_diff", "ev_until_diff"]
    frames = np.flatnonzero(event_team >= 0)
    alt = np.zeros(len(b))
    if len(frames) > 1:
        changed = event_team[frames[1:]] != event_team[frames[:-1]]
        alt[frames[1:][changed]] = 1.0
    for h in (25, 50):
        for side, name in ((0, "home"), (1, "away")):
            cols.append(rolling(ev[side].astype(float), h, "sum"))
            names.append("ev_{}_count{}".format(name, h))
        cols.append(rolling(alt, h, "sum"))
        names.append("ev_alternations{}".format(h))

    # control: the ball at a player's feet and moving with him
    ctrl = {}
    for side in (0, 1):
        with np.errstate(invalid="ignore"):
            c = (nearer[side] & (near[side] < CONTROL_REACH)
                 & (rel[side] < CONTROL_REL_SPEED)).astype(float)
        ctrl[side] = np.where(seen, c, np.nan)
    for h in (12, 25):
        home, away = rolling(ctrl[0], h, "mean"), rolling(ctrl[1], h, "mean")
        cols += [home, away, home - away]
        names += ["control_home{}".format(h), "control_away{}".format(h), "control_diff{}".format(h)]

    NAMES = names
    return np.column_stack(cols).astype(np.float32)


def n_features():
    """Number of columns compute() returns (without running it)."""
    return 3 * len(SCALES) + 2 + 2 * (1 + len(WINDOWS) + 2) + len(WINDOWS) + 6 + 6 + 6
