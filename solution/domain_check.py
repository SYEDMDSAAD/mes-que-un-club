"""How close is the damaged DFL tracking to ours? Aggregates only, never a frame or a position.

Every learned model is trained on DFL tracking damaged to look like BAR-ATH (dfl_eval.degrade),
plus Metrica's sample matches as they are. A model is only as good as that imitation, feature by
feature. This prints the statistics the features are built on, side by side, for BAR-ATH (fused
ball), clean DFL, damaged DFL and Metrica:

    seen        share of graded frames with a ball, and by state where labels exist: DFL and
                Metrica truth, BAR-ATH's public window, and the full match from --labels (your
                own submission file: predicted states, not ground truth)
    gap live    mean / median / p90 length, in frames, of missing-ball runs inside live play
    gap dead    the same inside stoppages (mean and share missing are what dfl_eval's
                MISS_* and GAP_* constants want)
    v live      per-frame ball speed on live frames, |step| * 25, p10 / p50 / p90. Independent
                position noise of s metres per axis adds about s * 44 m/s on its own
    still dead  1-s ball displacement on dead frames with a ball, p50 (TRACAB freezes the ball)
    near live   ball to the nearest player on live frames, p10 / p50 / p90
    dv live     change of ball velocity between frames on live frames, p50 / p90: a smoothed
                ball has a small one, a noisy ball a large one

If damaged DFL and BAR-ATH disagree on a row, every feature built on that row means something
different in training and in the submission.

--more prints a second line per source, for what touch features and the dead model read:

    turn k      ball turn 1 - cos(angle) between the velocity over the k frames before and after,
                p50 / p90, live frames where the ball moves > 2 m/s, k = 2 / 4 / 8. A smoothed ball
                turns gently at k = 2 and fully at k = 8; matching dv (k = 1) alone does not
                make the kinks alike
    impulse4    |v_out - v_in| at k = 4, p50 / p90 / p99 -- the raw material of a touch
    acf|dv|     autocorrelation of |dv| at lags 1..4: a ball interpolated between keyframes
                shows its keyframe period here (Metrica's sample ball: 0.5 / 0 / 0 / 0)
    kick near   ball to the nearest player at kick onsets, p50: a blurred kink lands farther
                from the kicker
    out         share of seen ball frames more than 1 m outside the lines, live / dead: does the
                tracker follow a ball off the pitch, or clamp / drop it (the dead model reads it)
    players     tracked per frame in play (mean), track births per minute (identity churn), and
                player speed p50 / p90 in play and in stoppages (seq.py uses raw speed)
    spells      possessions per minute of play, share shorter than 1 / 2 / 5 s, stoppages per
                graded minute: the labels' convention -- yours on BAR-ATH (--labels) against
                each labelled source and the public window
    near right  share of in-play frames with a ball where the nearest player's team is the team
                in possession, overall and when that player is within 1.5 m: how often the ball
                is on the wrong team. BAR-ATH's public window against damaged DFL calibrates
                variant "wrongball" (on BAR-ATH only the public window has true labels)
    drop len    mean length, in frames, of a player's missing stretch: with track births a
                minute it calibrates variant "droplen"
    v1s dead    1-s mean ball speed on dead frames with a ball, p50 / p90 (the dead model and
                the second stage read it)

    python solution/domain_check.py [--labels "Mes Que Un Club.csv"] [--more]

With MQUC_VARIANT=testbar each DFL match gets a third line, its BAR-like test copy
(dfl_eval.degrade_test): "seen" by state, v live, near live and dv live should sit near BAR-ATH's.

    python solution/domain_check.py --calibrate

calibrates that copy (variant parts testblurS / testslowS, also the training ball of "seqbar"):
for each (blur, slow) in CALIBRATE, the rows the second stage and the touch features read, as the
median over the DFL matches' test copies, against BAR-ATH's public window, and "off": the mean
|log(DFL / BAR-ATH)| over those rows (0 = identical). No variant needed; nothing is cached.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import dfl_eval  # noqa: E402
import external_metrica  # noqa: E402
from external_dfl import load_match as load_dfl  # noqa: E402
from games import GAME, half_video_blocks  # noqa: E402
from load import load  # noqa: E402

# Params
FPS = 25
GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def pct(x, qs):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, q)) if len(x) else np.nan for q in qs]


def runs(mask):
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def stats(ball, players, labels, blocks):
    n = len(ball)
    graded = np.zeros(n, bool)
    for first, last in blocks:
        graded[first:last + 1] = True
    lab = np.asarray(labels) if labels is not None else np.full(n, "X")
    live = graded & np.isin(lab, ["H", "A"])
    dead = graded & (lab == "D")
    x, y = ball[:, 0].astype(float), ball[:, 1].astype(float)
    seen = np.isfinite(x)

    vx = np.r_[np.nan, np.diff(x)] * FPS
    vy = np.r_[np.nan, np.diff(y)] * FPS
    v = np.hypot(vx, vy)
    dv = np.hypot(np.r_[np.nan, np.diff(vx)], np.r_[np.nan, np.diff(vy)])
    still = np.full(n, np.nan)
    still[FPS:] = np.hypot(x[FPS:] - x[:-FPS], y[FPS:] - y[:-FPS])
    d = np.hypot(players[..., 0] - x[:, None], players[..., 1] - y[:, None])
    near = np.nanmin(np.where(np.isfinite(d), d, np.nan), axis=1)

    gaps = {"live": [], "dead": []}
    for lo, hi in zip(*runs(graded & ~seen)):
        state = lab[(lo + hi) // 2]
        if state in ("H", "A"):
            gaps["live"].append(hi - lo)
        elif state == "D":
            gaps["dead"].append(hi - lo)

    def share(m):
        return seen[m].mean() if m.any() else np.nan

    return {
        "seen": share(graded), "seen live": share(live), "seen dead": share(dead),
        "gap live": [np.mean(gaps["live"]) if gaps["live"] else np.nan] + pct(gaps["live"], (50, 90)),
        "gap dead": [np.mean(gaps["dead"]) if gaps["dead"] else np.nan] + pct(gaps["dead"], (50, 90)),
        "v live": pct(v[live], (10, 50, 90)), "still dead": pct(still[dead], (50,)),
        "near live": pct(near[live], (10, 50, 90)), "dv live": pct(dv[live], (50, 90)),
    }


def show(name, s):
    def f(v):
        if isinstance(v, list):
            return "/".join("{:.1f}".format(a) for a in v)
        return "{:.2f}".format(v)
    print("{:<26} seen {} (live {}, dead {})  gap live {}  gap dead {}  v live {}  "
          "still dead {}  near live {}  dv live {}".format(
              name, f(s["seen"]), f(s["seen live"]), f(s["seen dead"]), f(s["gap live"]),
              f(s["gap dead"]), f(s["v live"]), f(s["still dead"]), f(s["near live"]),
              f(s["dv live"])), flush=True)


def more_stats(ball, players, labels, blocks, team=None):
    """The --more line: kink shape at several scales, interpolation, lag, players, spells."""
    n = len(ball)
    graded = np.zeros(n, bool)
    for first, last in blocks:
        graded[first:last + 1] = True
    lab = np.asarray(labels) if labels is not None else np.full(n, "X")
    live = graded & np.isin(lab, ["H", "A"])
    dead = graded & (lab == "D")
    b = ball[:, :2].astype(float)
    out = {}

    def shifted(x, k):
        y = np.full(x.shape, np.nan)
        if k > 0:
            y[:-k] = x[k:]
        else:
            y[-k:] = x[:k]
        return y

    for k in (2, 4, 8):
        v_in = (b - shifted(b, -k)) / k * FPS
        v_out = (shifted(b, k) - b) / k * FPS
        s_in, s_out = np.hypot(*v_in.T), np.hypot(*v_out.T)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos = (v_in * v_out).sum(axis=1) / (s_in * s_out)
            moving = live & (np.minimum(s_in, s_out) > 2.0)
        out["turn {}".format(k)] = pct(1.0 - cos[moving], (50, 90))
        if k == 4:
            out["impulse4"] = pct(np.hypot(*(v_out - v_in).T)[live], (50, 90, 99))

    v = np.vstack([np.r_[np.nan, np.diff(b[:, 0])], np.r_[np.nan, np.diff(b[:, 1])]]).T * FPS
    dv = np.hypot(*np.vstack([np.r_[np.nan, np.diff(v[:, 0])], np.r_[np.nan, np.diff(v[:, 1])]]))
    ok = live & np.isfinite(dv)
    z = np.where(ok, dv - np.nanmean(dv[ok]), 0.0)
    out["acf|dv|"] = [float(np.sum(z[k:] * z[:-k]) / max(np.sum(z * z), 1e-12)) for k in (1, 2, 3, 4)]

    d = np.hypot(players[..., 0] - b[:, None, 0], players[..., 1] - b[:, None, 1])
    d = np.where(np.isfinite(d), d, np.inf)
    j = np.argmin(d, axis=1)
    dmin = d[np.arange(n), j]
    gain = np.nan_to_num(np.hypot(*((shifted(b, 4) - b) / 4 * FPS).T)
                         - np.hypot(*((b - shifted(b, -4)) / 4 * FPS).T), nan=0.0)
    onset = live & (gain >= 3.0)
    onset[1:-1] &= (gain[1:-1] >= gain[:-2]) & (gain[1:-1] >= gain[2:])
    out["kick near"] = pct(np.where(np.isfinite(dmin), dmin, np.nan)[onset], (50,))

    seen = np.isfinite(b[:, 0])
    with np.errstate(invalid="ignore"):
        outside = (b[:, 0] < -1) | (b[:, 0] > 106) | (b[:, 1] < -1) | (b[:, 1] > 69)
    out["out"] = [float(outside[live & seen].mean()) if (live & seen).any() else np.nan,
                  float(outside[dead & seen].mean()) if (dead & seen).any() else np.nan]

    present = np.isfinite(players[..., 0])
    births = (present[1:] & ~present[:-1])[graded[1:]].sum()
    minutes = graded.sum() / FPS / 60
    sp = players[..., 2].astype(float)
    with np.errstate(invalid="ignore"):
        out["players"] = [float(present[live].sum(axis=1).mean()) if live.any() else np.nan,
                          float(births / max(minutes, 1e-9))]
    out["speed live"] = pct(sp[live].ravel(), (50, 90))
    out["speed dead"] = pct(sp[dead].ravel(), (50, 90))
    runs_len = []
    for k in range(players.shape[1]):
        miss = graded & ~present[:, k]
        if present[graded, k].any():                 # a slot that is used in this stretch
            lo, hi = runs(miss)
            inner = (lo > 0) & (hi < n)
            runs_len += (hi - lo)[inner].tolist()
    out["drop len"] = float(np.mean(runs_len)) if runs_len else np.nan
    v1 = np.hypot(*np.vstack([np.r_[np.nan, np.diff(b[:, 0])], np.r_[np.nan, np.diff(b[:, 1])]])) * FPS
    from features import rolling_mean
    out["v1s dead"] = pct(rolling_mean(np.clip(v1, 0, 40), FPS // 2)[dead & seen], (50, 90))
    if team is not None:
        team = np.asarray(team)
        dh = np.min(np.where((team == 0)[None, :], d, np.inf), axis=1)
        da = np.min(np.where((team == 1)[None, :], d, np.inf), axis=1)
        ok = live & seen & np.isfinite(dh) & np.isfinite(da)
        right = (dh < da) == (lab == "H")
        close = ok & (np.fmin(dh, da) < 1.5)
        out["near right"] = [float(right[ok].mean()) if ok.any() else np.nan,
                             float(right[close].mean()) if close.any() else np.nan]
    else:
        out["near right"] = [np.nan, np.nan]

    spells, stops = [], 0
    for first, last in blocks:
        w = lab[first:last + 1]
        br = np.flatnonzero(w[1:] != w[:-1]) + 1
        st, en = np.r_[0, br], np.r_[br, len(w)]
        for a, e in zip(st, en):
            if w[a] in ("H", "A"):
                spells.append(e - a)
            elif w[a] == "D":
                stops += 1
    spells = np.array(spells) / FPS
    live_min = live.sum() / FPS / 60
    out["spells"] = [len(spells) / max(live_min, 1e-9)] + \
        [float((spells < c).mean()) if len(spells) else np.nan for c in (1, 2, 5)] + \
        [stops / max(minutes, 1e-9)]
    return out


def show_more(name, s):
    def f(v, fmt="{:.2f}"):
        return "/".join(fmt.format(a) for a in v) if isinstance(v, list) else fmt.format(v)
    print("{:<26} turn2 {}  turn4 {}  turn8 {}  impulse4 {}  acf|dv| {}  kick near {}  "
          "out {}  players {}  speed live {}  dead {}  spells/min, <1/2/5 s, stops/min {}  "
          "near right {}  drop len {}  v1s dead {}".format(
              name, f(s["turn 2"]), f(s["turn 4"]), f(s["turn 8"]), f(s["impulse4"], "{:.1f}"),
              f(s["acf|dv|"]), f(s["kick near"]), f(s["out"], "{:.3f}"),
              f(s["players"], "{:.1f}"), f(s["speed live"], "{:.1f}"),
              f(s["speed dead"], "{:.1f}"), f(s["spells"]), f(s["near right"], "{:.3f}"),
              f(s["drop len"], "{:.0f}"), f(s["v1s dead"], "{:.2f}")), flush=True)


# (blur frames, slow offset m) tried by --calibrate; (3, 1.15) is the testbar default (metatd)
CALIBRATE = ((3.0, 1.15), (6.0, 0.3), (6.0, 0.6), (4.0, 0.3), (8.0, 0.3), (6.0, 0.15))
# (row, index, label): what --calibrate compares and scores
CAL_ROWS = (("v live", 1, "v live p50"), ("dv live", 0, "dv p50"), ("near live", 0, "near p10"),
            ("near live", 1, "near p50"), ("near live", 2, "near p90"), ("turn 4", 1, "turn4 p90"),
            ("turn 8", 1, "turn8 p90"), ("impulse4", 0, "imp4 p50"), ("impulse4", 1, "imp4 p90"),
            ("impulse4", 2, "imp4 p99"), ("kick near", 0, "kick near"),
            ("near right", 0, "right"), ("near right", 1, "right<1.5m"))


def calibration_row(s, m):
    return [(m if row in m else s)[row][i] for row, i, _ in CAL_ROWS]


def calibrate():
    """--calibrate: see the module docstring."""
    ball, players, our_team = load()
    public = pd.read_csv(GT)["label"].values
    idx = np.flatnonzero(public != "X")
    bl = [(idx[0], idx[-1])]
    bar = calibration_row(stats(ball, players, public, bl),
                          more_stats(ball, players, public, bl, our_team))
    rows = {c: [] for c in CALIBRATE}
    for m in dfl_eval.cached_matches():
        b, p, tm, truth, blocks = load_dfl(m)
        for blur, slow in CALIBRATE:
            dfl_eval.TEST_BLUR, dfl_eval.TEST_SLOW = blur, slow
            t_b, t_p = dfl_eval.degrade_test(b, p, blocks, truth, tm)
            rows[(blur, slow)].append(calibration_row(stats(t_b, t_p, truth, blocks),
                                                      more_stats(t_b, t_p, truth, blocks, tm)))
        print("  {} done".format(m), flush=True)
    print("{:<22}".format("") + "".join("{:>11}".format(label) for _, _, label in CAL_ROWS)
          + "{:>8}".format("off"))
    print("{:<22}".format("BAR-ATH public") + "".join("{:>11.2f}".format(v) for v in bar))
    bar = np.array(bar, float)
    for (blur, slow), values in sorted(rows.items()):
        med = np.nanmedian(np.array(values, float), axis=0)
        off = np.nanmean(np.abs(np.log(np.clip(med, 1e-6, None) / np.clip(bar, 1e-6, None))))
        print("{:<22}".format("testblur{:g}+testslow{:g}".format(blur, slow))
              + "".join("{:>11.2f}".format(v) for v in med) + "{:>8.3f}".format(off), flush=True)


def main(labels_path=None, more=False):
    ball, players, our_team = load()
    blocks = [(first, last) for _, first, last in half_video_blocks(GAME)]
    public = pd.read_csv(GT)["label"].values
    idx = np.flatnonzero(public != "X")
    def both(name, b, p, labels, bl, team=None):
        show(name, stats(b, p, labels, bl))
        if more:
            show_more(name, more_stats(b, p, labels, bl, team))

    both("BAR-ATH public window", ball, players, public, [(idx[0], idx[-1])], our_team)
    if labels_path:
        mine = pd.read_csv(labels_path)["label"].astype(str).values
        both("BAR-ATH full, our labels", ball, players, mine, blocks)

    for m in dfl_eval.cached_matches():
        b, p, tm, truth, bl = load_dfl(m)
        both("DFL {} clean".format(m), b, p, truth, bl, tm)
        d_b, d_p = dfl_eval.degrade(b, p, bl, truth=truth, team=tm)
        both("DFL {} damaged".format(m), d_b, d_p, truth, bl, tm)
        if dfl_eval.TEST_BAR:                   # the bench's BAR-like test copy (testbar)
            t_b, t_p = dfl_eval.degrade_test(b, p, bl, truth, tm)
            both("DFL {} testbar".format(m), t_b, t_p, truth, bl, tm)
    for m in external_metrica.MATCHES:
        b, p, tm, truth, bl = external_metrica.load_match(m)
        both("Metrica {}".format(m), b, p, truth, bl, tm)


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--calibrate" in args:
        calibrate()
    else:
        main(args[args.index("--labels") + 1] if "--labels" in args else None, "--more" in args)
