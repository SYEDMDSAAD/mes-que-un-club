"""Ball actions spotted on the broadcast video (T-DEED), moved onto the tactical timeline.

T-DEED's SoccerNet ball-action model (github.com/arturxe2/T-DEED, GPL-3.0; checkpoint
SoccerNetBall_challenge1 or _challenge2) marks the frame of every PASS, DRIVE, HEADER, HIGH PASS,
OUT, CROSS, THROW IN, SHOT, BALL PLAYER BLOCK, PLAYER SUCCESSFUL TACKLE, FREE KICK and GOAL. Each is
a touch (or the ball leaving the pitch) -- the moment the tracking blurs. Run it on the broadcast
video, where the ball is large and in focus, on your own machine (the video never leaves it), then:

    python solution/video_events.py build <T-DEED predictions .json>   # -> cache/video_events.csv
    python solution/video_events.py check                              # aggregates only

build
  1. broadcast frame -> tactical frame with broadcast.OFFSETS (one constant per half, stable to
     +-1 frame), only where the broadcast frame is live footage by the same fit as the ball fusion
     (replays, close-ups and crowd shots match no tactical frame: a replayed pass must not land on
     the wrong second), and not within CUT_GUARD frames of footage that is not live (T-DEED's 8-s
     context straddles the cut there). Graded frames only.
  2. the acting team: the team of the player nearest the ball (fused ball) at the mapped frame,
     the closest over +-ATTR_WINDOW frames; -1 if no ball or nobody within ATTR_REACH.
  3. quantised and sorted (frame, label, confidence to 2 decimals, team, distance to 0.1 m), so a
     rerun on other hardware writes the same bytes. Its SHA-256 is printed: put it in
     EXPECTED_SHA256 and load() refuses any other file.
  The CSV is derived from the gated video: it stays in cache/ (gitignored), never in the repo.

check (aggregates only; stats() computes them, for dfl_events.py sim-check too)
  per kind   events a graded minute per T-DEED label
  timing     offset from each kick-type event to the nearest kick onset in the tactical ball,
             where the ball is tracked: the median is a constant to remove (EVENT_SHIFT), the
             spread is T-DEED's timing error on this match
  public     on the public window: the share of true team changes with an event by the new team
             within +-6 / 12 / 25 frames, how often a PASS / DRIVE team matches the possession,
             the new team's first event after a change, and events by the team without the ball
             away from any change (false events or a wrong team)
  build also prints how the live footage covers the graded frames and play (coverage()).

Decoder evidence, two kinds:
  who   (model.emissions, PARAMS k_video_team > 0): team log-odds +-k_video_team x confidence
        for the touching team over [event - TEAM_PRE, event + TEAM_POST], and with video_hold > 0
        on until the next touch (team_evidence). This is the one that matters: perfect frame-by-
        frame team evidence is worth far more than perfect change times
  when  (model.switch_costs, PARAMS k_video > 0): a change to team T is made cheaper from
        VIDEO_PRE frames before to VIDEO_POST frames after an event by T, by k_video x confidence,
        twice that when the previous attributed event was by the other team, never below video_min

Simulated on a Metrica replica with the ball made to look like ours (pseudo_events, 3 matches,
spellF1 0.594 without): who 2 + hold 50 gained +0.068 with "good" events (80% of changes caught,
+-2 frames, 5% wrong team, 0.5 false per minute, up to 12 frames late) and +0.038 with "fair" ones
(60%, +-4, 15%, 1.5); without hold +0.022 / +0.019; "when" alone at most +0.01; "poor" events
(40%, +-8, 30%, 3) nothing or worse. With touch features in the second stage as well (touchfeat,
+0.042 alone): 0.704 with good events (who 3, hold 50), 0.660 with fair ones. So `check` decides
whether this is worth it. Tune on DFL -- dfl_events.py: DFL's own event logs, aligned to the
tracking and degraded to look like what `check` measures on BAR-ATH -- not on the public window.
(pseudo_events below draws touches at random from the labels instead; kept for the replica.)

Producing T-DEED's output reproducibly: video/tdeed_infer.py (num_workers 0, fp32, a counted
frame total checked against the broadcast tracking's 164,488, rounded scores).
"""

import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]

# Params
CSV = os.path.join(ROOT, "cache", "video_events.csv")
EXPECTED_SHA256 = None     # set to the hash `build` prints, once you adopt a file
CUT_GUARD = 50             # frames
ATTR_WINDOW = 3            # frames
ATTR_REACH = 3.0           # m
EVENT_SHIFT = 0            # frames added to every event after `check` measured the offset
KICKS = ("PASS", "HIGH PASS", "CROSS", "SHOT", "FREE KICK", "THROW IN")
TOUCHES = KICKS + ("DRIVE", "HEADER", "BALL PLAYER BLOCK", "PLAYER SUCCESSFUL TACKLE")
VIDEO_PRE = 25             # frames before an event over which a change to its team is cheaper
VIDEO_POST = 3
TEAM_PRE = 3               # frames around an event that count as its team's (team_evidence)
TEAM_POST = 6
FPS = 25


# -- build ------------------------------------------------------------------------------------------

def read_predictions(path):
    """[(broadcast frame, label, confidence)] from T-DEED's inference JSON."""
    with open(path) as handle:
        data = json.load(handle)
    preds = data["predictions"] if isinstance(data, dict) else data
    return [(int(p["frame"]), str(p["label"]), float(p["confidence"])) for p in preds]


def live_mask(cost, b_index, live_cost, relax_share, rolling_mean, window):
    """Tactical frames whose aligned broadcast frame is live footage (the ball fusion's test)."""
    tested = np.isfinite(cost)
    share = rolling_mean(np.where(tested, (cost < live_cost).astype(float), np.nan), window)
    return (b_index >= 0) & ((cost < live_cost) | (share > relax_share))


def guard(live, n_guard):
    """live, minus every frame within n_guard frames of a frame that is not live."""
    bad = (~live).astype(np.int32)
    c = np.concatenate(([0], np.cumsum(bad)))
    i = np.arange(len(live))
    lo, hi = np.maximum(0, i - n_guard), np.minimum(len(live), i + n_guard + 1)
    return live & ((c[hi] - c[lo]) == 0)


def to_tactical(preds, offsets, blocks, live, n_tactical):
    """Each prediction on the tactical timeline: (tactical frame, label, conf), live graded only.

    A broadcast frame b maps to b + offset of the half whose graded frames contain it.
    """
    out = []
    for b, label, conf in preds:
        for (first, last), offset in zip(blocks, offsets):
            t = b + offset + EVENT_SHIFT
            if first <= t <= last and 0 <= t < n_tactical and live[t]:
                out.append((t, label, conf))
                break
    return out


def attribute(events, ball, players, team):
    """(frame, label, conf, team, dist): the team of the nearest player around each event."""
    out = []
    n = len(ball)
    for t, label, conf in events:
        lo, hi = max(0, t - ATTR_WINDOW), min(n, t + ATTR_WINDOW + 1)
        b = ball[lo:hi, :2]
        d = np.hypot(players[lo:hi, :, 0] - b[:, None, 0], players[lo:hi, :, 1] - b[:, None, 1])
        d = np.where(np.isfinite(d), d, np.inf)
        if not np.isfinite(d).any():
            out.append((t, label, conf, -1, np.nan))
            continue
        k = np.unravel_index(np.argmin(d), d.shape)
        dist = float(d[k])
        out.append((t, label, conf, int(team[k[1]]) if dist < ATTR_REACH else -1, dist))
    return out


def write_csv(rows, path=CSV):
    """Quantised, sorted, written; returns the SHA-256 of the bytes written."""
    rows = sorted((int(t), label, round(conf, 2), int(tm), d) for t, label, conf, tm, d in rows)
    lines = ["frame,label,confidence,team,dist"]
    for t, label, conf, tm, d in rows:
        lines.append("{},{},{:.2f},{},{}".format(t, label, conf, tm,
                                                 "" if not np.isfinite(d) else "{:.1f}".format(d)))
    data = ("\n".join(lines) + "\n").encode()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(data)
    return hashlib.sha256(data).hexdigest()


def build(json_path):
    import broadcast
    import load as loader
    from features import rolling_mean
    from games import GAME, half_video_blocks
    ball, players, team = loader.load()
    blocks = [(first, last) for _, first, last in half_video_blocks(GAME)]
    b_index, cost, _ = broadcast.aligned()
    live = live_mask(cost, b_index, broadcast.LIVE_COST, 0.8, rolling_mean,
                     broadcast.RELAX_WINDOW)
    coverage(live, blocks, ball)
    live = guard(live, CUT_GUARD)
    events = to_tactical(read_predictions(json_path), broadcast.OFFSETS, blocks, live, len(ball))
    sha = write_csv(attribute(events, ball, players, team))
    print("{} events on live, graded footage -> {}\nsha256 {}".format(len(events), CSV, sha))
    return sha


def coverage(live, blocks, ball=None):
    """Print how the live footage covers the graded frames (before CUT_GUARD), overall and where
    the tracked ball moves faster than 3 m/s over 1 s -- a proxy for play, which is where a gap
    costs events: dfl_events.SIM's play_coverage and gap_len imitate the second. Aggregates only."""
    graded = np.zeros(len(live), bool)
    for first, last in blocks:
        graded[first:last + 1] = True
    off = graded & ~live
    edges = np.flatnonzero(np.diff(np.r_[0, off.astype(np.int8), 0]))
    starts, stops = edges[::2], edges[1::2]
    print("live footage covers {:.1%} of graded frames; {} stretches without it, median {:.1f} s"
          .format(1 - off.sum() / max(graded.sum(), 1), len(starts),
                  np.median(stops - starts) / FPS if len(starts) else 0.0))
    if ball is None:
        return
    from features import rolling_mean
    v = np.full(len(ball), np.nan)
    v[1:] = np.hypot(*np.diff(ball[:, :2].astype(float), axis=0).T) * FPS
    with np.errstate(invalid="ignore"):
        play = graded & (rolling_mean(np.clip(np.nan_to_num(v, nan=0.0), 0, 40), FPS // 2) > 3.0)
    touching = [(a, b) for a, b in zip(starts, stops) if play[a:b].any()]
    lengths = np.array([b - a for a, b in touching])
    print("where the ball moves (> 3 m/s over 1 s, a proxy for play): live footage on {:.1%}; {} "
          "stretches without it reach into play, median {:.1f} s, mean {:.1f} s "
          "(dfl_events.SIM play_coverage, gap_len)".format(
              1 - (off & play).sum() / max(play.sum(), 1), len(touching),
              np.median(lengths) / FPS if len(lengths) else 0.0,
              np.mean(lengths) / FPS if len(lengths) else 0.0))


def load(path=CSV):
    """Events as an int/float structured array: frame, label, confidence, team."""
    with open(path, "rb") as handle:
        data = handle.read()
    if EXPECTED_SHA256 and hashlib.sha256(data).hexdigest() != EXPECTED_SHA256:
        raise ValueError("{} is not the adopted video-event file (SHA-256 differs)".format(path))
    rows = [line.split(",") for line in data.decode().strip().split("\n")[1:]]
    return [(int(r[0]), r[1], float(r[2]), int(r[3])) for r in rows]


# -- decoder evidence --------------------------------------------------------------------------------

def switch_relief(n, events, k, labels=TOUCHES, pre=VIDEO_PRE, post=VIDEO_POST):
    """(to_away, to_home): how much cheaper a change to each team is on every frame."""
    relief = {0: np.zeros(n), 1: np.zeros(n)}
    attributed = [(t, lab, c, tm) for t, lab, c, tm in events if tm in (0, 1) and lab in labels]
    attributed.sort()
    previous = None
    for t, lab, conf, tm in attributed:
        weight = k * conf * (2.0 if previous is not None and previous != tm else 1.0)
        lo, hi = max(0, t - pre), min(n, t + post + 1)
        np.maximum(relief[tm][lo:hi], weight, out=relief[tm][lo:hi])
        previous = tm
    return relief[1], relief[0]


def team_evidence(n, events, k, labels=TOUCHES, pre=TEAM_PRE, post=TEAM_POST, hold=0):
    """Team log-odds (H vs A) per frame from the events: +k x confidence for a home touch over
    [frame - pre, frame + post], -k x confidence for an away one. A touch by team T means T has
    the ball at that moment under the ground truth's convention (a change sits on the new team's
    first touch) -- evidence about *who*, which the tracking lacks exactly where it is wrong.

    hold > 0 carries a touch's evidence on until the next attributed touch, at most hold frames:
    after a pass the passer's team keeps the ball until someone touches it -- the rule your
    event-built Metrica labels follow. Overlapping evidence of both teams cancels; per team the
    strongest counts."""
    best = {0: np.zeros(n), 1: np.zeros(n)}
    touches = sorted((t, conf, tm) for t, lab, conf, tm in events if tm in (0, 1) and lab in labels)
    for i, (t, conf, tm) in enumerate(touches):
        stop = t + post
        if hold > 0:
            nxt = touches[i + 1][0] if i + 1 < len(touches) else n
            stop = max(stop, min(t + hold, nxt - 1))
        lo, hi = max(0, t - pre), min(n, stop + 1)
        np.maximum(best[tm][lo:hi], k * conf, out=best[tm][lo:hi])
    return best[0] - best[1]


def pseudo_events(truth, rng, recall=0.7, jitter=2.0, team_error=0.1, fp_per_min=1.0,
                  touches_per_s=0.4, conf=(0.3, 0.9), delay=0):
    """Events simulated from a labelled match, for tuning the decoder where no video exists (DFL).

    An event of the new team at every true team change (kept with probability `recall`), events
    of the possessing team at `touches_per_s` inside possessions, timing jitter (frames, normal),
    the team wrong with probability `team_error`, false events at `fp_per_min` of live play.
    `delay` (frames): the new team's event comes up to that much after the change -- T-DEED has
    no reception class, so a controlled reception shows up as the receiver's PASS or DRIVE.
    Set the rates from what `check` measures on BAR-ATH.
    """
    n = len(truth)
    live = np.isin(truth, ["H", "A"])
    code = np.where(truth == "H", 0, 1)
    times, teams = [], []
    change = np.flatnonzero(live[1:] & (truth[1:] != truth[:-1])) + 1
    times += (change + (rng.integers(0, delay + 1, len(change)) if delay else 0)).tolist()
    teams += code[change].tolist()
    inside = np.flatnonzero(live & (rng.random(n) < touches_per_s / FPS))
    times += inside.tolist()
    teams += code[inside].tolist()
    keep = rng.random(len(times)) < recall
    times = np.array(times)[keep]
    teams = np.array(teams)[keep]
    times = np.clip(np.round(times + rng.normal(0, jitter, len(times))).astype(int), 0, n - 1)
    flip = rng.random(len(teams)) < team_error
    teams = np.where(flip, 1 - teams, teams)
    n_fp = rng.poisson(fp_per_min * live.sum() / FPS / 60)
    fp_t = rng.choice(np.flatnonzero(live), size=n_fp) if n_fp and live.any() else np.zeros(0, int)
    times = np.r_[times, fp_t]
    teams = np.r_[teams, rng.integers(0, 2, len(fp_t))]
    confs = np.round(rng.uniform(*conf, len(times)), 2)
    return sorted((int(t), "PASS", float(c), int(tm)) for t, c, tm in zip(times, confs, teams))


# -- check -------------------------------------------------------------------------------------------

def kick_onsets(ball, k=4, rise=3.0):
    """Frames where the ball's speed over the next k frames exceeds that over the last k by
    `rise` m/s, local maxima only."""
    b = ball[:, :2].astype(float)
    v_in = np.full(len(b), np.nan)
    v_out = np.full(len(b), np.nan)
    v_in[k:] = np.hypot(*(b[k:] - b[:-k]).T) / k * FPS
    v_out[:-k] = np.hypot(*(b[k:] - b[:-k]).T) / k * FPS
    gain = np.nan_to_num(v_out - v_in, nan=0.0)
    peak = gain >= rise
    peak[1:-1] &= (gain[1:-1] >= gain[:-2]) & (gain[1:-1] >= gain[2:])
    return np.flatnonzero(peak)


def stats(events, ball, gt, minutes):
    """The figures `check` prints, as a dict -- one function for BAR-ATH's real events and for
    the DFL stand-ins (dfl_events.py sim-check), so the two are measured the same way.

    events   (frame, label, confidence, team) rows
    ball     the tracked ball the events are timed against (ours; the damaged one on DFL)
    gt       labels H / A / D on the frames to score against, 'X' elsewhere
    minutes  graded minutes the events were looked for in
    """
    frames = np.array([e[0] for e in events], dtype=int)
    labels = np.array([e[1] for e in events], dtype=object)
    teams = np.array([e[3] for e in events], dtype=int)
    seen = np.isfinite(ball[:, 0])
    onsets = kick_onsets(ball)
    offs = []
    for t, lab in zip(frames, labels):
        if lab in KICKS and seen[max(0, t - 15):t + 16].mean() > 0.8 and len(onsets):
            j = np.searchsorted(onsets, t)
            o = min(onsets[max(0, j - 1):j + 1], key=lambda x: abs(x - t))
            if abs(o - t) <= 15:
                offs.append(o - t)
    out = {"n": len(events), "per_min": len(events) / max(minutes, 1e-9),
           "attributed": float(np.mean(teams >= 0)) if len(events) else np.nan,
           "per_kind": {k: float(np.sum(labels == k)) / max(minutes, 1e-9)
                        for k in sorted(set(labels.tolist()))},
           "offsets": np.array(offs)}
    gt = np.asarray(gt)
    live = np.isin(gt, ["H", "A"])
    changes = np.array([i for i in np.flatnonzero(gt[1:] != gt[:-1]) + 1
                        if gt[i - 1] in ("H", "A") and gt[i] in ("H", "A")], dtype=int)
    new_team = np.where(gt[changes] == "H", 0, 1) if len(changes) else np.zeros(0, int)
    out["changes"] = len(changes)
    out["hits"] = {}
    for w in (6, 12, 25):
        out["hits"][w] = int(sum(np.any((np.abs(frames - i) <= w) & (teams == tm))
                                 for i, tm in zip(changes, new_team)))
    inside = [(t, tm) for t, lab, tm in zip(frames, labels, teams)
              if lab in ("PASS", "DRIVE") and tm >= 0 and live[t]]
    out["pass_drive_n"] = len(inside)
    out["pass_drive_right"] = (float(np.mean([(gt[t] == "H") == (tm == 0) for t, tm in inside]))
                               if inside else np.nan)
    delays = []
    for i, tm in zip(changes, new_team):
        after = frames[(teams == tm) & (frames >= i - 6) & (frames <= i + 50)]
        if len(after):
            delays.append(int(after.min() - i))
    out["delays"] = np.array(delays)
    near_change = np.zeros(len(gt), bool)
    for i in changes:
        near_change[max(0, i - 25):i + 26] = True
    lone = [t for t, tm in zip(frames, teams) if live[t] and not near_change[t] and tm >= 0
            and (gt[t] == "H") != (tm == 0)]
    out["play_minutes"] = live.sum() / FPS / 60
    out["lone_per_min"] = len(lone) / max(out["play_minutes"], 1e-9)
    return out


def print_stats(s, scope):
    """`stats` in words. scope names the labelled frames ("public window", "DFL")."""
    print("{} events, {:.1f} per graded minute; attributed to a team: {:.0%}".format(
        s["n"], s["per_min"], s["attributed"]))
    print("per kind, per graded minute: " + ", ".join(
        "{} {:.2f}".format(k, v) for k, v in sorted(s["per_kind"].items(), key=lambda kv: -kv[1])))
    offs = s["offsets"]
    if len(offs):
        print("kick-type events with a tracked kick within 15 frames: {}; offset (kick - event) "
              "median {:+.0f}, IQR {:.0f}..{:.0f}, within +-3 frames {:.0%}".format(
                  len(offs), np.median(offs), *np.percentile(offs, [25, 75]),
                  (np.abs(offs) <= 3).mean()))
    for w, k in s["hits"].items():
        print("{}: true team changes with an event by the new team within +-{:>2}: {} of {} "
              "({:.0%})".format(scope, w, k, s["changes"], k / max(s["changes"], 1)))
    if s["pass_drive_n"]:
        print("{}: PASS / DRIVE events whose team is the possessing team: {:.0%} of {}".format(
            scope, s["pass_drive_right"], s["pass_drive_n"]))
    d = s["delays"]
    if len(d):
        print("{}: first event of the new team after a change: median {:+.0f} frames, max {:+d} "
              "({} of {} changes)".format(scope, np.median(d), int(d.max()), len(d), s["changes"]))
    print("{}: events by the team without the ball, away from any change: {:.1f} a minute of play "
          "(false events or wrong team)".format(scope, s["lone_per_min"]))


def check(path=CSV):
    import pandas as pd
    import load as loader
    from games import GAME, half_video_blocks
    events = load(path)
    ball, _, _ = loader.load()
    graded = sum(last - first + 1 for _, first, last in half_video_blocks(GAME))
    gt = pd.read_csv(os.path.join(ROOT, "evaluation", "gt",
                                  "BAR-ATH_possession_gt_public.csv"))["label"].values
    print_stats(stats(events, ball, gt, graded / FPS / 60), "public window")
    print("-> calibrate dfl_events.SIM until `python solution/dfl_events.py sim-check` prints "
          "figures like these (per kind, kick offset, hits, PASS / DRIVE team, lone events)")


if __name__ == "__main__":
    if sys.argv[1:2] == ["build"] and len(sys.argv) == 3:
        build(sys.argv[2])
    elif sys.argv[1:2] == ["check"]:
        check()
    else:
        print(__doc__)
