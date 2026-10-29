"""DFL's own event logs as stand-ins for T-DEED's video events: the decoder's video weights tuned
on the seven DFL matches before the GPU run (Step 4c). Aggregates only.

The DFL open data ships each match's official event log next to the tracking (external/dfl/
DFL_03_02_events_raw_*.xml, fetched by external_dfl.py; the positions file is never needed here).
Its passes, crosses, shots, set pieces, ball claims and won tackles are ball actions of the kinds
T-DEED marks on the broadcast video. They are

  1. parsed       time, kind (as T-DEED's label), acting team (home / away from the match info)
  2. put on our frames, against the CLEAN tracking
                  The logged times are a person's: in DFL's own sample they trail the tracking by
                  2.0-3.4 s, and the lag wanders. Each half gets one clock offset -- the one that
                  puts the most kicks on kick onsets of the clean ball by the kicker's team -- and
                  then every event is snapped to the nearest frame where its team touches the
                  ball (a kick onset for a kick, the first frame at the ball otherwise): first
                  within SNAP_WIDE of the half's offset, then within SNAP of that offset corrected
                  by the median shift of its LOCAL neighbours. An event that finds no touch is
                  dropped. `check` reports the offsets, how many snapped, how far, how often the
                  logged team holds the ball at the snapped frame, and the agreement with DFL's
                  own CalculatedFrame where an event carries one (shots).
  3. as T-DEED would see them (SIM)
                  kept with probability recall per kind; timing jitter and delay; confidences;
                  false events; stretches of play without live footage (replays, close-ups)
                  with video_events.CUT_GUARD around them -- and the team NOT from the log but
                  attributed by video_events.attribute on damaged tracking, exactly as BAR-ATH's
                  events get theirs. Carries (T-DEED's DRIVE), which no event log records, come
                  from the clean tracking: one player at a moving ball for CARRY_MIN frames.
                  attr_ball "bar" attributes with a ball missing on BAR-ATH's share of live and
                  dead frames (dfl_eval.MISS_*: 7.4% of live frames), not the bench's -- whose gaps,
                  laid by clock time, miss ~37% of DFL's live frames and would leave a third of
                  the events without a team.
  4. tuned        (bench) the pipeline on the DFL matches with feat["video_events"] filled, the
                  second stage and change detector fitted leave-one-match-out over DFL + Metrica
                  (as touch_bench.py), every SETTINGS row for every PROFILES row, paired against
                  "no video", averaged over SEEDS.

    python solution/dfl_events.py check        # alignment, per match
    python solution/dfl_events.py sim-check    # the figures video_events.py check prints on
                                               # BAR-ATH, for the simulated DFL events
    python solution/dfl_events.py bench        # the tuning grid (first run fits ~30 min)

Calibrate SIM until sim-check prints what `video_events.py check` prints on BAR-ATH's real
T-DEED events -- events a minute per kind, the kick offset spread, the share of changes with an
event by the new team, PASS / DRIVE team agreement, lone wrong-team events -- the domain_check
idea, for events. Then read the bench on the "measured" profile, with "good" and "fair" around
it for how much the answer depends on the calibration.
"""

import os
import pickle
import sys
import warnings
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import external_dfl  # noqa: E402
import video_events  # noqa: E402

# Params
FPS = 25
SECTIONS = {"firstHalf": 0, "secondHalf": 1}
HALF_STARTS = (10000, 100000)   # DFL frame number of each half's first frame (CalculatedFrame)
SEARCH = 375              # frames: each half's clock offset is searched within +-15 s
SMOOTH = 4.0              # frames: Gaussian smoothing of the offset histogram
SNAP_WIDE = 50            # frames: first snap, around the half's offset
SNAP = 25                 # frames: second snap, around the neighbour-corrected offset
LOCAL = 15                # snapped events each side whose median shift corrects an event
KICK_REACH = 2.0          # m: the kicker's distance to the clean ball around a kick onset
CONTACT = 1.5             # m: nearest player this close to the clean ball = a touch
CARRY_REACH = 1.0         # m
CARRY_MIN = 12            # frames one player stays at a moving ball for a carry (DRIVE)
CARRY_SPEED = (1.0, 8.0)  # m/s, the clean ball's speed during a carry
KICK_KINDS = ("PASS", "HIGH PASS", "CROSS", "SHOT", "FREE KICK", "THROW IN")
# T-DEED on BAR-ATH, as the simulation imitates it. A starting point, NOT a measurement: set it
# from `video_events.py check` (BAR-ATH) against `dfl_events.py sim-check` (here).
SIM = {
    "recall": {"PASS": 0.8, "HIGH PASS": 0.7, "CROSS": 0.7, "SHOT": 0.7, "FREE KICK": 0.5,
               "THROW IN": 0.8, "DRIVE": 0.5, "PLAYER SUCCESSFUL TACKLE": 0.3},
    "jitter": 2.0,            # frames, standard deviation
    "delay": 0,               # frames added to every event
    "conf_true": (0.3, 0.95),  # uniform
    "conf_false": (0.05, 0.5),
    "false_per_min": 1.0,     # false events a minute of play, kinds drawn like the true ones
    "play_coverage": 0.93,    # share of in-play frames on live footage (video_events build)
    "gap_len": 125,           # frames, mean length of a stretch without it
    "cut_guard": video_events.CUT_GUARD,
    "attr_ball": "bar",       # "bar" or "bench", see the docstring
}
PROFILES = {
    "measured (SIM)": {},
    "good": {"recall": {k: min(1.0, v + 0.1) for k, v in SIM["recall"].items()},
             "jitter": 1.5, "false_per_min": 0.5},
    "fair": {"recall": {k: v * 0.75 for k, v in SIM["recall"].items()},
             "jitter": 4.0, "false_per_min": 2.0, "play_coverage": 0.85},
}
SETTINGS = {"no video": {}}
for _k in (1.0, 2.0, 3.0):
    for _hold in (0, 25, 50):
        SETTINGS["who {:.0f}, hold {}".format(_k, _hold)] = {"k_video_team": _k, "video_hold": _hold}
SETTINGS["who 2, hold 50, when 1"] = {"k_video_team": 2.0, "video_hold": 50, "k_video": 1.0}
SETTINGS["who 2, hold 50, alive first"] = {"k_video_team": 2.0, "video_hold": 50,
                                           "alive_level": "first"}
SEEDS = (0, 1)


# -- 1. parse ---------------------------------------------------------------------------------------

def event_paths(match_id):
    """(events XML, match information XML) of a DFL match, fetched if missing (never positions)."""
    def find():                        # the file kinds as external_dfl.download tells them apart
        names = os.listdir(external_dfl.DFL_DIR) if os.path.isdir(external_dfl.DFL_DIR) else []
        mine = [n for n in names if match_id in n and n.endswith(".xml")]
        info = sorted(n for n in mine if "matchinformation" in n)
        ev = sorted((n for n in mine if "matchinformation" not in n and "positions" not in n),
                    key=lambda n: ("event" not in n.lower(), n))
        return ev, info
    ev, info = find()
    if not ev or not info:
        os.makedirs(external_dfl.DFL_DIR, exist_ok=True)
        for f in external_dfl.match_ids()[1]:
            if match_id in f["name"] and "positions" not in f["name"]:
                path = os.path.join(external_dfl.DFL_DIR, f["name"])
                if not os.path.isfile(path):
                    external_dfl.fetch(f["download_url"], path)
        ev, info = find()
    if not ev or not info:
        raise FileNotFoundError("no events / match information file for " + match_id)
    return os.path.join(external_dfl.DFL_DIR, ev[0]), os.path.join(external_dfl.DFL_DIR, info[0])


def chain_of(event):
    """[(tag, attributes)] from <Event> down its nested elements (DFL nests one child a level)."""
    chain = [(event.tag, dict(event.attrib))]
    elem = event
    while len(elem):
        elem = elem[0]
        chain.append((elem.tag, dict(elem.attrib)))
    return chain


def kind_of(tags, attrs):
    """(T-DEED label, attribute naming the acting team) of a DFL event; (None, None) when it is no
    ball action (fouls, cards, substitutions, offsides, whistles, ...)."""
    if "ShotAtGoal" in tags:
        return "SHOT", "Team"
    if "Play" in tags:
        if "ThrowIn" in tags:
            return "THROW IN", "Team"
        if "FreeKick" in tags:
            return "FREE KICK", "Team"
        if "Cross" in tags or "CornerKick" in tags:
            return "CROSS", "Team"
        if attrs.get("Height") == "high":
            return "HIGH PASS", "Team"
        return "PASS", "Team"                     # open play, kick-offs, goal kicks
    if "TacklingGame" in tags:
        if attrs.get("WinnerRole") == "withoutBallControl":
            return "PLAYER SUCCESSFUL TACKLE", "WinnerTeam"
        return "DRIVE", "WinnerTeam"              # the carrier kept it: a touch of his
    if "BallClaiming" in tags or "OtherBallAction" in tags:
        return "DRIVE", "Team"
    return None, None


def parse(events_path, info_path):
    """(rows, starts, ends): rows [(time, half, kind, team 0 home / 1 away, CalculatedFrame or
    -1)] in time order; starts / ends {half: time} from the kick-off and final-whistle entries."""
    home = external_dfl.home_team_id(info_path)
    starts, ends, raw = {}, {}, []
    for event in ET.parse(events_path).getroot().iter("Event"):
        if not event.get("EventTime"):
            continue
        chain = chain_of(event)
        tags = [t for t, _ in chain]
        attrs = {}
        for _, a in reversed(chain):              # the outer element wins, as in kloppy
            attrs.update(a)
        time = datetime.fromisoformat(event.get("EventTime"))
        for tag, a in chain:
            half = SECTIONS.get(a.get("GameSection"))
            if half is not None and tag == "KickOff":
                starts[half] = time
            elif half is not None and tag == "FinalWhistle":
                ends[half] = time
        kind, key = kind_of(tags, attrs)
        if kind is None or key not in attrs:
            continue
        calc = event.get("CalculatedFrame")
        raw.append((time, kind, 0 if attrs[key] == home else 1, int(calc) if calc else -1))
    rows = []
    order = sorted(starts, key=starts.get)
    for time, kind, team, calc in sorted(raw, key=lambda r: r[0]):
        begun = [h for h in order if starts[h] <= time]
        if not begun:
            continue                              # before the first kick-off
        half = begun[-1]
        if half in ends and time > ends[half]:
            continue                              # half-time, or after the final whistle
        rows.append((time, half, kind, team, calc))
    return rows, starts, ends


# -- 2. align ---------------------------------------------------------------------------------------

def nearest_player(ball, players, team):
    """Per frame: the nearest player's team (-1: no ball / nobody) and distance (inf)."""
    d = np.hypot(players[..., 0] - ball[:, None, 0], players[..., 1] - ball[:, None, 1])
    d = np.where(np.isfinite(d), d, np.inf)
    k = np.argmin(d, axis=1)
    dist = d[np.arange(len(d)), k]
    return np.where(np.isfinite(dist), np.asarray(team)[k], -1), dist, k


def candidates(ball, players, team):
    """{(kind group, team): sorted frames}: kick onsets of the clean ball with that team's player
    within KICK_REACH just before, and the first frames of that team's runs at the ball."""
    near_team, dist, _ = nearest_player(ball, players, team)
    out = {}
    onsets = video_events.kick_onsets(ball)
    for side in (0, 1):
        kicks = []
        for o in onsets:
            lo = max(0, o - 3)
            j = lo + int(np.argmin(dist[lo:o + 1]))
            if near_team[j] == side and dist[j] <= KICK_REACH:
                kicks.append(o)
        at = (near_team == side) & (dist <= CONTACT)
        first = np.flatnonzero(at & ~np.r_[False, at[:-1]])
        out[("kick", side)] = np.array(kicks, dtype=int)
        out[("touch", side)] = first
    return out


def closest(frames, target, window):
    """The frame of `frames` (sorted) nearest target within window, or None."""
    if not len(frames):
        return None
    j = np.searchsorted(frames, target)
    best = None
    for c in frames[max(0, j - 1):j + 1]:
        if abs(c - target) <= window and (best is None or abs(c - target) < abs(best - target)):
            best = int(c)
    return best


def half_offset(frame0, groups, teams, cands):
    """The clock offset (frames) that puts the most of a half's kicks on kick onsets of their team:
    the peak of the smoothed histogram of (onset - logged frame) within +-SEARCH."""
    from scipy.ndimage import gaussian_filter1d
    diffs = []
    for f, g, tm in zip(frame0, groups, teams):
        if g != "kick":
            continue
        c = cands[("kick", tm)]
        lo, hi = np.searchsorted(c, [f - SEARCH, f + SEARCH + 1])
        diffs.append(c[lo:hi] - f)
    if not diffs or not sum(len(d) for d in diffs):
        return 0, 0.0
    hist = np.bincount(np.concatenate(diffs) + SEARCH, minlength=2 * SEARCH + 1).astype(float)
    hist = gaussian_filter1d(hist, SMOOTH)
    peak = int(np.argmax(hist))
    return peak - SEARCH, float(hist[peak] / max(np.median(hist), 1e-9))


def align(rows, starts, truth, blocks, ball, players, team):
    """(events, report). events: [(frame, kind, team, calc frame on our index or -1)] snapped onto
    the clean tracking; report: aggregates for `check`."""
    cands = candidates(ball, players, team)
    report = {"halves": [], "kinds": {}}
    events = []
    for half, (first, last) in enumerate(blocks):
        if half not in starts:
            continue
        live = np.flatnonzero(np.isin(truth[first:last + 1], ["H", "A"]))
        if not len(live):
            continue
        anchor = first + int(live[0])                # the kick-off, by the labels
        rs = [r for r in rows if r[1] == half]
        if not rs:
            continue
        frame0 = np.array([anchor + int(round((r[0] - starts[half]).total_seconds() * FPS))
                           for r in rs])
        groups = ["kick" if r[2] in KICK_KINDS else "touch" for r in rs]
        teams = [r[3] for r in rs]
        offset, sharp = half_offset(frame0, groups, teams, cands)
        base = frame0 + offset
        first_snap = [closest(cands[(g, tm)], b, SNAP_WIDE) for g, tm, b in zip(groups, teams, base)]
        resid = np.array([np.nan if s is None else s - b for s, b in zip(first_snap, base)])
        ok = np.flatnonzero(np.isfinite(resid))
        snapped, after = [], []
        for i, (g, tm, b) in enumerate(zip(groups, teams, base)):
            j = int(np.searchsorted(ok, i))
            right = j + 1 if j < len(ok) and ok[j] == i else j
            near = np.r_[ok[max(0, j - LOCAL):j], ok[right:right + LOCAL]]
            local = int(round(np.median(resid[near]))) if len(near) else 0
            s = closest(cands[(g, tm)], b + local, SNAP)
            if s is None and g == "kick":            # a restart from a ball not tracked before it
                s = closest(cands[("touch", tm)], b + local, SNAP)
            snapped.append(s)
            if s is not None:
                after.append(s - b - local)
        for r, s in zip(rs, snapped):
            k = report["kinds"].setdefault(r[2], [0, 0])
            k[0] += 1
            if s is None or not first <= s <= last:
                continue
            k[1] += 1
            calc = (r[4] - HALF_STARTS[half] + first) if r[4] >= 0 else -1
            events.append((int(s), r[2], r[3], int(calc)))
        report["halves"].append({
            "offset_s": offset / FPS, "sharpness": sharp, "events": len(rs),
            "snapped": len(after),
            "iqr_first": float(np.subtract(*np.nanpercentile(resid, [75, 25]))) if len(ok) else np.nan,
            "iqr_local": float(np.subtract(*np.percentile(after, [75, 25]))) if after else np.nan})
    events.sort()
    dedup = []
    for e in events:                              # one event per frame and team
        if dedup and dedup[-1][0] == e[0] and dedup[-1][2] == e[2]:
            continue
        dedup.append(e)
    return dedup, report


def carries(ball, players, team, truth):
    """[(frame, "DRIVE", team, -1)]: a carry starts where one player stays within CARRY_REACH of
    the clean ball, nearest it, for CARRY_MIN frames while it moves at CARRY_SPEED, in play."""
    near_team, dist, k = nearest_player(ball, players, team)
    v = np.full(len(ball), np.nan)
    v[1:-1] = np.hypot(*(ball[2:, :2] - ball[:-2, :2]).T) / 2 * FPS
    with np.errstate(invalid="ignore"):
        ok = (np.isin(truth, ["H", "A"]) & (dist <= CARRY_REACH) & (v >= CARRY_SPEED[0])
              & (v <= CARRY_SPEED[1]))
    start = ok & ~np.r_[False, ok[:-1] & (k[1:] == k[:-1])]
    out = []
    for s in np.flatnonzero(start):
        e = s
        while e + 1 < len(ok) and ok[e + 1] and k[e + 1] == k[s]:
            e += 1
        if e - s + 1 >= CARRY_MIN:
            out.append((int(s), "DRIVE", int(near_team[s]), -1))
    return out


def match_events(match_id):
    """(clean tracking, labels, blocks, aligned events + carries, report) for one DFL match."""
    ball, players, team, truth, blocks = external_dfl.load_match(match_id)
    rows, starts, ends = parse(*event_paths(match_id))
    events, report = align(rows, starts, truth, blocks, ball, players, team)
    report["carries"] = carries(ball, players, team, truth)
    return (ball, players, team), truth, blocks, events, report


def check(matches=None):
    """Alignment aggregates per match and in total."""
    import dfl_eval
    matches = matches or dfl_eval.cached_matches()
    kinds, calc, team_ok, n_live = {}, [], [], 0
    for m in matches:
        (ball, players, team), truth, blocks, events, rep = match_events(m)
        for h in rep["halves"]:
            print("{} half: clock offset {:+.2f} s (peak {:.1f}x the median), {} ball actions, "
                  "{} snapped; shift spread (IQR) {:.0f} frames at the half's offset, {:.0f} "
                  "after the local correction".format(m, h["offset_s"], h["sharpness"], h["events"],
                                                       h["snapped"], h["iqr_first"], h["iqr_local"]))
        for k, (n, s) in rep["kinds"].items():
            kinds.setdefault(k, [0, 0])
            kinds[k][0] += n
            kinds[k][1] += s
        for f, kind, tm, c in events:
            if c >= 0:
                calc.append(f - c)
            if kind in KICK_KINDS and truth[min(f + 2, len(truth) - 1)] in ("H", "A"):
                team_ok.append((truth[min(f + 2, len(truth) - 1)] == "H") == (tm == 0))
        n_live += np.isin(truth, ["H", "A"]).sum()
        print("{}: {} snapped events, {} carries ({:.1f} a minute of play)".format(
            m, len(events), len(rep["carries"]),
            len(rep["carries"]) / max(np.isin(truth, ["H", "A"]).sum() / FPS / 60, 1e-9)))
    print("snapped, per kind: " + ", ".join("{} {}/{} ({:.0%})".format(k, s, n, s / max(n, 1))
                                             for k, (n, s) in sorted(kinds.items())))
    if team_ok:
        print("kicks whose logged team holds the ball 2 frames later by the labels: {:.1%} of {}"
              .format(np.mean(team_ok), len(team_ok)))
    if calc:
        calc = np.array(calc)
        print("against DFL's CalculatedFrame ({} events): median {:+.0f} frames, within +-3 "
              "{:.0%}, within +-12 {:.0%}".format(len(calc), np.median(calc),
                                                  np.mean(np.abs(calc) <= 3),
                                                  np.mean(np.abs(calc) <= 12)))


# -- 3. simulate ------------------------------------------------------------------------------------

def attribution_ball(clean, damaged, truth, blocks, rng, how):
    """The ball T-DEED's events are attributed with: "bench" the bench's damaged ball; "bar" the
    damaged ball where it exists, else the clean one, then missing on BAR-ATH's share of live and
    dead frames (dfl_eval.state_seen)."""
    if how == "bench":
        return damaged
    import dfl_eval
    out = np.where(np.isfinite(damaged[:, :1]), damaged, clean).astype(float)
    for first, last in blocks:
        seen = dfl_eval.state_seen(np.asarray(truth[first:last + 1]), rng)
        out[first:last + 1][~seen] = np.nan
    return out


def simulate(true_events, truth, blocks, ball, players, team, rng, sim=None):
    """T-DEED-like events [(frame, label, confidence, team)] from aligned true ones; the team is
    attributed on `ball` / `players` (video_events.attribute), as on BAR-ATH."""
    sim = dict(SIM, **(sim or {}))
    n = len(truth)
    kept = [(f, k) for f, k, _, _ in true_events if rng.random() < sim["recall"].get(k, 0.0)]
    frames = np.array([f for f, _ in kept], dtype=int)
    frames = frames + int(sim["delay"]) + np.round(rng.normal(0, sim["jitter"], len(kept))).astype(int)
    kinds = [k for _, k in kept]
    confs = rng.uniform(*sim["conf_true"], len(kept))
    live = np.flatnonzero(np.isin(truth, ["H", "A"]))
    n_false = rng.poisson(sim["false_per_min"] * len(live) / FPS / 60) if len(live) else 0
    if n_false:
        frames = np.r_[frames, rng.choice(live, n_false)]
        kinds = kinds + list(rng.choice(kinds or ["PASS"], n_false))
        confs = np.r_[confs, rng.uniform(*sim["conf_false"], n_false)]
    graded = np.zeros(n, bool)
    for first, last in blocks:
        graded[first:last + 1] = True
    shown = graded.copy()
    # stretches without live footage reaching into play, as many as leave play_coverage of the
    # in-play frames on live footage (before the guard), each widened by cut_guard
    n_gaps = rng.poisson(len(live) * (1 - sim["play_coverage"]) / max(sim["gap_len"], 1)) \
        if len(live) else 0
    guard = int(sim["cut_guard"])
    for s in rng.choice(live, n_gaps) if n_gaps else []:
        length = 1 + int(rng.exponential(max(sim["gap_len"] - 1, 1e-9)))
        shown[max(0, s - guard):s + length + guard] = False
    keep = [(int(f), k, float(c)) for f, k, c in zip(frames, kinds, confs) if 0 <= f < n and shown[f]]
    rows = video_events.attribute(keep, ball, players, team)
    return sorted((t, lab, round(c, 2), tm) for t, lab, c, tm, _ in rows)


def sim_check(matches=None, sim=None, seed=0):
    """video_events.stats over the simulated events of every match, pooled, printed like
    `video_events.py check` prints BAR-ATH's."""
    import dfl_eval
    matches = matches or dfl_eval.cached_matches()
    sim = dict(SIM, **(sim or {}))
    pooled = None
    for i, m in enumerate(matches):
        (ball, players, team), truth, blocks, events, rep = match_events(m)
        d_ball, d_players = dfl_eval.degrade(ball, players, blocks, truth=truth, team=team)
        rng = np.random.default_rng(1000 * seed + i)
        a_ball = attribution_ball(ball, d_ball, truth, blocks, rng, sim["attr_ball"])
        sim_events = simulate(events + rep["carries"], truth, blocks, a_ball, d_players, team,
                              rng, sim)
        graded = (truth != "X").sum() / FPS / 60
        pooled = pool(pooled, video_events.stats(sim_events, a_ball, truth, graded), graded)
    video_events.print_stats(pooled, "DFL ({} matches)".format(len(matches)))


def pool(acc, s, minutes):
    """Pool two video_events.stats dicts (counts summed, rates re-weighted by minutes)."""
    s = dict(s, minutes=minutes)
    if acc is None:
        return s
    out = dict(acc)
    m0, m1 = acc["minutes"], minutes
    out["minutes"] = m0 + m1
    out["n"] = acc["n"] + s["n"]
    out["per_min"] = out["n"] / out["minutes"]
    na, ns = acc["attributed"] * acc["n"], s["attributed"] * s["n"]
    out["attributed"] = (na + ns) / max(out["n"], 1)
    kinds = set(acc["per_kind"]) | set(s["per_kind"])
    out["per_kind"] = {k: (acc["per_kind"].get(k, 0) * m0 + s["per_kind"].get(k, 0) * m1)
                       / out["minutes"] for k in kinds}
    out["offsets"] = np.r_[acc["offsets"], s["offsets"]]
    out["changes"] = acc["changes"] + s["changes"]
    out["hits"] = {w: acc["hits"][w] + s["hits"][w] for w in acc["hits"]}
    n0, n1 = acc["pass_drive_n"], s["pass_drive_n"]
    out["pass_drive_n"] = n0 + n1
    out["pass_drive_right"] = ((np.nan_to_num(acc["pass_drive_right"]) * n0
                                + np.nan_to_num(s["pass_drive_right"]) * n1) / max(n0 + n1, 1))
    out["delays"] = np.r_[acc["delays"], s["delays"]]
    p0, p1 = acc["play_minutes"], s["play_minutes"]
    out["play_minutes"] = p0 + p1
    out["lone_per_min"] = (acc["lone_per_min"] * p0 + s["lone_per_min"] * p1) / max(p0 + p1, 1e-9)
    return out


# -- 4. bench ---------------------------------------------------------------------------------------

def fills(data):
    """seq_team / seq_dead / change_logodds leave-one-match-out over DFL + Metrica
    (touch_bench.fill), cached per variant: video evidence never changes them."""
    import touch_bench
    import variant
    path = variant.cache_path(os.path.join(ROOT, "cache", "bench_video_fills.pkl"))
    keys = ("seq_team", "seq_dead", "change_logodds")
    if os.path.isfile(path):
        with open(path, "rb") as handle:
            saved = pickle.load(handle)
        for k, v in data.items():
            v[5].update(saved[k])
        return
    touch_bench.fill(data)
    with open(path, "wb") as handle:
        pickle.dump({k: {key: v[5][key] for key in keys} for k, v in data.items()}, handle)


def scores(truth, labels):
    import possession_metrics as pm
    r = pm.evaluate(pm.label_codes(truth), pm.label_codes(labels), bootstrap_n=0)
    return np.array([r["spell_f1_mean"], r["macro_f1"], r["per_class"]["D"]["f1"],
                     r["n_pred_spells"] / max(r["n_gt_spells"], 1)])


def bench(profiles=PROFILES, settings=SETTINGS, seeds=SEEDS):
    import dfl_eval
    import touch_bench
    data = touch_bench.first_stages()
    fills(data)
    dfl = sorted((m, v) for (source, m), v in data.items() if source == "dfl")
    aligned = {m: match_events(m) for m, _ in dfl}
    blind = {name for name, params in settings.items()
             if not params.get("k_video_team") and not params.get("k_video")}
    fixed = {(name, m): scores(truth, dfl_eval.run_pipeline(b, p, tm, bl, settings[name], feat))
             for name in blind for m, (b, p, tm, truth, bl, feat) in dfl}   # no video: decode once
    print("{:<34} {:<52} {}".format("", "DFL spellF1 (paired diff, matches up)  per match",
                                    "macroF1  D-F1   poss/truth"))
    for pname, overrides in profiles.items():
        sim = dict(SIM, **overrides)
        per = {name: [] for name in settings}
        for seed in seeds:
            for i, (m, (b, p, tm, truth, bl, feat)) in enumerate(dfl):
                (ball, players, team), _, _, events, rep = aligned[m]
                rng = np.random.default_rng(1000 * seed + i)
                a_ball = attribution_ball(ball, b, truth, bl, rng, sim["attr_ball"])
                feat["video_events"] = simulate(events + rep["carries"], truth, bl, a_ball, p, tm,
                                                rng, sim)
                for name, params in settings.items():
                    per[name].append(fixed[(name, m)] if name in blind else scores(
                        truth, dfl_eval.run_pipeline(b, p, tm, bl, params, feat)))
                feat.pop("video_events", None)
        print("-- profile: {}".format(pname))
        base = np.array(per["no video"])
        n = len(dfl)
        for name, rows in per.items():
            rows = np.array(rows)
            diff = (rows[:, 0] - base[:, 0]).reshape(len(seeds), n).mean(axis=0)
            match = rows[:, 0].reshape(len(seeds), n).mean(axis=0)
            print("{:<34} {:.4f} ({:+.4f}, {} of {})  {}   {:.4f}   {:.4f}  {:.2f}".format(
                name, rows[:, 0].mean(), diff.mean(), int((diff > 0).sum()), n,
                " ".join("{:.3f}".format(v) for v in match), rows[:, 1].mean(), rows[:, 2].mean(),
                rows[:, 3].mean()), flush=True)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "check":
        check(sys.argv[2:] or None)
    elif command == "sim-check":
        sim_check(sys.argv[2:] or None)
    elif command == "bench":
        bench()
    else:
        print(__doc__)
