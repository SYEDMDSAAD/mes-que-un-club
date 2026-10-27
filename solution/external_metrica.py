"""Metrica's public sample matches as a second test bed -- tracking from the company behind our
ground truth, with possession labels rebuilt from their event data.

Three matches (github.com/metrica-sports/sample-data; "be responsible, acknowledge the source"):
Metrica tracking at 25 fps in pitch-normalised coordinates, and hand-tagged events with start and
end frames. They have no possession labels, so labels() rebuilds them from the events:

    PASS / SHOT / BALL OUT by T     T has the ball from the start frame to the end frame
    RECOVERY / BALL LOST by T       T has the ball at the start frame
    BALL OUT, a goal, END HALF      the ball is dead from the end frame ...
    FAULT RECEIVED                  ... or from the foul's start frame ...
    SET PIECE by T                  ... until the set piece, which T takes
    CHALLENGE, CARD                 skipped: their team attribution is ambiguous

Between two events possession stays with the last team that had the ball -- the challenge's own
rule for a loose ball -- so a change happens at the first event of the new team. The labels are
approximate (events mark moments, not frames of first control), but they follow Metrica's own
tagging, not DFL's.

The arrays match external_dfl's, so the whole pipeline and dfl_eval's tools run on them:

    ball      (N, 3)       x, y in metres from a corner, NaN speed
    players   (N, S, 3)    x, y, speed in m/s per person
    team      (S,)         0 home, 1 away
    labels    (N,)         'H' / 'A' / 'D', and 'X' between the halves
    blocks    (2, 2)       first and last frame of each half

    python solution/external_metrica.py        # convert the three matches, print aggregates
"""

import csv
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Params
DIR = os.path.join(ROOT, "external", "metrica")
CACHE_DIR = os.path.join(ROOT, "cache", "metrica")
URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/"
PITCH = (105.0, 68.0)
FPS = 25
HALF_GAP = 5000           # frames of 'X' between the halves, as for DFL
SPEED_HALF = 2            # frames each side for player speed
REFINE_REACH = 2.0        # m: a player this close and nearest the ball has reached it
REFINE_MAX_BACK = 75      # frames: how far back a change may move (3 s)
MATCHES = ("Sample_Game_1", "Sample_Game_2", "Sample_Game_3")
FILES = {"Sample_Game_1": ["RawEventsData.csv", "RawTrackingData_Home_Team.csv",
                           "RawTrackingData_Away_Team.csv"],
         "Sample_Game_2": ["RawEventsData.csv", "RawTrackingData_Home_Team.csv",
                           "RawTrackingData_Away_Team.csv"],
         "Sample_Game_3": ["events.json", "tracking.txt", "metadata.xml"]}
HOME_NAMES = ("Home", "Team A")


def download(match):
    import urllib.request
    os.makedirs(DIR, exist_ok=True)
    for name in FILES[match]:
        path = os.path.join(DIR, "{}_{}".format(match, name))
        if not os.path.isfile(path):
            urllib.request.urlretrieve("{}{}/{}_{}".format(URL, match, match, name), path)


def path(match, name):
    return os.path.join(DIR, "{}_{}".format(match, name))


# --- events ---------------------------------------------------------------------------------

def events(match):
    """[(period, start, end, type, subtype, home)] sorted by period and start frame."""
    out = []
    if match == "Sample_Game_3":
        for e in json.load(open(path(match, "events.json")))["data"]:
            sub = e.get("subtypes") or {}
            sub = sub.get("name", "") if isinstance(sub, dict) else "-".join(s["name"] for s in sub)
            end = (e.get("end") or {}).get("frame") or e["start"]["frame"]
            out.append((e["period"], e["start"]["frame"], end, e["type"]["name"], sub,
                        e["team"]["name"] in HOME_NAMES))
    else:
        with open(path(match, "RawEventsData.csv")) as handle:
            for r in csv.DictReader(handle):
                out.append((int(r["Period"]), int(r["Start Frame"]), int(r["End Frame"]),
                            r["Type"], r["Subtype"] or "", r["Team"] in HOME_NAMES))
    return sorted(out, key=lambda e: (e[0], e[1]))


def ends_dead(kind, sub):
    return (kind == "BALL OUT" or "END HALF" in sub
            or (kind == "SHOT" and ("GOAL" in sub or "OUT" in sub) and "SAVED" not in sub))


# --- tracking --------------------------------------------------------------------------------

def read_csv_team(file):
    """(period, frame, players (F, K, 2), ball (F, 2)) from one of Metrica's CSV tracking files."""
    with open(file) as handle:
        reader = csv.reader(handle)
        next(reader), next(reader)
        header = next(reader)
        rows = np.array([[float(v) if v not in ("", "NaN") else np.nan for v in row]
                         for row in reader], dtype=np.float64)
    n_players = (len(header) - 5) // 2
    players = rows[:, 3:3 + 2 * n_players].reshape(len(rows), n_players, 2)
    return rows[:, 0].astype(int), rows[:, 1].astype(int), players, rows[:, -2:]


def read_games_1_2(match):
    period, frame, home, ball = read_csv_team(path(match, "RawTrackingData_Home_Team.csv"))
    period_a, frame_a, away, _ = read_csv_team(path(match, "RawTrackingData_Away_Team.csv"))
    if not (np.array_equal(frame, frame_a) and np.array_equal(period, period_a)):
        raise ValueError(match + ": home and away tracking frames differ")
    team = np.r_[np.zeros(home.shape[1]), np.ones(away.shape[1])].astype(np.int8)
    return period, frame, np.concatenate([home, away], axis=1), ball, team


def read_game_3(match):
    root = ET.parse(path(match, "metadata.xml")).getroot()
    home_id = next(root.iter("Team")).get("id")
    player_team = {p.get("id"): p.get("teamId") for p in root.iter("Player")}
    channel_player = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    ids = sorted(player_team, key=lambda q: (player_team[q] != home_id, q))
    slot = {q: k for k, q in enumerate(ids)}
    team = np.array([0 if player_team[q] == home_id else 1 for q in ids], np.int8)
    halves = {}
    for p in root.iter("ProviderParameter"):
        name, value = p.findtext("Name"), p.findtext("Value")
        if name in ("first_half_start", "second_half_start") and value:
            halves[int(value)] = 1 if name.startswith("first") else 2
    specs = []                                 # (start, end, [slot per position])
    for d in root.iter("DataFormatSpecification"):
        refs = [r.get("playerChannelId") for r in d.iter("PlayerChannelRef")]
        order = [slot[channel_player[r]] for r in refs[::2]]
        specs.append((int(d.get("startFrame")), int(d.get("endFrame")), order))

    frames, rows_p, rows_b = [], [], []
    with open(path(match, "tracking.txt")) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            frames.append(int(parts[0]))
            rows_p.append([float(v) for g in parts[1].split(";") for v in g.split(",")])
            rows_b.append([float(v) for v in parts[2].split(",")[:2]])
    frame = np.array(frames)
    flat = np.array(rows_p)
    players = np.full((len(frame), len(ids), 2), np.nan)
    for start, end, order in specs:
        rows = (frame >= start) & (frame <= end)
        players[np.ix_(rows, order)] = flat[rows].reshape(rows.sum(), len(order), 2)
    second = min(f for f, h in halves.items() if h == 2)
    period = np.where(frame >= second, 2, 1)
    return period, frame, players, np.array(rows_b), team


def speed(xy):
    """m/s per frame from a centred difference, NaN where the neighbours are missing."""
    k = SPEED_HALF
    out = np.full(xy.shape[:-1], np.nan)
    d = xy[2 * k:] - xy[:-2 * k]
    out[k:-k] = np.hypot(d[..., 0], d[..., 1]) * FPS / (2 * k)
    return out


def assemble(match):
    read = read_game_3 if match == "Sample_Game_3" else read_games_1_2
    period, frame, players_xy, ball_xy, team = read(match)
    scale = np.array(PITCH)
    sections = sorted(set(period.tolist()))[:2]
    lengths = [int((period == s).sum()) for s in sections]
    total = sum(lengths) + HALF_GAP * (len(sections) - 1)
    ball = np.full((total, 3), np.nan, np.float32)
    players = np.full((total, len(team), 3), np.nan, np.float32)
    blocks, index, offset = [], {}, 0
    for s, length in zip(sections, lengths):
        rows = np.flatnonzero(period == s)
        idx = np.arange(length) + offset
        ball[idx, :2] = ball_xy[rows] * scale
        xy = players_xy[rows] * scale
        players[idx, :, :2] = xy
        players[idx, :, 2] = speed(xy)
        index[s] = (int(frame[rows[0]]), offset, length)
        blocks.append((offset, offset + length - 1))
        offset += length + HALF_GAP

    def frame_of(p, f):
        if p not in index:
            return None
        first, off, length = index[p]
        return off + min(max(f - first, 0), length - 1)

    lab = np.full(total, "X", dtype="<U1")
    evs = events(match)
    for (s, (first, off, length)), (lo, hi) in zip(index.items(), blocks):
        lab[lo:hi + 1] = period_labels([e for e in evs if e[0] == s],
                                       lambda f: frame_of(s, f) - off, length)
    return ball, players, team, lab, np.array(blocks)


def period_labels(evs, index_of, n):
    """(n,) 'H' / 'A' / 'D' for one half from its events (see the module docstring)."""
    marks = []
    for _, start, end, kind, sub, home in evs:
        side = "H" if home else "A"
        s, e = index_of(start), index_of(max(end, start))
        if kind in ("SET PIECE", "PASS", "SHOT", "BALL OUT", "RECOVERY", "BALL LOST"):
            marks.append((s, 1, side))
            if ends_dead(kind, sub):
                marks.append((e, 0, "D"))
        elif kind == "FAULT RECEIVED":
            marks.append((s, 0, "D"))
    out = np.full(n, "D", dtype="<U1")
    state, last = "D", 0
    for idx, _, new in sorted(marks):
        out[last:idx] = state
        state, last = new, idx
    out[last:] = state
    return out


def convert(match):
    out = os.path.join(CACHE_DIR, match + ".npz")
    if os.path.isfile(out):
        return out
    download(match)
    ball, players, team, lab, blocks = assemble(match)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(out, ball=ball, players=players, team=team, labels=lab, blocks=blocks)
    return out


def load_match(match):
    d = np.load(convert(match))
    return d["ball"], d["players"], d["team"], d["labels"], [tuple(b) for b in d["blocks"]]


def refine(labels, ball, players, team, reach=REFINE_REACH, max_back=REFINE_MAX_BACK):
    """Copy of labels with each direct H <-> A change moved back to the new team's arrival.

    An event-built change sits on the new team's first event, often its first pass; control
    started earlier, when its player reached the ball. Walking back from the change, every frame
    where the ball is tracked and a player of the new team is nearest it, within reach, joins the
    new possession. The walk stops at the first frame that is not, after max_back frames, or at
    the start of the old team's run.
    """
    out = labels.copy()
    with np.errstate(invalid="ignore"):
        d = np.hypot(players[..., 0] - ball[:, None, 0], players[..., 1] - ball[:, None, 1])
        d = np.where(np.isfinite(d), d, np.inf)
    nearest = np.argmin(d, axis=1)
    near_team = np.where(d[np.arange(len(d)), nearest] < reach, team[nearest], -1)
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    for prev, b in zip(starts[:-1], starts[1:]):
        old, new = labels[b - 1], labels[b]
        if old not in ("H", "A") or new not in ("H", "A") or old == new:
            continue
        side = 0 if new == "H" else 1
        t = b - 1
        while t > prev and b - t <= max_back and near_team[t] == side:
            t -= 1
        out[t + 1:b] = new
    return out


def seq_training_set():
    """(features, labels) per Metrica match for the second-stage model (seq.py).

    The first stage on these matches is the DFL-trained one (dead_ml, team_ml) plus the
    ball-finder cross-fitted within the match: out of sample, as on BAR-ATH.
    """
    import load as loader
    import model
    import seq
    from features import compute
    xs, labels = [], []
    for match in MATCHES:
        ball, players, team, truth, blocks = load_match(match)
        loader.OVERRIDE, model.BLOCKS = (ball, players, team), blocks
        try:
            feat = compute()
            model.dead_ml_log_lr(feat)
            model.team_ml_log_odds(feat)
            model.ballfinder_log_odds(feat)
        finally:
            loader.OVERRIDE, model.BLOCKS = None, None
        xs.append(seq.features(feat))
        labels.append(truth)
    return xs, labels


if __name__ == "__main__":
    for match in sys.argv[1:] or MATCHES:
        ball, players, team, lab, blocks = load_match(match)
        graded = lab != "X"
        live = np.isin(lab, ["H", "A"])
        starts = np.flatnonzero(np.r_[True, lab[1:] != lab[:-1]])
        runs = np.diff(np.r_[starts, len(lab)])
        poss = runs[np.isin(lab[starts], ["H", "A"])] / FPS
        minutes = graded.sum() / FPS / 60
        print("{}: {:.1f} min, slots {}+{}, ball seen {:.1%} (live {:.1%}, dead {:.1%}), "
              "players tracked {:.1f} | poss {} ({:.2f}/min), median {:.1f}s, <2s {:.1%}, "
              "D share {:.2f}, H share of live {:.2f}".format(
                  match, minutes, int((team == 0).sum()), int((team == 1).sum()),
                  np.isfinite(ball[graded, 0]).mean(), np.isfinite(ball[live, 0]).mean(),
                  np.isfinite(ball[graded & ~live, 0]).mean(),
                  np.isfinite(players[graded, :, 0]).sum(axis=1).mean(), len(poss),
                  len(poss) / minutes, np.median(poss), (poss < 2).mean(),
                  (lab[graded] == "D").mean(), (lab[live] == "H").mean()))
