"""Possession timelines rebuilt from Metrica's public sample event data -- a convention check.

Our ground truth is Metrica's; the DFL labels are not, and the two disagree on how short a
possession can be. Metrica publishes three sample matches with event data
(github.com/metrica-sports/sample-data; "be responsible, acknowledge the source"). Rebuilding a
possession timeline from those events gives Metrica-style possession statistics to compare with.

Rules: a pass, shot, recovery or ball-lost event by the other team switches possession at its
start frame; ball out, a foul received or a goal makes the ball dead until the next set piece;
challenges and cards are skipped (their team attribution is ambiguous).

    python solution/metrica_events.py
"""

import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DIR = os.path.join(ROOT, "external", "metrica")
FPS = 25
LIVE_TYPES = {"PASS", "SHOT", "RECOVERY", "BALL LOST"}
DEAD_TYPES = {"BALL OUT", "FAULT RECEIVED"}


def events_csv(path):
    with open(path) as handle:
        for row in csv.DictReader(handle):
            yield (int(row["Period"]), int(row["Start Frame"]), row["Type"], row["Subtype"] or "",
                   row["Team"])


def events_json(path):
    data = json.load(open(path))["data"]
    for e in data:
        sub = (e.get("subtypes") or {})
        sub = sub.get("name", "") if isinstance(sub, dict) else " ".join(s["name"] for s in sub)
        yield (e["period"], e["start"]["frame"], e["type"]["name"], sub, e["team"]["name"])


def timeline(events):
    """Per period: list of (frame, state) with state a team name or 'D'."""
    out = {}
    for period, frame, kind, sub, team in sorted(events, key=lambda e: (e[0], e[1])):
        seq = out.setdefault(period, [])
        state = seq[-1][1] if seq else "D"
        if kind == "SET PIECE":
            new = team
        elif kind in DEAD_TYPES or (kind == "SHOT" and "GOAL" in sub and "OUT" not in sub):
            new = "D"
        elif kind in LIVE_TYPES and state != "D":
            new = team
        else:
            continue
        if new != state:
            seq.append((frame, new))
    return out


def stats(timelines, end_frames):
    poss, dead, total = [], [], 0
    for period, seq in timelines.items():
        frames = [f for f, _ in seq] + [end_frames[period]]
        for (f, state), nxt in zip(seq, frames[1:]):
            (dead if state == "D" else poss).append((nxt - f) / FPS)
        total += frames[-1] - frames[0]
    poss, dead = np.array(poss), np.array(dead)
    minutes = total / FPS / 60
    return ("{:5.1f} min | poss {:4d} ({:.2f}/min) mean {:4.1f}s median {:4.1f}s  <1s {:4.1%}  "
            "<2s {:4.1%}  <4s {:4.1%} | D share {:.2f}").format(
        minutes, len(poss), len(poss) / minutes, poss.mean(), np.median(poss), (poss < 1).mean(),
        (poss < 2).mean(), (poss < 4).mean(), dead.sum() / (minutes * 60))


if __name__ == "__main__":
    for name, reader in (("Sample_Game_1_RawEventsData.csv", events_csv),
                         ("Sample_Game_2_RawEventsData.csv", events_csv),
                         ("Sample_Game_3_events.json", events_json)):
        events = list(reader(os.path.join(DIR, name)))
        ends = {}
        for period, frame, *_ in events:
            ends[period] = max(ends.get(period, 0), frame)
        print("{:<34} {}".format(name, stats(timeline(events), ends)))
