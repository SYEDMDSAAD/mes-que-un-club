"""DFL open data (Bassek et al. 2025, CC BY 4.0) as an external, fully labelled test bed.

Seven Bundesliga / 2. Bundesliga matches with TRACAB tracking at 25 fps, and on every ball frame
the official BallStatus (0 dead, 1 alive) and BallPossession (1 home, 2 away). That is the exact
shape of our labels, on whole matches -- the validation set the 5-minute public window cannot be.

    python solution/external_dfl.py            # download + convert every match not yet cached

Each match is streamed out of its XML into the arrays our pipeline reads, then the XML can go:

    ball      (N, 3)       x, y in metres from a corner, NaN speed (as in our ATD)
    players   (N, S, 3)    x, y, speed in m/s per person
    team      (S,)         0 home, 1 away
    labels    (N,)         'H' / 'A' / 'D', and 'X' between the halves
    blocks    (2, 2)       first and last frame of each half

The data is clean, manually checked tracking. degrade() makes it look like ours before it is used.
"""

import os
import sys
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Params
DFL_DIR = os.path.join(ROOT, "external", "dfl")
CACHE_DIR = os.path.join(ROOT, "cache", "dfl")
ARTICLE = "https://api.figshare.com/v2/articles/28196177"
PITCH = (105.0, 68.0)
HALF_GAP = 5000          # frames of 'X' inserted between the halves


def match_ids():
    import json
    with urllib.request.urlopen(ARTICLE) as response:
        files = json.load(response)["files"]
    ids = sorted({f["name"].split("DFL-MAT-")[1].split(".")[0] for f in files})
    return ids, files


def download(match_id, files):
    os.makedirs(DFL_DIR, exist_ok=True)
    paths = {}
    for f in files:
        if match_id not in f["name"]:
            continue
        kind = "info" if "matchinformation" in f["name"] else (
            "positions" if "positions" in f["name"] else "events")
        path = os.path.join(DFL_DIR, f["name"])
        if not os.path.isfile(path):
            urllib.request.urlretrieve(f["download_url"], path)
        paths[kind] = path
    return paths


def home_team_id(info_path):
    for team in ET.parse(info_path).getroot().iter("Team"):
        if team.get("Role") == "home":
            return team.get("TeamId")
    raise ValueError("no home team in " + info_path)


def parse(positions_path, home_id):
    """Stream the positions XML into per-half dicts of frame-indexed series."""
    halves = {}                                  # section -> {"ball": [...], "people": {...}}
    current = None
    for event, elem in ET.iterparse(positions_path, events=("start", "end")):
        if event == "start" and elem.tag == "FrameSet":
            section = elem.get("GameSection")
            team = elem.get("TeamId")
            current = {"section": section, "team": team, "person": elem.get("PersonId"),
                       "n": [], "x": [], "y": [], "s": [], "poss": [], "status": []}
        elif event == "end" and elem.tag == "Frame" and current is not None:
            current["n"].append(int(elem.get("N")))
            current["x"].append(float(elem.get("X")))
            current["y"].append(float(elem.get("Y")))
            current["s"].append(float(elem.get("S", "nan")))
            if current["team"] == "BALL":
                current["poss"].append(int(elem.get("BallPossession", "0")))
                current["status"].append(int(elem.get("BallStatus", "0")))
            elem.clear()
        elif event == "end" and elem.tag == "FrameSet":
            half = halves.setdefault(current["section"], {"ball": None, "people": []})
            if current["team"] == "BALL":
                half["ball"] = current
            else:
                current["home"] = current["team"] == home_id
                half["people"].append(current)
            current = None
            elem.clear()
    return halves


def assemble(halves):
    """Concatenate the halves on one frame index, with HALF_GAP frames of nothing between."""
    sections = [s for s in ("firstHalf", "secondHalf") if s in halves]
    persons = {}
    for s in sections:
        for p in halves[s]["people"]:
            persons.setdefault(p["person"], p["home"])
    slots = {pid: k for k, pid in enumerate(sorted(persons, key=lambda q: (not persons[q], q)))}
    team = np.array([0 if persons[pid] else 1 for pid in sorted(slots, key=slots.get)], np.int8)

    lengths, starts = [], []
    for s in sections:
        n = np.array(halves[s]["ball"]["n"])
        lengths.append(n.max() - n.min() + 1)
    total = sum(lengths) + HALF_GAP * (len(sections) - 1)
    ball = np.full((total, 3), np.nan, np.float32)
    players = np.full((total, len(slots), 3), np.nan, np.float32)
    labels = np.full(total, "X", dtype="<U1")
    blocks, offset = [], 0
    for s, length in zip(sections, lengths):
        b = halves[s]["ball"]
        n0 = min(b["n"])
        idx = np.array(b["n"]) - n0 + offset
        ball[idx, 0] = np.array(b["x"]) + PITCH[0] / 2
        ball[idx, 1] = np.array(b["y"]) + PITCH[1] / 2
        status, poss = np.array(b["status"]), np.array(b["poss"])
        labels[idx] = np.where(status == 0, "D", np.where(poss == 1, "H", "A"))
        for p in halves[s]["people"]:
            k = slots[p["person"]]
            pidx = np.array(p["n"]) - n0 + offset
            ok = (pidx >= offset) & (pidx < offset + length)
            players[pidx[ok], k, 0] = (np.array(p["x"]) + PITCH[0] / 2)[ok]
            players[pidx[ok], k, 1] = (np.array(p["y"]) + PITCH[1] / 2)[ok]
            players[pidx[ok], k, 2] = (np.array(p["s"]) / 3.6)[ok]     # km/h -> m/s
        blocks.append((offset, offset + length - 1))
        offset += length + HALF_GAP
    return ball, players, team, labels, np.array(blocks)


def convert(match_id, files, keep_xml=False):
    out = os.path.join(CACHE_DIR, match_id + ".npz")
    if os.path.isfile(out):
        return out
    paths = download(match_id, files)
    ball, players, team, labels, blocks = assemble(
        parse(paths["positions"], home_team_id(paths["info"])))
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(out, ball=ball, players=players, team=team, labels=labels, blocks=blocks)
    if not keep_xml:
        os.remove(paths["positions"])
    return out


def load_match(match_id):
    d = np.load(os.path.join(CACHE_DIR, match_id + ".npz"))
    return d["ball"], d["players"], d["team"], d["labels"], [tuple(b) for b in d["blocks"]]


if __name__ == "__main__":
    ids, files = match_ids()
    only = sys.argv[1:] or ids
    for match_id in only:
        print(match_id, "->", convert(match_id, files))
