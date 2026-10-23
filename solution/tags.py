"""Smart Tagging as video-frame intervals, and a full-match proxy score built from it.

The public ground truth is 5 minutes and 17 possessions -- too few to choose between two versions
of a model. The Smart Tagging covers the whole match. It is automatic and noisy, so it is not an
answer key, but it is independent of our model and of the private ground truth, which makes it a
useful second opinion when comparing variants:

    attack    BUILD UP / PROGRESSION / CREATING CHANCES / ATTACKING TRANSITION by team T
              -> live frames inside should mostly be T
    defend    DEFENDING IN ... THIRD by team T -> live frames inside should mostly be the other team
    restart   SET PIECES -> the ball is dead somewhere in the opening of the tag, then live
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "starter"))
import load_smart_tagging  # noqa: E402

# Params
HOME_NAMES = {"barcelona"}
AWAY_NAMES = {"atletic club", "athletic club"}
ATTACK = {"build up", "progression", "creating chances", "attacking transition"}
DEFEND = {"defending in attacking third", "defending in middle third",
          "defending in defensive third"}
SET_PIECE = "set pieces"
# Frames from a set-piece tag opening to the restart itself: ~3.5 s, read off the player and ball
# speed profile averaged over every set piece in the match.
RESTART_OFFSET = 88


def side(event):
    team = (event["tags"].get("Team") or "").strip().lower()
    return "H" if team in HOME_NAMES else "A" if team in AWAY_NAMES else None


def load():
    """[{code (lower case), start, end, side ('H'/'A'/None), type (lower case)}]."""
    return [{"code": (e["code"] or "").strip().lower(),
             "start": e["start_frame"], "end": e["end_frame"],
             "side": side(e), "type": (e["tags"].get("Type") or "").strip().lower()}
            for e in load_smart_tagging.load()]


def proxy_scores(labels, events):
    """Agreement of a label array with the tagging, over the whole match. Higher is better."""
    agree = total = 0
    for e in events:
        if e["side"] is None or e["code"] not in ATTACK | DEFEND:
            continue
        window = labels[e["start"]:e["end"]]
        live = window[np.isin(window, ["H", "A"])]
        expected = e["side"] if e["code"] in ATTACK else {"H": "A", "A": "H"}[e["side"]]
        agree += (live == expected).sum()
        total += len(live)

    # A set piece is a restart ~3.5 s after the tag opens: the second before it should contain
    # dead ball, and the seconds after it should be live. The share of set pieces where both hold.
    restarts = [e for e in events if e["code"] == SET_PIECE]
    ok = 0
    for e in restarts:
        restart = e["start"] + RESTART_OFFSET
        before = labels[max(restart - 50, 0):restart]
        after = labels[restart + 50:restart + 125]
        ok += (before == "D").any() and np.isin(after, ["H", "A"]).mean() > 0.5
    # Restart lag: first live frame after the tag opens, minus the restart estimated from the tag.
    lags = []
    for e in restarts:
        live = np.flatnonzero(np.isin(labels[e["start"]:e["start"] + 500], ["H", "A"]))
        if len(live):
            lags.append(live[0] - RESTART_OFFSET)
    return {"team_agree": agree / max(total, 1), "restart_ok": ok / max(len(restarts), 1),
            "restart_lag": float(np.median(lags)) if lags else float("nan")}
