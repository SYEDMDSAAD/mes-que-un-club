"""Is Smart Tagging a source of *who*? Its team phases against the public ground truth.

The tags mark phases by team: BUILD UP / PROGRESSION / CREATING CHANCES / ATTACKING TRANSITION by
T (T has the ball), DEFENDING IN ... THIRD by T (the other team has it); ATTACKING and DEFENSIVE
TRANSITION open right after the ball changes hands. Metrica produced them, as it produced the
ground truth, so they may follow its possession convention. If, where the tags and our labels
disagree, the tags are usually the right ones, they are a per-frame team prior worth adding (it
would then stop being an independent check). If not, they stay the check they are. The public
window answers it:

    coverage     share of the window's in-play frames inside an attack or defend phase
    tags right   on those frames, how often the phase's team is the truth's
    ours right   the same for our labels (--labels), on the same frames
    disagree     frames where the tags and our labels disagree: how often the tags are right
    transitions  ATTACKING TRANSITION by T: offset from its start to the nearest true change to
                 T within +-3 s (median, and how many fall within 1 s); DEFENSIVE TRANSITION by
                 T: the same for the nearest change away from T

17 possessions: read it as "clearly useful" or "clearly not", never as a weight to tune.
Aggregates only.

    python solution/tags_check.py --labels "Mes Que Un Club.csv"
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]

# Params
GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
SEARCH = 75               # frames either side of a transition tag's start
ATTACK = ("build up", "progression", "creating chances", "attacking transition")
DEFEND = ("defending in attacking third", "defending in middle third",
          "defending in defensive third")


def phase_team(events, n):
    """Per frame +1 (home has the ball by the tags), -1 (away), 0 (no phase, or phases disagree)."""
    vote = np.zeros(n)
    conflict = np.zeros(n, bool)
    for e in events:
        if e["side"] is None or e["code"] not in ATTACK + DEFEND:
            continue
        sign = 1.0 if e["side"] == "H" else -1.0
        if e["code"] in DEFEND:
            sign = -sign
        lo, hi = max(0, e["start"]), min(n, e["end"])
        clash = (vote[lo:hi] != 0) & (vote[lo:hi] != sign)
        conflict[lo:hi] |= clash
        vote[lo:hi] = np.where(vote[lo:hi] == 0, sign, vote[lo:hi])
    vote[conflict] = 0.0
    return vote


def changes(gt):
    """[(frame, new team)] of the direct H <-> A changes in the ground truth."""
    i = np.flatnonzero(gt[1:] != gt[:-1]) + 1
    return [(int(k), gt[k]) for k in i if gt[k - 1] in ("H", "A") and gt[k] in ("H", "A")]


def report(gt, events, labels=None):
    """The figures above, as a dict (aggregates)."""
    n = len(gt)
    window = gt != "X"
    live = window & np.isin(gt, ["H", "A"])
    truth = np.where(gt == "H", 1.0, -1.0)
    tags = phase_team(events, n)
    covered = live & (tags != 0)
    out = {"coverage": covered.sum() / max(live.sum(), 1),
           "tags right": float((tags[covered] == truth[covered]).mean()) if covered.any() else np.nan}
    if labels is not None:
        ours = np.where(labels == "H", 1.0, np.where(labels == "A", -1.0, 0.0))
        out["ours right"] = float((ours[covered] == truth[covered]).mean()) if covered.any() else np.nan
        disagree = covered & (ours != 0) & (ours != tags)
        out["disagree frames"] = int(disagree.sum())
        out["tags right where they disagree"] = (float((tags[disagree] == truth[disagree]).mean())
                                                 if disagree.any() else np.nan)
    ch = changes(gt)
    for code, into in (("attacking transition", True), ("defensive transition", False)):
        offs = []
        for e in events:
            if e["code"] != code or e["side"] is None or not window[min(max(e["start"], 0), n - 1)]:
                continue
            target = e["side"] if into else {"H": "A", "A": "H"}[e["side"]]
            near = [f - e["start"] for f, new in ch if new == target and abs(f - e["start"]) <= SEARCH]
            offs.append(min(near, key=abs) if near else None)
        found = [o for o in offs if o is not None]
        out[code] = {"tags": len(offs), "with a true change within 3 s": len(found),
                     "median offset": float(np.median(found)) if found else np.nan,
                     "within 1 s": int(np.sum(np.abs(found) <= 25)) if found else 0}
    return out


def main(labels_path=None):
    import tags
    gt = pd.read_csv(GT)["label"].values
    labels = pd.read_csv(labels_path)["label"].astype(str).values if labels_path else None
    for key, value in report(gt, tags.load(), labels).items():
        print("{:<34} {}".format(key, value if not isinstance(value, float) else "{:.3f}".format(value)))


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[args.index("--labels") + 1] if "--labels" in args else None)
