"""DFL possession labels rewritten to the challenge's rule of *controlled* possession.

CHALLENGE.md: a change of possession needs the other team to gain controlled possession; a touch
without control, a deflection, or an aerial duel won without keeping the ball is not one, and a
contested or loose ball stays with the last team that had it. The DFL labels change team on any
touch: 4.6 possessions a minute, 30% under 2 s, against 3.4 a minute and 12% in the public ground
truth. Models trained on them learn to look for changes the ground truth does not contain.

rewrite() gives a short spell of one team back to the team around it:

    A A A H A A A    -> A A A A A A A      the H touch is a deflection: A never lost the ball
    A A A H D D D    -> A A A A D D D      with to_dead: a touch that only knocks the ball out

Only spells shorter than max_frames, and only between spells of the other team (or that team and
a stoppage), so every real possession -- however short -- that the other team then kept stays.
"""

import numpy as np


def runs(labels):
    """(start, end exclusive, label) of each maximal run."""
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    ends = np.r_[starts[1:], len(labels)]
    return [(int(s), int(e), labels[s]) for s, e in zip(starts, ends)]


def rewrite(labels, max_frames, to_dead=False):
    """Copy of labels with uncontrolled spells given back to the surrounding team."""
    out = labels.copy()
    r = runs(labels)
    for k in range(1, len(r) - 1):
        start, end, lab = r[k]
        before, after = r[k - 1][2], r[k + 1][2]
        if lab not in ("H", "A") or end - start >= max_frames:
            continue
        if before in ("H", "A") and before != lab and (after == before
                                                       or (to_dead and after == "D")):
            out[start:end] = before
    return out
