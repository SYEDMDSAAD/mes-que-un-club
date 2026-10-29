"""Which experimental variant a run belongs to, so its cached models never mix with another's.

    MQUC_VARIANT=state python solution/mbr_bench.py

Unset (the default) is the submitted pipeline, byte for byte. Any other value tags every cached
learned model and bench file with it (cache/dead_ml_model.state.pkl, ...), so a variant trains
its own models and the submission's stay untouched. Variants read TAG to switch themselves on:

    state   dfl_eval.degrade damages DFL the way BAR-ATH's tracking actually loses the ball --
            mostly in stoppages -- with slow ball noise instead of independent noise per frame
    taper   the dead-ball drift fades in and out over ~1 s instead of switching on and off in one
            frame (which made a median 44 m/s jump at every stoppage edge of the training ball)
    smoothS the clean DFL ball blurred by a Gaussian of S frames before it is damaged, so its
            kinks are as soft as our ATD's (smooth2, smooth3, ...)
    metatd  Metrica's sample ball (drawn from its events) blurred and offset to look like our
            ATD before anything reads it, for training and for the Metrica bench alike
    touch   ball-touch features (touchfeat.py) in the second stage, blanked on Metrica's rows
    touchblurS  the touch features of training and bench matches computed from their ball
            blurred by S frames; every other model keeps the v13 training ball
    droplenS   player dropouts S frames long (v13: 50, ~46 track births a minute; ours 6-11)
    wrongballR R stretches a minute of play where the training ball sits on a player of the team
            without it, as our tracker does
    motiondropS  droplenS for the motion model's players only (train, and test on a bench); v15
            uses 250, motiondrop0 goes back to DROP_LEN
    nomotion  without the ball-blind, player-only possession model (motion_ml.py), on since v15
    motionline  + back-line / front-line velocities in the motion model (motion_ml.lines)
    motioncopiesN  N randomly damaged training copies per DFL match for the motion model alone

Parts combine with "+": MQUC_VARIANT=motion+wrongball3+droplen250.
"""

import os

TAG = os.environ.get("MQUC_VARIANT", "").strip()
PARTS = [part for part in TAG.split("+") if part]


def has(name):
    """True when the variant includes this part."""
    return name in PARTS


def value(prefix, default):
    """The number after `prefix` in a part (smooth2 -> 2.0), or default."""
    for part in PARTS:
        if part.startswith(prefix) and part[len(prefix):].replace(".", "", 1).isdigit():
            return float(part[len(prefix):])
    return default


def cache_path(path):
    """path, with the variant's tag before the extension when a variant is active."""
    if not TAG:
        return path
    root, ext = os.path.splitext(path)
    return "{}.{}{}".format(root, TAG, ext)
