"""Ground-truth-free test of how well a model copes with a missing ball.

The public window has the ball on nearly every frame, so it cannot tell two ways of handling a
lost ball apart -- yet over the full match the ball is missing on 41% of graded frames, mostly in
gaps longer than ten seconds. This test manufactures that situation where we can check it:

  1. run the model on the real tracking -> reference labels
  2. delete the ball over extra stretches where it *was* seen, with gap lengths drawn from this
     match's real gaps, until ~15% more of the graded frames are ball-less
  3. run the model again on the damaged tracking and compare with the reference

Every variant is scored against the same reference, so that ball-independent evidence cannot
win just by being stable: the first-pass ball-only labels on deleted frames where the real ball
sat within ANCHOR_DIST of a player (nearest-team is right ~90% of the time there, checked against
the public ground truth). Also reported: spellF1 of the damaged run against the variant's own
undamaged run, i.e. how much the possessions move when the ball goes missing.

    python solution/gapsim.py
"""

import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
import load as loader  # noqa: E402
from ballfinder import gap_lengths  # noqa: E402
from games import GAME, half_video_blocks  # noqa: E402
from model import predict  # noqa: E402

# Params
DELETE_SHARE = 0.15
ANCHOR_DIST = 1.5
BALL_ONLY = {"k_form_missing": 0, "k_form_seen": 0, "k_bf": 0}
SEEDS = (0, 1)


def damage(ball, blocks, seed):
    """Copy of ball with extra real-length gaps cut out of seen stretches; the deleted mask."""
    rng = np.random.default_rng(seed)
    seen = np.isfinite(ball[:, 0])
    gaps = gap_lengths(seen, blocks)
    graded = sum(last - first + 1 for first, last in blocks)
    deleted = np.zeros(len(ball), dtype=bool)
    while deleted.sum() < DELETE_SHARE * graded:
        first, last = blocks[rng.integers(len(blocks))]
        length = int(rng.choice(gaps))
        start = int(rng.integers(first, max(first + 1, last - length)))
        deleted[start:start + length] |= seen[start:start + length]
    out = ball.copy()
    out[deleted] = np.nan
    return out, deleted


def compare(anchor, own, damaged, deleted):
    spell = pm.evaluate(pm.label_codes(own), pm.label_codes(damaged),
                        bootstrap_n=0)["spell_f1_mean"]
    live = deleted & np.isin(anchor, ["H", "A"]) & np.isin(damaged, ["H", "A"])
    return (damaged[live] == anchor[live]).mean(), spell


def run(variants):
    blocks = [(first, last) for _, first, last in half_video_blocks(GAME)]
    ball, players, team = loader.load()
    d = np.hypot(players[..., 0] - ball[:, None, 0], players[..., 1] - ball[:, None, 1])
    with np.errstate(invalid="ignore"):
        at_feet = np.fmin.reduce(np.where(np.isfinite(d), d, np.inf), axis=1) < ANCHOR_DIST
    anchor = predict(BALL_ONLY)
    anchor[~at_feet] = "X"
    print("{:<32} teamAgree(anchor)  spellF1 vs own   (mean over {} seeds)".format(
        "", len(SEEDS)))
    for name, params in variants.items():
        loader.OVERRIDE = None
        own = predict(params)
        agree, spell = [], []
        for seed in SEEDS:
            damaged_ball, deleted = damage(ball, blocks, seed)
            loader.OVERRIDE = (damaged_ball, players, team)
            a, s = compare(anchor, own, predict(params), deleted)
            agree.append(a)
            spell.append(s)
        loader.OVERRIDE = None
        print("{:<32} {:.3f}              {:.4f}".format(name, np.mean(agree), np.mean(spell)))


if __name__ == "__main__":
    run({
        "carry forward (no form, no bf)": {"k_form_missing": 0, "k_form_seen": 0, "k_bf": 0},
        "formation only": {"k_bf": 0},
        "ball-finder only": {"k_form_missing": 0, "k_form_seen": 0},
        "formation + ball-finder": {},
    })
