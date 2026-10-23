"""Decoder settings and second-stage weight, re-checked now that the second stage exists.

The switch cost (9) and the second-stage weight (0.5) were chosen before or alongside the second
stage; better evidence can move their best values. The second stage is fitted leave-one-match-out
once, then every variant is decoded on the 7 DFL matches and on BAR-ATH (public window, Smart
Tagging checks, possession count). Aggregates only.

    python solution/seq_bench.py
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
import dfl_eval  # noqa: E402
import experiment  # noqa: E402
import model  # noqa: E402
import seq  # noqa: E402
import tags  # noqa: E402
from convention_bench import dfl_first_stage  # noqa: E402
from features import compute  # noqa: E402


def switch(cost):
    """All three H <-> A costs at once: the decoder only uses per-frame costs when they differ."""
    return {"switch": cost, "switch_contact": cost, "switch_quiet": cost}


VARIANTS = {"v10": {}}
VARIANTS.update({"seq {}".format(w): {"k_seq_team": w} for w in (0.75, 1.0)})
VARIANTS.update({"switch {}".format(c): switch(c) for c in (7, 8, 10, 12)})
VARIANTS.update({"seq 1.0, switch {}".format(c): dict(switch(c), k_seq_team=1.0) for c in (11, 13)})
VARIANTS.update({"stop/start {}".format(c): {"stop": c, "start": c} for c in (12, 18)})


def fill_seq(data, x):
    for m, (_, _, _, _, _, feat) in data.items():
        fitted = seq.fit([x[k] for k in data if k != m], [data[k][3] for k in data if k != m])
        feat["seq_team"], feat["seq_dead"] = seq.log_odds(fitted, x[m])


def main(variants=VARIANTS):
    data = dfl_first_stage()
    x = {m: seq.features(v[5]) for m, v in data.items()}
    fill_seq(data, x)
    feat_bar, events = compute(), tags.load()
    gt = pm.label_codes(pm.read_label_csv(experiment.GT))
    print("{:<24} DFL spellF1 (sd)   macroF1 poss/truth | BAR-ATH spellF1 macroF1 tagTeam "
          "restart rLag poss dur Dshare".format(""))
    for name, params in variants.items():
        per = np.array([dfl_eval.metrics(t, dfl_eval.run_pipeline(b, p, tm, bl, params, feat))
                        for b, p, tm, t, bl, feat in data.values()])
        m = per.mean(axis=0)
        bar = experiment.score(model.predict(params, feat_bar, events), gt, events)
        print("{:<24} {:.4f} ({:.3f})  {:.4f}  {:.2f}     | {}".format(
            name, m[0], per[:, 0].std(), m[1], m[4], bar), flush=True)


if __name__ == "__main__":
    main()
