"""Score the pipeline on Metrica's sample matches, against labels rebuilt from their events.

Metrica's tracking already has our ball-gap pattern (the ball is dropped when it is dead), so it
is used as it is. Every learned model is the one trained on the DFL matches, exactly as for
BAR-ATH: these matches are a test bed no model has seen. Aggregates only.

    python solution/metrica_eval.py
"""

import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import dfl_eval  # noqa: E402
from external_metrica import MATCHES, load_match  # noqa: E402

VARIANTS = {"v10": {}, "no second stage": {"k_seq_team": 0.0},
            "seq 1.0": {"k_seq_team": 1.0}}


def main(variants=VARIANTS):
    data = {m: load_match(m) + ({},) for m in MATCHES}
    rows = {"baseline smooth50": [dfl_eval.metrics(t, dfl_eval.baseline(b, p, tm, bl))
                                  for b, p, tm, t, bl, _ in data.values()]}
    for name, params in variants.items():
        rows[name] = [dfl_eval.metrics(t, dfl_eval.run_pipeline(b, p, tm, bl, params, feat))
                      for b, p, tm, t, bl, feat in data.values()]
    print("{:<22} spellF1 (sd)    macroF1  aliveDead  teamAcc  poss/truth  per-match spellF1"
          .format("Metrica, 3 matches"))
    for name, per in rows.items():
        per = np.array(per)
        m = per.mean(axis=0)
        print("{:<22} {:.4f} ({:.3f})  {:.4f}   {:.3f}      {:.3f}    {:.2f}       {}".format(
            name, m[0], per[:, 0].std(), m[1], m[2], m[3], m[4],
            " ".join("{:.3f}".format(v) for v in per[:, 0])), flush=True)


if __name__ == "__main__":
    main()
