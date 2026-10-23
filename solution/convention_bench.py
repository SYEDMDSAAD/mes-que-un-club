"""Second stage trained on DFL labels rewritten to the challenge's rule of controlled possession.

For each rewrite (convention.py), the second stage (seq.py) is retrained leave-one-match-out and
the pipeline is scored on the DFL matches against the original labels and against rewritten ones,
then retrained on all seven and run on BAR-ATH (public window, Smart Tagging checks, possession
count) at two switch costs. Aggregates only.

    python solution/convention_bench.py          # ~30 min; caches the DFL first stage in cache/
"""

import os
import pickle
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
import convention  # noqa: E402
import dfl_eval  # noqa: E402
import experiment  # noqa: E402
import load as loader  # noqa: E402
import model  # noqa: E402
import seq  # noqa: E402
import tags  # noqa: E402
from features import compute  # noqa: E402

# Params
CACHE = os.path.join(ROOT, "cache", "bench_dfl_prepared.pkl")
SCORE_REWRITE = (50, True)       # the rewrite the "vs rewritten" column is scored against
VARIANTS = {"v10 (original labels)": None, "rewrite <1s +to_dead": (25, True),
            "rewrite <2s": (50, False), "rewrite <2s +to_dead": (50, True)}
PARAMS = {"k_seq_team": 0.5}
SWITCHES = (9, 7)


def dfl_first_stage():
    """{match: (ball, players, team, truth, blocks, feat)} with the first stage filled, cached."""
    if os.path.isfile(CACHE):
        with open(CACHE, "rb") as handle:
            return pickle.load(handle)
    data = dfl_eval.prepare(dead_ml=True, team_ml=True)
    for ball, players, team, _, blocks, feat in data.values():
        loader.OVERRIDE, model.BLOCKS = (ball, players, team), blocks
        try:
            feat.update(compute())
            model.ballfinder_log_odds(feat)
        finally:
            loader.OVERRIDE, model.BLOCKS = None, None
    with open(CACHE, "wb") as handle:
        pickle.dump(data, handle)
    return data


def main():
    data = dfl_first_stage()
    x = {m: seq.features(v[5]) for m, v in data.items()}
    feat_bar, events = compute(), tags.load()
    model.dead_ml_log_lr(feat_bar)
    model.team_ml_log_odds(feat_bar)
    model.ballfinder_log_odds(feat_bar)
    x_bar = seq.features(feat_bar)
    graded = model.graded_mask(len(x_bar))
    gt = pm.label_codes(pm.read_label_csv(experiment.GT))

    for name, rewrite in VARIANTS.items():
        labels = {m: v[3] if rewrite is None else convention.rewrite(v[3], *rewrite)
                  for m, v in data.items()}
        vs_orig, vs_rew = [], []
        for m, (ball, players, team, truth, blocks, feat) in data.items():
            fitted = seq.fit([x[k] for k in data if k != m], [labels[k] for k in data if k != m])
            feat["seq_team"], feat["seq_dead"] = seq.log_odds(fitted, x[m])
            pred = dfl_eval.run_pipeline(ball, players, team, blocks, PARAMS, feat)
            vs_orig.append(dfl_eval.metrics(truth, pred))
            vs_rew.append(dfl_eval.metrics(convention.rewrite(truth, *SCORE_REWRITE), pred))
        o, r = np.mean(vs_orig, axis=0), np.mean(vs_rew, axis=0)
        fitted = seq.fit(list(x.values()), list(labels.values()))
        fb = dict(feat_bar)
        team_lo, dead_lo = seq.log_odds(fitted, x_bar)
        fb["seq_team"], fb["seq_dead"] = np.where(graded, team_lo, 0.0), dead_lo
        print("== {}".format(name))
        print("   DFL vs original labels    spellF1 {:.4f}  macroF1 {:.4f}  poss/truth {:.2f}"
              .format(o[0], o[1], o[4]))
        print("   DFL vs rewritten labels   spellF1 {:.4f}  macroF1 {:.4f}  poss/truth {:.2f}"
              .format(r[0], r[1], r[4]))
        for switch in SWITCHES:
            row = experiment.score(model.predict(dict(PARAMS, switch=switch), fb, events),
                                   gt, events)
            print("   BAR-ATH switch {}: spellF1 macroF1 tagTeam restart rLag poss dur Dshare  {}"
                  .format(switch, row), flush=True)


if __name__ == "__main__":
    main()
