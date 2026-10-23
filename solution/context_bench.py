"""Auto-context: a second round of the second stage that also sees the first round's own outputs.

Round 1 is seq.py as in v10. Round 2 adds round 1's team and dead log-odds at offsets from -2 s to
+2 s and as window means (seq.context_features). Round 2's training inputs must be out of sample
like everything else, so round 1 is fitted leave-one-match-out first; on each test match round 2
is fitted on the other six. Aggregates only.

    python solution/context_bench.py
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

# Params
VARIANTS = {"round 1 (v10)": ("r1", {}), "round 2, weight 0.5": ("r2", {}),
            "round 2, weight 1.0": ("r2", {"k_seq_team": 1.0})}


def main():
    data = dfl_first_stage()
    x1 = {m: seq.features(v[5]) for m, v in data.items()}
    r1 = {}
    for m in data:
        fitted = seq.fit([x1[k] for k in data if k != m], [data[k][3] for k in data if k != m])
        r1[m] = seq.log_odds(fitted, x1[m])
    x2 = {m: seq.context_features(x1[m], *r1[m]) for m in data}
    r2 = {}
    for m in data:
        fitted = seq.fit([x2[k] for k in data if k != m], [data[k][3] for k in data if k != m])
        r2[m] = seq.log_odds(fitted, x2[m])

    # BAR-ATH: round 1 trained on all seven, round 2 on the seven out-of-sample round-1 outputs
    feat_bar, events = compute(), tags.load()
    model.dead_ml_log_lr(feat_bar)
    model.team_ml_log_odds(feat_bar)
    model.ballfinder_log_odds(feat_bar)
    xb1 = seq.features(feat_bar)
    rb1 = seq.log_odds(seq.fit(list(x1.values()), [v[3] for v in data.values()]), xb1)
    xb2 = seq.context_features(xb1, *rb1)
    rb2 = seq.log_odds(seq.fit(list(x2.values()), [v[3] for v in data.values()]), xb2)
    graded = model.graded_mask(len(xb1))
    gt = pm.label_codes(pm.read_label_csv(experiment.GT))

    team_acc = {name: [] for name in ("r1", "r2")}
    for m, v in data.items():
        live = np.isin(v[3], ["H", "A"])
        for name, r in (("r1", r1), ("r2", r2)):
            team_acc[name].append(((r[m][0][live] > 0) == (v[3][live] == "H")).mean())
    print("per-frame team accuracy on DFL live frames: round 1 {:.3f}, round 2 {:.3f}".format(
        np.mean(team_acc["r1"]), np.mean(team_acc["r2"])))
    print("{:<22} DFL spellF1 (sd)   macroF1 poss/truth | BAR-ATH spellF1 macroF1 tagTeam "
          "restart rLag poss dur Dshare".format(""))
    for name, (rnd, params) in VARIANTS.items():
        per = []
        for m, (b, p, tm, t, bl, feat) in data.items():
            feat["seq_team"], feat["seq_dead"] = (r1 if rnd == "r1" else r2)[m]
            per.append(dfl_eval.metrics(t, dfl_eval.run_pipeline(b, p, tm, bl, params, feat)))
        per = np.array(per)
        fb = dict(feat_bar)
        team, dead = rb1 if rnd == "r1" else rb2
        fb["seq_team"], fb["seq_dead"] = np.where(graded, team, 0.0), dead
        bar = experiment.score(model.predict(params, fb, events), gt, events)
        print("{:<22} {:.4f} ({:.3f})  {:.4f}  {:.2f}     | {}".format(
            name, per[:, 0].mean(), per[:, 0].std(), per[:, 1].mean(), per[:, 4].mean(), bar),
            flush=True)


if __name__ == "__main__":
    main()
