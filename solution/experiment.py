"""Compare model variants on two independent yardsticks, printing aggregates only.

    public   spellF1 / macroF1 on the 5-minute public ground truth (17 possessions: noisy)
    proxy    agreement with Smart Tagging over the full match (noisy too, but independent)
    poss     predicted possessions over the full match, and their mean length

Neither is the private score. A change worth keeping should not hurt either of them.

    python solution/experiment.py
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
import tags  # noqa: E402
from features import compute  # noqa: E402
from model import predict  # noqa: E402

GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
VARIANTS = {
    "v3 (no ball-finder)": {"k_bf": 0},
    "v4 bf 0.05": {"k_bf": 0.05},
    "v4 bf 0.1 (default)": {},
    "v4 bf 0.2": {"k_bf": 0.2},
    "v4 bf 0.1, no formation": {"k_form_missing": 0, "k_form_seen": 0},
}


def possessions(labels):
    seq = labels[labels != "X"]
    starts = np.r_[0, np.flatnonzero(seq[1:] != seq[:-1]) + 1]
    n = sum(seq[s] in "HA" for s in starts)
    return n, np.isin(seq, ["H", "A"]).sum() / 25.0 / max(n, 1), (seq == "D").mean()


def score(labels, gt_codes, events):
    result = pm.evaluate(gt_codes, pm.label_codes(labels), bootstrap_n=0)
    proxy = tags.proxy_scores(labels, events)
    n, dur, dead = possessions(labels)
    lat = result["transitions"]
    return "{:.4f}   {:.4f}   {:.3f}   {:.3f}   {:+5.0f}   {:4d}  {:5.1f}  {:.2f}".format(
        result["spell_f1_mean"], result["macro_f1"], proxy["team_agree"],
        proxy["restart_ok"], proxy["restart_lag"], n, dur, dead)


def run(variants):
    feat, events = compute(), tags.load()
    gt_codes = pm.label_codes(pm.read_label_csv(GT))
    print("{:<28} spellF1  macroF1  tagTeam  restart  rLag    poss   dur  Dshare".format(""))
    for name, params in variants.items():
        print("{:<28} {}".format(name, score(predict(params, feat, events), gt_codes, events)))


if __name__ == "__main__":
    run(VARIANTS)
