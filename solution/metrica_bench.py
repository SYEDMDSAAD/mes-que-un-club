"""The second stage trained with Metrica's sample matches added to the DFL ones.

Metrica's tracking is the closest thing to ours (same provider, same dropped-ball pattern), and
labels rebuilt from their events follow Metrica's tagging rather than DFL's. The first stage on
the Metrica matches is the DFL-trained one, out of sample as on BAR-ATH; the second stage is
refitted per variant and scored out of sample on every match:

    DFL        leave-one-match-out over the 7, plus the Metrica matches when the variant uses them
    Metrica    leave-one-match-out over the 3, plus the DFL matches when the variant uses them
    BAR-ATH    trained on everything the variant uses

It also splits Metrica's alive/dead errors by whether the ball is tracked. Aggregates only.

    python solution/metrica_bench.py
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
import dfl_eval  # noqa: E402
import experiment  # noqa: E402
import load as loader  # noqa: E402
import model  # noqa: E402
import seq  # noqa: E402
import tags  # noqa: E402
from convention_bench import dfl_first_stage  # noqa: E402
from external_metrica import MATCHES, load_match  # noqa: E402
from features import compute  # noqa: E402

# Params
CACHE = os.path.join(ROOT, "cache", "bench_metrica_prepared.pkl")
SOURCES = {"DFL only (v10)": ("dfl",), "DFL + Metrica": ("dfl", "metrica"),
           "Metrica only": ("metrica",)}
WEIGHTS = (0.5, 1.0)


def metrica_first_stage():
    """{match: (ball, players, team, truth, blocks, feat)} with the DFL-trained first stage."""
    if os.path.isfile(CACHE):
        with open(CACHE, "rb") as handle:
            return pickle.load(handle)
    data = {}
    for m in MATCHES:
        ball, players, team, truth, blocks = load_match(m)
        feat = {}
        loader.OVERRIDE, model.BLOCKS = (ball, players, team), blocks
        try:
            feat.update(compute())
            model.dead_ml_log_lr(feat)
            model.team_ml_log_odds(feat)
            model.ballfinder_log_odds(feat)
        finally:
            loader.OVERRIDE, model.BLOCKS = None, None
        data[m] = (ball, players, team, truth, blocks, feat)
    with open(CACHE, "wb") as handle:
        pickle.dump(data, handle)
    return data


def dead_split(data):
    """Metrica alive/dead agreement of v10, split by whether the ball is tracked."""
    print("v10 on Metrica, truth vs prediction, by ball tracked (share of graded frames):")
    for m, (b, p, tm, t, bl, feat) in data.items():
        feat.pop("seq_team", None), feat.pop("seq_dead", None)
        pred = dfl_eval.run_pipeline(b, p, tm, bl, {}, feat)
        graded = t != "X"
        seen = np.isfinite(b[:, 0])
        cells = []
        for name, s in (("seen", seen), ("missing", ~seen)):
            for tl, pl in (("D", "D"), ("D", "live"), ("live", "D"), ("live", "live")):
                tt = (t == "D") if tl == "D" else np.isin(t, ["H", "A"])
                pp = (pred == "D") if pl == "D" else np.isin(pred, ["H", "A"])
                cells.append("{} {}->{} {:.3f}".format(name, tl, pl, (graded & s & tt & pp).sum()
                                                       / graded.sum()))
        print("  {}: {}".format(m, "  ".join(cells)))


def main():
    met = metrica_first_stage()
    dfl = dfl_first_stage()
    dead_split(met)
    x = {("dfl", m): seq.features(v[5]) for m, v in dfl.items()}
    x.update({("metrica", m): seq.features(v[5]) for m, v in met.items()})
    y = {("dfl", m): v[3] for m, v in dfl.items()}
    y.update({("metrica", m): v[3] for m, v in met.items()})
    feat_bar, events = compute(), tags.load()
    model.dead_ml_log_lr(feat_bar)
    model.team_ml_log_odds(feat_bar)
    model.ballfinder_log_odds(feat_bar)
    x_bar = seq.features(feat_bar)
    graded = model.graded_mask(len(x_bar))
    gt = pm.label_codes(pm.read_label_csv(experiment.GT))

    print("{:<26} DFL spellF1 macroF1 p/t | Metrica spellF1 (per match) macroF1 p/t | "
          "BAR-ATH spellF1 macroF1 tagTeam restart rLag poss dur Dshare".format(""))
    for name, sources in SOURCES.items():
        fills = {}
        for key in x:
            train = [k for k in x if k != key and k[0] in sources]
            fills[key] = seq.log_odds(seq.fit([x[k] for k in train], [y[k] for k in train]),
                                      x[key])
        fitted = seq.fit([x[k] for k in x if k[0] in sources], [y[k] for k in x if k[0] in sources])
        team, dead = seq.log_odds(fitted, x_bar)
        for w in WEIGHTS:
            params = {"k_seq_team": w}
            out = {}
            for src, data in (("dfl", dfl), ("metrica", met)):
                per = []
                for m, (b, p, tm, t, bl, feat) in data.items():
                    feat["seq_team"], feat["seq_dead"] = fills[(src, m)]
                    per.append(dfl_eval.metrics(t, dfl_eval.run_pipeline(b, p, tm, bl, params,
                                                                         feat)))
                out[src] = np.array(per)
            fb = dict(feat_bar)
            fb["seq_team"], fb["seq_dead"] = np.where(graded, team, 0.0), dead
            bar = experiment.score(model.predict(params, fb, events), gt, events)
            d, mt = out["dfl"].mean(axis=0), out["metrica"].mean(axis=0)
            print("{:<26} {:.4f}  {:.4f}  {:.2f} | {:.4f} ({})  {:.4f}  {:.2f} | {}".format(
                "{}, w {}".format(name, w), d[0], d[1], d[4], mt[0],
                " ".join("{:.3f}".format(v) for v in out["metrica"][:, 0]), mt[1], mt[4], bar),
                flush=True)


if __name__ == "__main__":
    main()
