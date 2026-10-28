"""Touch features and the change detector, on every labelled match. Aggregates only.

Run it twice -- as the submission, and as the variant -- and compare the per-match columns:

    python solution/touch_bench.py                                  # v13 second stage
    MQUC_VARIANT=touch+taper+smooth2 python solution/touch_bench.py # + touch features, fixed damage

Each run: the first stage of every DFL and Metrica match (cached per variant), the second stage
fitted leave-one-match-out over all ten (touch features blanked on Metrica's rows, whose ball is
drawn from the events), the change detector (change_ml.py) fitted the same way on the same rows,
then every decoder setting below on every match, paired against k_change 0. Judge on DFL first:
Metrica's labels and ball come from the same events, so a touch feature can read them there.
Then BAR-ATH: public window, Smart Tagging agreement and restarts, possession count -- for
alarms, never to choose.

    python solution/touch_bench.py --bar-ath
"""

import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import change_ml  # noqa: E402
import dfl_eval  # noqa: E402
import model  # noqa: E402
import seq  # noqa: E402

# Params
SETTINGS = {"k_change 0": {}, "k_change 1": {"k_change": 1.0}, "k_change 2": {"k_change": 2.0},
            "k_change 3": {"k_change": 3.0}, "k_change 2, switch 11": {"k_change": 2.0,
                                                                     "switch": 11.0,
                                                                     "switch_contact": 11.0,
                                                                     "switch_quiet": 11.0}}


def first_stages():
    from convention_bench import dfl_first_stage
    from metrica_bench import metrica_first_stage
    data = {("dfl", m): v for m, v in dfl_first_stage().items()}
    data.update({("metrica", m): v for m, v in metrica_first_stage().items()})
    return data


def fill(data):
    """seq_team / seq_dead and change_logodds per match, leave-one-match-out."""
    xs = {k: seq.features(v[5]) for k, v in data.items()}
    for key, value in data.items():
        others = [k for k in data if k != key]
        train_x, train_y = [xs[k] for k in others], [data[k][3] for k in others]
        feat = value[5]
        feat["seq_team"], feat["seq_dead"] = seq.log_odds(seq.fit(train_x, train_y), xs[key])
        feat["change_logodds"] = change_ml.log_odds(change_ml.fit(train_x, train_y), xs[key])
        print("  {} {} fitted".format(*key), flush=True)


def main(data=None, settings=SETTINGS):
    data = data if data is not None else first_stages()
    fill(data)
    base = None
    print("{:<24} {:<46} {}".format("", "DFL spellF1 (paired diff, up)  per match",
                                    "Metrica spellF1 (paired diff, up)  per match"))
    for name, params in settings.items():
        per = {}
        for (source, match), (b, p, tm, truth, bl, feat) in data.items():
            labels = dfl_eval.run_pipeline(b, p, tm, bl, params, feat)
            per.setdefault(source, []).append(dfl_eval.metrics(truth, labels)[0])
        base = base or per
        cells = []
        for source in ("dfl", "metrica"):
            f = np.array(per.get(source, []))
            if not len(f):
                cells.append("{:<46}".format("-"))
                continue
            d = f - np.array(base[source])
            cells.append("{:.4f} ({:+.4f}, {} of {})  {}".format(
                f.mean(), d.mean(), int((d > 0).sum()), len(d),
                " ".join("{:.3f}".format(v) for v in f)))
        print("{:<24} {}   {}".format(name, *cells), flush=True)


def bar_ath(settings=SETTINGS):
    import experiment
    import possession_metrics as pm
    import tags
    from features import compute
    feat, events = compute(), tags.load()
    gt = pm.label_codes(pm.read_label_csv(experiment.GT))
    print("{:<24} spellF1  macroF1  tagTeam  restart  rLag    poss   dur  Dshare".format("BAR-ATH"))
    for name, params in settings.items():
        print("{:<24} {}".format(name, experiment.score(model.predict(params, feat, events), gt,
                                                      events)), flush=True)


if __name__ == "__main__":
    main()
    if "--bar-ath" in sys.argv[1:]:
        bar_ath()
