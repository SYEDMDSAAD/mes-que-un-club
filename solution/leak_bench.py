"""The stoppage leak: how the team evidence's strength moves frames between alive and dead.
Aggregates only.

v15's motion model and the touch features both raised DFL's spellF1 at second-stage weight 0.5,
and both pulled BAR-ATH's stoppage frames into play (public D-class F1 0.888 -> 0.823). The cause
is in the decoder (model.PARAMS alive_level): Viterbi scores a live frame by the better of H and A,
-log(1 + e^-|team|), so the surer the team evidence, the more alive a frame looks -- by up to 0.69
a frame, about what the dead evidence itself weighs. Inside a stoppage the second stage is often
sure of the team (the players line up for the restart), so the stoppage turns into possession.
Lowering k_seq_team hides it and gives up the team evidence everywhere else.

Every SETTINGS row on every labelled match (second stage and change detector leave-one-match-out,
the cache dfl_events.py and ceiling_bench.py share), against the first row (v15.1):

    spellF1   DFL and Metrica, paired difference, matches up
    D-F1      the D class's F1, the leak's own figure, and macroF1
    moved     share of graded frames that changed between D and alive against each REFS row:
              v15.1, and "first, seq 0.4" -- with alive_level "first" the stoppages should stay
              put (moved ~0) however high the second-stage weight goes

    python solution/leak_bench.py               # the labelled matches
    python solution/leak_bench.py --bar-ath     # + BAR-ATH: public window spellF1 / macroF1 /
                                                # D-F1, and over the FULL match the D share and
                                                # the frames moved between D and alive

The full-match columns are the gate the public window cannot be (it holds a few stoppages). They
need no labels, so they read all 158,857 graded frames: once the fix is in, raising the team
weight (or adding touch or video evidence) must leave BAR-ATH's stoppages where they are --
moved against "first, seq 0.4" well under 0.5% of the match -- while the one-off move from v15.1
to "first" is judged on the public window's D-F1 and on the labelled matches' D-F1.
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
import model  # noqa: E402

# Params
SETTINGS = {
    "v15.1 (seq 0.4)": {},
    "seq 0.5": {"k_seq_team": 0.5},
    "seq 0.7": {"k_seq_team": 0.7},
    "first, seq 0.4": {"alive_level": "first"},
    "first, seq 0.5": {"alive_level": "first", "k_seq_team": 0.5},
    "first, seq 0.7": {"alive_level": "first", "k_seq_team": 0.7},
    "first, seq 1.0": {"alive_level": "first", "k_seq_team": 1.0},
    "seq_gate 1, seq 0.5": {"seq_gate": 1.0, "k_seq_team": 0.5},
    "seq_gate 1, seq 0.7": {"seq_gate": 1.0, "k_seq_team": 0.7},
    "k_seq_dead 0.3, seq 0.5": {"k_seq_dead": 0.3, "k_seq_team": 0.5},
    "first + k_seq_dead 0.3, seq 0.7": {"alive_level": "first", "k_seq_dead": 0.3,
                                        "k_seq_team": 0.7},
}
REFS = ("v15.1 (seq 0.4)", "first, seq 0.4")   # rows "moved" is measured against


def scores(truth, labels):
    r = pm.evaluate(pm.label_codes(truth), pm.label_codes(labels), bootstrap_n=0)
    return np.array([r["spell_f1_mean"], r["macro_f1"], r["per_class"]["D"]["f1"]])


def moved(labels, ref, graded):
    """Share of graded frames that are D in one labelling and not in the other."""
    return ((labels == "D") != (ref == "D"))[graded].mean()


def main(settings=SETTINGS, refs=REFS):
    import dfl_events
    import touch_bench
    data = touch_bench.first_stages()
    dfl_events.fills(data)
    labels = {name: {key: dfl_eval.run_pipeline(b, p, tm, bl, params, feat)
                     for key, (b, p, tm, truth, bl, feat) in data.items()}
              for name, params in settings.items()}
    refs = [r for r in refs if r in labels]
    first = next(iter(settings))
    print("{:<32} {:<44} {:<44} {}".format(
        "", "DFL spellF1 (diff, up)  macro   D-F1", "Metrica spellF1 (diff, up)  macro   D-F1",
        "moved D<->alive vs " + " / ".join(refs) + " (DFL, Metrica)"))
    for name in settings:
        cells, mv = [], []
        for source in ("dfl", "metrica"):
            keys = [k for k in data if k[0] == source]
            if not keys:
                cells.append("{:<44}".format("-"))
                continue
            rows = np.array([scores(data[k][3], labels[name][k]) for k in keys])
            base = np.array([scores(data[k][3], labels[first][k]) for k in keys])
            d = rows[:, 0] - base[:, 0]
            cells.append("{:.4f} ({:+.4f}, {} of {})  {:.4f}  {:.4f}".format(
                rows[:, 0].mean(), d.mean(), int((d > 0).sum()), len(d), rows[:, 1].mean(),
                rows[:, 2].mean()))
            mv.append("/".join("{:.2%}".format(np.mean([moved(labels[name][k], labels[r][k],
                                                              data[k][3] != "X") for k in keys]))
                               for r in refs))
        print("{:<32} {:<44} {:<44} {}".format(name, cells[0], cells[1], "  ".join(mv)),
              flush=True)


def bar_ath(settings=SETTINGS, refs=REFS):
    import experiment
    import tags
    from features import compute
    feat, events = compute(), tags.load()
    gt = pm.label_codes(pm.read_label_csv(experiment.GT))
    graded = model.graded_mask(len(feat["ball_x"]))
    labels = {name: model.predict(params, feat, events) for name, params in settings.items()}
    refs = [r for r in refs if r in labels]
    print("{:<32} public: spellF1  macroF1  D-F1   | full match: D share  moved vs {}".format(
        "BAR-ATH", " / ".join(refs)))
    for name, lab in labels.items():
        r = pm.evaluate(gt, pm.label_codes(lab), bootstrap_n=0)
        print("{:<32}         {:.4f}   {:.4f}   {:.4f} |             {:.3f}    {}".format(
            name, r["spell_f1_mean"], r["macro_f1"], r["per_class"]["D"]["f1"],
            (lab[graded] == "D").mean(),
            " / ".join("{:.2%}".format(moved(lab, labels[ref], graded)) for ref in refs)),
            flush=True)


if __name__ == "__main__":
    main()
    if "--bar-ath" in sys.argv[1:]:
        bar_ath()
