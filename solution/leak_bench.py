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
bench_data.labelled(); variant "testbar" scores DFL on a BAR-like ball), against the first row
(v15.1):

    spellF1   DFL and Metrica, paired difference, matches up
    D-F1      the D class's F1 and macroF1
    stops     true stoppages (D runs of STOP_MIN frames or more) missed -- less than half of them
              called D -- out of all of them, and stoppages called where the truth has less than
              half D: the leak's own signature is a stoppage lost whole, which D-F1 and the frame
              counts dilute
    moved     share of graded frames that changed between D and alive against each REFS row

    python solution/leak_bench.py               # the labelled matches
    python solution/leak_bench.py --bar-ath     # + BAR-ATH: public window spellF1 / macroF1 /
                                                # D-F1, and over the FULL match the D share, the
                                                # stoppages, how many of v15.1's are lost and how
                                                # many are new, and the frames moved
    python solution/leak_bench.py --bar-only --ref "Mes Que Un Club.csv"
                                                # BAR-ATH only, against a labels file -- e.g.
                                                # rows of a touch variant against the v15.1 file

The full-match columns read all 158,857 graded frames without labels. "moved" turned out a weak
alarm (v15's leak at 0.5 moved 0.23% of the match yet cost the public window 0.07 of D-F1: few
stoppages, each lost whole); "lost" counts exactly those. A row should lose no more of v15.1's
stoppages than ordinary decoding changes do -- compare it with "seq 0.5", the known leak.
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
    "seq_gate 2, seq 0.7": {"seq_gate": 2.0, "k_seq_team": 0.7},
    "k_seq_dead 0.3, seq 0.5": {"k_seq_dead": 0.3, "k_seq_team": 0.5},
    "first + k_seq_dead 0.3, seq 0.7": {"alive_level": "first", "k_seq_dead": 0.3,
                                        "k_seq_team": 0.7},
}
REFS = ("v15.1 (seq 0.4)", "first, seq 0.4")   # rows "moved" is measured against
STOP_MIN = 25             # frames: a stoppage is a D run at least this long


def scores(truth, labels):
    r = pm.evaluate(pm.label_codes(truth), pm.label_codes(labels), bootstrap_n=0)
    return np.array([r["spell_f1_mean"], r["macro_f1"], r["per_class"]["D"]["f1"]])


def moved(labels, ref, graded):
    """Share of graded frames that are D in one labelling and not in the other."""
    return ((labels == "D") != (ref == "D"))[graded].mean()


def stoppages(labels, graded):
    """[(start, stop)] of the D runs of STOP_MIN frames or more on graded frames."""
    d = (np.asarray(labels) == "D") & graded
    edges = np.flatnonzero(np.diff(np.r_[0, d.astype(np.int8), 0]))
    return [(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a >= STOP_MIN]


def lost(runs, labels):
    """How many of runs keep less than half their frames D in labels."""
    labels = np.asarray(labels)
    return sum(int((labels[a:b] == "D").mean() < 0.5) for a, b in runs)


def main(settings=SETTINGS, refs=REFS):
    import bench_data
    data = bench_data.labelled()
    labels = {name: {key: dfl_eval.run_pipeline(b, p, tm, bl, params, feat)
                     for key, (b, p, tm, truth, bl, feat) in data.items()}
              for name, params in settings.items()}
    refs = [r for r in refs if r in labels]
    first = next(iter(settings))
    true_stops = {k: stoppages(v[3], v[3] != "X") for k, v in data.items()}
    print("{:<32} {:<44} {:<44} {:<30} {}".format(
        "", "DFL spellF1 (diff, up)  macro   D-F1", "Metrica spellF1 (diff, up)  macro   D-F1",
        "stops missed / false (DFL; Metrica)", "moved vs " + " / ".join(refs) + " (DFL; Metrica)"))
    for name in settings:
        cells, mv, st = [], [], []
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
            missed = sum(lost(true_stops[k], labels[name][k]) for k in keys)
            false = sum(lost(stoppages(labels[name][k], data[k][3] != "X"), data[k][3])
                        for k in keys)
            st.append("{}/{} {}".format(missed, sum(len(true_stops[k]) for k in keys), false))
            mv.append("/".join("{:.2%}".format(np.mean([moved(labels[name][k], labels[r][k],
                                                              data[k][3] != "X") for k in keys]))
                               for r in refs))
        print("{:<32} {:<44} {:<44} {:<30} {}".format(name, cells[0], cells[1], "; ".join(st),
                                                      "; ".join(mv)), flush=True)


def bar_ath(settings=SETTINGS, refs=REFS, ref_csv=None):
    """ref_csv: a labels file (e.g. the v15.1 submission) to count lost / new stoppages and moved
    frames against instead of the first REFS row -- to compare across variants (touch...)."""
    if dfl_eval.TEST_BAR:
        print("(--bar-ath skipped: testbar changes only how the DFL matches are scored)")
        return
    import pandas as pd
    import experiment
    import tags
    from features import compute
    feat, events = compute(), tags.load()
    gt = pm.label_codes(pm.read_label_csv(experiment.GT))
    graded = model.graded_mask(len(feat["ball_x"]))
    labels = {name: model.predict(params, feat, events) for name, params in settings.items()}
    refs = [r for r in refs if r in labels]
    if ref_csv:
        labels_ref = pd.read_csv(ref_csv)["label"].astype(str).values
        ref_name, compare = os.path.basename(ref_csv), {os.path.basename(ref_csv): labels_ref}
    else:
        ref_name = refs[0] if refs else None
        compare = {r: labels[r] for r in refs}
    reference = compare.get(ref_name)
    ref_stops = stoppages(reference, graded) if reference is not None else []
    print("{:<32} public: spellF1  macroF1  D-F1   | full match: D share  stops  lost  new  "
          "(vs {})  moved vs {}".format("BAR-ATH", ref_name or "-", " / ".join(compare)))
    if reference is not None:
        print("{:<32} {:>58} {:.3f}   {:4d}".format(ref_name, "|", (reference[graded] == "D").mean(),
                                                    len(ref_stops)))
    for name, lab in labels.items():
        r = pm.evaluate(gt, pm.label_codes(lab), bootstrap_n=0)
        mine = stoppages(lab, graded)
        print("{:<32}         {:.4f}   {:.4f}   {:.4f} |             {:.3f}   {:4d}  {:4d} {:4d}"
              "            {}".format(
                  name, r["spell_f1_mean"], r["macro_f1"], r["per_class"]["D"]["f1"],
                  (lab[graded] == "D").mean(), len(mine),
                  lost(ref_stops, lab) if reference is not None else 0,
                  lost(mine, reference) if reference is not None else 0,
                  " / ".join("{:.2%}".format(moved(lab, ref, graded)) for ref in compare.values())),
              flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--bar-only" not in args:
        main()
    if "--bar-ath" in args or "--bar-only" in args:
        bar_ath(ref_csv=args[args.index("--ref") + 1] if "--ref" in args else None)
