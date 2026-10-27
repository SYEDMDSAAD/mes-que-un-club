"""Expected-spellF1 decoding (mbr.py) against plain Viterbi, on every labelled match we have.

The second stage is the v11 one (DFL + Metrica), fitted leave-one-match-out over all ten matches,
so every match is scored by models that never saw it. The final emissions of each match are built
once, exactly as model.predict builds them (first pass Viterbi -> formation -> second pass), and
then decoded by every variant. Paired differences against Viterbi are what to read: the matches
differ far more from each other than the decoders do.

Calibration columns, because MBR is only as good as the posterior it samples:

    E[n]/n    possessions per sampled path over the true count. Near 1 means the sampled
              "truths" look like the real one; far above 1 means the samples flicker (TAU too
              high), far below means they are copies of Viterbi (TAU too low).
    expF1     the spellF1 the decoder expects to reach against its own samples; compare with
              the realised one. Much higher = overconfident posterior.

Pick the variant from the DFL and Metrica columns together (it must not lose on either), then
check BAR-ATH (public window, Smart Tagging agreement, restart check, possession count) for
anything alarming. Never pick from the public window. Aggregates only.

    python solution/mbr_bench.py          # first run fits the second stage 10 times (~30 min)
"""

import os
import pickle
import sys
import time
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
import mbr  # noqa: E402
import model  # noqa: E402
import seq  # noqa: E402
import tags  # noqa: E402
import variant  # noqa: E402
from convention_bench import dfl_first_stage  # noqa: E402
from features import compute  # noqa: E402
from metrica_bench import metrica_first_stage  # noqa: E402

# Params
CACHE = variant.cache_path(os.path.join(ROOT, "cache", "bench_mbr_emissions.pkl"))
# name: (tau, prior_scale). prior_scale None tempers Viterbi's whole score by tau; a number keeps
# that multiple of the decoder's change costs as the samples' prior while the evidence is / tau.
VARIANTS = {
    "mbr tau 0.5": (0.5, None),
    "mbr tau 0.75": (0.75, None),
    "mbr tau 1": (1.0, None),
    "mbr tau 1.5": (1.5, None),
    "mbr tau 2": (2.0, None),
    "mbr tau 2, prior 0.5": (2.0, 0.5),
    "mbr tau 4, prior 0.5": (4.0, 0.5),
    "mbr tau 4, prior 0.25": (4.0, 0.25),
    "mbr tau 8, prior 0.25": (8.0, 0.25),
}
N_SAMPLES = 128


def second_stage_fills(dfl, met):
    """v11 second stage, leave-one-match-out over DFL + Metrica, written into each feat dict."""
    data = {("dfl", m): v for m, v in dfl.items()}
    data.update({("metrica", m): v for m, v in met.items()})
    x = {k: seq.features(v[5]) for k, v in data.items()}
    for key, value in data.items():
        others = [k for k in data if k != key]
        fitted = seq.fit([x[k] for k in others], [data[k][3] for k in others])
        value[5]["seq_team"], value[5]["seq_dead"] = seq.log_odds(fitted, x[key])
        print("  second stage for {} fitted".format(key[1]), flush=True)
    return data


def final_emissions(ball, players, team, blocks, feat, params=None):
    """The emissions model.predict decodes last: first pass Viterbi -> formation -> second pass."""
    loader.OVERRIDE, model.BLOCKS = (ball, players, team), blocks
    try:
        if "ball_x" not in feat:
            feat.update(model.compute())
        p = dict(model.PARAMS, **(params or {}))
        mask = model.graded_mask(len(feat["ball_x"]))
        first = model.decode(model.emissions(feat, p, mask, []), p, feat)
        form = model.formation_log_odds(feat, first, p)
        return model.emissions(feat, p, mask, [], form)
    finally:
        loader.OVERRIDE, model.BLOCKS = None, None


def prepared():
    """{(source, match): (truth, blocks, emissions)}, cached."""
    if os.path.isfile(CACHE):
        with open(CACHE, "rb") as handle:
            return pickle.load(handle)
    data = second_stage_fills(dfl_first_stage(), metrica_first_stage())
    out = {k: (v[3], v[4], final_emissions(v[0], v[1], v[2], v[4], v[5]))
           for k, v in data.items()}
    with open(CACHE, "wb") as handle:
        pickle.dump(out, handle)
    return out


def labels_from(paths, blocks, n):
    labels = np.full(n, "X", dtype="<U1")
    for (first, last), path in zip(blocks, paths):
        labels[first:last + 1] = model.STATES[path]
    return labels


def decode(emit, blocks, setting):
    """(labels, report): Viterbi when setting is None, else MBR with setting = (tau, prior_scale)."""
    cost = model.transition_costs(model.PARAMS)
    halves = [emit[first:last + 1] for first, last in blocks]
    if setting is None:
        return labels_from([model.viterbi(h, cost) for h in halves], blocks, len(emit)), {}
    tau, prior = setting
    report = {}
    paths = mbr.decode_match(halves, cost, tau=tau, prior_scale=prior, n_samples=N_SAMPLES,
                             report=report)
    return labels_from(paths, blocks, len(emit)), report


def main(variants=VARIANTS):
    data = prepared()
    results = {}
    for name, setting in [("viterbi (v12)", None)] + list(variants.items()):
        rows, t0 = [], time.time()
        for (source, match), (truth, blocks, emit) in data.items():
            labels, report = decode(emit, blocks, setting)
            r = pm.evaluate(pm.label_codes(truth), pm.label_codes(labels), bootstrap_n=0)
            rows.append((source, r["spell_f1_mean"], r["macro_f1"],
                         r["n_pred_spells"] / max(r["n_gt_spells"], 1),
                         report.get("n_truth_expected", np.nan) / max(r["n_gt_spells"], 1),
                         report.get("expected_spellF1", np.nan)))
        results[name] = rows
        base = results["viterbi (v12)"]
        line = []
        for source in ("dfl", "metrica"):
            mine = [r for r in rows if r[0] == source]
            ref = [r for r in base if r[0] == source]
            diff = np.array([a[1] - b[1] for a, b in zip(mine, ref)])
            line.append("{:.4f} ({:+.4f}, {} of {} up)  macro {:.4f}  p/t {:.2f}  E[n]/n {:.2f}  "
                        "expF1 {:.3f}".format(np.mean([r[1] for r in mine]), diff.mean(),
                                              int((diff > 0).sum()), len(diff),
                                              np.mean([r[2] for r in mine]),
                                              np.mean([r[3] for r in mine]),
                                              np.nanmean([r[4] for r in mine]),
                                              np.nanmean([r[5] for r in mine])))
        print("{:<24} DFL {} | Metrica {}  [{:.0f}s]".format(name, line[0], line[1],
                                                             time.time() - t0), flush=True)
    return results


def bar_ath(variants=VARIANTS):
    """The BAR-ATH checks for each variant: public window, Smart Tagging, possession count."""
    feat, events = compute(), tags.load()
    gt = pm.label_codes(pm.read_label_csv(experiment.GT))
    print("{:<24} spellF1  macroF1  tagTeam  restart  rLag    poss   dur  Dshare".format("BAR-ATH"))
    print("{:<24} {}".format("viterbi (v12)", experiment.score(model.predict({}, feat, events),
                                                             gt, events)))
    for name, (tau, prior) in variants.items():
        params = {"mbr_tau": tau, "mbr_prior_scale": prior, "mbr_samples": N_SAMPLES}
        print("{:<24} {}".format(name, experiment.score(model.predict(params, feat, events),
                                                      gt, events)), flush=True)


if __name__ == "__main__":
    main()
    if "--bar-ath" in sys.argv[1:]:
        bar_ath()
