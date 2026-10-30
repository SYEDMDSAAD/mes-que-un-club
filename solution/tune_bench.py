"""Decode-only retune on the labelled matches. Aggregates only.

Every value swept here moves the decoder, or the weight of evidence that is already computed --
never a trained model -- so a sweep costs decodes only (bench_data.labelled() is cached). The
values in model.PARAMS were set on DFL damaged with BAR-ATH's ball gaps laid by clock time, where
30-37% of in-play frames lack a ball (BAR-ATH: 7.4%). Under a BAR-like test ball
(MQUC_VARIANT=...+testbar+testblurS+testslowS, see domain_check.py --calibrate) the sweep asks
which settings suit our ball; without it, it checks that they do not break the usual bench.

    python solution/tune_bench.py [--jobs N] [--only switch,k_seq_team] [--base "k=v,..."]
    python solution/tune_bench.py --baseline [--against TAG]
    python solution/tune_bench.py --combo "switch=10,k_seq_team=0.9" [--jobs N] [--against TAG]
    python solution/tune_bench.py --bar-ath [--combo "..."] [--ref "Mes Que Un Club.csv"]

Every run saves its default row's per-match scores to cache/tune_base[.variant].json (numbers
only), and --against TAG scores each row against the default row saved by the run with
MQUC_VARIANT=TAG, paired match by match -- the way to compare models trained differently
(seqbar against the submission's, both on the same BAR-like ball), which one run cannot.
--baseline runs that default row alone.

Each row, against the default (model.PARAMS, or --base on top of it):

    DFL / Metrica   mean spellF1, paired difference, matches up
    2 s             the paired difference against labels with every possession under 2 s given
                    back to the surrounding team (convention_bench.SCORE_REWRITE): CHALLENGE.md's
                    "controlled possession", which the public window suggests (12% of its
                    possessions under 2 s, DFL 30%)
    macro  D-F1     mean over every labelled match
    p/t             predicted / true possessions

After a sweep, per parameter, the NESTED gain: for each labelled match, the value (default
included) with the best mean difference on the OTHER matches -- original and 2 s labels weighted
equally -- scored on that match. That is what tuning the parameter is worth on a match it was not
tuned on; a row's own best difference is optimistic, this is not. "suggested" combines every
parameter whose nested gain is positive on DFL with both labels and not negative on Metrica, at
its best value over all matches. Confirm it with --combo, which also catches interactions, then on
BAR-ATH with --bar-ath -- under a variant WITHOUT testbar, which only changes how the DFL matches
are scored (keep seqbar / seqmix if the run had them: they change the model BAR-ATH gets).
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
import convention  # noqa: E402
import convention_bench  # noqa: E402
import dfl_eval  # noqa: E402

# Params
# name: the PARAMS overrides of each value tried (one parameter at a time; the default is implied)
SWEEP = {
    "switch": [{"switch": v, "switch_contact": v, "switch_quiet": v} for v in (7.0, 8.0, 10.0, 11.0)],
    "stop": [{"stop": v} for v in (12.0, 18.0)],
    "start": [{"start": v} for v in (12.0, 18.0)],
    "k_seq_team": [{"k_seq_team": v} for v in (0.5, 0.9, 1.2)],
    "seq_gate": [{"seq_gate": v} for v in (0.5, 2.0)],
    "k_change": [{"k_change": v} for v in (1.0, 3.0)],
    "change_min": [{"change_min": v} for v in (1.0, 3.0)],
    "k_tml": [{"k_tml": v} for v in (0.1, 0.35)],
    "k_team": [{"k_team": v} for v in (0.5, 1.5)],
    "near/far": [{"near": a, "far": b} for a, b in ((0.5, 2.0), (1.5, 4.0), (2.0, 5.0))],
    "k_bf": [{"k_bf": v} for v in (0.1, 0.4)],
    "k_form_missing": [{"k_form_missing": v} for v in (0.2, 0.6)],
    "k_form_seen": [{"k_form_seen": v} for v in (0.0, 0.2)],
    "k_ml": [{"k_ml": v} for v in (0.15, 0.3)],
    "min_spell": [{"min_spell": v} for v in (25, 50)],
}
DATA, TRUTH2 = None, None        # set by load(); module globals so forked workers share them


def load():
    global DATA, TRUTH2
    import bench_data
    DATA = bench_data.labelled()
    TRUTH2 = {k: convention.rewrite(v[3], *convention_bench.SCORE_REWRITE) for k, v in DATA.items()}


def parse(text):
    """'switch=10,k_seq_team=0.9' -> {"switch": 10.0, ..., "k_seq_team": 0.9}. "switch" alone sets
    switch_contact and switch_quiet too, as the sweep does: the decoder charges those wherever the
    ball is seen, so "switch" by itself would move the cost only where it is missing."""
    out = {}
    for item in filter(None, (t.strip() for t in (text or "").split(","))):
        key, value = item.split("=")
        try:
            out[key.strip()] = float(value)
        except ValueError:
            out[key.strip()] = value.strip()
    if "switch" in out:
        for key in ("switch_contact", "switch_quiet"):
            out.setdefault(key, out["switch"])
    return out


def show_params(params):
    return ",".join("{}={:g}".format(k, v) if isinstance(v, float) else "{}={}".format(k, v)
                    for k, v in params.items())


def run(params):
    """{key: (spellF1, spellF1 vs 2 s labels, macroF1, D-F1, predicted / true possessions)}."""
    out = {}
    for key, (ball, players, team, truth, blocks, feat) in DATA.items():
        labels = dfl_eval.run_pipeline(ball, players, team, blocks, params, feat)
        codes = pm.label_codes(labels)
        r = pm.evaluate(pm.label_codes(truth), codes, bootstrap_n=0)
        r2 = pm.evaluate(pm.label_codes(TRUTH2[key]), codes, bootstrap_n=0)
        out[key] = (r["spell_f1_mean"], r2["spell_f1_mean"], r["macro_f1"],
                    r["per_class"]["D"]["f1"], r["n_pred_spells"] / max(r["n_gt_spells"], 1))
    return out


def _work(item):
    return item[0], run(item[1])


def saved_path(tag):
    root, ext = os.path.splitext(os.path.join(ROOT, "cache", "tune_base.json"))
    return "{}.{}{}".format(root, tag, ext) if tag else root + ext


def save(res):
    import json
    with open(saved_path(os.environ.get("MQUC_VARIANT", "").strip()), "w") as handle:
        json.dump({"{}|{}".format(*k): list(v) for k, v in res.items()}, handle)


def against(tag):
    """The default row's per-match scores saved by the run of variant TAG."""
    import json
    if not os.path.isfile(saved_path(tag)):
        sys.exit("no saved default for {!r}: run MQUC_VARIANT={!r} python solution/tune_bench.py "
                 "--baseline first".format(tag, tag))
    with open(saved_path(tag)) as handle:
        return {tuple(k.split("|", 1)): tuple(v) for k, v in json.load(handle).items()}


def run_all(settings, jobs=1, other=None):
    """{name: run(params)} for an ordered {name: params}, printing each row as it lands -- against
    the first row, and against `other` ((tag, saved scores)) when given."""
    items = list(settings.items())
    first = items[0][0]
    results = {first: run(items[0][1])}            # the base first: it also fills feat caches
    if not items[0][1]:                            # the variant's own default (no --base)
        save(results[first])

    def show(name, res):
        row(name, res, results[first])
        if other:
            row("  vs {}".format(other[0] or "(no variant)"), res, other[1])

    show(first, results[first])
    rest = items[1:]
    if jobs > 1 and rest:
        import multiprocessing
        with multiprocessing.get_context("fork").Pool(jobs) as pool:
            for name, res in pool.imap(_work, rest):
                results[name] = res
                show(name, res)
    else:
        for name, params in rest:
            results[name] = run(params)
            show(name, results[name])
    return results


def header():
    print("{:<34} {:<38} {:<38} {:>6} {:>6} {:>5}".format(
        "", "DFL spellF1 (diff, up)  2 s diff (up)", "Metrica spellF1 (diff, up)  2 s diff (up)",
        "macro", "D-F1", "p/t"), flush=True)


def row(name, res, base):
    cells = []
    for source in ("dfl", "metrica"):
        keys = [k for k in res if k[0] == source and k in base]
        if not keys:
            cells.append("-")
            continue
        s = np.array([res[k][0] for k in keys])
        d = s - np.array([base[k][0] for k in keys])
        d2 = np.array([res[k][1] - base[k][1] for k in keys])
        cells.append("{:.4f} ({:+.4f}, {}/{})  {:+.4f} ({})".format(
            s.mean(), d.mean(), int((d > 0).sum()), len(d), d2.mean(), int((d2 > 0).sum())))
    allr = np.array(list(res.values()))
    print("{:<34} {:<38} {:<38} {:>6.4f} {:>6.4f} {:>5.2f}".format(
        name[:34], cells[0], cells[1], allr[:, 2].mean(), allr[:, 3].mean(), allr[:, 4].mean()),
        flush=True)


def nested(results, groups, base_name):
    """Per parameter: held-out gain by source and label set, and its best value over all matches."""
    keys = list(results[base_name])
    base = results[base_name]

    def diff(name, k):                       # original and 2 s labels weighted equally
        return 0.5 * ((results[name][k][0] - base[k][0]) + (results[name][k][1] - base[k][1]))

    print("\nnested gain (value chosen on the other matches, scored on the held-out one)")
    print("{:<16} {:>10} {:>10} {:>10} {:>10}   {}".format(
        "", "DFL", "DFL 2 s", "Metrica", "Met. 2 s", "best value over all matches"))
    suggested = {}
    for group, names in groups.items():
        cands = [base_name] + names
        gains = {k: None for k in keys}
        for k in keys:
            others = [j for j in keys if j != k]
            score = [np.mean([diff(c, j) for j in others]) for c in cands]
            best = cands[int(np.argmax(score))] if max(score) > 0 else base_name
            gains[k] = (results[best][k][0] - base[k][0], results[best][k][1] - base[k][1])
        cell = []
        for source in ("dfl", "metrica"):
            ks = [k for k in keys if k[0] == source]
            for i in (0, 1):
                cell.append(np.mean([gains[k][i] for k in ks]) if ks else float("nan"))
        overall = [np.mean([diff(c, j) for j in keys]) for c in cands]
        best_all = cands[int(np.argmax(overall))]
        print("{:<16} {:>+10.4f} {:>+10.4f} {:>+10.4f} {:>+10.4f}   {}".format(
            group, cell[0], cell[1], cell[2], cell[3],
            "(default)" if best_all == base_name else best_all), flush=True)
        met = np.nanmean([cell[2], cell[3]]) if np.isfinite(cell[2]) else 0.0
        if best_all != base_name and cell[0] > 0 and cell[1] > 0 and met >= 0:
            suggested.update(SETTINGS_OF[best_all])
    print("suggested: --combo \"{}\"".format(show_params(suggested)) if suggested
          else "suggested: nothing (keep the default)", flush=True)
    return suggested


SETTINGS_OF = {}


def sweep(only=None, jobs=1, base=None):
    base = base or {}
    settings, groups = {"default" + (" + base" if base else ""): dict(base)}, {}
    for group, values in SWEEP.items():
        if only and group not in only:
            continue
        groups[group] = []
        for overrides in values:
            shown = {k: v for k, v in overrides.items() if k in group.split("/")} or overrides
            name = "{}: {}".format(group, show_params(shown))  # "switch: switch=10" sets all 3
            settings[name] = dict(base, **overrides)
            SETTINGS_OF[name] = overrides
            groups[group].append(name)
    header()
    results = run_all(settings, jobs)
    return nested(results, groups, next(iter(settings)))


def combo(params, jobs=1, base=None, other=None):
    """The default (and --base) row, then the combination if any."""
    base = base or {}
    header()
    settings = {"default" + (" + base" if base else ""): dict(base)}
    if params:
        settings["combo " + show_params(params)] = dict(base, **params)
    run_all(settings, jobs, other)


def bar(params, ref_csv=None, base=None):
    """BAR-ATH guards (leak_bench.bar_ath): the default (and --base), and the combination if any."""
    import leak_bench
    base = base or {}
    settings = {"default": dict(base)}
    if params:
        settings["combo"] = dict(base, **params)
    leak_bench.bar_ath(settings, refs=("default",), ref_csv=ref_csv)


if __name__ == "__main__":
    args = sys.argv[1:]

    def arg(name, default=None):
        return args[args.index(name) + 1] if name in args else default

    jobs = int(arg("--jobs", 1))
    base = parse(arg("--base"))
    if "--bar-ath" in args:
        bar(parse(arg("--combo")), arg("--ref"), base)
    else:
        other = (arg("--against"), against(arg("--against"))) if "--against" in args else None
        load()
        print("variant {!r}: {} labelled matches".format(os.environ.get("MQUC_VARIANT", ""),
                                                          len(DATA)), flush=True)
        if "--combo" in args or "--baseline" in args:
            combo(parse(arg("--combo")), jobs, base, other)
        else:
            only = set(arg("--only").split(",")) if arg("--only") else None
            sweep(only, jobs, base)
