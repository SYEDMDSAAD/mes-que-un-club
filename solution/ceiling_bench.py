"""How far could each kind of better evidence take spellF1? Oracle ceilings on the labelled matches.

Each row hands the decoder one part of the truth and decodes again, on the emissions that
mbr_bench.prepared() caches (DFL + Metrica, second stage leave-one-match-out, final pass exactly
as model.predict builds it):

    pipeline             Viterbi on the emissions as they are
    perfect dead         the D evidence replaced by the truth (+-M per frame); team evidence as is
    perfect team         the team evidence replaced by the truth on live frames; D evidence as is
    perfect change times an H <-> A change costs 1 within +-TOL frames of a true change in that
                         direction and 2 x switch elsewhere; the evidence as is. The ceiling of a
                         touch / change detector that is never wrong about *when*
    perfect boundaries   the pipeline's labels with every change moved onto a true change of the
                         same kind within +-SNAP frames: nothing found or lost, edges placed
    + short possessions  the pipeline's labels with every true possession shorter than SHORT
                         frames written in from the truth: the ceiling of catching the scrambles
    perfect team + dead  both evidence oracles: what is left is the decoder's own costs

Read the gaps to the first row, not the levels. A big gap means that lever has room; a small one
means even perfect evidence of that kind would not move the score -- whatever it does on the
public window. F1 at IoU 0.3 and 0.7 are printed too: a lever that lifts 0.7 much more than 0.3
is about edges, one that lifts both is about finding possessions. Aggregates only.

    python solution/ceiling_bench.py          # reuses cache/bench_mbr_emissions.pkl if present
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
import model  # noqa: E402

# Params
M = 5.0                   # log-likelihood per frame of an oracle frame
TOL = 3                   # frames: the cheap window around a true change
SNAP = 50                 # frames: how far a boundary may be moved onto a true one
SHORT = 75                # frames: true possessions shorter than this are written in (3 s)
H, A, D = 0, 1, 2
CODES = {"H": H, "A": A, "D": D}


def codes(labels, first, last):
    return np.array([CODES[c] for c in labels[first:last + 1]], dtype=np.int8)


def labels_of(paths, blocks, n):
    out = np.full(n, "X", dtype="<U1")
    for (first, last), path in zip(blocks, paths):
        out[first:last + 1] = model.STATES[path]
    return out


def team_of(emit):
    """The team log-odds an emission row was built from: log P(H|alive) - log P(A|alive)."""
    return emit[:, H] - emit[:, A]


def with_team(emit, team):
    out = emit.copy()
    out[:, H] = -np.logaddexp(0.0, -team)
    out[:, A] = -np.logaddexp(0.0, team)
    return out


def viterbi_half(emit, truth=None, change_oracle=False):
    cost = model.transition_costs(model.PARAMS)
    if not change_oracle:
        return model.viterbi(emit, cost)
    n = len(emit)
    to_away = np.full(n, 2 * cost[H, A])
    to_home = np.full(n, 2 * cost[A, H])
    for i in np.flatnonzero(truth[1:] != truth[:-1]) + 1:
        a, b = truth[i - 1], truth[i]
        if a == H and b == A:
            to_away[max(0, i - TOL):i + TOL + 1] = 1.0
        elif a == A and b == H:
            to_home[max(0, i - TOL):i + TOL + 1] = 1.0
    return model.viterbi(emit, cost, to_away, to_home)


def boundaries(path):
    i = np.flatnonzero(path[1:] != path[:-1]) + 1
    return [(int(k), int(path[k - 1]), int(path[k])) for k in i]


def snap_boundaries(pred, truth):
    """pred with each boundary moved onto the nearest unused true boundary of the same kind within
    SNAP frames, never past a neighbouring boundary, so no segment appears or disappears."""
    pb, tb = boundaries(pred), boundaries(truth)
    by_kind = {}
    for pos, a, b in tb:
        by_kind.setdefault((a, b), []).append(pos)
    for kind in by_kind:
        by_kind[kind] = np.array(by_kind[kind])
    cand = []
    for j, (pos, a, b) in enumerate(pb):
        tp = by_kind.get((a, b))
        if tp is None:
            continue
        near = tp[np.abs(tp - pos) <= SNAP]
        for t in near.tolist():
            cand.append((abs(t - pos), j, t))
    cand.sort()
    new = [pos for pos, _, _ in pb]
    used_t, used_j = set(), set()
    for _, j, t in cand:
        if j in used_j or t in used_t:
            continue
        lo = new[j - 1] if j else 0
        hi = new[j + 1] if j + 1 < len(new) else len(pred)
        if lo < t < hi:
            new[j] = t
            used_j.add(j)
            used_t.add(t)
    out = np.empty_like(pred)
    starts = [0] + new
    labs = [pred[0]] + [b for _, _, b in pb]
    stops = new + [len(pred)]
    for s, e, lab in zip(starts, stops, labs):
        out[s:e] = lab
    return out


def add_short(pred, truth):
    out = pred.copy()
    for s, e, lab in boundaries_to_runs(truth):
        if lab != D and e - s < SHORT:
            out[s:e] = truth[s:e]
    return out


def boundaries_to_runs(path):
    b = np.flatnonzero(path[1:] != path[:-1]) + 1
    starts, stops = np.r_[0, b], np.r_[b, len(path)]
    return [(int(s), int(e), int(path[s])) for s, e in zip(starts, stops)]


def variants(truth_half, emit_half):
    """{name: decoded path} for one half."""
    t = truth_half
    out = {"pipeline": viterbi_half(emit_half)}
    dead = np.where(t == D, M, -M)
    oracle_dead = emit_half.copy()
    oracle_dead[:, D] = dead
    out["perfect dead"] = viterbi_half(oracle_dead)
    team = np.where(t == H, M, np.where(t == A, -M, 0.0))
    out["perfect team"] = viterbi_half(with_team(emit_half, team))
    out["perfect change times"] = viterbi_half(emit_half, t, change_oracle=True)
    out["perfect boundaries"] = snap_boundaries(out["pipeline"], t)
    out["+ short possessions"] = add_short(out["pipeline"], t)
    both = with_team(emit_half, team)
    both[:, D] = dead
    out["perfect team + dead"] = viterbi_half(both)
    return out


def evaluate(truth, blocks, emit):
    """{name: pm.evaluate result} for one match."""
    per = {}
    for first, last in blocks:
        t = codes(truth, first, last)
        for name, path in variants(t, emit[first:last + 1]).items():
            per.setdefault(name, []).append(path)
    return {name: pm.evaluate(pm.label_codes(truth),
                              pm.label_codes(labels_of(paths, blocks, len(truth))), bootstrap_n=0)
            for name, paths in per.items()}


def main(data=None):
    if data is None:
        import mbr_bench
        data = mbr_bench.prepared()
    results = {}
    for key, (truth, blocks, emit) in data.items():
        results[key] = evaluate(truth, blocks, emit)
    names = list(next(iter(results.values())))
    sources = sorted({k[0] for k in results})
    print("{:<22}".format("") + "".join(
        "| {:<52}".format("{} ({} matches): spellF1  gain  F1@0.3  F1@0.7  p/t".format(
            s, sum(k[0] == s for k in results))) for s in sources))
    for name in names:
        line = "{:<22}".format(name)
        for s in sources:
            rows = [r for k, r in results.items() if k[0] == s]
            f = np.mean([r[name]["spell_f1_mean"] for r in rows])
            base = np.mean([r["pipeline"]["spell_f1_mean"] for r in rows])
            lo = np.mean([r[name]["spell_f1"][0.3]["f1"] for r in rows])
            hi = np.mean([r[name]["spell_f1"][0.7]["f1"] for r in rows])
            pt = np.mean([r[name]["n_pred_spells"] / max(r[name]["n_gt_spells"], 1) for r in rows])
            line += "| {:.4f}  {:+.3f}  {:.3f}   {:.3f}   {:.2f}{}".format(f, f - base, lo, hi, pt,
                                                                        " " * 13)
        print(line, flush=True)
    return results


if __name__ == "__main__":
    main()
