"""How far could each kind of better evidence take spellF1? Oracle ceilings on the labelled matches.

Each row hands the decoder one part of the truth and decodes again, on the final emissions and
per-frame switch costs of the submission's decoder (DFL + Metrica; second stage AND change
detector fitted leave-one-match-out, as touch_bench.py and dfl_events.py fit them, shared cache;
final pass exactly as model.predict builds it, k_change relief included):

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

and, splitting "perfect team" by where the evidence goes wrong:

    team: ball missing   the truth's team only on in-play frames without a ball
    team: nearest wrong  only on frames with a ball whose nearest player is on the wrong team --
                         the tracking errors no ball-based model can see through
    team: short poss.    only inside true possessions shorter than SHORT frames

The team rows change only H against A. Up to v15 they rebuilt H and A from the oracle log-odds,
which also lifted every true live frame to the alive level of a certain frame (+0.69 a frame
over a frame without team evidence, model.PARAMS alive_level): part of each team row was really
alive / dead information. LEVEL = "keep" (v16) keeps each frame's level, so a team row is "who"
only; LEVEL = "posterior" gives the v15 rows.

Then, per frame type of the true in-play frames, how the pipeline's labels do there: the share of
in-play frames of that type, how often the team is right where the pipeline says H / A, and how
often it says D instead. Compare the shares with BAR-ATH's before reading a row's gain as ours:
the DFL damage lays our ball gaps over DFL by clock time, so far more of DFL's in-play frames
lack a ball than of ours (domain_check.py, the "seen" row by state: live frames with a ball).

Read the gaps to the first row, not the levels. A big gap means that lever has room; a small one
means even perfect evidence of that kind would not move the score -- whatever it does on the
public window. F1 at IoU 0.3 and 0.7 are printed too: a lever that lifts 0.7 much more than 0.3
is about edges, one that lifts both is about finding possessions. Aggregates only.

    python solution/ceiling_bench.py          # reuses cache/bench_ceiling.<PARAMS hash>.pkl
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
LEVEL = "keep"            # team rows: "keep" each frame's alive level (v16), "posterior" (v15)
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
    """emit with H - A = team. LEVEL "keep": the better of H and A stays where it was."""
    out = emit.copy()
    if LEVEL == "keep":
        level = np.maximum(emit[:, H], emit[:, A])
        out[:, H] = level + np.minimum(0.0, team)
        out[:, A] = level + np.minimum(0.0, -team)
    else:
        out[:, H] = -np.logaddexp(0.0, -team)
        out[:, A] = -np.logaddexp(0.0, team)
    return out


def viterbi_half(emit, truth=None, change_oracle=False, costs=None):
    """costs: the decoder's per-frame (to_away, to_home) switch costs for this half, or None."""
    cost = model.transition_costs(model.PARAMS)
    if not change_oracle:
        if costs is None:
            return model.viterbi(emit, cost)
        return model.viterbi(emit, cost, costs[0], costs[1])
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


def team_where(emit_half, truth_half, mask):
    """Emissions with the team evidence replaced by the truth on mask only."""
    team = team_of(emit_half)
    team[mask] = np.where(truth_half[mask] == H, M, -M)
    return with_team(emit_half, team)


def variants(truth_half, emit_half, flags_half=None, costs_half=None):
    """{name: decoded path} for one half. flags_half: (ball seen, nearest player home) or None;
    costs_half: the decoder's (to_away, to_home) or None (plain transition costs)."""
    t = truth_half
    c = costs_half
    out = {"pipeline": viterbi_half(emit_half, costs=c)}
    dead = np.where(t == D, M, -M)
    oracle_dead = emit_half.copy()
    oracle_dead[:, D] = dead
    out["perfect dead"] = viterbi_half(oracle_dead, costs=c)
    team = np.where(t == H, M, np.where(t == A, -M, 0.0))
    out["perfect team"] = viterbi_half(with_team(emit_half, team), costs=c)
    out["perfect change times"] = viterbi_half(emit_half, t, change_oracle=True)
    out["perfect boundaries"] = snap_boundaries(out["pipeline"], t)
    out["+ short possessions"] = add_short(out["pipeline"], t)
    both = with_team(emit_half, team)
    both[:, D] = dead
    out["perfect team + dead"] = viterbi_half(both, costs=c)
    live = t != D
    if flags_half is not None:
        seen, near_home = flags_half
        out["team: ball missing"] = viterbi_half(team_where(emit_half, t, live & ~seen), costs=c)
        wrong = live & seen & (near_home != (t == H))
        out["team: nearest wrong"] = viterbi_half(team_where(emit_half, t, wrong), costs=c)
    out["team: short poss."] = viterbi_half(team_where(emit_half, t, short_mask(t)), costs=c)
    return out


def short_mask(t):
    """True possessions shorter than SHORT frames."""
    short = np.zeros(len(t), bool)
    for s0, e0, lab in boundaries_to_runs(t):
        if lab != D and e0 - s0 < SHORT:
            short[s0:e0] = True
    return short


def frame_types(t, path, flags_half):
    """{type: [in-play frames, team right where the path says H / A, path says H / A, path says
    D]} for one half: the pipeline's labels by the kind of evidence each true in-play frame has."""
    live = t != D
    types = {"all in play": live, "inside short poss.": live & short_mask(t)}
    if flags_half is not None:
        seen, near_home = flags_half
        types["ball missing"] = live & ~seen
        types["nearest right"] = live & seen & (near_home == (t == H))
        types["nearest wrong"] = live & seen & (near_home != (t == H))
    out = {}
    for name, mask in types.items():
        alive = mask & (path != D)
        out[name] = np.array([mask.sum(), (alive & (path == t)).sum(), alive.sum(),
                              (mask & (path == D)).sum()], dtype=float)
    return out


def evaluate(truth, blocks, emit, flags=None, costs=None):
    """({name: pm.evaluate result}, frame-type counts) for one match. flags: (ball seen, nearest
    home) arrays or None; costs: the decoder's per-frame (to_away, to_home) or None."""
    per, types = {}, {}
    for first, last in blocks:
        t = codes(truth, first, last)
        fl = None if flags is None else tuple(f[first:last + 1] for f in flags)
        ch = None if costs is None else tuple(np.asarray(c)[first:last + 1] for c in costs)
        for name, path in variants(t, emit[first:last + 1], fl, ch).items():
            per.setdefault(name, []).append(path)
        for name, counts in frame_types(t, per["pipeline"][-1], fl).items():
            types[name] = types.get(name, 0) + counts
    return ({name: pm.evaluate(pm.label_codes(truth),
                               pm.label_codes(labels_of(paths, blocks, len(truth))), bootstrap_n=0)
             for name, paths in per.items()}, types)


def prepared(params=None):
    """{(source, match): (truth, blocks, final emissions, (to_away, to_home), (ball seen, nearest
    player home))} -- the final decode of model.predict on every labelled match, cached for the
    submission's PARAMS (params=None; the file name carries a hash of them, so a changed default
    never reads a stale cache)."""
    import hashlib
    import pickle
    import variant
    tag = hashlib.md5(repr(sorted(model.PARAMS.items())).encode()).hexdigest()[:8]
    path = variant.cache_path(os.path.join(ROOT, "cache", "bench_ceiling.{}.pkl".format(tag)))
    if params is None and os.path.isfile(path):
        with open(path, "rb") as handle:
            return pickle.load(handle)
    import dfl_events
    import load as loader
    import mbr_bench
    import touch_bench
    data = touch_bench.first_stages()
    dfl_events.fills(data)
    p = dict(model.PARAMS, **(params or {}))
    out = {}
    for key, (ball, players, team, truth, blocks, feat) in data.items():
        emit = mbr_bench.final_emissions(ball, players, team, blocks, feat, params)
        loader.OVERRIDE, model.BLOCKS = (ball, players, team), blocks
        try:
            costs = tuple(model.switch_costs(feat, p))
        finally:
            loader.OVERRIDE, model.BLOCKS = None, None
        with np.errstate(invalid="ignore"):
            flags = (np.asarray(feat["ball_seen"], bool), np.asarray(feat["d_home"] < feat["d_away"]))
        out[key] = (truth, blocks, emit, costs, flags)
    if params is None:
        with open(path, "wb") as handle:
            pickle.dump(out, handle)
    return out


def main(data=None, params=None):
    """data: {key: (truth, blocks, emit, costs, flags)}, as prepared() returns (costs / flags may
    be None)."""
    data = data if data is not None else prepared(params)
    results, types = {}, {}
    for key, (truth, blocks, emit, costs, flags) in data.items():
        results[key], types[key] = evaluate(truth, blocks, emit, flags, costs)
    names = []
    for r in results.values():
        names += [k for k in r if k not in names]
    sources = sorted({k[0] for k in results})
    print("{:<22}".format("") + "".join(
        "| {:<52}".format("{} ({} matches): spellF1  gain  F1@0.3  F1@0.7  p/t".format(
            s, sum(k[0] == s for k in results))) for s in sources))
    for name in names:
        line = "{:<22}".format(name)
        for s in sources:
            rows = [r for k, r in results.items() if k[0] == s and name in r]
            if not rows:
                line += "| {:<52}".format("-")
                continue
            f = np.mean([r[name]["spell_f1_mean"] for r in rows])
            base = np.mean([r["pipeline"]["spell_f1_mean"] for r in rows])
            lo = np.mean([r[name]["spell_f1"][0.3]["f1"] for r in rows])
            hi = np.mean([r[name]["spell_f1"][0.7]["f1"] for r in rows])
            pt = np.mean([r[name]["n_pred_spells"] / max(r[name]["n_gt_spells"], 1) for r in rows])
            line += "| {:.4f}  {:+.3f}  {:.3f}   {:.3f}   {:.2f}{}".format(f, f - base, lo, hi, pt,
                                                                        " " * 13)
        print(line, flush=True)
    print()
    print("{:<22}".format("pipeline, by frame") + "".join(
        "| {:<52}".format("{}: share of in-play  team right  called D".format(s)) for s in sources))
    kinds = []
    for t in types.values():
        kinds += [k for k in t if k not in kinds]
    for kind in kinds:
        line = "{:<22}".format(kind)
        for s in sources:
            tot = sum(t[kind] for k, t in types.items() if k[0] == s and kind in t)
            allp = sum(t["all in play"] for k, t in types.items() if k[0] == s)
            if np.isscalar(tot) or not allp[0]:
                line += "| {:<52}".format("-")
                continue
            line += "| {:<52}".format("{:.3f}               {:.3f}       {:.3f}".format(
                tot[0] / allp[0], tot[1] / max(tot[2], 1), tot[3] / max(tot[0], 1)))
        print(line, flush=True)
    return results


if __name__ == "__main__":
    main()
