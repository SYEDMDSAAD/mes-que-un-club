"""Decode for expected spellF1 (minimum Bayes risk) instead of the single most likely path.

Viterbi returns the one label sequence with the highest score. The ranking metric does not reward
that sequence as such: it counts possessions, matches them one-to-one by temporal IoU, and averages
F1 over IoU 0.3 ... 0.7. Two things follow that a single most-likely path cannot express:

  * F1 is maximised by predicting a possession whenever the chance that it matches a true one is
    above about F/2 (Lipton, Elkan & Narayanaswamy 2014) -- around 0.3 at our scores. The most
    likely path only keeps a short possession when it is more likely than not.
  * What a doubtful short possession costs depends on where it lands. In a burst of short
    possessions a wrong one costs one false positive. In the middle of a long true possession it
    also cuts that possession in two and can lose its match. One switch cost cannot price both,
    which is why lowering it "adds false changes as fast as true ones".

So this module:

  1. turns the decoder's score into a distribution over label paths,
     p(path) proportional to exp(score(path) / TAU), score = sum of emissions - sum of costs;
  2. draws N_SAMPLES paths per half from it (forward filtering, backward sampling) -- each one a
     plausible "truth" given our evidence;
  3. builds candidate label sequences: Viterbi at several H <-> A costs, and the posterior
     marginal argmax with short spells removed;
  4. cuts each half into blocks at stoppages that every candidate agrees on and the posterior is
     sure of (no possession can cross one, so blocks are scored independently);
  5. picks for each block the candidate with the highest expected 2 * TP - lam * (possessions it
     predicts) against the samples, TP averaged over the five IoU thresholds exactly as the
     scorer does. lam is the expected F1 itself, solved over the whole match (Dinkelbach's method
     for maximising a ratio), because spellF1 pools both halves.

A small TAU collapses every sample onto the Viterbi path, and the decoder returns Viterbi, so the
family contains the current decoder (keep TAU >= 0.3: below that exp() underflows). TAU and
prior_scale are what to tune, on the labelled benches (mbr_bench.py), never on the public window.

It is only as good as the posterior it samples. On synthetic matches whose evidence strength
varies (ball seen / ball missing) and whose posterior is calibrated, it beat the best fixed-cost
Viterbi by about 0.02 spellF1 on held-out matches, every match up; with uniform evidence, a
miscalibrated posterior, or confident errors, it only ties. So mbr_bench checks calibration
instead of assuming it: sampled possessions per true possession should be close to 1.

Deterministic: seeded generator, no BLAS in the recursions, ties go to the base Viterbi.
"""

import numpy as np

# Params
TAU = 1.0                 # evidence temperature of the path distribution (see decode_match)
N_SAMPLES = 128           # posterior paths drawn per half
SEED = 0
SWITCH_GRID = (1.0, 0.45, 0.6, 0.75, 0.9, 1.2, 1.5)   # candidate H <-> A cost multipliers; 1.0 first
MIN_SPELL = (13, 25)      # frames: marginal-argmax candidates with runs shorter than this merged
CUT_PDEAD = 0.98          # a block boundary needs P(dead) at least this, and D in every candidate
IOUS = (0.3, 0.4, 0.5, 0.6, 0.7)   # possession_metrics.SPELL_IOUS
H, A, D = 0, 1, 2


# -- the path distribution ----------------------------------------------------------------------

def _transition(cost, tau):
    """exp(-cost / tau) as nested Python floats: m[i][j] for i -> j."""
    return np.exp(-np.asarray(cost, dtype=float) / tau).tolist()


def _likelihoods(emit, tau):
    """exp(emit / tau), each row divided by its maximum (the scale cancels in every posterior)."""
    e = np.asarray(emit, dtype=float) / tau
    return np.exp(e - e.max(axis=1, keepdims=True))


def forward_backward(emit, cost, tau):
    """(alpha, marginals), both (T, 3). alpha[t] is P(state at t | evidence up to t), normalised.

    Plain Python arithmetic over 3 states: fast enough (T = 80k in well under a second) and free
    of BLAS, whose rounding can differ between machines.
    """
    m = _transition(cost, tau)
    e = _likelihoods(emit, tau).tolist()
    n = len(e)
    alpha = [None] * n
    a0, a1, a2 = e[0]
    s = a0 + a1 + a2
    a0, a1, a2 = a0 / s, a1 / s, a2 / s
    alpha[0] = (a0, a1, a2)
    for t in range(1, n):
        e0, e1, e2 = e[t]
        p0 = (a0 * m[0][0] + a1 * m[1][0] + a2 * m[2][0]) * e0
        p1 = (a0 * m[0][1] + a1 * m[1][1] + a2 * m[2][1]) * e1
        p2 = (a0 * m[0][2] + a1 * m[1][2] + a2 * m[2][2]) * e2
        s = p0 + p1 + p2
        a0, a1, a2 = p0 / s, p1 / s, p2 / s
        alpha[t] = (a0, a1, a2)

    marg = [None] * n
    b0 = b1 = b2 = 1.0
    for t in range(n - 1, -1, -1):
        a0, a1, a2 = alpha[t]
        q0, q1, q2 = a0 * b0, a1 * b1, a2 * b2
        s = q0 + q1 + q2
        marg[t] = (q0 / s, q1 / s, q2 / s)
        if t:
            e0, e1, e2 = e[t]
            c0, c1, c2 = e0 * b0, e1 * b1, e2 * b2
            n0 = m[0][0] * c0 + m[0][1] * c1 + m[0][2] * c2
            n1 = m[1][0] * c0 + m[1][1] * c1 + m[1][2] * c2
            n2 = m[2][0] * c0 + m[2][1] * c1 + m[2][2] * c2
            s = n0 + n1 + n2
            b0, b1, b2 = n0 / s, n1 / s, n2 / s
    return np.array(alpha), np.array(marg)


def sample_paths(alpha, cost, tau, n_samples, rng):
    """(n_samples, T) int8 paths drawn from p(path) by backward sampling from the forward messages."""
    m = np.exp(-np.asarray(cost, dtype=float) / tau)
    n = len(alpha)
    out = np.empty((n_samples, n), dtype=np.int8)
    w = alpha[n - 1]
    c = np.cumsum(w / w.sum())
    u = rng.random(n_samples)
    s = (u > c[0]).astype(np.int8) + (u > c[1]).astype(np.int8)
    out[:, n - 1] = s
    for t in range(n - 1, 0, -1):
        a = alpha[t - 1]
        w0 = a[0] * m[0][s]
        w1 = a[1] * m[1][s]
        w2 = a[2] * m[2][s]
        u = rng.random(n_samples) * (w0 + w1 + w2)
        s = (u > w0).astype(np.int8) + (u > w0 + w1).astype(np.int8)
        out[:, t - 1] = s
    return out


# -- candidates -----------------------------------------------------------------------------------

def viterbi_batch(emit, costs):
    """Most likely path for each (3, 3) cost matrix in costs: (C, T) int8.

    Same arithmetic and tie-breaking as model.viterbi, so a cost matrix equal to the model's gives
    the model's path exactly.
    """
    emit = np.asarray(emit, dtype=float)
    costs = np.asarray(costs, dtype=float)
    n, c = len(emit), len(costs)
    score = np.repeat(emit[:1], c, axis=0)
    back = np.zeros((n, c, 3), dtype=np.int8)
    for t in range(1, n):
        cand = score[:, :, None] - costs                       # [k, from, to]
        b = np.argmax(cand, axis=1)
        back[t] = b
        score = np.take_along_axis(cand, b[:, None, :], axis=1)[:, 0, :] + emit[t]
    path = np.empty((c, n), dtype=np.int8)
    path[:, -1] = np.argmax(score, axis=1)
    rows = np.arange(c)
    for t in range(n - 1, 0, -1):
        path[:, t - 1] = back[t, rows, path[:, t]]
    return path


def merge_short(path, min_len):
    """Copy of path with every run shorter than min_len frames given to the run before it (to the
    run after it at the very start), in one left-to-right pass."""
    if not len(path):
        return path.copy()
    b = np.flatnonzero(path[1:] != path[:-1]) + 1
    starts, stops = np.r_[0, b], np.r_[b, len(path)]
    labels, lengths = [], []
    for lab, length in zip(path[starts].tolist(), (stops - starts).tolist()):
        if labels and (length < min_len or labels[-1] == lab):
            lengths[-1] += length
        else:
            labels.append(lab)
            lengths.append(length)
    if len(labels) > 1 and lengths[0] < min_len:
        lengths[1] += lengths[0]
        labels, lengths = labels[1:], lengths[1:]
    return np.repeat(np.array(labels, dtype=path.dtype), lengths)


def candidates(emit, cost, marg):
    """(C, T) candidate paths; row 0 is Viterbi at the model's own costs."""
    costs = []
    for mult in SWITCH_GRID:
        c = np.array(cost, dtype=float)
        c[H, A] *= mult
        c[A, H] *= mult
        costs.append(c)
    rows = list(viterbi_batch(emit, costs))
    mpm = np.argmax(marg, axis=1).astype(np.int8)
    rows += [merge_short(mpm, k) for k in MIN_SPELL]
    return np.array(rows)


# -- scoring ---------------------------------------------------------------------------------------

def spells(path, lo, hi):
    """(starts, stops, teams) of the maximal H or A runs of path[lo:hi], in absolute frames."""
    w = path[lo:hi]
    if not len(w):
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, np.int8)
    b = np.flatnonzero(w[1:] != w[:-1]) + 1
    starts, stops = np.r_[0, b], np.r_[b, len(w)]
    team = w[starts]
    keep = team != D
    return starts[keep] + lo, stops[keep] + lo, team[keep]


def matches(pred, truth):
    """Mean over IOUS of the one-to-one matches between two spell lists, as possession_metrics
    counts them: same team, greedy by descending IoU, ties by truth then prediction index."""
    ps, pe, pt = pred
    gs, ge, gt = truth
    if not len(ps) or not len(gs):
        return 0.0
    inter = np.minimum(pe[:, None], ge[None, :]) - np.maximum(ps[:, None], gs[None, :])
    ok = (inter > 0) & (pt[:, None] == gt[None, :])
    if not ok.any():
        return 0.0
    pi, gi = np.nonzero(ok)
    union = np.maximum(pe[pi], ge[gi]) - np.minimum(ps[pi], gs[gi])
    iou = inter[pi, gi] / union
    order = np.lexsort((pi, gi, -iou))
    used_p, used_g, got = set(), set(), []
    for k in order.tolist():
        if pi[k] in used_p or gi[k] in used_g:
            continue
        used_p.add(pi[k])
        used_g.add(gi[k])
        got.append(iou[k])
    got = np.array(got)
    return float(np.mean([(got >= thr).sum() for thr in IOUS]))


def _runs(mask):
    """[lo, hi) of the maximal True runs of a boolean array."""
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def blocks_of(cands, marg):
    """[lo, hi) stretches separated by stoppages that every candidate calls dead and that contain
    a frame with P(dead) >= CUT_PDEAD -- so almost no sampled possession runs across one."""
    cut = np.zeros(cands.shape[1], dtype=bool)
    for lo, hi in _runs((cands == D).all(axis=0)):
        if marg[lo:hi, D].max() >= CUT_PDEAD:
            cut[lo:hi] = True
    return _runs(~cut)


def block_table(cands, samples, blocks):
    """Per block: expected matches (C,) and predicted possessions (C,) of every candidate.

    Identical candidate or sample stretches are scored once and weighted by their count, which is
    most of them in quiet blocks.
    """
    etp, npred = [], []
    for lo, hi in blocks:
        s_keys, s_weight, s_spells = {}, [], []
        for row in samples:
            key = row[lo:hi].tobytes()
            if key not in s_keys:
                s_keys[key] = len(s_spells)
                s_spells.append(spells(row, lo, hi))
                s_weight.append(0)
            s_weight[s_keys[key]] += 1
        s_weight = np.array(s_weight, dtype=float) / len(samples)
        c_cache, e_row, n_row = {}, [], []
        for row in cands:
            key = row[lo:hi].tobytes()
            if key not in c_cache:
                sp = spells(row, lo, hi)
                c_cache[key] = (sum(w * matches(sp, g) for w, g in zip(s_weight, s_spells)),
                                len(sp[0]))
            e, k = c_cache[key]
            e_row.append(e)
            n_row.append(k)
        etp.append(e_row)
        npred.append(n_row)
    return np.array(etp, dtype=float).reshape(len(blocks), len(cands)), \
        np.array(npred, dtype=float).reshape(len(blocks), len(cands))


def choose(etp, npred, n_truth, iters=100):
    """Candidate per block maximising 2 * sum(TP) / (sum(predicted) + n_truth), by Dinkelbach:
    with lam fixed the objective 2 TP - lam * predicted splits over blocks; lam <- the F1 reached.
    Returns (choice per block, expected spellF1)."""
    lam, best = 0.5, None
    for _ in range(iters):
        gain = np.round(2.0 * etp - lam * npred, 9)
        choice = np.argmax(gain, axis=1)                  # first maximum: the base Viterbi
        rows = np.arange(len(choice))
        tp, pred = etp[rows, choice].sum(), npred[rows, choice].sum()
        f1 = 2.0 * tp / max(pred + n_truth, 1e-9)
        if best is not None and f1 <= lam + 1e-12:
            break
        best, lam = choice, f1
    return best, lam


# -- entry point ------------------------------------------------------------------------------------

def decode_match(emits, cost, tau=TAU, n_samples=N_SAMPLES, seed=SEED, prior_scale=None,
                 report=None):
    """Label paths (int8 arrays, 0 H / 1 A / 2 D) for each half's emissions, chosen jointly.

    emits        [(T_k, 3) log-likelihoods per half], the same arrays Viterbi decodes
    cost         (3, 3) state-change costs, the same matrix Viterbi uses; the candidates use it
    tau          evidence temperature: the samples see emissions / tau
    prior_scale  the samples pay cost * prior_scale per change, i.e.
                 p(path) proportional to exp(emissions / tau - costs * prior_scale).
                 None means 1 / tau: Viterbi's whole score tempered alike. A larger value keeps
                 the prior on changes while the evidence -- counted on 25 strongly correlated
                 frames a second -- is discounted.
    report       optional dict, filled with diagnostics (expected F1, counts, blocks)
    """
    rng = np.random.default_rng(seed)
    scale = 1.0 / tau if prior_scale is None else prior_scale
    sample_cost = np.asarray(cost, dtype=float) * scale * tau     # forward_backward divides by tau
    halves = []
    for emit in emits:
        alpha, marg = forward_backward(emit, sample_cost, tau)
        samples = sample_paths(alpha, sample_cost, tau, n_samples, rng)
        cands = candidates(emit, cost, marg)
        halves.append((cands, samples, blocks_of(cands, marg)))

    tables = [block_table(c, s, b) for c, s, b in halves]
    etp = np.concatenate([t[0] for t in tables])
    npred = np.concatenate([t[1] for t in tables])
    n_truth = np.mean([sum(len(spells(row, 0, len(row))[0]) for row in rows)
                       for rows in zip(*[s for _, s, _ in halves])])
    choice, f1 = choose(etp, npred, n_truth)

    out, k = [], 0
    for cands, _, blocks in halves:
        path = np.full(cands.shape[1], D, dtype=np.int8)
        for lo, hi in blocks:
            path[lo:hi] = cands[choice[k], lo:hi]
            k += 1
        out.append(path)

    if report is not None:
        rows = np.arange(len(choice))
        report.update({
            "expected_spellF1": float(f1),
            "expected_spellF1_viterbi": float(2 * etp[:, 0].sum() / (npred[:, 0].sum() + n_truth)),
            "n_pred": int(npred[rows, choice].sum()),
            "n_pred_viterbi": int(npred[:, 0].sum()),
            "n_truth_expected": float(n_truth),
            "blocks": len(choice),
            "blocks_changed": int((choice != 0).sum()),
            "candidate_use": np.bincount(choice, minlength=etp.shape[1]).tolist(),
        })
    return out
