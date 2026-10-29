"""Three-state possession model (H / A / D) decoded with Viterbi, one half at a time.

Each frame contributes a log-likelihood for each state; changing state costs a fixed penalty.
Viterbi then finds the label sequence that best explains the whole half, which is what turns
per-frame flicker into possessions with a beginning and an end.

Evidence
--------
team   When the ball is tracked and someone is close to it, the nearest team gets the frame.
       The weight falls off with distance: a ball 1 m from a player says much more than a ball
       8 m from everybody, which is usually a ball in flight and says nothing.
dead   Players slow down when the ball is out of play. A two-component Gaussian mixture on log
       mean player speed, fitted on the whole match *without labels*, gives P(dead) per frame.
       A ball clearly outside the pitch adds to it; a ball travelling fast counts against it.
tags   Smart Tagging set pieces. Averaged over the match, the restart sits ~3.5 s after the tag
       opens (the same convention as the kickoff tags), so the seconds before it lean dead and
       the seconds after it lean live -- and towards the restarting team, when the tag names one.

Nothing here is fitted to the ground truth. The mixture is unsupervised; the remaining numbers
are in PARAMS and are set from football reasoning, not tuned on the public window.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from games import ANCHOR_FPS, GAME, GAMES, half_video_blocks  # noqa: E402
from features import compute  # noqa: E402
from load import load  # noqa: E402
import ballfinder  # noqa: E402
import dead_ml  # noqa: E402
import team_ml  # noqa: E402
import seq  # noqa: E402
import touches  # noqa: E402
import formation  # noqa: E402
import tags  # noqa: E402

# Params
PITCH = (105.3, 68.0)
PARAMS = {
    # team evidence: full weight within NEAR m of the ball, fading to zero by FAR m
    "near": 1.0,
    "far": 3.0,
    "gap_scale": 1.0,     # m of distance gap between the two teams that counts as decisive
    "k_team": 1.0,        # log-likelihood of one fully confident frame
    # dead evidence
    "k_dead": 1.0,        # multiplier on the mixture log-odds
    "out_margin": 1.0,    # m beyond a line before a tracked ball counts as out
    "k_out": 4.0,
    "ball_v_mid": 3.0,    # m/s: a ball moving faster than this over 1 s is probably in play
    "k_ball_v": 0.5,
    "k_at_feet": 1.0,     # a ball within reach of a player is a ball in play
    "feet_near": 1.0,
    "feet_far": 5.0,
    "dead_bias": 0.0,     # added to the dead log-odds on every frame
    "k_rules": 0.0,       # weight of all the hand-built dead evidence above -- replaced by:
    "k_ml": 0.2,          # weight of the dead-ball classifier learned on DFL (dead_ml.py)
    "speed_lead": 40,     # frames: players slow down after the whistle and speed up after the
                          # restart, so player speed at t + lead is the evidence for frame t
    # set-piece tags, in frames relative to the tag start
    "restart": tags.RESTART_OFFSET,
    "k_tag_dead": 1.0,
    "tag_dead_before": 175,   # frames before the restart that lean dead (7 s)
    "k_tag_live": 1.0,
    "k_tag_team": 1.0,
    # formation evidence (second pass), per frame, scaled down because neighbouring frames are
    # nearly copies of each other and would otherwise be counted 25 times a second
    "k_form_missing": 0.4,    # where the ball is not tracked
    "k_form_seen": 0.1,       # where it is: shapes lag a turnover, so the ball should lead
    "k_team_rules": 1.0,      # weight of the hand-built team evidence (nearest, formation, bf)
    "k_tml": 0.2,             # weight of the team classifier learned on DFL (team_ml.py)
    "k_seq_team": 0.4,        # weights of the second-stage temporal model (seq.py); v14: 0.5
    "k_seq_dead": 0.0,
    "pseudo_dmin": 2.0,       # m: first-pass labels used for training only this close to the ball
    # ball-finder evidence: which team's player is nearest the unseen ball, per frame
    "k_bf": 0.2,
    # state change costs, in the same log units as the evidence
    "switch_contact": 9.0,  # H <-> A near a contact by the team taking over (see switch_costs)
    "switch_quiet": 9.0,    # H <-> A where the ball is seen and no such contact is near
    "contact_reach": 1.5,   # m
    "contact_window": 12,   # frames
    "snap_window": 0,     # frames: move each H <-> A change to a touch this near (0 = off)
    "snap_reach": 1.5,    # m: a player this near the ball, and nearer than the other side
    "snap_rule": "nearest",   # which touch: "nearest" to the change, or "after_old" (see snap)
    "switch": 9.0,        # H <-> A
    "stop": 15.0,         # H/A -> D
    "start": 15.0,        # D -> H/A
    # final decode for expected spellF1 instead of the most likely path (mbr.py); 0 = Viterbi.
    # Set from mbr_bench.py on the DFL and Metrica benches, never from the public window.
    "mbr_tau": 0.0,
    "mbr_prior_scale": None,  # None: the whole Viterbi score tempered by mbr_tau
    "mbr_samples": 128,
    # the ball is dead before each half's kickoff (games.py anchors), up to this many frames
    # before it -- the last second is left to the evidence. None = off.
    "kickoff_margin": 25,
    # per-frame relief of the H <-> A cost (0 = off, the submission): from ball actions spotted on
    # the broadcast video (video_events.py; feat["video_events"], else cache/video_events.csv), and
    # from a change detector's log-odds (feat["change_logodds"]). Never below the *_min cost.
    "k_video": 0.0,
    "video_min": 2.0,
    "k_video_team": 0.0,      # team log-odds per frame around a video event (video_events.py)
    "video_hold": 0,          # frames a video touch's team evidence is carried towards the next
    "k_change": 2.0,
    "change_min": 2.0,
    # How team evidence meets the alive / dead decision (v16 options, off: "all" and 0 are v15.1).
    # Viterbi keeps the better of H and A, which scores -log(1 + e^-|team|): -0.69 a frame with no
    # team evidence, 0 with strong evidence. So the *strength* of the team evidence pulls a frame
    # towards alive by up to 0.69 a frame, about what the dead evidence itself weighs (k_ml 0.2).
    # Where the second stage is sure of the team inside a stoppage -- the players line up for the
    # restart, which the motion model reads as possession -- the stoppage leaks into H / A.
    #   "all"    the level from all the team evidence (v15.1)
    #   "first"  the level from the evidence before the second stage (nearest player, ball-finder,
    #            formation, team_ml, tags): the second stage, touch and video evidence then decide
    #            only H against A, and can no longer move a frame between alive and dead
    #   "flat"   one constant level, -alive_flat, on every frame: alive against dead from the
    #            dead evidence alone
    "alive_level": "all",
    "alive_flat": 0.6931,
    # > 0: the second stage's team log-odds times sigmoid(-seq_gate x its own dead log-odds), so
    # they fade where the second stage itself calls the ball dead (0 = off, v15.1)
    "seq_gate": 0.0,
}
STATES = np.array(["H", "A", "D"])
# Set to [(first, last), ...] to run on a match other than BAR-ATH (see dfl_eval.py).
BLOCKS = None


def blocks():
    """(first, last) frame of each half."""
    if BLOCKS is not None:
        return list(BLOCKS)
    return [(first, last) for _, first, last in half_video_blocks(GAME)]
H, A, D = 0, 1, 2


def fit_two_gaussians(x, iters=200):
    """1-D two-component Gaussian mixture by EM. Returns (weights, means, stds), means ascending."""
    x = x[np.isfinite(x)]
    mu = np.percentile(x, [20, 70]).astype(float)
    sd = np.array([x.std(), x.std()]) / 2
    wt = np.array([0.5, 0.5])
    for _ in range(iters):
        dens = wt / (sd * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x[:, None] - mu) / sd) ** 2)
        resp = dens / dens.sum(axis=1, keepdims=True)
        nk = resp.sum(axis=0)
        wt = nk / len(x)
        mu = (resp * x[:, None]).sum(axis=0) / nk
        sd = np.sqrt((resp * (x[:, None] - mu) ** 2).sum(axis=0) / nk)
    order = np.argsort(mu)
    return wt[order], mu[order], sd[order]


def graded_mask(n):
    mask = np.zeros(n, dtype=bool)
    for first, last in blocks():
        mask[first:last + 1] = True
    return mask


def dead_log_odds(feat, mask):
    """log P(dead)/P(alive) per frame from player speed, mixture fitted on graded frames."""
    log_speed = np.log(np.clip(feat["mean_speed_2s"], 0.05, None))
    wt, mu, sd = fit_two_gaussians(log_speed[mask])

    def log_density(k):
        return np.log(wt[k]) - np.log(sd[k]) - 0.5 * ((log_speed - mu[k]) / sd[k]) ** 2

    odds = log_density(0) - log_density(1)
    return np.nan_to_num(odds, nan=0.0), (wt, np.exp(mu), sd)


def proximity(feat, near, far):
    """1 when a player is within `near` m of the ball, fading to 0 at `far` m; 0 with no ball."""
    d_min = np.fmin(feat["d_home"], feat["d_away"])
    return np.nan_to_num(np.clip((far - d_min) / (far - near), 0.0, 1.0), nan=0.0)


def team_log_odds(feat, p):
    """log P(H)/P(A) per frame from who is nearest the ball. 0 where the ball is not seen."""
    gap = np.nan_to_num(np.tanh((feat["d_away"] - feat["d_home"]) / p["gap_scale"]), nan=0.0)
    return p["k_team"] * proximity(feat, p["near"], p["far"]) * gap


def ball_out(feat, p):
    x, y, m = feat["ball_x"], feat["ball_y"], p["out_margin"]
    with np.errstate(invalid="ignore"):
        return (x < -m) | (x > PITCH[0] + m) | (y < -m) | (y > PITCH[1] + m)


def tag_evidence(n, events, p):
    """(team log-odds H vs A, dead log-odds) per frame from the set-piece tags."""
    team, dead = np.zeros(n), np.zeros(n)
    for e in events:
        if e["code"] != tags.SET_PIECE:
            continue
        restart = e["start"] + p["restart"]
        dead[max(restart - int(p["tag_dead_before"]), 0):restart] += p["k_tag_dead"]
        dead[restart + 25:restart + 150] -= p["k_tag_live"]
        if e["side"] is not None:
            sign = 1.0 if e["side"] == "H" else -1.0
            team[restart:restart + 50] += sign * p["k_tag_team"]
    return team, dead


def kickoff_frames():
    """Video frame of each half's kickoff from the games.py anchors, or [] off BAR-ATH."""
    if BLOCKS is not None:
        return []
    game = GAMES[GAME]
    return [(a["minute"] * 60 + a["second"]) * ANCHOR_FPS + a["frame"]
            for a in (game["kickoff"], game["kickoff_2nd"])]


def before_kickoff(n, p):
    """Graded frames before a half's kickoff (less the margin): the ball is not in play yet."""
    out = np.zeros(n, dtype=bool)
    if p["kickoff_margin"] is None:
        return out
    for (first, _), kick in zip(blocks(), kickoff_frames()):
        out[first:max(first, kick - int(p["kickoff_margin"]))] = True
    return out


def emissions(feat, p, mask, events, form=None):
    """(N, 3) log-likelihoods for H, A, D. `form` is formation log-odds H vs A, if available."""
    team = team_log_odds(feat, p)
    if p["k_bf"]:
        team = team + p["k_bf"] * ballfinder_log_odds(feat)
    if form is not None:
        weight = np.where(feat["ball_seen"], p["k_form_seen"], p["k_form_missing"])
        team = team + weight * form
    team = p["k_team_rules"] * team
    if p["k_tml"]:
        team = team + p["k_tml"] * team_ml_log_odds(feat)
    dead, _ = dead_log_odds(feat, mask)
    lead = int(p["speed_lead"])
    if lead > 0:
        dead = np.concatenate((dead[lead:], np.full(lead, dead[-1])))
    dead = p["k_dead"] * dead + p["k_out"] * ball_out(feat, p)
    moving = np.clip((feat["ball_v_1s"] - p["ball_v_mid"]) / p["ball_v_mid"], -1.0, 1.0)
    dead -= p["k_ball_v"] * np.nan_to_num(moving, nan=0.0)
    dead -= p["k_at_feet"] * proximity(feat, p["feet_near"], p["feet_far"])
    dead += p["dead_bias"]
    dead *= p["k_rules"]
    if p["k_ml"]:
        dead += p["k_ml"] * dead_ml_log_lr(feat)
    first_team = team.copy()                  # the evidence before the second stage
    if p["k_seq_team"] or p["k_seq_dead"]:
        seq_team, seq_dead = seq_log_odds(feat)
        if p.get("seq_gate", 0.0) > 0:
            seq_team = seq_team / (1.0 + np.exp(np.clip(p["seq_gate"] * seq_dead, -50.0, 50.0)))
        team = team + p["k_seq_team"] * seq_team
        dead += p["k_seq_dead"] * seq_dead

    tag_team, tag_dead = tag_evidence(len(team), events, p)
    team += tag_team
    first_team = first_team + tag_team
    dead += tag_dead
    if p.get("k_video_team", 0.0) > 0:
        import video_events
        team = team + video_events.team_evidence(len(team), video_events_of(feat),
                                                 p["k_video_team"], hold=int(p["video_hold"]))

    # log P(H|alive) and log P(A|alive): their difference is the team log-odds, and they sum to
    # P(alive) = 1 -- but Viterbi takes the better of the two, not their sum, so strong team
    # evidence does lift a frame towards alive (PARAMS alive_level).
    out = np.zeros((len(team), 3))
    out[:, H] = -np.logaddexp(0.0, -team)
    out[:, A] = -np.logaddexp(0.0, team)
    level = p.get("alive_level", "all")
    if level != "all":
        # H - A stays the team log-odds; the better of the two is moved to the chosen level
        target = (-np.logaddexp(0.0, -np.abs(first_team)) if level == "first"
                  else np.full(len(team), -float(p["alive_flat"])))
        lift = target + np.logaddexp(0.0, -np.abs(team))
        out[:, H] += lift
        out[:, A] += lift
    out[:, D] = dead
    pre = before_kickoff(len(out), p)
    out[pre, H] = out[pre, A] = -1e3
    return out


def viterbi(emit, cost, to_away=None, to_home=None):
    """Most likely state path. emit (T, S) log-likelihoods, cost (S, S) penalty for i -> j.

    to_away / to_home, if given, are per-frame costs of H -> A and A -> H, overriding cost.
    """
    n, s = emit.shape
    score = emit[0].copy()
    back = np.zeros((n, s), dtype=np.int8)
    cost = cost.copy()
    for t in range(1, n):
        if to_away is not None:
            cost[H, A], cost[A, H] = to_away[t], to_home[t]
        cand = score[:, None] - cost             # cand[i, j]: come from i into j
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], np.arange(s)] + emit[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def transition_costs(p):
    cost = np.zeros((3, 3))
    cost[H, A] = cost[A, H] = p["switch"]
    cost[H, D] = cost[A, D] = p["stop"]
    cost[D, H] = cost[D, A] = p["start"]
    return cost


def switch_costs(feat, p):
    """Per-frame cost of H -> A and A -> H.

    Cheap (switch_contact) within contact_window frames of a contact by the team taking over --
    one of its players within contact_reach m of the ball and nearer than anyone on the other
    side. Dear (switch_quiet) where the ball is tracked and no such contact is near. The base
    switch cost where the ball is missing, since contacts cannot be seen there.
    """
    with np.errstate(invalid="ignore"):
        home = (feat["d_home"] < p["contact_reach"]) & (feat["d_home"] < feat["d_away"])
        away = (feat["d_away"] < p["contact_reach"]) & (feat["d_away"] < feat["d_home"])
    w = int(p["contact_window"])
    seen = touches.near(feat["ball_seen"].astype(np.int8), 1, w)
    out = []
    for side in (away, home):
        near = touches.near(side.astype(np.int8), 1, w)
        out.append(np.where(near, p["switch_contact"], np.where(seen, p["switch_quiet"],
                                                                 p["switch"])))
    if p.get("k_video", 0.0) > 0:
        import video_events
        relief = video_events.switch_relief(len(out[0]), video_events_of(feat), p["k_video"])
        out = [np.maximum(c - r, np.minimum(c, p["video_min"])) for c, r in zip(out, relief)]
    if p.get("k_change", 0.0) > 0:
        relief = p["k_change"] * np.clip(change_log_odds(feat), 0.0, None)
        out = [np.maximum(c - relief, np.minimum(c, p["change_min"])) for c in out]
    return out


def video_events_of(feat):
    """Video events for this match: feat["video_events"] (benches, simulated), else BAR-ATH's
    cached file (video_events.py build)."""
    if "video_events" not in feat:
        if BLOCKS is not None:
            raise ValueError("k_video > 0 off BAR-ATH needs feat['video_events']")
        import video_events
        feat["video_events"] = video_events.load()
    return feat["video_events"]


def decode(emit, p, feat=None, final=False):
    """Viterbi per half. Label array of length N: H/A/D on graded frames, X elsewhere.

    final=True with p["mbr_tau"] > 0 decodes both halves jointly for expected spellF1 instead
    (mbr.py). Only the last decode of predict() does; the first pass, which trains the formation
    model, stays Viterbi.
    """
    cost = transition_costs(p)
    labels = np.full(len(emit), "X", dtype="<U1")
    if final and p.get("mbr_tau", 0.0) > 0:
        import mbr
        halves = [(first, last + 1) for first, last in blocks()]
        paths = mbr.decode_match([emit[a:b] for a, b in halves], cost, tau=p["mbr_tau"],
                                 n_samples=int(p["mbr_samples"]),
                                 prior_scale=p["mbr_prior_scale"])
        for (a, b), path in zip(halves, paths):
            labels[a:b] = STATES[path]
        return labels
    varying = feat is not None and (p["switch_contact"] != p["switch"]
                                    or p["switch_quiet"] != p["switch"]
                                    or p.get("k_video", 0.0) > 0 or p.get("k_change", 0.0) > 0)
    to_away, to_home = switch_costs(feat, p) if varying else (None, None)
    for first, last in blocks():
        seg = slice(first, last + 1)
        labels[seg] = STATES[viterbi(emit[seg], cost,
                                     None if to_away is None else to_away[seg],
                                     None if to_home is None else to_home[seg])]
    return labels


def snap(labels, feat, p):
    """Move each direct H <-> A change onto the first frame of a touch by the team taking over.

    On the DFL matches, wherever the new team touches the ball within 3 s of a true change the
    change sits on that touch (85% within 5 frames); the decoder only knows roughly when the
    evidence turns. Each change moves at most snap_window frames and never past the start of the
    previous run or the end of the next one, so no possession appears or disappears.

        nearest     the touch onset nearest the decoded change
        after_old   the first onset after the old team's last touch before the change
    """
    w = int(p["snap_window"])
    with np.errstate(invalid="ignore"):
        touch = {"H": (feat["d_home"] < p["snap_reach"]) & (feat["d_home"] < feat["d_away"]),
                 "A": (feat["d_away"] < p["snap_reach"]) & (feat["d_away"] < feat["d_home"])}
    onset = {k: np.flatnonzero(v & ~np.r_[False, v[:-1]]) for k, v in touch.items()}
    labels = labels.copy()
    alive = np.isin(labels, ["H", "A"])
    for i in np.flatnonzero(alive[:-1] & alive[1:] & (labels[:-1] != labels[1:])) + 1:
        old, new = labels[i - 1], labels[i]
        if old == new:                            # an earlier move already shifted this one
            continue
        start = i - 1
        while start > 0 and labels[start - 1] == old:
            start -= 1
        end = i
        while end + 1 < len(labels) and labels[end + 1] == new:
            end += 1
        lo, hi = max(start + 1, i - w), min(end, i + w)
        cand = onset[new][(onset[new] >= lo) & (onset[new] <= hi)]
        if p["snap_rule"] == "after_old":
            before = np.flatnonzero(touch[old][lo:i]) + lo
            if len(before):
                cand = cand[cand > before[-1]]
            target = cand[0] if len(cand) else i
        else:
            target = cand[np.argmin(np.abs(cand - i))] if len(cand) else i
        if target < i:
            labels[target:i] = new
        elif target > i:
            labels[i:target] = old
    return labels


def formation_log_odds(feat, labels, p):
    """Formation model, cross-fitted: trained on one half's first-pass labels, applied to the other.

    Cross-fitting keeps a half from grading its own homework -- the model never scores frames
    whose pseudo-labels it was trained on.
    """
    halves = blocks()
    if "formation" not in feat:
        feat["formation"] = formation.features(halves, PITCH)
    x = feat["formation"]
    d_min = np.fmin(feat["d_home"], feat["d_away"])
    with np.errstate(invalid="ignore"):
        usable = (np.isfinite(x).all(axis=1) & np.isin(labels, ["H", "A"])
                  & feat["ball_seen"] & (d_min < p["pseudo_dmin"]))
    y = (labels == "H").astype(float)

    out = np.zeros(len(labels))
    for k, (first, last) in enumerate(halves):
        train = usable.copy()
        train[first:last + 1] = False           # train on every other half
        model = formation.fit_logistic(x[train], y[train])
        out[first:last + 1] = formation.log_odds(model, x[first:last + 1])
    return out


def dead_ml_log_lr(feat):
    """Learned dead-ball log-likelihood ratio per frame, cached in `feat`.

    For BAR-ATH the classifier is trained on all the DFL matches; dfl_eval fills feat["dead_ml"]
    itself with a model that never saw the match being scored.
    """
    if "dead_ml" not in feat:
        ball, players, _ = load()
        x = dead_ml.features(ball, players, blocks(), PITCH)
        feat["dead_ml"] = dead_ml.log_lr(dead_ml.trained(), x)
    return feat["dead_ml"]


def team_ml_log_odds(feat):
    """Learned team log-odds H vs A per frame, cached in `feat` (dfl_eval fills it per match)."""
    if "team_ml" not in feat:
        ball, players, team = load()
        x = team_ml.features(ball, players, team, blocks(), PITCH)
        feat["team_ml"] = team_ml.log_odds(team_ml.trained(), x)
    return feat["team_ml"]


def motion_ml_log_odds(feat):
    """Ball-blind team log-odds H vs A per frame (motion_ml.py), cached in `feat` (benches fill
    it leave-one-match-out)."""
    if "motion_ml" not in feat:
        import motion_ml
        ball, players, team = load()
        x = motion_ml.features(ball, players, team, blocks(), PITCH)
        feat["motion_ml"] = motion_ml.log_odds(motion_ml.trained(), x)
    return feat["motion_ml"]


def touch_features(feat):
    """Ball-touch features (touchfeat.py) per frame, cached in `feat`."""
    if "touch" not in feat:
        import touchfeat
        ball, players, team = load()
        if seq.TOUCH_BLUR > 0 and BLOCKS is not None:      # a training or bench match, not ours
            import dfl_eval
            ball = dfl_eval.blur_runs(ball, seq.TOUCH_BLUR)
        feat["touch"] = touchfeat.compute(ball, players, team)
    return feat["touch"]


def seq_log_odds(feat):
    """Second-stage (team, dead) log-odds per frame, cached in `feat` (dfl_eval fills them per
    match). Needs the first-stage evidence, so it fills that first."""
    if "seq_team" not in feat:
        dead_ml_log_lr(feat)
        team_ml_log_odds(feat)
        ballfinder_log_odds(feat)
        if seq.TOUCH:
            touch_features(feat)
        if seq.MOTION:
            motion_ml_log_odds(feat)
        team, dead = seq.log_odds(seq.trained(dfl_eval_training_set), seq.features(feat))
        feat["seq_team"], feat["seq_dead"] = np.where(graded_mask(len(team)), team, 0.0), dead
    return feat["seq_team"], feat["seq_dead"]


def dfl_eval_training_set():
    """The second stage learns from the DFL matches and Metrica's sample matches together (and
    the change detector, change_ml.py, from the same rows)."""
    import dfl_eval
    import external_metrica
    xs, labels = dfl_eval.seq_training_set()
    more_xs, more_labels = external_metrica.seq_training_set()
    return xs + more_xs, labels + more_labels


def change_log_odds(feat):
    """Change-detector log-odds per frame (change_ml.py), cached in `feat`; benches fill it
    leave-one-match-out themselves."""
    if "change_logodds" not in feat:
        import change_ml
        dead_ml_log_lr(feat)
        team_ml_log_odds(feat)
        ballfinder_log_odds(feat)
        if seq.TOUCH:
            touch_features(feat)
        if seq.MOTION:
            motion_ml_log_odds(feat)
        z = change_ml.log_odds(change_ml.trained(dfl_eval_training_set), seq.features(feat))
        feat["change_logodds"] = np.where(graded_mask(len(z)), z, 0.0)
    return feat["change_logodds"]


def ballfinder_log_odds(feat):
    """Ball-finder log-odds H vs A on every graded frame without a ball, 0 elsewhere.

    Cross-fitted like the formation model: each half is scored by a model trained on the other.
    The ball-finder learns from the tracking alone, never from our labels. Cached in `feat`.
    """
    if "ballfinder" in feat:
        return feat["ballfinder"]
    halves = blocks()
    ctx = ballfinder.Context(halves)
    gaps = ballfinder.gap_lengths(ctx.seen, halves)
    out = np.zeros(len(ctx.seen))
    for k, (first, last) in enumerate(halves):
        other_first, other_last = halves[1 - k]
        model = ballfinder.fit(ctx, np.arange(other_first, other_last + 1, 4), gaps)
        frames = np.flatnonzero(~ctx.seen[first:last + 1]) + first
        out[frames], _ = ballfinder.team_log_odds(ctx, model, frames, ctx.last[frames],
                                                  ctx.next[frames])
    feat["ballfinder"] = out
    return out


def predict(params=None, feat=None, events=None):
    """Label array of length N: H/A/D on graded frames, X elsewhere.

    Two passes. The first uses the ball, the players' speed and the tags. Its labels, where the
    ball is close to a player, train the formation model; the second pass adds that model's
    evidence, which is what carries the frames where the ball is not tracked.
    """
    p = dict(PARAMS, **(params or {}))
    feat = feat if feat is not None else compute()
    events = events if events is not None else tags.load()
    mask = graded_mask(len(feat["ball_x"]))

    one_pass = p["k_form_missing"] == 0 and p["k_form_seen"] == 0
    labels = decode(emissions(feat, p, mask, events), p, feat, final=one_pass)
    if one_pass:
        return labels
    form = formation_log_odds(feat, labels, p)
    labels = decode(emissions(feat, p, mask, events, form), p, feat, final=True)
    return snap(labels, feat, p) if p["snap_window"] else labels
