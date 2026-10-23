# Who has the ball — technical document

**Team Mes Que Un Club** · More Than A Hack 2027, Phase 1 · BAR-ATH

## In one paragraph

Every frame of each half gets three log-likelihoods: home has the ball (H), away has it (A), or it
is dead (D). A Viterbi decoder then finds the label sequence that best explains the whole half,
paying a fixed price for every change of state. The decoder is what turns per-frame flicker into
possessions with a beginning and an end, which is what spellF1 rewards. Most of the work went
into the evidence, because the hard part of this match is not the frames where the ball is
visible. It is the 41% of graded frames where it is not. We attack that from five directions:

1. borrow the ball from the broadcast tracking
2. read possession from the shape of the two teams
3. guess which player is nearest the unseen ball
4. learn possession and stoppages from seven fully labelled professional matches in open data,
   plus three of Metrica's own public sample matches
5. a second-stage model that reads all of that over a ±4 s window, to catch the order of touches
   that a single frame cannot show

| | public 5 min (17 possessions) | 7 DFL matches (≈3,000 possessions)¹ | 3 Metrica matches (≈1,100 possessions)³ |
|---|---|---|---|
| organisers' baseline (`smooth50`) | spellF1 0.326 · macroF1 0.621 | spellF1 0.201 | spellF1 0.499 |
| ours without the second stage (v9) | spellF1 0.514 · macroF1 0.880 | spellF1 0.466 · macroF1 0.854 | spellF1 0.574 |
| ours, second stage trained on DFL (v10) | spellF1 0.581 · macroF1 0.874 | spellF1 0.526² · macroF1 0.874 | spellF1 0.621 |
| **ours (submitted, v11): second stage trained on DFL + Metrica** | **spellF1 0.613 · macroF1 0.899** | **spellF1 0.530**² · macroF1 0.874 | **spellF1 0.633** |

¹ Leave-one-match-out, on DFL tracking damaged to look like ours (§3). This bench, not the public
window, is what every design decision was made on. The public window is 17 possessions of mostly
Barcelona build-up: too few, and too unusual, to choose between models.

² The second stage is scored leave-one-match-out too. Its first-stage inputs on each training
match come from models that did see the test match, which is a slight optimistic bias; on BAR-ATH
every input is out of sample.

³ Metrica's public sample matches, with possession labels rebuilt from their event data (§3).
Leave-one-match-out for the second stage; the first-stage models never see them.

## 1. Approach

### The model

```
tracking ──► features ──► emissions (T × 3 log-likelihoods) ──► Viterbi per half ──► H / A / D
                 ▲                                                   costs: H↔A 9 · →D 15 · D→ 15
   broadcast ball, formation, ball-finder, learned team / dead models, Smart Tagging
```

The team evidence and the alive/dead evidence are kept apart on purpose. Team evidence `t` (log
P(H)/P(A)) enters as `log P(H|alive) = −log(1+e^−t)` and `log P(A|alive) = −log(1+e^t)`. These sum
to 1 in probability space, so even overwhelming team evidence cannot make a dead frame look alive,
or the other way round. Before this change, strong team evidence kept shrinking the stoppages.

We chose an HMM and Viterbi over an end-to-end sequence model for three reasons:

- We have no labelled data from this match beyond 5 minutes.
- Our only large labelled source (DFL) uses a different tracking provider and a slightly different
  possession convention.
- Explicit, separately weighted evidence lets every source be tested on its own and dropped when it
  does not pay (§2).

### The evidence

| evidence | where it matters | what it is |
|---|---|---|
| **Nearest player** | ball tracked | the team of the nearest player, full weight within 1 m, fading to nothing at 3 m |
| **Broadcast ball** (`broadcast.py`) | ball missing in the tactical feed | the broadcast tracking is in the same pitch coordinates but on its own unsynchronised clock, with cuts and replays. We align it by fingerprinting player positions: for each 4 s sample, the offset at which broadcast and tactical players coincide. Each half has one constant offset (4014 and 26,920 frames). Replays and close-ups match nothing and are skipped. Filling tactical ball gaps from live broadcast footage cuts missing-ball frames from **40.7% to 36.8%** |
| **Formation** (`formation.py`) | ball missing | the team in possession spreads wider and longer and pushes up. A logistic regression on 15 smoothed shape features, expressed in the home team's attacking direction. It is trained without ground truth on our own first-pass labels where the ball is within 2 m of a player, and cross-fitted: each half is scored by a model trained on the other |
| **Ball-finder** (`ballfinder.py`) | ball missing | "which player is nearest the unseen ball?" asked as a ranking over the 22 players (crowding, pressing, role, distance to where the ball was last and will next be seen). It is trained on frames where the ball *is* seen, with the ball memory simulated as if a real-length gap had just started. P(home) is the summed probability of the home players. Cross-fitted between halves |
| **Learned team model** (`team_ml.py`) | everywhere | gradient-boosted classifier on DFL BallPossession: both teams' nearest and second-nearest distances and counts within 3/6 m, now, at the last-seen ball and at the next-seen ball, smoothed, plus the formation features. Output minus the training prior, weight 0.2 |
| **Learned dead model** (`dead_ml.py`) | everywhere | gradient-boosted classifier on DFL BallStatus: player speed from 4 s before to 4 s after the frame (scaled by the match's own typical speed), ball seen, moving, out of bounds, distance to lines and corners, crowding. It replaced our hand-built stoppage rules, which called slow build-up dead |
| **Second stage** (`seq.py`) | everywhere, mostly around changes | gradient-boosted H / A / D classifier on the DFL labels and on Metrica's three sample matches (labels rebuilt from their events, `external_metrica.py`) over a ±4 s window of the evidence above: the nearest-team distance gap, the learned team and dead outputs, the ball-finder, ball speed and player speed at offsets from −2 s to +2 s and as window means; frames since the last and until the next touch by each team; how often the nearest team flips within ±1 s and ±2 s (a scramble). Its inputs on the training matches are out of sample, as they are on BAR-ATH. Only its team log-odds are used, at weight 0.5. Per-frame team accuracy on DFL rises 2–4 points over the first-stage team model on every match |
| **Smart Tagging** (`tags.py`) | set pieces | a set-piece tag's restart sits about 88 frames (3.5 s) after the tag opens, the same as the kickoff tags. The seconds before lean dead, the seconds after lean live and towards the restarting team when the tag names one |

## 2. Assets: used, and tried and dropped

### Used

- **Tactical ATD and homography**: the backbone. Ball speed is recomputed from positions. The
  ball's third ATD value is filled on only ~8% of tracked frames (positive, 0.4–6, most likely a
  height estimate) and is not used.
- **Broadcast ATD and homography**: ball gap filling (above).
- **Smart Tagging**: set-piece evidence. Its attack and defence phases also give a full-match
  *consistency check* that needs no ground truth: live frames inside "BUILD UP by T" should mostly
  be T. We used that check to compare variants, never as training labels.
- **DFL open data**, 7 Bundesliga matches. Bassek, Rein, Weber & Memmert (2025), *An integrated
  dataset of spatiotemporal and event data in elite soccer*, Scientific Data; figshare
  [10.6084/m9.figshare.28196177](https://doi.org/10.6084/m9.figshare.28196177), **CC BY 4.0**. It
  has 25 fps TRACAB tracking with official per-frame BallPossession and BallStatus. It is
  downloaded automatically by `external_dfl.py`. Used to train the three learned models and as the
  benchmark. These are labels of *other* matches, which the rules allow in any amount; nothing is
  hand-labelled on BAR-ATH.
- **Metrica Sports sample data**, 3 matches
  ([github.com/metrica-sports/sample-data](https://github.com/metrica-sports/sample-data)): Metrica
  tracking and hand-tagged events, downloaded automatically by `external_metrica.py`. Possession
  labels are rebuilt from the events (§3). They train the second stage together with DFL, and are
  a second benchmark (`metrica_eval.py`, `metrica_bench.py`). It is the closest data to ours: the
  same provider, and the same habit of dropping the ball when it is dead.

### Not used in the submission

- **The videos**: the submitted pipeline decodes no video. We did try a ball detector on the tactical
  video (`video/ball_detector.py`, below), and left it out.

### Tried and dropped

All of these were measured on the DFL bench and on BAR-ATH. Their code is kept and switched off,
so the figures can be rebuilt.

| idea | result |
|---|---|
| Hand-built stoppage rules (player-speed mixture, ball out, ball speed) | replaced by the learned dead model: DFL spellF1 0.365 → 0.431 |
| Kinematic touch detection (ball path bends or speeds up next to a player), and making the H↔A cost cheap near such touches | the ATD ball is heavily smoothed (median 1.8 m from the nearest player), so touches barely register. No gain on either yardstick (`touches.py`, `switch_costs`) |
| Lower switch penalty, to find more short possessions | DFL flat (0.466 → 0.470), BAR-ATH public 0.514 → 0.41. It adds false changes as fast as true ones |
| Training the learned models on 3 randomly damaged copies per match | DFL 0.460 vs 0.466; the BAR-ATH checks were mixed. Kept 1 copy |
| Snapping each H↔A change onto the nearest touch by the new team (`snap` in `model.py`) | on DFL the true change sits on that touch 85% of the time, and snapping does bring more of our changes within 5 frames (20% → 28%). spellF1 does not move (0.4664 → 0.4669), because the badly placed edges are mostly in bursts of several short possessions, where one boundary cannot be right (§4) |
| Using the second stage's dead/alive output as well | DFL +0.013, but the set-piece restart check on BAR-ATH fell from 0.81 to 0.63. The first-stage dead model stays |
| A better change detector alone (tested with an *oracle*: switching made cheap exactly at the true changes) | even perfect knowledge of where changes happen only reached DFL 0.52, and a realistic 60%-recall detector 0.49. That showed the per-frame evidence was the bottleneck, not the decoder, and led to the second stage, which reached 0.526 without any oracle |
| **Ball detector on the tactical video** (`video/ball_detector.py`, `video_value.py`): YOLO fine-tuned on this match's own frames, with the ATD ball projected through the homography as automatic labels (no hand labels); cleaned labels (no height, away from gap edges, steady path) and three frames t−2, t, t+2 as the input channels, so a moving ball shows as a streak | on held-out tracked frames, 36% of top detections are within 10 px, and confident ones (≥ 0.3) are right 80–86% of the time. But 89% of the frames where our ball is still missing are stoppages, and only 7.4% of live frames lack a ball after the broadcast fill, so there is little left to fill. Filling moving detections changed public spellF1 0.581 → 0.587 (no extra possession found, one false one fewer) and the restart check 0.814 → 0.797. **Left out**: the gain is small, and a GPU-trained detector cannot be guaranteed to retrain to the same weights, which would put the byte-identical reproduction at risk |
| **DFL labels rewritten to "controlled possession"** (`convention.py`, `convention_bench.py`): a spell of one team shorter than 1 or 2 s between spells of the other team given back to that team, as a deflection, then the second stage retrained | no gain. Scored against rewritten DFL labels, v10 already reaches 0.629 (the rewrites 0.621–0.630). On BAR-ATH the rewrites predict fewer possessions (301 → 270–276) and public spellF1 falls 0.581 → 0.573. v10 already ignores most one-touch spells |
| Second stage at weight 1.0 instead of 0.5 | DFL 0.526 → 0.537 and Metrica 0.621 → 0.633, but BAR-ATH macroF1 0.874 → 0.855 and the tag agreement fell; the same with Metrica added (public 0.613 → 0.600, 323 → 347 possessions). Kept 0.5 |
| Retuning the decoder now the second stage exists: H↔A cost 7–13, stop/start 12 and 18 (`seq_bench.py`) | all within ±0.004 on DFL, none better on BAR-ATH. The costs stay |
| A second round of the second stage that also reads the first round's own outputs (auto-context, `context_bench.py`) | per-frame team accuracy on DFL 0.872 → 0.867, spellF1 0.526 → 0.523–0.524. Dropped, although the public window rose |
| Second stage trained on Metrica's matches only | best on Metrica (0.660), but DFL 0.503 and public 0.563: it learns Metrica's event-tagging timing rather than possession. Kept both sources (`metrica_bench.py`) |
| Training on DFL without the ball position noise | DFL bench up to 0.475, BAR-ATH collapsed (public 0.31, dead share 0.48 → 0.34). TRACAB freezes the ball during stoppages, and the model learned "still ball = dead", a shortcut our tracking never offers. The noise stays, documented in `dfl_eval.degrade` |

## 3. Possession rules we implemented

The ground truth is mechanical: corrected tracking names an owning team, or says the ball is dead.
In the DFL labels, a team change sits on the new team's first touch whenever that touch is visible
(85% within 5 frames). In the public window, most changes fall within a second of a touch by the
new team or a kick (`gt_rules.py`). Our rules follow from that.

- **Team change.** Possession changes only when the evidence for the other team outweighs a fixed
  cost (9 in log-likelihood). In practice the new team must have the ball for a sustained moment,
  or be clearly nearest it on both sides of a gap.
- **Loose balls and challenges.** A loose ball stays with the team that last had it. A challenge
  that does not end in control by the other team is not a change of possession, and a 50/50 that
  bounces between the two sides stays with whoever wins it in the end. This is the main deliberate
  simplification (§4).
- **Ball in flight.** A pass, cross or long ball belongs to the team that played it until someone
  receives it. A ball more than 3 m from everyone is not team evidence at all.
- **Stoppages.** The ball is dead from the moment play stops until the restart is taken. The point
  where play stops comes from the learned dead model, which looks 4 s ahead and behind (players
  slowing, the ball out of bounds). The restart comes from the same model plus the set-piece tags.
  The restart frame belongs to the team taking it.
- **Fouls, offsides and injuries** are dead time like any other stoppage. No separate rule.
- **Abstention.** We write `X` only outside the halves (before kickoff, half-time, after the final
  whistle). Inside a half every frame is H, A or D, because the scorer fills `X` with the last
  label anyway.

### Whose convention: DFL's or Metrica's?

The ground truth is Metrica's; DFL's labels come from another provider. We therefore also learn
from Metrica's three public sample matches. They have tracking and hand-tagged events but no
possession labels, so `external_metrica.py` rebuilds them from the events:

- a pass, shot, recovery, ball lost or set piece by a team means that team has the ball at that
  frame; a pass or shot keeps it until its end frame
- a ball out, a goal, a foul or the end of a half makes the ball dead, until the next set piece
- challenges and cards are skipped, because their team is ambiguous
- between events the ball stays with the last team that had it, the challenge's rule for a loose
  ball, so a change happens at the new team's first event

That gives 3.8–4.2 possessions per minute, a median length of 4–5 s and about 26% under 2 s,
between DFL (4.6 a minute, 30%) and the public window (3.4, 12%). The labels are approximate,
since events mark moments rather than the frame of first control, which is why Metrica alone was
not enough (§2): mixed with DFL, it raised all three labelled yardsticks.

One difference we checked and kept: in Metrica's sample matches the ball is almost never tracked
during a stoppage, while in BAR-ATH it is tracked during 14% of the public window's dead time.
The dead-ball model therefore still learns from DFL, damaged to our gap pattern, not from Metrica. The public
window, with 17 long Barcelona possessions, is the unusual one. The learned dead model never sees
which team has the ball, so possession conventions cannot leak into it at all.

## 4. Limitations

**Short possessions are where we lose.** On the DFL bench (`spelldiag.py`):

- About 86% of what we predict is a real possession.
- We recover 46% of the real possessions.
- 86% of the ones we miss are under 5 s, and half are under 2 s: the scrambles after a duel, a
  block or a clearance.
- Stoppage edges are placed to within a median of 5 frames.
- Team-change edges are off by a median of 32 frames. That is mostly the same problem seen from
  the other side: where the truth has H → A (1 s) → H (1 s) → A, we draw a single boundary
  somewhere in the middle.

These are not a tuning problem: lowering the change cost finds them and an equal number of false
ones. The second stage (v10) is what moved them: on the public window it finds 11 of 17 true
possessions instead of 9, and its team-change edges are off by a median of 35 frames instead of 66.
Recovering the rest needs evidence of an individual touch, and the tactical ATD ball is smoothed
and missing on 37% of frames even after the broadcast fill.

**One check disagrees with the second stage.** Smart Tagging's attack and defence phases agree
with our live labels on 87% of frames for v9, 84% for v10 and 83% for v11, and the set-piece
restart check falls from 0.81 to 0.80 with v11. Where the versions differ, the ball is visible but
moving (median 6.9 m/s) and away from players, in short stretches around passes and
interceptions. That is a question of exactly when a change happens. All three labelled sources we
have (DFL, Metrica's matches and the public window) favour the second stage there, so we kept it,
at a conservative weight. But the tagging is the only full-match signal from this match itself.

**What we would do with more time:**

- A neural sequence model (the second stage is gradient boosting over time-shifted features).
- Decoding for expected spellF1 rather than the single most likely path.
- Ball-touch detection from the broadcast video, where the ball is large and in focus. The
  tactical-video detector above locates the ball, but not precisely enough to see a touch.

**Other limitations:**

- **Tracking provider.** The learned models are trained on TRACAB tracking damaged to resemble
  this match's ATD (our real ball-gap pattern, ball noise 0.3 m, player noise 0.1 m, 7% player
  dropouts in 2 s chunks). Any difference the damage does not imitate is a risk. The
  frozen-ball artefact above is the one we found.
- **Validation.** We have no full-match ground truth. Decisions rest on:
  - the DFL bench
  - the public window
  - two ground-truth-free checks on BAR-ATH: agreement with Smart Tagging phases, and `gapsim.py`,
    which deletes the ball where it was seen and measures how stable the labels stay

  A change was kept only if it did not hurt any of them.

## Data sources and licences

| source | licence / terms | how we use it | changes we make |
|---|---|---|---|
| Challenge bundle, BAR-ATH: tactical and broadcast tracking, homography, Smart Tagging | competition terms of data use; not redistributed, not in this repository | the input of the pipeline | none to the files; derived arrays are cached locally in the gitignored `cache/` |
| DFL open data: Bassek, M., Rein, R., Weber, H. & Memmert, D. (2025). *An integrated dataset of spatiotemporal and event data in elite soccer.* Scientific Data. Data: [doi.org/10.6084/m9.figshare.28196177](https://doi.org/10.6084/m9.figshare.28196177) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | training the dead-ball, team and second-stage models; the 7-match benchmark | downloaded from the source at run time and not redistributed. Converted to arrays, and the tracking deliberately degraded to resemble BAR-ATH: ball deleted on our match's real gap pattern, 0.3 m ball and 0.1 m player position noise, 7% player dropouts (`dfl_eval.degrade`). The possession labels are used as published; a rewritten version was tried and dropped (§2) |
| Metrica Sports sample data (3 matches), [github.com/metrica-sports/sample-data](https://github.com/metrica-sports/sample-data) | published by Metrica Sports for public use, with the request to acknowledge the source | training the second-stage model with DFL; a 3-match benchmark | downloaded from the source at run time and not redistributed. Converted to arrays in metres; player speeds computed from positions; possession labels rebuilt from the events (§3) |
| Challenge repository code (scorer, validator, loaders), Metrica Sports | MIT (see `LICENSE`) | scoring, validation, data paths | none |

## 5. Reproduction

```bash
pip install -r solution/requirements.txt           # Python 3.12.3, pinned versions
# put the challenge bundle's data/tactical/ and data/broadcast/ files in data/ (no video needed)
python solution/run.py --out "Mes Que Un Club.csv"   # ≈20 min on the first run, ≈1 min after
python evaluation/validate_submission.py "Mes Que Un Club.csv"
```

- The first run downloads the DFL matches from figshare and Metrica's sample matches from GitHub,
  converts them, trains the learned models and caches everything in `cache/`.
- All randomness is seeded, and the file is **byte-identical** across runs, including from an empty
  cache. Submitted file md5: `c322c1bfad9ef0e077847ce0c2fec8c2`.
- Evaluation tools, none of them needed for reproduction:
  - `solution/dfl_eval.py --bench`: the DFL benchmark
  - `solution/metrica_eval.py`, `solution/metrica_bench.py`: the Metrica benchmark
  - `solution/experiment.py`: public window and Smart Tagging checks
  - `solution/gapsim.py`
  - `solution/spelldiag.py`: where spellF1 is lost

No match data, derived data or labels are in the repository. `data/`, `cache/` and `external/` are
gitignored. Nothing was fitted to the private ground truth, and the public 5 minutes were used
only as a check.
