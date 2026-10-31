# Who has the ball — technical document

**Team Mes Que Un Club** · More Than A Hack 2027, Phase 1 · BAR-ATH

## In one paragraph

Every frame of each half gets three log-likelihoods: home has the ball (H), away has it (A), or it
is dead (D). A Viterbi decoder then finds the label sequence that best explains the whole half,
paying a fixed price for every change of state. The decoder is what turns per-frame flicker into
possessions with a beginning and an end, which is what spellF1 rewards. Most of the work went
into the evidence, because the hard part of this match is not the frames where the ball is
visible. It is the 41% of graded frames where it is not, and the frames where our tracking puts
the ball next to the wrong team. We attack that from seven directions:

1. borrow the ball from the broadcast tracking
2. read possession from the shape of the two teams
3. guess which player is nearest the unseen ball
4. learn possession and stoppages from seven fully labelled professional matches in open data,
   plus three of Metrica's own public sample matches
5. a second-stage model that reads all of that over a ±4 s window, to catch the order of touches
   that a single frame cannot show
6. a change detector, trained on the same rows, that tells the decoder *when* possession is
   likely to change hands, so a change costs less there
7. a possession model that never looks at the ball: it reads only how the two teams move, which
   still follows the real ball where the tracked one is wrong or missing
8. ball-touch features (where the ball kicks, turns or moves with a player) in the second stage,
   with its team evidence faded where it itself calls the ball dead, so that sure team evidence
   cannot pull a stoppage into play (v16)

| | public 5 min (17 possessions) | 7 DFL matches (≈3,000 possessions)¹ | 3 Metrica matches (≈1,100 possessions)³ |
|---|---|---|---|
| organisers' baseline (`smooth50`) | spellF1 0.326 · macroF1 0.621 | spellF1 0.201 | spellF1 0.499 |
| ours without the second stage (v9) | spellF1 0.514 · macroF1 0.880 | spellF1 0.466 · macroF1 0.854 | spellF1 0.574 |
| ours, second stage trained on DFL (v10) | spellF1 0.581 · macroF1 0.874 | spellF1 0.526² · macroF1 0.874 | spellF1 0.621 |
| ours, second stage trained on DFL + Metrica (v11) | spellF1 0.613 · macroF1 0.899 | spellF1 0.530² · macroF1 0.874 | spellF1 0.633 |
| ours, v11 + 7 s of dead-ball lean before a tagged set piece (v12) | spellF1 0.634 · macroF1 0.912 | spellF1 0.530²˒⁴ · macroF1 0.874 | spellF1 0.633⁴ |
| ours, v12 with realistic ball damage in training (§3), dead before kickoff (v13) | spellF1 0.632 · macroF1 0.902 | spellF1 0.527²˒⁵ · macroF1 0.873 | spellF1 0.650 |
| ours, v13 + a change detector (v14) | spellF1 0.625 · macroF1 0.905 | spellF1 0.537²˒⁶ | spellF1 0.661 |
| ours, v14 + the player-only model (v15) | spellF1 0.635 · macroF1 0.906 | spellF1 0.557²˒⁷ | spellF1 0.671 |
| ours, v15, the change detector trained on changes into possessions of 2 s or more (v15.1) | spellF1 0.647 · macroF1 0.907 | spellF1 0.558²˒⁸ | spellF1 0.671 |
| **ours (submitted, v16): v15.1 + ball-touch features, the second stage's team evidence gated by its own dead output, at weight 0.7 (§1)** | **spellF1 0.647 · macroF1 0.904** | **spellF1 0.589²˒⁹** | **spellF1 0.678** |

¹ Leave-one-match-out, on DFL tracking damaged to look like ours (§3). This bench, not the public
window, is what every design decision was made on. The public window is 17 possessions of mostly
Barcelona build-up: too few, and too unusual, to choose between models.

² The second stage is scored leave-one-match-out too. Its first-stage inputs on each training
match come from models that did see the test match, which is a slight optimistic bias; on BAR-ATH
every input is out of sample.

³ Metrica's public sample matches, with possession labels rebuilt from their event data (§3).
Leave-one-match-out for the second stage; the first-stage models never see them.

⁴ v12 changes only the use of Smart Tagging, which the DFL and Metrica matches do not have, so
their scores are v11's.

⁵ v13 changes how the DFL test matches are damaged as well, so this is a different test, not
comparable with the rows above. Metrica's matches are untouched, which makes them the yardstick
for v13: up on all three (0.669 / 0.641 / 0.641 against 0.644 / 0.637 / 0.619). On BAR-ATH the
public window is level (one possession either way), and both full-match Smart Tagging checks rise:
team agreement 0.829 → 0.843 and set-piece restarts 0.797 → 0.847.

⁶ Same test matches as v13, so directly comparable: DFL 0.527 → 0.537 (6 of 7 matches up),
Metrica 0.650 → 0.661 (all 3 up: 0.690 / 0.642 / 0.649). On BAR-ATH the public window is level
(0.632 → 0.625, one boundary), macroF1 0.902 → 0.905, the restart check unchanged (0.847) and team
agreement 0.843 → 0.837. It finds more possessions on the full match (307 → 338), closer to the
4–6 per minute of play the labelled sources show.

⁷ Same test matches as v14: DFL 0.537 → 0.557 (6 of 7 up: 0.478 / 0.486 / 0.616 / 0.574 / 0.566 /
0.600 / 0.580), Metrica 0.661 → 0.671 (all 3 up). With Metrica's ball made to look like ours
(`metatd`) 0.571 → 0.595 (all 3 up), and DFL in that run 0.519 → 0.545 (7 of 7). The second
stage's team weight went from 0.5 to 0.4: at 0.5 the benches were higher still (DFL 0.567, Metrica
0.676, `metatd` 0.606), but on BAR-ATH the stronger team evidence pulled stoppage frames into play
(public dead-class F1 0.888 → 0.823, macroF1 0.878, past our alarm of −0.02); 0.4 is the strongest
weight that does not (0.3: DFL 0.547; 0.2: 0.533). On BAR-ATH 326 possessions, restart check
0.847, team agreement 0.845.

⁸ Level with v15 on the labels as published (DFL 0.557 → 0.558, Metrica 0.671 → 0.671), better
where every possession under 2 s is given back to the surrounding team, the convention the public
window suggests (DFL 0.614 → 0.620, 6 of 7 up; Metrica 0.703 → 0.707, 3 of 3 up): a hedge on the
private labels' convention. BAR-ATH: 325 possessions, restart check 0.847, team agreement 0.845.

⁹ Same test matches as v15.1, second stage and change detector leave-one-match-out
(`leak_bench.py`): DFL 0.558 → 0.589 (7 of 7 up against touch at the old setting), Metrica 0.671 →
0.678 (3 of 3). Scored on a DFL ball made to look like BAR-ATH's (`testbar`, §4) the gain is
smaller, 0.536 → 0.542. On BAR-ATH the public window is level (0.647, macroF1 0.907 → 0.904, dead-class F1 0.890 → 0.887; 335 possessions), and of v15.1's 113
full-match stoppages (D runs of 1 s or more) 3 are lost and 1 is new, none of them with a Smart
Tagging set piece restarting in or just after it; the restart check is unchanged (0.847).

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

That separation is not complete. Viterbi scores a live frame by the better of H and A,
`−log(1+e^−|t|)`, so the surer the team evidence, the more alive a frame looks — by up to 0.69 a
frame, about what the dead evidence itself weighs. Inside a stoppage the second stage is often sure
of the team (the players line up for the restart), and at weight 0.5 (v15's motion signal, or the
touch features) BAR-ATH's public dead-class F1 fell 0.888 → 0.823. Since v16 the second stage's
team log-odds are multiplied by `sigmoid(−seq_dead)`, its own dead log-odds (`seq_gate` 1): they
fade where the second stage itself calls the ball dead. That allows a team weight of 0.7.
Two other cures were tried and dropped (§2).

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
| **Second stage** (`seq.py`) | everywhere, mostly around changes | gradient-boosted H / A / D classifier on the DFL labels and on Metrica's three sample matches (labels rebuilt from their events, `external_metrica.py`) over a ±4 s window of the evidence above: the nearest-team distance gap, the learned team and dead outputs, the ball-finder, ball speed and player speed at offsets from −2 s to +2 s and as window means; frames since the last and until the next touch by each team; how often the nearest team flips within ±1 s and ±2 s (a scramble). Since v16 it also reads ball-touch features (`touchfeat.py`; see the player-only model for the weight). Its inputs on the training matches are out of sample, as they are on BAR-ATH. Only its team log-odds are used, at weight 0.7, faded where its own dead output is high (§1). Per-frame team accuracy on DFL rises 2–4 points over the first-stage team model on every match |
| **Player-only model** (`motion_ml.py`) | where the tracked ball is on the wrong team, or missing | gradient-boosted classifier on DFL BallPossession that never sees the ball: both teams' shape (centroids, depth, width, push, lines, from `formation.py`) and pressing (for each team its most-pressed player: opponents within 10 m running straight at him), now, ±1 and ±2 s, and their changes over the second before and after. The players react to the real ball, so it is right where ball-based models copy a tracking error. Trained on DFL damaged as for the others but with player dropouts 250 frames long (ours are few and long: 6–11 track births a minute, against 43–49 with the 50-frame dropouts). Its log-odds enter the second stage as one more signal; the second stage's team weight was 0.4 in v15 (0.5 before), 0.7 with the gate since v16 |
| **Change detector** (`change_ml.py`) | at changes of possession | gradient-boosted classifier on the second stage's input rows (DFL + Metrica, live frames): "is there an H↔A change within ±3 frames, into a possession that lasts at least 2 s?" (`MIN_KEEP` 50 frames since v15.1: DFL changes hands on every touch, the challenge counts controlled possessions). Its log-odds above the training prior, × 2, are taken off the H↔A cost on that frame, never below 2 (`k_change`, `change_min`). The second stage says *who*; this says *when*, which the per-frame team log-odds blur over the second either side of a change |
| **Smart Tagging** (`tags.py`) | set pieces | a set-piece tag's restart sits about 88 frames (3.5 s) after the tag opens, the same as the kickoff tags. The 7 s before lean dead (in Metrica's matches the ball is dead for at least 7 s before 76% of set pieces: median 13 s, 9 s before a throw-in, 23 s before a free kick, 29 s before a corner), the seconds after lean live and towards the restarting team when the tag names one |

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
  video (`video/ball_detector.py`, below), and left it out. `video_events.py` and
  `video/tdeed_infer.py` can turn T-DEED's ball-action spotting on the broadcast video into team
  evidence (`k_video_team`) or cheaper changes (`k_video`); both are off, and nothing in the
  submission reads them.

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
| A longer dead-ball lean before tagged set pieces: 10 or 15 s instead of 7 (4 in v11) | 7 s removes the one fake restart the public window shows (public 0.613 → 0.634) and changes the full match in a dozen places, mostly stoppages starting a few seconds earlier. 10 and 15 s leave the public window as it is but remove 7 and 16 more possessions on the full match, cutting into live play before quick throw-ins, a third of which come less than 10 s after the ball went out. Kept 7 s |
| Second stage at weight 1.0 instead of 0.5 | DFL 0.526 → 0.537 and Metrica 0.621 → 0.633, but BAR-ATH macroF1 0.874 → 0.855 and the tag agreement fell; the same with Metrica added (public 0.613 → 0.600, 323 → 347 possessions). Kept 0.5 |
| Retuning the decoder now the second stage exists: H↔A cost 7–13, stop/start 12 and 18 (`seq_bench.py`) | all within ±0.004 on DFL, none better on BAR-ATH. The costs stay |
| A second round of the second stage that also reads the first round's own outputs (auto-context, `context_bench.py`) | per-frame team accuracy on DFL 0.872 → 0.867, spellF1 0.526 → 0.523–0.524. Dropped, although the public window rose |
| Second stage trained on Metrica's matches only | best on Metrica (0.660), but DFL 0.503 and public 0.563: it learns Metrica's event-tagging timing rather than possession. Kept both sources (`metrica_bench.py`) |
| Metrica labels sharpened with the tracking (`external_metrica.refine`): each team change moved back from the new team's first event to the moment its player reached the ball (a third of changes move, median 4 frames); and Metrica's matches counted twice in training | sharpened: DFL 0.530 → 0.529, public 0.613 → 0.600. Counted twice: Metrica 0.633 → 0.639 but DFL 0.527 and public 0.568. Both together: DFL and public unchanged, Metrica 0.642, which only reflects the extra weight on Metrica. Kept v11 (`metrica_bench.py`) |
| Training on DFL without the ball position noise | DFL bench up to 0.475, BAR-ATH collapsed (public 0.31, dead share 0.48 → 0.34). TRACAB freezes the ball during stoppages, and the model learned "still ball = dead", a shortcut our tracking never offers. Solved in v13 by a drifting dead ball instead (§3) |
| Slow ball noise without the dead-ball drift (`MQUC_VARIANT=noise`) | Metrica 0.633 → 0.648, but BAR-ATH public macroF1 0.90 → 0.67: the frozen-ball shortcut came back as soon as the per-frame jitter that hid it was gone |
| The ball removed mostly in stoppages, at fixed rates (`MQUC_VARIANT=state`, `state40`) | Metrica up to 0.683, BAR-ATH public macroF1 0.72–0.74. In Metrica's matches the ball is almost never tracked when dead; in BAR-ATH's public window it still is on 67% of dead frames, so "no ball = stoppage" does not carry over |
| Decoding for expected spellF1 instead of the most likely path (`mbr.py`, `mbr_bench.py`): posterior paths sampled, a candidate path chosen per stretch between sure stoppages | +0.0025 on DFL and Metrica at the best temperature, public window 0.621 → 0.600. Too small to be worth it; the posterior is overconfident (it expects F1 0.99), as 25 strongly correlated frames a second would predict. Off (`mbr_tau` 0) |
| Change detector at other strengths, and with a higher H↔A cost (`touch_bench.py`) | strength 1: DFL +0.004, Metrica +0.006; strength 3: DFL +0.006, Metrica −0.002, public 0.600 and 358 possessions; strength 2 with H↔A cost 11: DFL +0.005, Metrica +0.007. Kept strength 2 at cost 9 |
| Change detector targets: every change (`MIN_KEEP` 0, v15), or only changes into a possession of at least 1 s (25) or 2 s (50) | a hedge against the private labels' convention. Each scored against the original labels and against labels with every possession under 2 s given back to the surrounding team (`convention.rewrite`): 0: DFL 0.557 / 0.614, Metrica 0.671 / 0.703; 25: DFL 0.559 / 0.615, Metrica 0.669 / 0.704; **50: DFL 0.558 / 0.620 (rewritten 6 of 7 up), Metrica 0.671 / 0.707 (3 of 3 up)**. 50 is level on the original labels and better on the rewritten ones, so it is adopted (v15.1); BAR-ATH public in the bench 0.647 / 0.907, 325 possessions |
| Ball-touch features in the second stage (`touchfeat.py`, `MQUC_VARIANT=touch`): ball impulse at 2 / 4 / 8 frames given to the nearer team, touch events, alternations, "ball moving with the player"; with the DFL training ball blurred (`smoothS`) so its kinks look like our smoothed ATD, and the dead-ball drift faded in over ½ s (`taper`) instead of switched on in one frame | blurring is what decides it. `domain_check.py --more` shows our ball's touches are small (impulse over 4 frames p50 / p90 / p99 1.2 / 2.8 / 5.3 m/s; v13's damaged DFL p90 / p99 4.1–6.7 / 18–22); a 5-frame blur matches them, a 2-frame one does not (p90 / p99 4–5 / 13). But with the 5-frame blur BAR-ATH macroF1 fell 0.90 → 0.86 with or without touch features (public 0.55–0.59), and with the 2-frame blur 0.875–0.882 (public 0.62–0.63), against Metrica 0.652–0.663 and v14's 0.661. Touch features add ~+0.04 on BAR-ATH public on top of the blurred ball, so the loss comes from the blur; a blurred ball also loses the frame-to-frame jitter ours has (`acf\|dv\|` 0.97 against 0.50). Off until a damage that matches both is found |
| Ball-touch features with their own ball (`MQUC_VARIANT=touch+touchblur6`): the touch features of training and bench matches read their damaged ball blurred by 6 frames, every other model keeps the v15 training ball | the 6-frame blur matches our ball on every kink row (impulse over 4 frames p90 / p99 2.98 / 5.96 m/s against 2.8 / 5.3, turn8 p90 0.123 against 0.12, kick near 1.97 m against 1.70). On v15 (team weight 0.4): DFL 0.557 → 0.568 (6 of 7 up), and in the `metatd` run 0.545 → 0.561 (7 of 7); but Metrica with a ball like ours only 0.595 → 0.599 (1 of 3 up), and BAR-ATH public spellF1 0.635 → 0.642 with macroF1 0.906 → 0.891. About +0.01 where it shows at all, with the stoppage leak the motion model had at weight 0.5: kept off in v15.1, **adopted in v16** with the gate (§1, footnote ⁹) |
| Other cures for the stoppage leak (`leak_bench.py`): the alive level taken from the first-stage team evidence only (`alive_level` "first"), or the second stage's dead output added (`k_seq_dead` 0.3) | "first": DFL down on 7 of 7 matches at weight 0.4, and the restart check 0.847 → 0.831. `k_seq_dead` 0.3: strong on DFL and Metrica, but BAR-ATH public dead-class F1 0.826, 7 of v15.1's 113 stoppages lost (2 confirmed by a set-piece tag), restart check 0.712. The gate (§1) passes every check |
| Team evidence from logged events, as if from video event spotting (`dfl_events.py`): DFL's event log aligned to the tracking by time and location (70% within ±3 frames), then fed through `video_events.py` as T-DEED events would be | on the BAR-like DFL ball, against no events at 0.536: events at their exact frame with the logged team +0.110; with the team attributed from our tracking −0.047; at measured timing with the logged team +0.049, with attributed team −0.023. The gain is all in *who* touched the ball, which our tracking cannot supply; video events stay off until a spotter that names the team exists |
| Second stage and change detector trained on a BAR-like DFL ball (`MQUC_VARIANT=seqbar`), or on both balls (`seqmix`); the BAR-like ball calibrated against our public window (`domain_check.py --calibrate`: 6-frame blur, 0.3 m slow offset, 0.112 off against 0.320 for the earlier setting) | scored on that ball, both look strong: DFL +0.026 (seqbar, 7 of 7) and +0.029 (seqmix, 7 of 7). seqbar also gains on Metrica, the only real tracking we can score (+0.004, 3 of 3); it loses 0.028 on the usual bench, which is expected of a model trained on a ball that is rarely missing in play. seqmix loses 0.005 on Metrica (0 of 3): mixing halves Metrica's share of the training rows. On BAR-ATH, seqbar raises public spellF1 0.647 → 0.663 and keeps the full-match checks (restart 0.847, 312 possessions, no tagged stoppage lost, tag agreement 0.841 → 0.852), but the public dead-class F1 falls 0.887 → 0.852 (macroF1 0.904 → 0.891): trained on a ball missing mainly in stoppages, it moves stoppage edges. Dropped; the second stage keeps learning from the usual damage |
| Decoder and evidence weights retuned on the BAR-like ball (`tune_bench.py`): 15 settings one at a time, each scored by its held-out (nested) gain, then the suggested combination (H↔A cost 11, `k_team` 0.5, `k_tml` 0.1, `k_change` 1, near / far 0.5 / 2 m, `k_bf` 0.1, `k_form_missing` 0.2, `seq_gate` 2) | each setting is worth +0.001 to +0.004 on its own. The combination: DFL +0.006 (6 of 7) and Metrica +0.013 on that ball, usual bench −0.004; most of its gain is against labels with possessions under 2 s merged (+0.028 / +0.046). On BAR-ATH it merges the same short possessions: 335 → 288 on the full match, public window 0.647 → 0.635, set-piece restart check 0.847 → 0.814. Dropped: the gain is far inside the private score's noise, and our own match's checks go the other way. Giving our own possessions under 1 s back to the surrounding team (`min_spell` 25) was not suggested: Metrica −0.022 |
| Smart Tagging team phases as a per-frame prior (`tags_check.py`) | on the public window the tags cover 55% of in-play frames and are right on 0.774 of them (our labels 0.917); on the 729 frames where they disagree with ours they are right 0.162 of the time. Kept as an independent check only |
| Training ball put on the wrong team for stretches (`MQUC_VARIANT=wrongballR`, R a minute) | not needed: damaged DFL already has the nearest player on the wrong team about as often as ours ("near right" 0.746 / 0.888 all / within 1.5 m, BAR-ATH public 0.781 / 0.889); every R from 1 to 6 moves it further away (R 1: 0.732 / 0.854, R 6: 0.675 / 0.727) |
| Longer player dropouts in the DFL damage (`droplen250`) | track births 43–49 → 9–10 a minute, dropout length ~150 → 650–820 frames (ours 5.6–11 a minute, 1235 frames); on its own DFL 0.531 (−0.006), Metrica 0.665, BAR-ATH public 0.588 / 0.871 |
| Player-only possession model in the second stage at the v14 team weight 0.5, and with all models trained on the longer dropouts (`motion_ml.py`): formation and pressing features of both teams, blind to the ball, trained on DFL's true labels | the biggest lever found since v10: the ceiling bench puts perfect team evidence on frames whose nearest player is on the wrong team at +0.195 (DFL) / +0.181 (Metrica), and on frames without a ball at +0.168 / +0.011. motion: DFL 0.537 → 0.565 (7 of 7), Metrica 0.669; with `droplen250` 0.564 (7 of 7 against droplen250 alone), Metrica 0.669, Metrica with `metatd` 0.571 → 0.597 (3 of 3); with the longer dropouts in the motion model's players only (`motiondrop250`) DFL 0.567 (7 of 7), Metrica 0.676 (3 of 3), with `metatd` 0.606 (3 of 3). But BAR-ATH public macroF1 0.905 → 0.878–0.888 (spellF1 0.612–0.630), the loss all in the dead class (F1 0.888 → 0.823; team classes −0.01); 1.3% of the match's frames change. A second stage whose dead output comes from a model without the motion signal changed nothing: that output's weight is 0, the dead frames move with the team evidence. Adopted at team weight 0.4 instead (v15, footnote ⁷) |
| The broadcast tracking's players as a second opinion where the tactical tracking puts the wrong team nearest the ball (`broadcast_check.py`) | where tactical is wrong, broadcast is right on only 35% of frames. Not used |

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
- **Before kickoff.** The ball is dead until each half's kickoff, taken from the organisers'
  anchors in `games.py` (the last second is left to the evidence). The second half's graded frames
  begin 117 s before its kickoff, while players walk on; v12 found six possessions there.
- **Abstention.** We write `X` only outside the halves (before kickoff, half-time, after the final
  whistle). Inside a half every frame is H, A or D, because the scorer fills `X` with the last
  label anyway.

### Making the training ball look like ours (v13)

The learned models train on DFL tracking damaged to resemble BAR-ATH. A review of that damage
(`domain_check.py`, which prints the statistics the features are built on side by side) found
that the 0.3 m of independent noise we added per frame was a median 12.5 m/s of fake ball speed:

| | ball speed in play, median | frame-to-frame change of ball velocity, median | dead ball moved in 1 s, median |
|---|---|---|---|
| BAR-ATH (full match, our labels) | 5.8 m/s | 0.3 | 1.5 m |
| DFL clean | 6.3 | 0.6–0.8 | 0.0 (frozen) |
| DFL damaged, v12 | 14.6 | 21.8 | 0.5 |
| **DFL damaged, v13** | **6.3** | **0.8** | **2.1** |

So no model could learn from ball speed, which is half of what a touch looks like. The noise was
there for a reason, though: TRACAB freezes a dead ball, and without the jitter the models learn
"a still ball is dead". v13 uses slow noise (smoothed over half a second) and adds a slowly
wandering ball in stoppages only, the way our ball is carried to the restart. Both halves are
needed; slow noise alone collapsed BAR-ATH (§2).

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
  this match's ATD (our real ball-gap pattern, slow 0.3 m ball noise with a wandering ball in
  stoppages, player noise 0.1 m, 7% player dropouts in 2 s chunks; `domain_check.py` compares the
  statistics side by side). Any difference the damage does not imitate is a risk. The
  frozen-ball artefact above is the one we found.
- **The DFL bench flatters the ball.** On BAR-ATH the ball is seen in 96% of live frames; in the
  damaged DFL matches 64–70%. Scoring DFL on a ball made to look like ours (`MQUC_VARIANT=testbar`:
  gaps by state, ATD-like smoothing; every model trained as submitted) shrinks most gains, which
  is why v16 was also checked against v15.1's stoppages on the full match (footnote ⁹). Training
  the second stage on that ball instead (v17, §2) did not survive the real-tracking checks, so the
  gap remains.
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
| DFL open data: Bassek, M., Rein, R., Weber, H. & Memmert, D. (2025). *An integrated dataset of spatiotemporal and event data in elite soccer.* Scientific Data. Data: [doi.org/10.6084/m9.figshare.28196177](https://doi.org/10.6084/m9.figshare.28196177) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | training the dead-ball, team and second-stage models and the change detector; the 7-match benchmark | downloaded from the source at run time and not redistributed. Converted to arrays, and the tracking deliberately degraded to resemble BAR-ATH: ball deleted on our match's real gap pattern; slow ball noise (0.3 m, smoothed over half a second) plus a slowly wandering ball in stoppages (1.5 m), 0.1 m player position noise, 7% player dropouts (`dfl_eval.degrade`). The possession labels are used as published; a rewritten version was tried and dropped (§2) |
| Metrica Sports sample data (3 matches), [github.com/metrica-sports/sample-data](https://github.com/metrica-sports/sample-data) | published by Metrica Sports for public use, with the request to acknowledge the source | training the second-stage model and the change detector with DFL; a 3-match benchmark | downloaded from the source at run time and not redistributed. Converted to arrays in metres; player speeds computed from positions; possession labels rebuilt from the events (§3) |
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
  cache. Submitted file md5: `7869e10ff29fd01aa57747b1fbe7e68c`.
- Evaluation tools, none of them needed for reproduction:
  - `solution/dfl_eval.py --bench`: the DFL benchmark
  - `solution/metrica_eval.py`, `solution/metrica_bench.py`: the Metrica benchmark
  - `solution/experiment.py`: public window and Smart Tagging checks
  - `solution/gapsim.py`
  - `solution/spelldiag.py`: where spellF1 is lost

No match data, derived data or labels are in the repository. `data/`, `cache/` and `external/` are
gitignored. Nothing was fitted to the private ground truth, and the public 5 minutes were used
only as a check.
