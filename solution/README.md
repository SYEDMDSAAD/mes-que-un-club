# Possession labelling — solution

Regenerates the submission file from the provided match data.

## Reproduce

```bash
pip install -r solution/requirements.txt          # Python 3.12
python solution/run.py --out "Mes Que Un Club.csv"   # ~20 min on first run, ~1 min after
python evaluation/validate_submission.py "Mes Que Un Club.csv"
```

Two runs produce byte-identical files, including from an empty `cache/`.

### Data it needs

| | where | how |
|---|---|---|
| tactical ATD, homography, Smart Tagging | `data/tactical/` | the challenge bundle (gated; not in this repo) |
| broadcast ATD and homography | `data/broadcast/` | the challenge bundle (gated; not in this repo). The broadcast video is not needed |
| DFL open data, 7 matches | `external/dfl/`, `cache/dfl/` | **downloaded automatically** from [figshare](https://doi.org/10.6084/m9.figshare.28196177) (Bassek et al. 2025, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); degraded to resemble our tracking, see TECHNICAL.md) |
| Metrica sample data, 3 matches | `external/metrica/`, `cache/metrica/` | **downloaded automatically** from [GitHub](https://github.com/metrica-sports/sample-data) (Metrica Sports, published for public use; possession labels rebuilt from the events, see TECHNICAL.md) |

No video is decoded. Everything derived from the data goes to `cache/`, which is gitignored, as are
`data/` and `external/`.

## Files

| file | what |
|---|---|
| `run.py` | entry point: writes the submission |
| `model.py` | the 3-state (H / A / D) model and its Viterbi decoder; every parameter is in `PARAMS` |
| `load.py`, `features.py` | tracking → arrays → per-frame features |
| `broadcast.py` | aligns the broadcast tracking to the tactical timeline and fills ball gaps from it |
| `formation.py` | team in possession from team shape (for frames without a ball) |
| `ballfinder.py` | which player is nearest the unseen ball |
| `dead_ml.py` | dead-ball classifier trained on DFL BallStatus |
| `team_ml.py` | possession classifier trained on DFL BallPossession |
| `seq.py` | second stage: H / A / D over a ±4 s window of the first-stage evidence, trained on DFL and Metrica |
| `tags.py` | Smart Tagging intervals and set-piece evidence |
| `external_dfl.py` | downloads and converts the DFL open data |
| `external_metrica.py` | downloads and converts Metrica's sample matches, rebuilds possession labels from their events |
| `mbr.py` | optional final decoder for expected spellF1 (posterior samples, block-wise choice among candidate paths); off while `PARAMS["mbr_tau"]` is 0 |
| `variant.py` | `MQUC_VARIANT=<tag>` runs an experimental variant with its own cached models (`state`, `taper`, `smoothS`, combined with `+`); unset = the submission |
| `touchfeat.py` | ball-touch features for the second stage (multi-scale impulse, attribution to a team, touch events, control); variant `touch`, blanked on Metrica's rows |
| `motion_ml.py` | possession from the players alone, blind to the ball, trained on DFL BallPossession: one more second-stage signal (v15; `MQUC_VARIANT=nomotion` = off) |
| `change_ml.py` | "H <-> A change within +-3 frames" detector on the second stage's rows; its log-odds relieve the H <-> A cost (`PARAMS["k_change"]`, 0 = off) |
| `video_events.py` | T-DEED ball actions from the broadcast video moved onto the tactical timeline, team-attributed, hashed; decoder relief of the H <-> A cost (`PARAMS["k_video"]`, 0 = off); `pseudo_events()` simulates them on labelled matches for tuning |
| `video/tdeed_infer.py` | runs T-DEED on the broadcast video reproducibly (decoded-frame check, no worker duplication, fp32, rounded scores); separate environment, `video/requirements-tdeed.txt` |

### Evaluation tools (not needed to reproduce)

| file | what |
|---|---|
| `dfl_eval.py --bench` | scores the pipeline on all 7 DFL matches (damaged to look like our tracking), leave-one-match-out |
| `metrica_eval.py`, `metrica_bench.py` | scores the pipeline on Metrica's 3 sample matches; second stage trained with and without them |
| `seq_bench.py`, `context_bench.py` | decoder retune and auto-context experiments (dropped) |
| `experiment.py` | public window + Smart Tagging consistency checks on BAR-ATH |
| `gapsim.py` | deletes the ball where it was seen and measures how the model copes |
| `gt_rules.py`, `diagnose.py` | what the public ground truth's boundaries line up with |
| `spelldiag.py` | where spellF1 is lost: missed / spurious possessions, edge errors |
| `metrica_events.py` | possession statistics rebuilt from Metrica's public sample events (convention check) |
| `touches.py` | kinematic touch detection (tried, inert by default) |
| `convention.py`, `convention_bench.py` | DFL labels rewritten to "controlled possession" (tried, dropped) |
| `video/ball_detector.py`, `video_value.py` | ball detector on the tactical video, GPU (tried, left out; see TECHNICAL.md) |
| `mbr_bench.py [--bar-ath]` | expected-spellF1 decoding vs Viterbi on the 7 DFL + 3 Metrica matches (v11 second stage, leave-one-match-out), with calibration columns |
| `domain_check.py [--labels file] [--more]` | ball-missing rates by state, gap lengths, ball speed / jerk, ball-to-player distance; with `--more` kink shape at 2 / 4 / 8 frames, interpolation, kick distance, out-of-bounds handling, players, and the labels' possession statistics: BAR-ATH vs clean and damaged DFL vs Metrica |
| `ceiling_bench.py` | oracle ceilings on the cached bench emissions: perfect dead / team evidence, perfect change times, perfect boundaries, perfect short possessions -- where the remaining spellF1 is |
| `touch_bench.py [--bar-ath]` | second stage and change detector leave-one-match-out on the 7 DFL + 3 Metrica matches, decoded at several `k_change`; run with and without `MQUC_VARIANT=touch...` |
| `tags_check.py --labels <csv>` | Smart Tagging's team phases against the public ground truth: is it a source of *who*? (aggregates) |
| `broadcast_check.py` | does the broadcast tracking get the nearest team right where the tactical one is wrong? (public window) |

The approach, the assets used and dropped, the possession rules and the limitations are in
[TECHNICAL.md](TECHNICAL.md).
