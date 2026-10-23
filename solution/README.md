# Possession labelling — solution

Regenerates the submission file from the provided match data.

## Reproduce

```bash
pip install -r solution/requirements.txt          # Python 3.12
python solution/run.py --out <team_name>.csv      # ~10 min on first run, ~1 min after
python evaluation/validate_submission.py <team_name>.csv
```

Two runs produce byte-identical files, including from an empty `cache/`.

### Data it needs

| | where | how |
|---|---|---|
| tactical ATD, homography, Smart Tagging | `data/tactical/` | the challenge bundle (gated; not in this repo) |
| broadcast ATD and homography | `data/broadcast/` | the challenge bundle (gated; not in this repo). The broadcast video is not needed |
| DFL open data, 7 matches | `external/dfl/`, `cache/dfl/` | **downloaded automatically** from [figshare](https://doi.org/10.6084/m9.figshare.28196177) (Bassek et al. 2025, CC BY 4.0) |

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
| `seq.py` | second stage: H / A / D over a ±4 s window of the first-stage evidence, trained on DFL |
| `tags.py` | Smart Tagging intervals and set-piece evidence |
| `external_dfl.py` | downloads and converts the DFL open data |

### Evaluation tools (not needed to reproduce)

| file | what |
|---|---|
| `dfl_eval.py --bench` | scores the pipeline on all 7 DFL matches (damaged to look like our tracking), leave-one-match-out |
| `experiment.py` | public window + Smart Tagging consistency checks on BAR-ATH |
| `gapsim.py` | deletes the ball where it was seen and measures how the model copes |
| `gt_rules.py`, `diagnose.py` | what the public ground truth's boundaries line up with |
| `spelldiag.py` | where spellF1 is lost: missed / spurious possessions, edge errors |
| `metrica_events.py` | possession statistics rebuilt from Metrica's public sample events (convention check) |
| `touches.py` | kinematic touch detection (tried, inert by default) |
| `convention.py`, `convention_bench.py` | DFL labels rewritten to "controlled possession" (tried, dropped) |
| `video/ball_detector.py`, `video_value.py` | ball detector on the tactical video, GPU (tried, left out; see TECHNICAL.md) |

The approach, the assets used and dropped, the possession rules and the limitations are in
[TECHNICAL.md](TECHNICAL.md).
