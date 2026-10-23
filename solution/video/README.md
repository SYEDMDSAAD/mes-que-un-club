# Ball detector v4 — guide for the GPU teammate

v4 = v2 (your laptop settings: batch 4, no cache, 16 px boxes, no shrinking) plus:

1. **Clean labels** — frames where the ATD has a ball height (ball in the air), frames within 5 of a
   tracking gap, short tracked stretches and jumpy paths are left out of training and scoring.
   That keeps ~90% of tracked frames (~8,500 training frames, every 10th clean one).
2. **Motion input** — each crop is three greyscale frames (t-2, t, t+2) stacked as its colour
   channels. Still things look grey; a moving ball becomes a small coloured streak.

**Data rules.** Everything stays on your machine. Never upload the video, frames, crops, the
`runs/*.jpg` images or the output CSV to any AI tool or website. Text reports (`valcheck_*.txt`,
`report.txt`) contain numbers only and are fine to share. No hand labelling.

## Steps (new folder `work_v4`, so v1–v3 stay untouched)

```
pip install "numpy<2.4"                    # the np.trapz crash fix, if not already done
python ball_detector_v4.py dataset  --data data --out work_v4
python ball_detector_v4.py train    --data data --out work_v4
```
The dataset step decodes the video once (~15–25 min) and writes ~8,500 motion crop pairs plus a
colour copy of the held-out crops (for comparing v2 fairly). Training: ~6–8 min per epoch on your
laptop, early stop as before.

Then score **both** detectors on the **same clean held-out crops**:
```
python ball_detector_v4.py valcheck --data data --out work_v4 --kind motion --weights work_v4/runs/ball/weights/best.pt
python ball_detector_v4.py valcheck --data data --out work_v4 --kind rgb    --weights work_v2/runs/ball/weights/best.pt
```
Send the two printouts (or the `work_v4/valcheck_*.txt` files).

**Stop there.** We pick the winner from those numbers; only then run (overnight, a few hours):
```
python ball_detector_v4.py detect --data data --out work_v4 --weights <winner best.pt>
python ball_detector_v4.py report --data data --out work_v4
```
(If v2 wins, run detect with the v2 script instead, since v2 expects colour frames.)
