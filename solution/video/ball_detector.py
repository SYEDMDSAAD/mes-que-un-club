"""Find the ball in the tactical video on the frames where the tracking lost it.  (v4)

The automatic tracking (ATD) has the ball on ~57% of frames. On those frames it also tells us
where the ball is in the picture -- the homography maps its pitch position to a pixel. That is a
free training set: thousands of frames of THIS match, THIS camera, with the ball marked by an
algorithm, not by hand. A small detector (YOLO) is trained on crops around those pixels and then
run over every graded frame, including the ~43% where the tracking has no ball.

What v4 changes, after v1-v3 stalled at ~1 in 3 balls found within 10 px:

  clean labels   Only frames where the label can be trusted are used for training and scoring:
                 no ball height in the ATD (an airborne ball projects onto the grass below it,
                 not onto the ball), not within EDGE frames of a tracking gap (the tracker is
                 guessing there), inside a tracked run of at least MIN_RUN frames, and a steady
                 pixel path (no jump of more than MAX_JERK px against its neighbours).
  motion         The detector sees three greyscale frames, t - GAP, t and t + GAP, as its three
                 colour channels. The pitch and the standing players look the same in all three,
                 so they come out grey; a moving ball sits in a different place in each channel
                 and shows up as a coloured streak -- the TrackNet idea for tiny, fast balls.
  laptop         BATCH 4, no image cache (v2 settings), --batch to change it.

Steps, each resumable:

    python ball_detector.py check     --data DATA --out OUT [--count]   # environment + video
    python ball_detector.py dataset   --data DATA --out OUT   # clean crops: motion + plain colour
    python ball_detector.py train     --data DATA --out OUT   # fine-tune YOLO on the motion crops
    python ball_detector.py valcheck  --data DATA --out OUT --weights W --kind motion|rgb
    python ball_detector.py detect    --data DATA --out OUT   # every graded frame
    python ball_detector.py report    --data DATA --out OUT   # accuracy summary, numbers only

valcheck scores any weights on the clean held-out crops: --kind rgb for the v2/v3 weights (they
expect a colour picture), --kind motion for v4. Same frames, same crops, so the numbers compare.

DATA is the challenge bundle's data/ folder (with tactical/ inside). Output:

    OUT/ball_video_tactical.csv     frame, rank, px, py, conf   (up to TOP_K rows per frame)
    OUT/report.txt, OUT/valcheck_*.txt   aggregate numbers, safe to share with the team

DATA RULES. The video and everything derived from it stay on this machine: never upload frames,
crops or the output to an online AI tool or website. The text reports hold aggregates only.
"""

import argparse
import glob
import hashlib
import json
import os
import random
import sys

import numpy as np

# Params
FPS = 25
N_FRAMES = 199319
WIDTH, HEIGHT = 1920, 1080
PITCH = (105.3, 68.0)                         # m
HALVES = ((11185, 90651), (110808, 190197))   # graded video frames (DATA.md)
STEM = "BAR-ATH_tactical"
BALL_DIAMETER = 0.22                          # m
BOX_SCALE, BOX_MIN, BOX_MAX = 1.6, 16, 48     # px: v2 -- a 4-9 px ball with 2-3 px jitter fits
# clean labels
EDGE = 5                                      # frames next to a tracking gap are not trusted
MIN_RUN = 25                                  # frames: shorter tracked stretches are not trusted
MAX_JERK = 3.0                                # px: |p(t-1) - 2 p(t) + p(t+1)| above this is a jump
# motion input
GAP = 2                                       # frames between the three channels
# training data
SAMPLE_EVERY = 10                             # every 10th clean frame (fewer frames are clean)
VAL_BLOCK = 3000                              # frames; every 5th block of time is held out
TILE = 640
TILE_X = (0, 427, 853, 1280)                  # 4 x 2 overlapping tiles cover 1920 x 1080
TILE_Y = (0, 440)
NEG_MIN_DIST = 120                            # px: negative crops keep the ball this far out
EPOCHS = 30
PATIENCE = 8                                  # stop early once 8 epochs bring no gain
BATCH = 4                                     # 4 GB laptop GPU: batch 8 ran out of memory
BASE_MODEL = "yolo11s.pt"
# detection
CONF_MIN = 0.01                               # low on purpose: weak candidates are linked over time later
TOP_K = 3
MERGE_PX = 12                                 # detections from overlapping tiles closer than this merge
FRAMES_PER_BATCH = 4
SEED = 0


# ---------------------------------------------------------------- data

def paths(data_dir):
    folder = os.path.join(data_dir, "tactical")
    return {"video": os.path.join(folder, STEM + ".mp4"),
            "atd": os.path.join(folder, STEM + "_atd.txt"),
            "homography": os.path.join(folder, STEM + "_homography.csv")}


def graded(frame):
    return any(first <= frame <= last for first, last in HALVES)


def graded_frames():
    return [f for first, last in HALVES for f in range(first, last + 1)]


def load_ball(atd_path):
    """(N, 3): normalised pitch x, y of the tracked ball (NaN where missing), and its third value.

    The third value is filled on ~8% of tracked frames, always positive, 0.4-6 typically: most
    likely the ball's height in metres where the tracker estimated one.
    """
    ball = np.full((N_FRAMES, 3), np.nan)
    with open(atd_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            x, y, z = parts[2].split(",")
            ball[int(parts[0]) - 1] = float(x), float(y), float(z)
    return ball


def load_homography(path):
    import pandas as pd
    table = pd.read_csv(path)
    cols = ["h{}{}".format(r, c) for r in range(3) for c in range(3)]
    return table[cols].values.astype(np.float64).reshape(-1, 3, 3)


def project(h, x, y):
    d = h[2, 0] * x + h[2, 1] * y + h[2, 2]
    return (h[0, 0] * x + h[0, 1] * y + h[0, 2]) / d, (h[1, 0] * x + h[1, 1] * y + h[1, 2]) / d


def ball_pixels(ball, hom):
    """(N, 3): pixel x, pixel y and label box side of the tracked ball, NaN where missing."""
    out = np.full((N_FRAMES, 3), np.nan)
    for f in np.flatnonzero(np.isfinite(ball[:, 0])):
        h = hom[f]
        if not np.isfinite(h).all():
            continue
        x, y = ball[f, :2]
        px, py = project(h, x, y)
        qx, qy = project(h, x + BALL_DIAMETER / PITCH[0], y)
        rx, ry = project(h, x, y + BALL_DIAMETER / PITCH[1])
        diam = 0.5 * (np.hypot(qx - px, qy - py) + np.hypot(rx - px, ry - py))
        if 0 <= px < WIDTH and 0 <= py < HEIGHT:
            out[f] = px, py, float(np.clip(BOX_SCALE * diam, BOX_MIN, BOX_MAX))
    return out


def clean_mask(ball, pix):
    """True where the tracked ball's pixel can be trusted as a label (see the module docstring)."""
    seen = np.isfinite(pix[:, 0])
    ok = seen & ~np.isfinite(ball[:, 2])                      # no height: ball on the ground
    # inside a long tracked run, away from its ends
    starts = np.flatnonzero(seen & ~np.r_[False, seen[:-1]])
    ends = np.flatnonzero(seen & ~np.r_[seen[1:], False])
    inner = np.zeros(N_FRAMES, bool)
    for s, e in zip(starts, ends):
        if e - s + 1 >= MIN_RUN:
            inner[s + EDGE:e - EDGE + 1] = True
    ok &= inner
    # steady pixel path
    jerk = np.full(N_FRAMES, np.inf)
    second = pix[:-2, :2] - 2 * pix[1:-1, :2] + pix[2:, :2]
    jerk[1:-1] = np.hypot(second[:, 0], second[:, 1])
    ok &= np.nan_to_num(jerk, nan=np.inf) <= MAX_JERK
    return ok & np.array([graded(f) for f in range(N_FRAMES)])


def is_val(frame):
    return (frame // VAL_BLOCK) % 5 == 4


def windows(video_path, centers, keep_colour=False):
    """Yield (frame, motion image, colour image or None) for each wanted centre frame.

    The motion image stacks greyscale frames t - GAP, t, t + GAP as its three channels. The video
    is decoded sequentially, never seeking: seeking in an mp4 can land on the wrong frame, and a
    one-frame shift would silently misalign everything.
    """
    import cv2
    centers = sorted(set(int(c) for c in centers))
    if not centers:
        return
    grey, colour = {}, {}
    k, frame = 0, 0
    cap = cv2.VideoCapture(video_path)
    while k < len(centers):
        ok, image = cap.read()
        if not ok:
            raise RuntimeError("video ended at frame {} before frame {}".format(
                frame, centers[k] + GAP))
        c = centers[k]
        if frame >= c - GAP:
            grey[frame] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if keep_colour:
                colour[frame] = image
        while k < len(centers) and frame == centers[k] + GAP:
            c = centers[k]
            motion = np.dstack([grey[c - GAP], grey[c], grey[c + GAP]])
            yield c, motion, colour.get(c)
            k += 1
            if k < len(centers):
                low = centers[k] - GAP
                for f in [f for f in grey if f < low]:
                    del grey[f]
                    colour.pop(f, None)
        frame += 1
    cap.release()


def require_gpu(args):
    import torch
    if not torch.cuda.is_available() and not args.allow_cpu:
        sys.exit("No CUDA GPU visible to PyTorch -- this would run on the CPU and take days.\n"
                 "Reinstall torch with CUDA, check with the 'check' step, or pass --allow-cpu.")
    if torch.cuda.is_available():
        print("using GPU:", torch.cuda.get_device_name(0))


# ---------------------------------------------------------------- steps

def check(args):
    import cv2
    p = paths(args.data)
    for name, path in p.items():
        print("{:<11} {}  {}".format(name, "ok" if os.path.isfile(path) else "MISSING", path))
    try:
        import torch
        print("torch {}  cuda {}  {}".format(torch.__version__, torch.cuda.is_available(),
                                            torch.cuda.get_device_name(0)
                                            if torch.cuda.is_available() else ""))
    except ImportError:
        print("torch MISSING")
    try:
        import ultralytics
        print("ultralytics {}  numpy {}".format(ultralytics.__version__, np.__version__))
    except ImportError:
        print("ultralytics MISSING")
    cap = cv2.VideoCapture(p["video"])
    print("video {}x{} at {:.2f} fps, header says {} frames".format(
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    if args.count:
        n = 0
        while cap.read()[0]:
            n += 1
        print("decoded {} frames (must be {})".format(n, N_FRAMES))
    cap.release()


def dataset(args):
    rng = random.Random(SEED)
    p = paths(args.data)
    print("reading tracking and homography ...")
    ball = load_ball(p["atd"])
    pix = ball_pixels(ball, load_homography(p["homography"]))
    clean = clean_mask(ball, pix)
    np.save(os.path.join(args.out, "atd_ball_pixels.npy"), pix)
    np.save(os.path.join(args.out, "clean_mask.npy"), clean)
    tracked = np.isfinite(pix[:, 0]) & np.array([graded(f) for f in range(N_FRAMES)])
    chosen = np.flatnonzero(clean)[::SAMPLE_EVERY]
    print("tracked graded frames {}, clean {} ({:.0%}); training frames {}".format(
        int(tracked.sum()), int(clean.sum()), clean.sum() / max(tracked.sum(), 1), len(chosen)))
    print("  removed: height {}, gap edge / short run or unsteady {}".format(
        int((tracked & np.isfinite(ball[:, 2])).sum()),
        int((tracked & ~np.isfinite(ball[:, 2]) & ~clean).sum())))

    root = os.path.join(args.out, "yolo")
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            os.makedirs(os.path.join(root, kind, split), exist_ok=True)
    for kind in ("images", "labels"):
        os.makedirs(os.path.join(args.out, "val_rgb", kind, "val"), exist_ok=True)

    for n, (frame, motion, colour) in enumerate(windows(p["video"], chosen, keep_colour=True)):
        split = "val" if is_val(frame) else "train"
        bx, by, side = pix[frame]
        crops = []
        # positive crop: the ball anywhere inside it, not always in the middle
        x0 = int(np.clip(bx - rng.uniform(40, TILE - 40), 0, WIDTH - TILE))
        y0 = int(np.clip(by - rng.uniform(40, TILE - 40), 0, HEIGHT - TILE))
        crops.append(("{}_p".format(frame), x0, y0, [(bx - x0, by - y0, side)]))
        # negative crop: same frame, ball well outside it
        for _ in range(20):
            x0 = rng.randrange(0, WIDTH - TILE + 1)
            y0 = rng.randrange(0, HEIGHT - TILE + 1)
            cx, cy = np.clip(bx, x0, x0 + TILE), np.clip(by, y0, y0 + TILE)
            if np.hypot(bx - cx, by - cy) > NEG_MIN_DIST:
                crops.append(("{}_n".format(frame), x0, y0, []))
                break
        for name, x0, y0, balls in crops:
            save_crop(root, split, name, motion, x0, y0, balls)
            if split == "val":                     # same crop in colour, to score v2 / v3 weights
                save_crop(os.path.join(args.out, "val_rgb"), split, name, colour, x0, y0, balls)
        if n % 500 == 0:
            print("  {} / {}".format(n, len(chosen)), flush=True)

    with open(os.path.join(root, "ball.yaml"), "w") as handle:
        handle.write("path: {}\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n"
                     .format(os.path.abspath(root)))
    print("dataset written to", root)


def save_crop(root, split, name, image, x0, y0, balls):
    import cv2
    crop = image[y0:y0 + TILE, x0:x0 + TILE]
    cv2.imwrite(os.path.join(root, "images", split, name + ".png"), crop)
    with open(os.path.join(root, "labels", split, name + ".txt"), "w") as handle:
        for cx, cy, side in balls:
            handle.write("0 {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                cx / TILE, cy / TILE, side / TILE, side / TILE))


def train(args):
    from ultralytics import YOLO
    require_gpu(args)
    model = YOLO(BASE_MODEL)
    # hsv_h / hsv_s off: the channels are three moments in time, not colours, and hue or
    # saturation jitter would mix them.
    model.train(data=os.path.join(args.out, "yolo", "ball.yaml"), imgsz=TILE, epochs=EPOCHS,
                batch=args.batch, seed=SEED, deterministic=True, workers=4, patience=PATIENCE,
                cache=False, device=0 if not args.allow_cpu else "cpu",
                project=os.path.join(args.out, "runs"), name="ball", exist_ok=True,
                mosaic=0.0, degrees=0.0, scale=0.0, fliplr=0.5,
                hsv_h=0.0, hsv_s=0.0, hsv_v=0.3, verbose=True)
    weights = os.path.join(args.out, "runs", "ball", "weights", "best.pt")
    with open(weights, "rb") as handle:
        print("weights", weights, "md5", hashlib.md5(handle.read()).hexdigest())


def valcheck(args):
    """Top detection vs the label on the clean held-out crops. Numbers only."""
    from ultralytics import YOLO
    root = os.path.join(args.out, "yolo" if args.kind == "motion" else "val_rgb")
    model = YOLO(args.weights)
    pos = sorted(glob.glob(os.path.join(root, "images", "val", "*_p.png")))
    neg = sorted(glob.glob(os.path.join(root, "images", "val", "*_n.png")))

    def top(files):
        out = []
        for i in range(0, len(files), 16):
            for r in model.predict(files[i:i + 16], imgsz=TILE, conf=0.001, verbose=False,
                                   half=False):
                b = r.boxes
                if len(b):
                    j = int(b.conf.argmax())
                    x, y = b.xywh[j, :2].cpu().numpy()
                    out.append((float(x), float(y), float(b.conf[j])))
                else:
                    out.append((np.nan, np.nan, 0.0))
        return np.array(out)

    tp, tn = top(pos), top(neg)
    lab = []
    for f in pos:
        v = open(f.replace("images", "labels").replace(".png", ".txt")).read().split()
        lab.append((float(v[1]) * TILE, float(v[2]) * TILE))
    lab = np.array(lab)
    err = np.hypot(tp[:, 0] - lab[:, 0], tp[:, 1] - lab[:, 1])
    err[np.isnan(err)] = np.inf
    lines = ["weights {} ({}) on {} clean ball crops / {} no-ball crops".format(
        args.weights, args.kind, len(pos), len(neg))]
    for conf in (0.001, 0.05):
        ok = tp[:, 2] >= conf
        lines.append("  conf >= {:<5}: top detection within 5 px {:.1%}  10 px {:.1%}  "
                     "20 px {:.1%}".format(conf, (ok & (err <= 5)).mean(),
                                           (ok & (err <= 10)).mean(), (ok & (err <= 20)).mean()))
    for conf in (0.1, 0.3, 0.5):
        s = tp[:, 2] >= conf
        lines.append("  conf >= {}: fires on {:.1%} of ball crops ({} within 10 px), "
                     "{:.1%} of no-ball crops".format(
                         conf, s.mean(),
                         "{:.1%}".format((err[s] <= 10).mean()) if s.any() else "n/a",
                         (tn[:, 2] >= conf).mean()))
    text = "\n".join(lines)
    print(text)
    name = "valcheck_{}_{}.txt".format(args.kind, hashlib.md5(args.weights.encode()).hexdigest()[:6])
    with open(os.path.join(args.out, name), "w") as handle:
        handle.write(text + "\n")


def merge(cands):
    """Detections from overlapping tiles -> distinct candidates, best first."""
    cands.sort(key=lambda c: -c[2])
    kept = []
    for x, y, c in cands:
        if all(np.hypot(x - kx, y - ky) > MERGE_PX for kx, ky, _ in kept):
            kept.append((x, y, c))
    return kept[:TOP_K]


def resume_point(out_path):
    """After a crash: the frame to restart from, with the file cut back to before it. None if
    there is nothing to resume. The last frame in the file may be half written, so it is redone;
    frames after it with no detection wrote no rows, so they are simply redone too."""
    if not os.path.isfile(out_path):
        return None
    with open(out_path) as handle:
        lines = handle.read().split("\n")
    rows = [l for l in lines[1:] if l.count(",") == 4 and not l.endswith(",")]
    if not rows:
        return None
    last = int(rows[-1].split(",")[0])
    keep = [l for l in rows if int(l.split(",")[0]) < last]
    with open(out_path, "w") as handle:
        handle.write(lines[0] + "\n" + "".join(l + "\n" for l in keep))
    return last


def detect(args):
    import torch
    from ultralytics import YOLO
    require_gpu(args)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    p = paths(args.data)
    weights = args.weights or os.path.join(args.out, "runs", "ball", "weights", "best.pt")
    model = YOLO(weights)
    out_path = os.path.join(args.out, "ball_video_tactical.csv")
    tiles = [(x, y) for y in TILE_Y for x in TILE_X]
    frames = graded_frames()
    start = resume_point(out_path)
    if start is not None:
        print("resuming at frame", start)
        frames = [f for f in frames if f >= start]

    with open(out_path, "a" if start is not None else "w") as out:
        if start is None:
            out.write("frame,rank,px,py,conf\n")
        batch_frames, batch_images = [], []

        def flush():
            crops = [img[y:y + TILE, x:x + TILE] for img in batch_images for x, y in tiles]
            results = model.predict(crops, imgsz=TILE, conf=CONF_MIN, verbose=False, half=False)
            for i, frame in enumerate(batch_frames):
                cands = []
                for t, (x0, y0) in enumerate(tiles):
                    boxes = results[i * len(tiles) + t].boxes
                    for (cx, cy, _, _), c in zip(boxes.xywh.cpu().numpy(),
                                                 boxes.conf.cpu().numpy()):
                        cands.append((float(cx) + x0, float(cy) + y0, float(c)))
                for rank, (x, y, c) in enumerate(merge(cands), 1):
                    out.write("{},{},{:.0f},{:.0f},{:.3f}\n".format(frame, rank, x, y, c))
            batch_frames.clear()
            batch_images.clear()

        for n, (frame, motion, _) in enumerate(windows(p["video"], frames)):
            batch_frames.append(frame)
            batch_images.append(motion)
            if len(batch_frames) == FRAMES_PER_BATCH:
                flush()
            if n % 5000 == 0:
                print("  {} / {}".format(n, len(frames)), flush=True)
        if batch_frames:
            flush()
    with open(out_path, "rb") as handle:
        print("wrote", out_path, "md5", hashlib.md5(handle.read()).hexdigest())


def report(args):
    """Accuracy on held-out time blocks, all tracked frames and clean ones. Numbers only."""
    import pandas as pd
    pix = np.load(os.path.join(args.out, "atd_ball_pixels.npy"))
    clean = np.load(os.path.join(args.out, "clean_mask.npy"))
    det = pd.read_csv(os.path.join(args.out, "ball_video_tactical.csv"))
    top = det[det["rank"] == 1].set_index("frame")
    frames = np.array(graded_frames())
    tracked = np.isfinite(pix[frames, 0])
    val_block = np.array([is_val(f) for f in frames])
    lines = []

    def share(mask):
        return "{:.1%}".format(mask.mean()) if len(mask) else "n/a"

    for label, sel in (("held-out frames with a tracked ball", tracked & val_block),
                       ("held-out CLEAN frames", tracked & val_block & clean[frames])):
        val = frames[sel]
        have = np.isin(val, top.index)
        err = np.full(len(val), np.inf)
        idx = val[have]
        err[have] = np.hypot(top.loc[idx, "px"].values - pix[idx, 0],
                             top.loc[idx, "py"].values - pix[idx, 1])
        conf = top.reindex(val)["conf"].fillna(0).values
        lines.append("{}: {}".format(label, len(val)))
        for r in (5, 10, 20):
            lines.append("  best detection within {:2d} px of the tracked ball: {}".format(
                r, share(err <= r)))
        for c in (0.1, 0.3, 0.5):
            s = conf >= c
            lines.append("  conf >= {}: {} of frames, of which {} within 10 px".format(
                c, share(s), share(err[s] <= 10)))
    missing = frames[~tracked]
    mconf = top.reindex(missing)["conf"].fillna(0).values
    lines.append("graded frames WITHOUT a tracked ball: {}".format(len(missing)))
    for c in (0.1, 0.3, 0.5):
        lines.append("  with a detection of conf >= {}: {}".format(c, share(mconf >= c)))
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.out, "report.txt"), "w") as handle:
        handle.write(text + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["check", "dataset", "train", "valcheck", "detect",
                                         "report"])
    parser.add_argument("--data", required=True, help="the bundle's data/ folder")
    parser.add_argument("--out", required=True, help="working folder for everything produced")
    parser.add_argument("--weights", help="valcheck / detect: weights to use")
    parser.add_argument("--kind", choices=["motion", "rgb"], default="motion",
                        help="valcheck: motion for v4 weights, rgb for v2 / v3 weights")
    parser.add_argument("--batch", type=int, default=BATCH, help="train: batch size")
    parser.add_argument("--allow-cpu", action="store_true", help="train/detect without a GPU")
    parser.add_argument("--count", action="store_true", help="check: decode and count every frame")
    args = parser.parse_args()
    if args.step == "valcheck" and not args.weights:
        parser.error("valcheck needs --weights")
    os.makedirs(args.out, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    {"check": check, "dataset": dataset, "train": train, "valcheck": valcheck, "detect": detect,
     "report": report}[args.step](args)
    with open(os.path.join(args.out, "steps.log"), "a") as handle:
        handle.write(json.dumps({"step": args.step, "argv": sys.argv[1:]}) + "\n")


if __name__ == "__main__":
    main()
