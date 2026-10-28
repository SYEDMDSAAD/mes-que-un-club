"""T-DEED on the broadcast video, reproducibly: per-frame class scores, then soft-NMS events.

T-DEED (github.com/arturxe2/T-DEED) is GPL-3.0, so it stays a separate checkout; this script only
imports its model and its video reader. Run it on the machine that has the video -- nothing here
uploads anything.

    git clone https://github.com/arturxe2/T-DEED ../T-DEED
    git -C ../T-DEED checkout d953b4de8a2ad941c565e1683c5b07255cbe644e
    # checkpoint from the README's Google Drive folder, kept at
    #   ../T-DEED/checkpoints/SoccerNetBall/SoccerNetBall_challenge2/checkpoint_best.pt
    pip install -r solution/video/requirements-tdeed.txt
    python solution/video/tdeed_infer.py --tdeed ../T-DEED --video data/broadcast/BAR-ATH_broadcast.mp4 \\
        --out cache/tdeed [--model SoccerNetBall_challenge2] [--device cuda|cpu]
    python solution/video_events.py build cache/tdeed/predictions.json

Writes into --out:
    scores.npz        per sampled frame (every STRIDE-th video frame) the class probabilities,
                      background first, averaged over the overlapping clips, rounded to 4 decimals
    predictions.json  soft-NMS events {"predictions": [{frame, label, confidence}]}, frame being
                      the broadcast video frame (decode order)
    run.json          decoded frame count, checkpoint and output SHA-256, versions, settings

What makes it repeatable, against T-DEED's own inference.py:
  * one decode pass counts the frames first; it must equal EXPECTED_FRAMES (the broadcast
    tracking's 164,488) -- cv2's CAP_PROP_FRAME_COUNT is read from the container and can be off
  * num_workers 0: T-DEED's video dataset is an IterableDataset that does not shard, so each
    DataLoader worker would decode and score the whole video again
  * fp32 throughout (its predict() autocasts to fp16 on GPU by default, and the displacement head
    is rounded to whole frames, so fp16 can move an event); cudnn deterministic, fixed batch size
  * scores rounded to 4 decimals before NMS, so the last bits of another GPU cannot reorder
    events; events sorted by frame, then label
  * T-DEED's model creation downloads ImageNet weights it then overwrites: switched off, every
    weight comes from the checkpoint (strict load)
"""

import argparse
import hashlib
import json
import os
import platform
import sys
from types import SimpleNamespace

import numpy as np

# Params
EXPECTED_FRAMES = 164488   # broadcast tracking records = broadcast video frames
STRIDE = 2                 # T-DEED's SoccerNet-ball frame stride (12.5 fps)
SIZE = (796, 448)          # the resolution the SoccerNet-ball checkpoints were trained at
THRESHOLD = 0.05           # keep low: thresholds belong to video_events / the decoder, not here
NMS_WINDOW = 12            # sampled frames (T-DEED's WINDOWS_SNB[1]: +-24 video frames)
DECIMALS = 4


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def count_frames(video):
    """Frames cv2 decodes from the file, one by one (the same reader T-DEED's dataset uses)."""
    import cv2
    stream = cv2.VideoCapture(video)
    n = 0
    while stream.grab():
        n += 1
    stream.release()
    return n


def accumulate(batches, n_sampled):
    """Average overlapping clip predictions: batches yields (scores (B, L, C), starts (B,)).

    A clip starting at a negative index (T-DEED pads the start with black frames) is cut; a row
    that is all zeros (the displacement head moved its score elsewhere) does not count towards
    the average, as in T-DEED.
    """
    total, support = None, np.zeros(n_sampled, np.int64)
    for pred, starts in batches:
        pred = np.asarray(pred, dtype=np.float64)
        if total is None:
            total = np.zeros((n_sampled, pred.shape[2]), np.float64)
        for p, s in zip(pred, np.asarray(starts).tolist()):
            if s < 0:
                p, s = p[-s:], 0
            e = min(s + len(p), n_sampled)
            if e <= s:
                continue
            p = p[:e - s]
            total[s:e] += p
            support[s:e] += p.sum(axis=1) != 0
    support[support == 0] = 1
    return total / support[:, None]


def soft_nms(scores, classes, threshold=THRESHOLD, window=NMS_WINDOW):
    """[(sampled frame, label, score)]: per class, take the best remaining frame, then scale every
    score within +-window by (distance / window) ** 2, until the best is below threshold."""
    events = []
    offsets = np.arange(-window, window + 1)
    decay = (offsets / float(window)) ** 2
    for j, label in enumerate(classes, start=1):          # column 0 is background
        s = np.where(scores[:, j] >= threshold, scores[:, j], 0.0)
        while True:
            i = int(np.argmax(s))
            if s[i] < threshold:
                break
            events.append((i, label, float(s[i])))
            lo, hi = max(0, i - window), min(len(s), i + window + 1)
            s[lo:hi] *= decay[lo - i + window:hi - i + window]
    events.sort(key=lambda e: (e[0], e[1]))
    return events


def tdeed_batches(args, n_sampled):
    """(scores, starts) per batch from the T-DEED model, deterministic settings."""
    sys.path.insert(0, args.tdeed)
    here = os.getcwd()
    os.chdir(args.tdeed)                       # T-DEED reads config/ and data/ relatively
    try:
        import timm
        import torch
        from torch.utils.data import DataLoader

        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True, warn_only=True)
        if args.device == "cuda":
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            torch.set_num_threads(args.threads)
            torch.nn.Module.cuda = lambda self, *a, **k: self   # update_pred_head calls .cuda()
        create = timm.create_model
        timm.create_model = lambda *a, **k: create(*a, **dict(k, pretrained=False))

        from util.io import load_json
        from util.dataset import load_classes
        from model.model import TDEEDModel
        from dataset.frame import ActionSpotInferenceDataset

        family = args.model.split("_")[0]
        cfg = load_json(os.path.join("config", family, args.model + ".json"))
        targs = SimpleNamespace(**cfg)
        targs.crop_dim = cfg["crop_dim"] if cfg["crop_dim"] > 0 else None
        targs.pretrain = cfg.get("pretrain")
        classes = load_classes(os.path.join("data", cfg["dataset"], "class.txt"))
        model = TDEEDModel(device=args.device, args=targs)
        if targs.pretrain is not None:
            pre = load_classes(os.path.join("data", targs.pretrain["dataset"], "class.txt"))
            n = [len(classes) + 1, len(pre) + 1]
            model._model.update_pred_head(n)
            model._model.to(args.device)
            model._num_classes = int(np.sum(n))
        ckpt = os.path.join("checkpoints", family, args.model, "checkpoint_best.pt")
        model.load(torch.load(ckpt, map_location=args.device))
        names = [k for k, _ in sorted(classes.items(), key=lambda kv: kv[1])]

        dataset = ActionSpotInferenceDataset(
            os.path.join(here, args.video) if not os.path.isabs(args.video) else args.video,
            clip_len=targs.clip_len, overlap_len=targs.clip_len // 4 * 3, stride=STRIDE,
            dataset=targs.dataset, size=SIZE)
        loader = DataLoader(dataset, batch_size=targs.batch_size, shuffle=False, num_workers=0)

        def gen():
            for frames, starts in loader:
                _, pred = model.predict(frames, use_amp=False)
                yield pred, starts.numpy()
        return gen(), names, sha256(ckpt), {"torch": torch.__version__, "timm": timm.__version__,
                                            "cuda": torch.version.cuda, "batch": targs.batch_size}
    finally:
        os.chdir(here)


def write_outputs(out, scores, events, meta):
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, "scores.npz"), scores=scores.astype(np.float32))
    preds = {"predictions": [{"frame": int(i) * STRIDE, "label": lab, "confidence": round(c, DECIMALS)}
                             for i, lab, c in events]}
    path = os.path.join(out, "predictions.json")
    with open(path, "w") as handle:
        json.dump(preds, handle, indent=1, sort_keys=True)
    meta = dict(meta, predictions_sha256=sha256(path), n_events=len(events))
    with open(os.path.join(out, "run.json"), "w") as handle:
        json.dump(meta, handle, indent=1, sort_keys=True)
    return meta


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tdeed", required=True, help="T-DEED checkout")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="SoccerNetBall_challenge2")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--threads", type=int, default=4, help="CPU threads (cpu only)")
    parser.add_argument("--expected-frames", type=int, default=EXPECTED_FRAMES)
    args = parser.parse_args(argv)

    import cv2
    n_frames = count_frames(args.video)
    if args.expected_frames and n_frames != args.expected_frames:
        raise SystemExit("decoded {} frames, expected {}: this decode is not the broadcast "
                         "tracking's timeline".format(n_frames, args.expected_frames))
    n_sampled = (n_frames + STRIDE - 1) // STRIDE
    batches, names, ckpt_sha, versions = tdeed_batches(args, n_sampled)
    scores = np.round(accumulate(batches, n_sampled), DECIMALS)
    events = soft_nms(scores, names)
    meta = write_outputs(args.out, scores, events, {
        "frames_decoded": n_frames, "stride": STRIDE, "size": list(SIZE), "model": args.model,
        "checkpoint_sha256": ckpt_sha, "device": args.device, "threshold": THRESHOLD,
        "nms_window": NMS_WINDOW, "opencv": cv2.__version__, "python": platform.python_version(),
        "numpy": np.__version__, **versions})
    print("{} events from {} frames -> {}/predictions.json (sha256 {})".format(
        meta["n_events"], n_frames, args.out, meta["predictions_sha256"]))


if __name__ == "__main__":
    main()
