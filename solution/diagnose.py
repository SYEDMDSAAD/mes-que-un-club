"""Side-by-side segments of a submission and the public ground truth, over the public window.

    python solution/diagnose.py evaluation/results/<file>.csv
"""
import sys

import numpy as np
import pandas as pd

GT = "evaluation/gt/BAR-ATH_possession_gt_public.csv"


def segs(labels, offset):
    breaks = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts, stops = np.r_[0, breaks], np.r_[breaks, len(labels)]
    return [(offset + a, offset + b, labels[a]) for a, b in zip(starts, stops)]


if __name__ == "__main__":
    gt = pd.read_csv(GT).label.values
    pred = pd.read_csv(sys.argv[1]).label.values
    idx = np.flatnonzero(gt != "X")
    lo, hi = idx[0], idx[-1] + 1
    g, p = segs(gt[lo:hi], lo), segs(pred[lo:hi], lo)
    events = sorted([(a, "GT  ", b - a, lab) for a, b, lab in g] +
                    [(a, "PRED", b - a, lab) for a, b, lab in p])
    for start, who, length, lab in events:
        print("{:>6}  {}  {}  {:>5.1f}s".format(start, who, lab, length / 25.0))
