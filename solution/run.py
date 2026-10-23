"""Regenerate the submission file from the provided data.

    python solution/run.py                         # -> evaluation/results/possession.csv
    python solution/run.py --out my_team.csv

Prints only aggregate counts, never frame-level labels.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from model import predict  # noqa: E402

# Params
DEFAULT_OUT = os.path.join(ROOT, "evaluation", "results", "possession.csv")


def write(labels, path):
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))


def summary(labels):
    graded = labels != "X"
    seq = labels[graded]
    breaks = np.flatnonzero(seq[1:] != seq[:-1]) + 1
    starts = np.concatenate(([0], breaks))
    spells = [s for s in starts if seq[s] in "HA"]
    return "H {}  A {}  D {}  possessions {}  mean {:.1f} s".format(
        (seq == "H").sum(), (seq == "A").sum(), (seq == "D").sum(), len(spells),
        np.isin(seq, ["H", "A"]).sum() / 25.0 / max(len(spells), 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    labels = predict()
    write(labels, args.out)
    print(summary(labels))
    print("wrote", args.out)
