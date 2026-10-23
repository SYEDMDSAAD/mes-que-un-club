"""Where spellF1 is lost: missed, spurious, or matched-but-misaligned possessions.

spellF1 averages possession F1 over IoU thresholds 0.3 ... 0.7. A possession found with sloppy
edges counts at 0.3 and not at 0.7, so the per-threshold F1 and the edge errors of matched
possessions separate "we did not find it" from "we found it but cut it wrong" -- two different
fixes. Edge errors are split by what is on the other side of the edge in the truth: a stoppage
(D) or the other team.

    python solution/spelldiag.py            # 7 DFL matches, leave-one-match-out, + BAR-ATH public
"""

import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402

# Params
MATCH_IOU = 0.3
FPS = 25


def analyse(truth, labels):
    gt, pr = pm.label_codes(truth), pm.label_codes(labels)
    runs = pm.valid_runs(gt)
    pr = pm.fill_abstentions(pr, runs)
    per_iou = pm.spell_scores(gt, pr, runs)["per_iou"]
    g_sp, p_sp = pm.possession_spells(gt, runs), pm.possession_spells(pr, runs)
    pairs = [o for o in pm.spell_overlaps(g_sp, p_sp) if o[0] >= MATCH_IOU]
    used_g, used_p, matched = set(), set(), []
    for iou, i, j in pairs:                       # greedy one-to-one, best IoU first
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        matched.append((iou, i, j))

    def neighbour(pos, before):
        k = pos - 1 if before else pos
        return "D" if 0 <= k < len(gt) and gt[k] == pm.D else "team"

    edges = {"D": [], "team": []}
    for iou, i, j in matched:
        (gs, ge, _), (ps, pe, _) = g_sp[i], p_sp[j]
        edges[neighbour(gs, True)].append(ps - gs)
        edges[neighbour(ge, False)].append(pe - ge)
    missed = [(g_sp[i][1] - g_sp[i][0]) / FPS for i in range(len(g_sp)) if i not in used_g]
    spurious = [(p_sp[j][1] - p_sp[j][0]) / FPS for j in range(len(p_sp)) if j not in used_p]
    ious = np.array([m[0] for m in matched])
    return {"per_iou": {k: v["f1"] for k, v in per_iou.items()}, "n_gt": len(g_sp),
            "n_pred": len(p_sp), "matched": len(matched), "ious": ious,
            "edges": edges, "missed": np.array(missed), "spurious": np.array(spurious)}


def report(results, name):
    per_iou = {k: np.mean([r["per_iou"][k] for r in results]) for k in results[0]["per_iou"]}
    n_gt = sum(r["n_gt"] for r in results)
    n_pred = sum(r["n_pred"] for r in results)
    matched = sum(r["matched"] for r in results)
    ious = np.concatenate([r["ious"] for r in results])
    missed = np.concatenate([r["missed"] for r in results])
    spurious = np.concatenate([r["spurious"] for r in results])
    print("\n== {}".format(name))
    print("F1 by IoU threshold: " + "  ".join("{}: {:.3f}".format(k, v) for k, v in per_iou.items()))
    print("true possessions {}, predicted {}, matched at IoU>={} : {} ({:.0%} of true)".format(
        n_gt, n_pred, MATCH_IOU, matched, matched / max(n_gt, 1)))
    print("matched pairs by IoU: <0.5 {:.0%}   0.5-0.7 {:.0%}   >=0.7 {:.0%}".format(
        (ious < 0.5).mean(), ((ious >= 0.5) & (ious < 0.7)).mean(), (ious >= 0.7).mean()))
    for kind in ("D", "team"):
        e = np.concatenate([np.array(r["edges"][kind]) for r in results])
        if len(e):
            print("edge next to {:<4}: n {:4d}  median {:+5.0f} frames  |err| median {:4.0f}  "
                  "|err|>25 {:.0%}  |err|>50 {:.0%}".format(kind, len(e), np.median(e),
                                                           np.median(np.abs(e)),
                                                           (np.abs(e) > 25).mean(),
                                                           (np.abs(e) > 50).mean()))
    for label, arr in (("missed true", missed), ("spurious predicted", spurious)):
        if len(arr):
            print("{:<19}: {:4d}  length <2s {:.0%}  2-5s {:.0%}  5-15s {:.0%}  >15s {:.0%}".format(
                label, len(arr), (arr < 2).mean(), ((arr >= 2) & (arr < 5)).mean(),
                ((arr >= 5) & (arr < 15)).mean(), (arr >= 15).mean()))


if __name__ == "__main__":
    import dfl_eval
    from model import predict
    data = dfl_eval.prepare(dead_ml=True, team_ml=True)
    results = [analyse(t, dfl_eval.run_pipeline(b, p, tm, bl, None, feat))
               for b, p, tm, t, bl, feat in data.values()]
    report(results, "7 DFL matches, damaged, v9")
    gt = pm.read_label_csv(os.path.join(ROOT, "evaluation", "gt",
                                        "BAR-ATH_possession_gt_public.csv"))["label"].values
    report([analyse(gt, predict())], "BAR-ATH public window, v9")
