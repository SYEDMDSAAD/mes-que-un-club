"""The labelled matches every v16 bench decodes: DFL + Metrica, first stage, second stage and change
detector, all leave-one-match-out. Aggregates only, no output of its own.

    labelled()   {(source, match): (ball, players, team, truth, blocks, feat)}: the first stage
                 from the cached bench first stages (convention_bench / metrica_bench), then
                 seq_team, seq_dead and change_logodds in each feat: the second stage and the
                 change detector fitted on every other labelled match -- as touch_bench.fill
                 does, cached in cache/bench_video_fills[.variant].pkl (the file v16 used)

Variant "testbar" (dfl_eval.TEST_BAR): the DFL matches are scored on a BAR-like ball while every
model keeps learning from the usual damage -- the first-stage classifiers inside dfl_eval.prepare,
and here the second stage and the change detector, which are fitted on the rows of the usual
first stage (the cache of the same variant without the test parts -- "testbar", "testblurS",
"testslowS" -- so "touch+touchblur6+testbar+testblur6" reads "touch+touchblur6"'s; keep the other
parts in the same order) and applied to the BAR-like rows. That is the submission's own
situation: trained on the drift damage, run on our ball.

Variants "seqbar" / "seqmix" (dfl_eval.SEQ_BAR / SEQ_MIX): the second stage and the change
detector learn from the BAR-like DFL rows (or from both copies), as the submission's would. They
change nothing in the first stage, so its caches are shared with the same variant without them
("touch+touchblur6+seqbar+testbar+testblur6" reads "touch+touchblur6+testbar+testblur6"'s), and a
match's other copy is left out with it.
"""

import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import change_ml  # noqa: E402
import dfl_eval  # noqa: E402
import seq  # noqa: E402
import variant  # noqa: E402

KEYS = ("seq_team", "seq_dead", "change_logodds")
# the variant parts that only shape the test damage: left out of the tag of the training caches
TEST_PARTS = tuple(p for p in variant.PARTS
                   if p == "testbar" or p.startswith("testblur") or p.startswith("testslow"))
# the parts that only change what the second stage learns from: left out of every first-stage cache
SECOND_STAGE_PARTS = ("seqbar", "seqmix")


def first_stage_path(name, test_bar):
    """cache/<name>.<tag>.pkl for a first stage on the usual damage (test_bar False) or on the
    BAR-like ball: the variant's other parts in their order, without the second-stage parts, then
    with test_bar "testbar" and the testblurS / testslowS parts -- so "x+seqbar+testblur6" (BAR-like
    training rows, usual test) and "x+testbar+testblur6" share one BAR-like first stage."""
    parts = [p for p in variant.PARTS if p not in SECOND_STAGE_PARTS and p not in TEST_PARTS]
    if test_bar:
        parts += ["testbar"] + [p for p in variant.PARTS if p in TEST_PARTS and p != "testbar"]
    return os.path.join(ROOT, "cache", "{}{}.pkl".format(name, "." + "+".join(parts) if parts
                                                          else ""))


def first_stages(test_bar=None, metrica=True):
    """{(source, match): (...)}: the cached first stages; test_bar as dfl_eval.prepare."""
    import convention_bench
    import metrica_bench
    test_bar = dfl_eval.TEST_BAR if test_bar is None else test_bar
    data = {("dfl", m): v for m, v in convention_bench.dfl_first_stage(
        first_stage_path("bench_dfl_prepared", test_bar), test_bar=test_bar).items()}
    if metrica:                                     # Metrica is never test-damaged
        data.update({("metrica", m): v for m, v in metrica_bench.metrica_first_stage(
            first_stage_path("bench_metrica_prepared", False)).items()})
    return data


def training_rows(test):
    """The rows the second stage and the change detector learn from, as in the submission
    (dfl_eval.seq_training_set): DFL on the usual damage, on the BAR-like ball ("seqbar") or both
    ("seqmix", the BAR-like copy keyed ("dflbar", match)), plus Metrica's matches. The copy being
    scored is reused, never rebuilt."""
    rows = {}
    for bar_like in ((False, True) if dfl_eval.SEQ_MIX else (dfl_eval.SEQ_BAR,)):
        stage = test if bar_like == dfl_eval.TEST_BAR else first_stages(bar_like, metrica=False)
        name = "dflbar" if bar_like and dfl_eval.SEQ_MIX else "dfl"
        rows.update({(name, m): v for (source, m), v in stage.items() if source == "dfl"})
    rows.update({k: v for k, v in test.items() if k[0] != "dfl"})
    return rows


def fill(test, train):
    """seq_team / seq_dead / change_logodds into each test feat: models fitted on the train rows of
    every other match (every copy of this one left out), applied to this key's test rows."""
    xs = {}

    def x_of(value):
        if id(value[5]) not in xs:
            xs[id(value[5])] = seq.features(value[5])
        return xs[id(value[5])]

    for key, value in test.items():
        others = [k for k in train if k[1] != key[1]]
        train_x, train_y = [x_of(train[k]) for k in others], [train[k][3] for k in others]
        feat = value[5]
        feat["seq_team"], feat["seq_dead"] = seq.log_odds(seq.fit(train_x, train_y), x_of(value))
        feat["change_logodds"] = change_ml.log_odds(change_ml.fit(train_x, train_y), x_of(value))
        print("  {} {} fitted ({} training rows)".format(key[0], key[1], len(others)), flush=True)


def labelled():
    """The labelled matches with every learned input filled leave-one-match-out (see the module
    docstring), the second stage and change detector cached per variant."""
    test = first_stages()
    path = variant.cache_path(os.path.join(ROOT, "cache", "bench_video_fills.pkl"))
    if os.path.isfile(path):
        with open(path, "rb") as handle:
            saved = pickle.load(handle)
        for k, v in test.items():
            v[5].update(saved[k])
        return test
    fill(test, training_rows(test))
    with open(path, "wb") as handle:
        pickle.dump({k: {key: v[5][key] for key in KEYS} for k, v in test.items()}, handle)
    return test
