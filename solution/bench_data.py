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


def first_stages(test_bar=None):
    """{(source, match): (...)}: the cached first stages; test_bar as dfl_eval.prepare."""
    import convention_bench
    import metrica_bench
    test_bar = dfl_eval.TEST_BAR if test_bar is None else test_bar
    dfl_path = (convention_bench.CACHE if test_bar == dfl_eval.TEST_BAR else
                variant.cache_path(os.path.join(ROOT, "cache", "bench_dfl_prepared.pkl"),
                                   without=TEST_PARTS))
    met_path = variant.cache_path(os.path.join(ROOT, "cache", "bench_metrica_prepared.pkl"),
                                  without=TEST_PARTS)       # Metrica is never test-damaged
    data = {("dfl", m): v for m, v in convention_bench.dfl_first_stage(
        dfl_path, test_bar=test_bar).items()}
    data.update({("metrica", m): v for m, v in metrica_bench.metrica_first_stage(met_path).items()})
    return data


def fill(test, train):
    """seq_team / seq_dead / change_logodds into each test feat: models fitted on the train rows of
    every other key, applied to this key's test rows. train is test unless "testbar"."""
    xs_train = {k: seq.features(v[5]) for k, v in train.items()}
    xs_test = xs_train if train is test else {k: seq.features(v[5]) for k, v in test.items()}
    for key, value in test.items():
        others = [k for k in train if k != key]
        train_x, train_y = [xs_train[k] for k in others], [train[k][3] for k in others]
        feat = value[5]
        feat["seq_team"], feat["seq_dead"] = seq.log_odds(seq.fit(train_x, train_y), xs_test[key])
        feat["change_logodds"] = change_ml.log_odds(change_ml.fit(train_x, train_y), xs_test[key])
        print("  {} {} fitted".format(*key), flush=True)


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
    train = first_stages(test_bar=False) if dfl_eval.TEST_BAR else test
    fill(test, train)
    with open(path, "wb") as handle:
        pickle.dump({k: {key: v[5][key] for key in KEYS} for k, v in test.items()}, handle)
    return test
