"""Which experimental variant a run belongs to, so its cached models never mix with another's.

    MQUC_VARIANT=state python solution/mbr_bench.py

Unset (the default) is the submitted pipeline, byte for byte. Any other value tags every cached
learned model and bench file with it (cache/dead_ml_model.state.pkl, ...), so a variant trains
its own models and the submission's stay untouched. Variants read TAG to switch themselves on:

    state   dfl_eval.degrade damages DFL the way BAR-ATH's tracking actually loses the ball --
            mostly in stoppages -- with slow ball noise instead of independent noise per frame
"""

import os

TAG = os.environ.get("MQUC_VARIANT", "").strip()


def cache_path(path):
    """path, with the variant's tag before the extension when a variant is active."""
    if not TAG:
        return path
    root, ext = os.path.splitext(path)
    return "{}.{}{}".format(root, TAG, ext)
