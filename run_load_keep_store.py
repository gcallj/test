"""Run ga_run in load mode but DO NOT cleanup the store (so we can diagnose after)."""
import os
os.environ["GA_RUN_MODE"] = "load"
os.environ["GA_EVAL_WORKERS"] = "1"

import ga_run
from payload_store import PayloadStore

# Monkey patch cleanup to no-op
_orig_cleanup = PayloadStore.cleanup
def _noop(self):
    print("[PATCHED] PayloadStore.cleanup() skipped")
PayloadStore.cleanup = _noop

ga_run.RUN_MODE = "load"
ga_run.GA_EVAL_WORKERS = 1
ga_run.run()
