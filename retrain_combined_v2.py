"""
Seeded retrain combining ALPHA_FOCUS + wr_target_bonus + bigger pop.

Objetivo: encontrar genoma que combine alto WR_target COM alpha positivo.
Atual base: WR_target 61.1%, alpha -1.43 (underperf B&H severo).
"""
import os
import sys

os.environ["GA_ALPHA_FOCUS"] = "on"  # amplifica excess_return weight
os.environ["GA_STAGE_GATE"] = "off"
os.environ["GA_AUTO_TUNE"] = "0"
os.environ["GA_FAST_MODE"] = "true"
os.environ["GA_EVAL_WORKERS"] = "1"

sys.argv = [
    "run_local_ga_staged.py",
    "--chunks", "4",
    "--ngen-per-chunk", "4",
    "--pop-size", "28",
    "--windows", "6",
    "--reset",
]

print("[WRAPPER v2] ALPHA_FOCUS + wr_target_bonus combinados", flush=True)
print(f"[WRAPPER] sys.argv: {sys.argv}", flush=True)

import run_local_ga_staged  # noqa: E402
import ga_run  # noqa: E402
print(f"[WRAPPER] ga_run.GA_ALPHA_FOCUS = {ga_run.GA_ALPHA_FOCUS}", flush=True)
print(f"[WRAPPER] ga_run.GA_STAGE_GATE  = {ga_run.GA_STAGE_GATE}", flush=True)

run_local_ga_staged.main()
