"""
Wrapper que ativa GA_STAGE_GATE=soft ANTES de importar ga_run / run_local_ga_staged
e roda retreino seeded leve.

4 chunks x 4 gens x pop 24 x 4 windows = ~90 min total local estimado.
"""
import os
import sys

# CRITICAL: set BEFORE any import of ga_run
os.environ["GA_STAGE_GATE"] = "soft"
os.environ["GA_AUTO_TUNE"] = "0"
os.environ["GA_FAST_MODE"] = "true"
os.environ["GA_EVAL_WORKERS"] = "1"

# Override sys.argv for run_local_ga_staged.main()
sys.argv = [
    "run_local_ga_staged.py",
    "--chunks", "4",
    "--ngen-per-chunk", "4",
    "--pop-size", "24",
    "--windows", "4",
    "--reset",  # fresh start with stage gate active
]

print("[WRAPPER] GA_STAGE_GATE=soft activated before import", flush=True)
print(f"[WRAPPER] sys.argv: {sys.argv}", flush=True)

import run_local_ga_staged  # noqa: E402
import ga_run  # noqa: E402
print(f"[WRAPPER] ga_run.GA_STAGE_GATE = {ga_run.GA_STAGE_GATE}", flush=True)

# Run
run_local_ga_staged.main()
