"""
Wrapper que ativa GA_ALPHA_FOCUS=on ANTES de importar ga_run / run_local_ga_staged.

Objetivo: forcar o GA a priorizar reducao de underperformance vs buy-and-hold
(alpha), aumentando o peso do excess_return na fitness (1.8 -> 2.5, clip -1.0 -> -2.5).

Config leve: 2 chunks x 3 gens x 20 pop x 4 windows (~60 min local).

Baseline atual (GA_ALPHA_FOCUS=on aplicado ao base):
  fit=-180.110 mean_excess=-17.643 med_excess=-0.611 pct_beat=35.3%

A meta e encontrar um genoma que o GA escolhe sob essa fitness modificada e
entao re-avalia-lo sob a fitness default para garantir que nao perdeu muito
em WR/MDD.
"""
import os
import sys

os.environ["GA_ALPHA_FOCUS"] = "on"
os.environ["GA_STAGE_GATE"] = "off"
os.environ["GA_AUTO_TUNE"] = "0"
os.environ["GA_FAST_MODE"] = "true"
os.environ["GA_EVAL_WORKERS"] = "1"

sys.argv = [
    "run_local_ga_staged.py",
    "--chunks", "2",
    "--ngen-per-chunk", "3",
    "--pop-size", "20",
    "--windows", "4",
    "--reset",
]

print("[WRAPPER] GA_ALPHA_FOCUS=on activated before import", flush=True)
print(f"[WRAPPER] sys.argv: {sys.argv}", flush=True)

import run_local_ga_staged  # noqa: E402
import ga_run  # noqa: E402
print(f"[WRAPPER] ga_run.GA_ALPHA_FOCUS = {ga_run.GA_ALPHA_FOCUS}", flush=True)
print(f"[WRAPPER] ga_run.GA_STAGE_GATE  = {ga_run.GA_STAGE_GATE}", flush=True)

run_local_ga_staged.main()
