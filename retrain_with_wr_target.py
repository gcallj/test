"""
Wrapper para retrain seeded com a nova fitness que reforca win_rate_target.

Baseline atual sob nova fitness:
  fit = -173.862 (era -175.355 sem wr_target_bonus)
  mean_wr_target = 45.18%
  med_wr_target  = 45.50%

O GA deve buscar genoma com mais alvo hits. Meta qualitativa: subir
mean_wr_target de 45% para 55%+ (delta fitness ~9 pontos).

Config: 3 chunks x 3 gens x 24 pop x 4 windows (~90-120 min local).
"""
import os
import sys

os.environ["GA_ALPHA_FOCUS"] = "off"
os.environ["GA_STAGE_GATE"] = "off"
os.environ["GA_AUTO_TUNE"] = "0"
os.environ["GA_FAST_MODE"] = "true"
os.environ["GA_EVAL_WORKERS"] = "1"

sys.argv = [
    "run_local_ga_staged.py",
    "--chunks", "3",
    "--ngen-per-chunk", "3",
    "--pop-size", "24",
    "--windows", "4",
    "--reset",
]

print("[WRAPPER] Seeded retrain com wr_target_bonus ativo (v8)", flush=True)
print(f"[WRAPPER] sys.argv: {sys.argv}", flush=True)

import run_local_ga_staged  # noqa: E402
import ga_run  # noqa: E402
print(f"[WRAPPER] ga_run.GA_ALPHA_FOCUS = {ga_run.GA_ALPHA_FOCUS}", flush=True)
print(f"[WRAPPER] ga_run.GA_STAGE_GATE  = {ga_run.GA_STAGE_GATE}", flush=True)

run_local_ga_staged.main()
