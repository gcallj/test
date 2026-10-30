#!/usr/bin/env python3
"""
Walk-forward do mercado sueco (docs/adr/0002, docs/sweden/plano.md).

Seis folds de 6 meses cobrem os ultimos 3 anos. Cada fold e operado por um
genoma treinado so com os 4 anos anteriores ao seu inicio.

    python walkforward.py plan                  # mostra o calendario
    python walkforward.py train --fold 3        # treina um fold (ga_run em modo fold)
    python walkforward.py train --all           # treina os 6 em sequencia
    python walkforward.py status                # quais folds ja tem genoma

Os genomas ficam em data/walkforward/fold_<k>.json e sao a entrada dos replays.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
FOLD_DIR = REPO_ROOT / "data" / "walkforward"

# Data ancora do calendario: o ultimo fold termina aqui. Fixa para que o
# calendario seja reproduzivel; mude-a de proposito quando quiser refazer o
# walk-forward com mais dados.
ANCHOR_END = pd.Timestamp("2026-10-30")
N_FOLDS = 6
FOLD_MONTHS = 6
TRAIN_YEARS = 4


@dataclass(frozen=True)
class Fold:
    k: int
    train_end: str      # ultimo dia que o GA pode ver
    trade_start: str    # primeiro dia operado com este genoma
    trade_end: str      # ultimo dia operado com este genoma

    @property
    def path(self) -> Path:
        return FOLD_DIR / f"fold_{self.k}.json"


def folds(anchor_end: pd.Timestamp = ANCHOR_END) -> List[Fold]:
    # Fronteiras contiguas por construcao: o fim de um fold e a vespera do
    # inicio do seguinte, qualquer que seja o tamanho dos meses.
    stop = anchor_end.normalize() + pd.Timedelta(days=1)
    bounds = [stop - pd.DateOffset(months=FOLD_MONTHS * (N_FOLDS - j)) for j in range(N_FOLDS + 1)]
    out = []
    for k in range(1, N_FOLDS + 1):
        start, nxt = bounds[k - 1], bounds[k]
        out.append(Fold(
            k=k,
            train_end=str((start - pd.Timedelta(days=1)).date()),
            trade_start=str(start.date()),
            trade_end=str((nxt - pd.Timedelta(days=1)).date()),
        ))
    return out


def get_fold(k: int) -> Fold:
    for f in folds():
        if f.k == k:
            return f
    raise SystemExit(f"fold {k} nao existe (1..{N_FOLDS})")


def load_fold_genome(k: int) -> Optional[dict]:
    p = get_fold(k).path
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def fold_env(f: Fold, base: Optional[dict] = None) -> dict:
    env = dict(os.environ if base is None else base)
    env.update({
        "GA_RUN_MODE": "train",
        "GA_TRAIN_END": f.train_end,
        "GA_TRAIN_YEARS": str(TRAIN_YEARS),
        "GA_NO_SEED": "1",
        "GA_HOLDOUT_DAYS": "0",          # o proprio fold e o periodo fora da amostra
        "GA_FOLD_OUT": str(f.path),
        "GA_DISABLE_TELEGRAM_SEND": "1",
        "GA_KEEP_STORE": "0",
        # Store proprio do fold (dados truncados): nunca confundir com o store
        # completo que os replays usam em output/ga_memmap.
        "GA_MEMMAP_DIR": str(REPO_ROOT / "output" / f"ga_memmap_fold{f.k}"),
    })
    env.setdefault("GA_TIME_BUDGET_HOURS", "3.5")
    return env


def train(f: Fold) -> int:
    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[walkforward] fold {f.k}: treino ate {f.train_end} ({TRAIN_YEARS} anos), "
          f"opera {f.trade_start} .. {f.trade_end}", flush=True)
    env = fold_env(f)
    proc = subprocess.run([sys.executable, "-u", "ga_run.py"], cwd=REPO_ROOT, env=env)
    shutil.rmtree(env["GA_MEMMAP_DIR"], ignore_errors=True)
    if proc.returncode == 0 and f.path.exists():
        g = json.loads(f.path.read_text(encoding="utf-8"))
        g.update(asdict(f))
        f.path.write_text(json.dumps(g, indent=2), encoding="utf-8")
        print(f"[walkforward] fold {f.k}: ok fit={g['fitness']:.4f}", flush=True)
    return proc.returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    sub.add_parser("status")
    tr = sub.add_parser("train")
    g = tr.add_mutually_exclusive_group(required=True)
    g.add_argument("--fold", type=int)
    g.add_argument("--all", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "plan":
        for f in folds():
            print(f"fold {f.k}: treino ate {f.train_end} | opera {f.trade_start} .. {f.trade_end}")
        return 0
    if args.cmd == "status":
        for f in folds():
            g = load_fold_genome(f.k)
            print(f"fold {f.k}: " + (f"fit={g['fitness']:.4f}" if g else "sem genoma"))
        return 0
    targets = folds() if args.all else [get_fold(args.fold)]
    rc = 0
    for f in targets:
        rc = rc or train(f)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
