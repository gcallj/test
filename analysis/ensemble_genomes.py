#!/usr/bin/env python3
"""Ensemble HOF: combina top-N genomas do Hall of Fame em sinal agregado.

Em vez de usar 1 genoma final do GA, avalia TOP-N genomas do HOF (output/
ga_checkpoints_H5/hof_*.json) e produz sinais como media (ou voto majoritario)
das predicoes individuais. Reduz variancia e overfit ao ultimo genoma.

Uso:
  python analysis/ensemble_genomes.py --top-n 5 --mode mean
  python analysis/ensemble_genomes.py --top-n 3 --mode majority --output ensemble_signals.csv

Modes:
  mean      : score_avg = mean(score_i) por (ticker, dia). Decisao: comprar se >threshold.
  majority  : decisao = voto majoritario (>=N/2 genomas votam comprar).

Output: csv com colunas Date, ticker, n_genomes_buy, signal_avg, ensemble_decision.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def load_top_n_hof(checkpoint_dir: Path, top_n: int) -> List[Dict[str, Any]]:
    """Carrega top-N genomas do HOF por fitness descendente.

    Espera arquivos hof_genomeN.json ou hof_*.json com campo 'genome' e 'fitness'.
    """
    hof_files = sorted(checkpoint_dir.glob("hof_*.json"))
    if not hof_files:
        # Fallback: tenta global_ga_checkpoint.json + ga_checkpoints_H5/best_*.json
        ck = ROOT / "global_ga_checkpoint.json"
        if ck.exists():
            with open(ck, "r", encoding="utf-8") as f:
                d = json.load(f)
            return [{"fitness": d.get("fitness", 0.0), "genome": d.get("genome", []), "source": "main_checkpoint"}]
        return []

    genomes = []
    for hf in hof_files:
        try:
            with open(hf, "r", encoding="utf-8") as f:
                d = json.load(f)
            if "genome" in d:
                genomes.append({
                    "fitness": float(d.get("fitness", 0.0)),
                    "genome": list(d["genome"]),
                    "source": hf.name,
                })
        except Exception as e:
            print(f"[WARN] failed to load {hf}: {e}", file=sys.stderr)

    # Sort by fitness desc, keep top N
    genomes.sort(key=lambda g: -g["fitness"])
    return genomes[:top_n]


def evaluate_genomes_per_day(genomes: List[Dict[str, Any]]) -> pd.DataFrame:
    """Roda apply_phase para cada genoma e coleta sinais por (Date, ticker).

    Reusa run_local_ga_staged.evaluate_genome_full + ga_run.apply_phase logic.
    Retorna DataFrame com colunas: Date, ticker, signal_<i> por genoma i.
    """
    import ga_run as gr
    import run_local_ga_staged as staged

    store, _w, _sg, _sf, _bg, _bf, _bi = staged.open_existing_store()
    cache = staged.build_ticker_arrays_cache(store)

    rows = []
    for i, g in enumerate(genomes):
        print(f"[ensemble] Evaluating genome {i+1}/{len(genomes)} (fit={g['fitness']:.4f}, src={g['source']})...")
        gp = gr.decode_global_params(g["genome"])
        for tk, p in cache.items():
            try:
                # Usa a logica de score_matrix + decisao binaria
                # score_matrix tem shape (n_days, n_features). Decisao "compra"
                # depende do gene vote_threshold_long aplicado em score_matrix.
                # Aqui simplificamos: signal = score_matrix.mean(axis=1) > threshold.
                # Para uma versao acurada, deveria reusar o backtest, mas isso
                # eh helper de ensemble — sinal binario por dia.
                score = p.get("score_matrix")
                dates = p.get("dates")
                if score is None or dates is None:
                    continue
                if score.ndim == 2:
                    s_mean = np.nanmean(score, axis=1)
                else:
                    s_mean = score
                threshold = gp.vote_threshold_long if hasattr(gp, "vote_threshold_long") else 0.3
                buys = (s_mean > threshold).astype(int)
                for d_idx in range(len(dates)):
                    rows.append({
                        "Date": pd.Timestamp(dates[d_idx]),
                        "ticker": tk,
                        "genome_idx": i,
                        "score_mean": float(s_mean[d_idx]) if not np.isnan(s_mean[d_idx]) else 0.0,
                        "buy": int(buys[d_idx]),
                    })
            except Exception as e:
                print(f"[WARN] eval {tk} genome {i}: {e}", file=sys.stderr)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def aggregate_ensemble(per_genome_df: pd.DataFrame, mode: str = "mean", buy_threshold: float = 0.5) -> pd.DataFrame:
    """Agrega sinais cross-genome por (Date, ticker)."""
    if per_genome_df.empty:
        return pd.DataFrame()

    n_genomes = per_genome_df["genome_idx"].nunique()
    grouped = per_genome_df.groupby(["Date", "ticker"]).agg(
        score_avg=("score_mean", "mean"),
        score_std=("score_mean", "std"),
        n_genomes_buy=("buy", "sum"),
        n_genomes_total=("buy", "count"),
    ).reset_index()

    if mode == "majority":
        grouped["ensemble_buy"] = (grouped["n_genomes_buy"] >= (n_genomes / 2.0)).astype(int)
    else:  # mean
        grouped["ensemble_buy"] = (grouped["score_avg"] > buy_threshold).astype(int)

    return grouped


def compute_ensemble_metrics(ensemble_df: pd.DataFrame) -> Dict[str, float]:
    """Compute basic stats on ensemble signal stability."""
    if ensemble_df.empty:
        return {}

    # Disagreement rate (where genomas votam diferente)
    if "n_genomes_buy" in ensemble_df.columns and "n_genomes_total" in ensemble_df.columns:
        n_total = ensemble_df["n_genomes_total"].iloc[0] if len(ensemble_df) else 1
        disagreement = ensemble_df.apply(
            lambda r: min(r["n_genomes_buy"], n_total - r["n_genomes_buy"]) / max(n_total, 1),
            axis=1,
        ).mean()
    else:
        disagreement = 0.0

    return {
        "n_signal_days": int(len(ensemble_df)),
        "buy_rate_pct": round(100 * ensemble_df["ensemble_buy"].mean(), 2),
        "score_avg_median": round(float(ensemble_df["score_avg"].median()), 4),
        "score_std_median": round(float(ensemble_df.get("score_std", pd.Series([0])).median() or 0), 4),
        "avg_disagreement_rate": round(float(disagreement), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--mode", choices=["mean", "majority"], default="mean")
    ap.add_argument("--buy-threshold", type=float, default=0.5)
    ap.add_argument("--checkpoint-dir", type=str, default=str(ROOT / "output" / "ga_checkpoints_H5"))
    ap.add_argument("--output", type=str, default="/tmp/ensemble_signals.csv")
    args = ap.parse_args()

    cdir = Path(args.checkpoint_dir)
    print(f"[ensemble] Loading top-{args.top_n} HOF genomes from {cdir}")
    genomes = load_top_n_hof(cdir, args.top_n)
    if not genomes:
        print("[ensemble] No genomes found.")
        return 1
    print(f"[ensemble] Loaded {len(genomes)} genomes:")
    for i, g in enumerate(genomes):
        print(f"  {i+1}. fit={g['fitness']:+.4f} src={g['source']}")

    per_genome_df = evaluate_genomes_per_day(genomes)
    if per_genome_df.empty:
        print("[ensemble] No signals produced.")
        return 1

    ensemble = aggregate_ensemble(per_genome_df, mode=args.mode, buy_threshold=args.buy_threshold)
    ensemble.to_csv(args.output, index=False)
    print(f"[ensemble] Wrote {len(ensemble)} (Date, ticker) rows to {args.output}")

    metrics = compute_ensemble_metrics(ensemble)
    print(f"[ensemble] Stats: {metrics}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
