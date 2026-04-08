"""
Validate Minervini Stage filter on the SAME walk-forward windows that GA trained on.

Duas modalidades de filtro:
  - AGRESSIVO (GATE_MODE="aggressive"): bloqueia TUDO exceto Stage 2 ideal/momentum/pullback
    Resultado conhecido: NO-GO (delta_fit -4.9 em OOS, elimina muitos trades bons)

  - NEGATIVO (GATE_MODE="negative"): bloqueia APENAS Stage 4 Queda confirmada
    Mantem Stage 1 base, Stage 2 em todos sub-timings, Stage 3 topo.
    Muito menos restritivo. Hipotese: Stage 4 e bear confirmado, evitar e trivial ganho.

Controle via env GATE_MODE (default "negative" apos primeira rodada).

Saida:
  - output/stage_filter_validation_<MODE>.csv
  - output/stage_filter_by_stage_<MODE>.csv
  - console: resumo + go/no-go
"""
import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("GA_RUN_MODE", "load")
os.environ.setdefault("GA_EVAL_WORKERS", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Ensure stdout handles unicode
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Add repo root to path
REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
os.chdir(REPO_DIR)

import ga_run  # noqa: E402
from ga_run import (  # noqa: E402
    decode_global_params,
    build_temporal_windows,
    backtest_stats_global_intraday,
    classify_minervini_stage,
    global_fitness_from_stats,
    rolling_mean_np,
    ATR_EPS,
    GA_MEMMAP_DIR,
)
from payload_store import PayloadStore, build_window_plans  # noqa: E402


OUTDIR = "output"
os.makedirs(OUTDIR, exist_ok=True)

# GATE_MODE: "aggressive" (default) or "negative"
# - aggressive: allow only Stage 2 ideal/momentum/pullback
# - negative:   block only Stage 4 Queda (everything else passes)
GATE_MODE = os.environ.get("GATE_MODE", "negative").lower()
print(f"[CONFIG] GATE_MODE={GATE_MODE}", flush=True)


def stage_allows_entry(stage_num: int, timing_sub: str) -> bool:
    """Decide if an entry is allowed given stage classification.

    aggressive: only Stage 2 ideal/momentum/pullback
    negative:   everything except Stage 4 Queda (hard block)
    soft:       everything passes, but bars in Stage 3/4 + Stage 2 atrasado
                get score_matrix attenuated by 50% (soft penalty)
    """
    if GATE_MODE == "aggressive":
        return stage_num == 2 and timing_sub in ("ideal", "momentum", "pullback")
    elif GATE_MODE == "soft":
        # Soft mode: "allowed" means NO attenuation applied.
        # Attenuate bars in Stage 3, Stage 4, and Stage 2 atrasado.
        if stage_num == 4:
            return False
        if stage_num == 3:
            return False
        if stage_num == 2 and timing_sub == "atrasado":
            return False
        return True
    else:  # negative
        return stage_num != 4


# ==============================================================================
# STAGE GATE: compute mask of bars where Stage 2 (ideal/momentum/pullback) holds
# ==============================================================================
def compute_stage_gate_mask(close: np.ndarray, high: np.ndarray) -> np.ndarray:
    """
    Retorna array booleano: True nos bars onde Stage 2 ideal/momentum/pullback.
    Entradas sao bloqueadas nos outros stages.

    Nota: classify_minervini_stage olha close/ma50/ma200/slopes e classifica
    o bar atual. Aqui computamos isso vetorizado em Python (ciclo sobre bars)
    e retornamos a mask.
    """
    n = len(close)
    ma50 = rolling_mean_np(close, 50, 10)
    ma200 = rolling_mean_np(close, 200, 50)

    mask = np.zeros(n, dtype=bool)
    blocked_by_stage = Counter()
    for i in range(n):
        if not (np.isfinite(ma50[i]) and np.isfinite(ma200[i])):
            continue
        ma50_prev = ma50[i - 20] if i >= 20 else np.nan
        ma200_prev = ma200[i - 20] if i >= 20 else np.nan
        stage_num, _, timing_sub, _ = classify_minervini_stage(
            close=float(close[i]),
            ma50=float(ma50[i]),
            ma200=float(ma200[i]),
            ma50_prev=float(ma50_prev) if np.isfinite(ma50_prev) else np.nan,
            ma200_prev=float(ma200_prev) if np.isfinite(ma200_prev) else np.nan,
        )
        allowed = stage_allows_entry(stage_num, timing_sub)
        if allowed:
            mask[i] = True
        else:
            # Track the stage type that blocked this bar
            blocked_by_stage[(stage_num, timing_sub)] += 1

    return mask, blocked_by_stage, ma50, ma200


def apply_gate_to_score_matrix(score_matrix: np.ndarray, gate_mask: np.ndarray) -> np.ndarray:
    """
    Modifica score_matrix nos bars bloqueados.

    GATE_MODE = "aggressive"/"negative": ZERA score_matrix (bloqueio total)
    GATE_MODE = "soft":                   REDUZ score_matrix 50% (atenuacao)

    Trick: votes_long sao computados como media de features acima de z_threshold.
    Reduzir magnitude do score_matrix reduz quantas features passam no threshold,
    atenuando naturalmente a probabilidade de entrada sem bloquear completamente.
    """
    sm_copy = np.asarray(score_matrix, dtype=np.float64).copy()
    if GATE_MODE == "soft":
        sm_copy[~gate_mask, :] *= 0.5  # atenuacao 50%
    else:
        sm_copy[~gate_mask, :] = 0.0   # bloqueio total
    return sm_copy


def compute_forward_returns(close: np.ndarray, horizon: int = 5) -> np.ndarray:
    """Forward return com horizonte N bars. Usado para classificar entries
    bloqueados como winners/losers hipoteticos."""
    n = len(close)
    fwd = np.full(n, np.nan, dtype=np.float64)
    if n > horizon:
        fwd[:-horizon] = (close[horizon:] / np.maximum(close[:-horizon], ATR_EPS)) - 1.0
    return fwd


# ==============================================================================
# MAIN VALIDATION LOOP
# ==============================================================================
def main():
    t0_total = time.time()

    # 1. Load checkpoint
    ck = json.load(open("global_ga_checkpoint.json"))
    gp = decode_global_params(ck["genome"])
    print(f"[INIT] checkpoint fitness={ck['fitness']:.4f}", flush=True)

    # 2. Open PayloadStore (prefer main, fallback to staged)
    candidates = [
        os.path.abspath(GA_MEMMAP_DIR),
        os.path.abspath("output/ga_memmap_staged"),
    ]
    store_path = None
    for p in candidates:
        if os.path.exists(os.path.join(p, "meta.json")):
            store_path = p
            break
    if store_path is None:
        raise RuntimeError(
            "No PayloadStore found. Run 'python ga_run.py' (load mode) first "
            "to build it."
        )
    print(f"[INIT] store: {store_path}", flush=True)
    store = PayloadStore(store_path, mode="r")
    print(f"[INIT] {len(store.tickers)} tickers, dates "
          f"{store.min_date.date()} -> {store.max_date.date()}", flush=True)

    # 3. Build walk-forward windows (same as GA)
    windows = build_temporal_windows(
        store.min_date, store.max_date,
        train_years=3, test_months=6, step_months=6,
    )
    window_plans = build_window_plans(store, windows)
    print(f"[INIT] {len(window_plans)} walk-forward windows built", flush=True)

    # 4. Pre-compute stage gate masks for every ticker (once per ticker on full history)
    print(f"[SETUP] Computing stage gate masks per ticker...", flush=True)
    t0 = time.time()
    ticker_gate = {}   # ticker -> {mask, blocked_dist, fwd_ret5}
    for tk in store.tickers:
        arr = store.get_ticker_arrays(tk)
        close = np.asarray(arr["close"], dtype=np.float64)
        high = np.asarray(arr["high"], dtype=np.float64)
        mask, blocked, _, _ = compute_stage_gate_mask(close, high)
        fwd = compute_forward_returns(close, horizon=5)
        ticker_gate[tk] = {
            "mask": mask,
            "blocked": blocked,
            "fwd_ret5": fwd,
        }
    print(f"[SETUP] gate masks computed in {time.time()-t0:.0f}s", flush=True)

    # 5. Aggregate global stage distribution
    global_blocked = Counter()
    for tk, data in ticker_gate.items():
        for k, v in data["blocked"].items():
            global_blocked[k] += v
    total_bars = sum(global_blocked.values())
    total_allowed = sum(data["mask"].sum() for data in ticker_gate.values())
    print(f"[GATE] Stage distribution across all tickers:", flush=True)
    print(f"  ALLOWED (Stage 2 ideal/momentum/pullback): {total_allowed:,} bars", flush=True)
    for (stage, timing), count in sorted(global_blocked.items(), key=lambda x: -x[1]):
        print(f"  BLOCKED Stage {stage} {timing}: {count:,} bars", flush=True)

    # 6. Count blocked winners vs losers (forward return > 0 = would have been winner)
    blocked_winners = 0
    blocked_losers = 0
    blocked_rets = []
    for tk, data in ticker_gate.items():
        mask = data["mask"]
        fwd = data["fwd_ret5"]
        blocked_idx = ~mask & np.isfinite(fwd)
        blocked_rets.extend(fwd[blocked_idx].tolist())
        blocked_winners += int((fwd[blocked_idx] > 0).sum())
        blocked_losers += int((fwd[blocked_idx] <= 0).sum())
    blocked_wr = (blocked_winners / max(blocked_winners + blocked_losers, 1)) * 100
    blocked_median_ret = float(np.median(blocked_rets)) * 100 if blocked_rets else 0.0
    print(f"[GATE] Hipotetical winrate in BLOCKED bars (5-day fwd): "
          f"{blocked_wr:.1f}% | median ret: {blocked_median_ret:+.2f}%", flush=True)
    print(f"[GATE] (se esse WR for < WR geral ~72%, filtro esta removendo perdedores - BOM)", flush=True)

    # 7. Save blocked stage distribution to CSV
    by_stage_rows = []
    for (stage, timing), count in global_blocked.items():
        by_stage_rows.append({
            "stage_num": stage,
            "timing_sub": timing,
            "blocked_count": count,
            "pct_of_total_blocked": count / max(total_bars, 1) * 100,
        })
    by_stage_rows.append({
        "stage_num": 2,
        "timing_sub": "ALLOWED (ideal/momentum/pullback)",
        "blocked_count": total_allowed,
        "pct_of_total_blocked": total_allowed / max(total_allowed + total_bars, 1) * 100,
    })
    pd.DataFrame(by_stage_rows).to_csv(
        os.path.join(OUTDIR, f"stage_filter_by_stage_{GATE_MODE}.csv"), index=False
    )
    print(f"[OUT] saved output/stage_filter_by_stage_{GATE_MODE}.csv", flush=True)

    # 8. Walk through each window, for each ticker, run backtest raw vs gated
    print(f"\n[WALK] Running backtest on {len(window_plans)} windows...", flush=True)
    rows = []
    t0_walk = time.time()

    for wi, wp in enumerate(window_plans):
        t0_win = time.time()
        try:
            train_payloads, test_payloads = store.get_window_payloads(wp)
        except Exception as e:
            print(f"[WIN {wi}] failed to load payloads: {e}", flush=True)
            continue

        for mode, payloads in [("train", train_payloads), ("test", test_payloads)]:
            if not payloads:
                continue
            stats_raw = []
            stats_gated = []
            for tk, p in payloads.items():
                close = np.asarray(p["close"], dtype=np.float64)
                if len(close) < 5:
                    continue
                sm = np.asarray(p["score_matrix"], dtype=np.float64)

                # Need to build per-window mask by slicing the ticker's full mask
                # The payload slice is a subset of the full history; we need the
                # corresponding slice of ticker_gate[tk]["mask"].
                # The payload already is a slice of the store arrays, so we
                # recompute the mask fresh on THIS slice (short time).
                high = np.asarray(p["high"], dtype=np.float64)
                win_mask, _, _, _ = compute_stage_gate_mask(close, high)

                # RAW baseline
                try:
                    st_raw = backtest_stats_global_intraday(
                        p["open"], p["high"], p["low"], p["close"],
                        sm, p["atr"], gp, precomputed=p,
                    )
                except Exception as e:
                    print(f"  [W{wi}/{mode}/{tk}] raw backtest err: {e}", flush=True)
                    continue

                # GATED version: zero out score_matrix on blocked bars
                sm_gated = apply_gate_to_score_matrix(sm, win_mask)
                try:
                    st_gated = backtest_stats_global_intraday(
                        p["open"], p["high"], p["low"], p["close"],
                        sm_gated, p["atr"], gp, precomputed=p,
                    )
                except Exception as e:
                    print(f"  [W{wi}/{mode}/{tk}] gated backtest err: {e}", flush=True)
                    continue

                stats_raw.append(st_raw)
                stats_gated.append(st_gated)

            if not stats_raw or not stats_gated:
                continue

            fit_raw = global_fitness_from_stats(stats_raw)
            fit_gated = global_fitness_from_stats(stats_gated)

            wr_raw_arr = [s["win_rate"] for s in stats_raw if s.get("n_trades", 0) > 5]
            wr_gated_arr = [s["win_rate"] for s in stats_gated if s.get("n_trades", 0) > 5]

            row = {
                "window": wi,
                "mode": mode,
                "n_tickers": len(stats_raw),
                "fit_raw": round(fit_raw, 4),
                "fit_gated": round(fit_gated, 4),
                "delta_fit": round(fit_gated - fit_raw, 4),
                "wr_raw": round(float(np.median(wr_raw_arr)) * 100, 2) if wr_raw_arr else 0,
                "wr_gated": round(float(np.median(wr_gated_arr)) * 100, 2) if wr_gated_arr else 0,
                "trades_raw_med": float(np.median([s["n_trades"] for s in stats_raw])),
                "trades_gated_med": float(np.median([s["n_trades"] for s in stats_gated])),
                "ret_raw_med": round(float(np.median([s["total_return"] for s in stats_raw])) * 100, 2),
                "ret_gated_med": round(float(np.median([s["total_return"] for s in stats_gated])) * 100, 2),
                "mdd_raw_med": round(float(np.median([s["mdd"] for s in stats_raw])) * 100, 2),
                "mdd_gated_med": round(float(np.median([s["mdd"] for s in stats_gated])) * 100, 2),
                "sharpe_raw_med": round(float(np.median([s["sharpe"] for s in stats_raw])), 3),
                "sharpe_gated_med": round(float(np.median([s["sharpe"] for s in stats_gated])), 3),
            }
            rows.append(row)

        elapsed_win = time.time() - t0_win
        if wi % 5 == 0 or wi == len(window_plans) - 1:
            print(f"  [W{wi}/{len(window_plans)-1}] done in {elapsed_win:.1f}s "
                  f"(total {(time.time()-t0_walk)/60:.1f}m)", flush=True)

    # 9. Save per-window CSV
    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUTDIR, f"stage_filter_validation_{GATE_MODE}.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[OUT] saved {out_csv}", flush=True)

    # 10. Summary analysis
    print(f"\n{'=' * 60}", flush=True)
    print(f"RESULTADO AGREGADO", flush=True)
    print(f"{'=' * 60}", flush=True)

    def summarize(sub: pd.DataFrame, label: str):
        if sub.empty:
            print(f"[{label}] no data", flush=True)
            return
        dfit = float(sub["delta_fit"].median())
        dfit_mean = float(sub["delta_fit"].mean())
        dret = float((sub["ret_gated_med"] - sub["ret_raw_med"]).median())
        dwr = float((sub["wr_gated"] - sub["wr_raw"]).median())
        dmdd = float((sub["mdd_gated_med"] - sub["mdd_raw_med"]).median())
        dtrades = float((sub["trades_gated_med"] - sub["trades_raw_med"]).median())
        dsharpe = float((sub["sharpe_gated_med"] - sub["sharpe_raw_med"]).median())
        n_win = int(sub.shape[0])
        n_pos_fit = int((sub["delta_fit"] > 0).sum())

        print(f"\n[{label}] n={n_win} windows", flush=True)
        print(f"  delta_fit        mediana={dfit:+.4f}  media={dfit_mean:+.4f}  "
              f"positivo em {n_pos_fit}/{n_win} janelas", flush=True)
        print(f"  delta_WR         {dwr:+.2f}pp (filtro esperado aumentar)", flush=True)
        print(f"  delta_return     {dret:+.2f}pp", flush=True)
        print(f"  delta_MDD        {dmdd:+.2f}pp (negativo = melhor)", flush=True)
        print(f"  delta_sharpe     {dsharpe:+.3f}", flush=True)
        print(f"  delta_n_trades   {dtrades:+.1f} (esperado diminuir)", flush=True)

    summarize(df[df["mode"] == "train"], "TRAIN windows")
    summarize(df[df["mode"] == "test"], "TEST windows (OOS)")

    # 11. GO/NO-GO decision
    test_df = df[df["mode"] == "test"]
    if not test_df.empty:
        median_delta_fit = float(test_df["delta_fit"].median())
        n_pos = int((test_df["delta_fit"] > 0).sum())
        n_total = int(test_df.shape[0])
        print(f"\n{'=' * 60}", flush=True)
        print(f"GO/NO-GO", flush=True)
        print(f"{'=' * 60}", flush=True)
        print(f"median delta_fit (OOS): {median_delta_fit:+.4f}", flush=True)
        print(f"positive in {n_pos}/{n_total} OOS windows", flush=True)

        if median_delta_fit > 0 and n_pos > n_total / 2:
            print(f"DECISAO: GO - filtro ajuda em OOS. Seguir para Fase 2 "
                  f"(integrar em ga_run.py + retreinar seeded).", flush=True)
        else:
            print(f"DECISAO: NO-GO - filtro nao ajuda em OOS. NAO integrar.", flush=True)

    print(f"\n[DONE] total time: {(time.time()-t0_total)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
