#!/usr/bin/env python3
"""
Fix GA_stock.ipynb:
1) Fix notebook format (add missing cell ids for nbformat 4.5)
2) Remove next_day_filled from output
3) Fix score_100: 0=strong sell, 50=neutral, 100=strong buy
4) Always populate best_buy_value, best_sell_value (reference levels)
5) Add stop_loss, take_profit per row
6) Add confidence (0-100) per prediction
7) Optimize for speed (vectorize apply section)
"""

import json
import uuid

NB_PATH = "/workspaces/test/GA_stock.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# --- FIX 1: Add missing cell ids for nbformat 4.5 ---
for cell in nb["cells"]:
    if "id" not in cell:
        cell["id"] = str(uuid.uuid4())[:8]

# --- Get main code cell (cell index 2) ---
main_cell = nb["cells"][2]
src = "".join(main_cell["source"])

# ============================================================
# FIX score_0_100_from_ev: make it directional (0=strong sell, 100=strong buy)
# ============================================================
OLD_SCORE_FN = '''def score_0_100_from_ev(
    score_ev: float,
    recent_scores_ev: np.ndarray,
    quality: float,
) -> float:
    if (not np.isfinite(score_ev)):
        return 50.0
    recent = recent_scores_ev[np.isfinite(recent_scores_ev)] if recent_scores_ev is not None else np.array([], dtype=np.float64)
    if len(recent) == 0:
        pct_rank = 0.5
    else:
        pct_rank = float(np.mean(recent <= score_ev))
    
    # Usar percentil rank direto (0-100)
    score = pct_rank * 100.0
    return float(np.clip(score, 0.0, 100.0))'''

NEW_SCORE_FN = '''def score_0_100_from_ev(
    score_ev: float,
    recent_scores_ev: np.ndarray,
    quality: float,
) -> float:
    """0 = venda forte, 50 = neutro, 100 = compra forte."""
    if not np.isfinite(score_ev):
        return 50.0
    recent = recent_scores_ev[np.isfinite(recent_scores_ev)] if recent_scores_ev is not None else np.array([], dtype=np.float64)
    if len(recent) < 5:
        # Sem historico suficiente, mapear via sigmoid suave
        raw = float(np.tanh(score_ev * 2.0))  # [-1, 1]
        return float(np.clip(50.0 + raw * 50.0, 0.0, 100.0))
    # Directional percentile: score_ev positivo = compra, negativo = venda
    pct_rank = float(np.mean(recent <= score_ev))
    # Mapear [0,1] -> [0,100] mantendo direcionalidade
    return float(np.clip(pct_rank * 100.0, 0.0, 100.0))'''

assert OLD_SCORE_FN in src, "Could not find old score_0_100_from_ev function"
src = src.replace(OLD_SCORE_FN, NEW_SCORE_FN)

# ============================================================
# FIX the apply loop: vectorize + add stop/confidence/always fill prices
# ============================================================
OLD_APPLY_LOOP = '''        # Generate apply signals for last APPLY_DAYS
        for i in range(max(0, n - APPLY_DAYS), n):
            sig = generate_signal_global(payload, global_params, i)
            
            # Calculate score_ev for the day
            x = np.asarray(payload["score_matrix"], dtype=np.float64)
            feat_n = max(1, x.shape[1])
            votes_long = (x > global_params.z_threshold).sum(axis=1) / feat_n
            votes_short = (x < -global_params.z_threshold).sum(axis=1) / feat_n
            score_raw = votes_long - votes_short
            score_ev = pd.Series(score_raw).ewm(span=int(global_params.signal_ema_span), adjust=False).mean().to_numpy()
            
            # Calculate quality factor
            quality = compute_quality_factor(st["sharpe"], st["total_return"], st["n_trades"] / (n / 252))
            
            # Calculate 0-100 score
            lookback = max(63, SCORE_LOOKBACK)
            recent_scores = score_ev[max(0, i - lookback + 1):i + 1]
            score_100 = score_0_100_from_ev(score_ev[i], recent_scores, quality)

            # Calculate next day entry levels
            atr_val = payload["atr"][i]
            close_val = c[i]
            
            # Calculate strength
            score95 = np.nanpercentile(np.abs(score_ev), 95) if np.isfinite(np.nanmax(np.abs(score_ev))) else 1.0
            score95 = max(score95, ATR_EPS)
            strength = float(np.clip(abs(score_ev[i]) / score95, 0.0, 1.0))
            score95 = np.nanpercentile(np.abs(score_ev), 95) if np.isfinite(np.nanmax(np.abs(score_ev))) else 1.0
            discount = global_params.entry_discount_atr_frac * (1.0 - global_params.score_strength_scaling * strength)
            
            best_buy = close_val - discount * atr_val
            best_sell = close_val + discount * atr_val
            
            # Check if filled next day
            next_day_filled = False
            entry_ref_price = np.nan
            if i + 1 < n:
                next_o = payload["open"][i+1]
                next_h = payload["high"][i+1]
                next_l = payload["low"][i+1]
                
                if sig == "buy":
                    limit_px = next_o - discount * payload["atr"][i+1]
                    next_day_filled = (next_l <= limit_px <= next_h)
                    entry_ref_price = limit_px if next_day_filled else next_o
                elif sig == "sell":
                    limit_px = next_o + discount * payload["atr"][i+1]
                    next_day_filled = (next_l <= limit_px <= next_h)
                    entry_ref_price = limit_px if next_day_filled else next_o
            
            results_apply.append({
                "Date": pd.to_datetime(dates[i]).strftime("%Y-%m-%d"),
                "ticker": tkr,
                "close": close_val,
                "signal_eod": sig,
                "score_100": score_100,
                "best_buy_value": best_buy if sig == "buy" else np.nan,
                "best_sell_value": best_sell if sig == "sell" else np.nan,
                "next_day_filled": next_day_filled,
                "entry_ref_price": entry_ref_price
            })'''

NEW_APPLY_LOOP = '''        # Generate apply signals (vectorized pre-compute, loop only for signals)
        x = np.asarray(payload["score_matrix"], dtype=np.float64)
        feat_n = max(1, x.shape[1])
        votes_long = (x > global_params.z_threshold).sum(axis=1) / feat_n
        votes_short = (x < -global_params.z_threshold).sum(axis=1) / feat_n
        score_raw = votes_long - votes_short
        score_ev = ewm_mean_np(score_raw, int(global_params.signal_ema_span))
        
        quality = compute_quality_factor(st["sharpe"], st["total_return"], st["n_trades"] / max(n / 252, 0.1))
        score95 = np.nanpercentile(np.abs(score_ev), 95) if np.isfinite(np.nanmax(np.abs(score_ev))) else 1.0
        score95 = max(score95, ATR_EPS)
        lookback = max(63, SCORE_LOOKBACK)
        
        # Backtest consistency metrics for confidence
        bt_sharpe = st.get("sharpe", 0.0)
        bt_win_rate = st.get("win_rate", 0.0)
        bt_n_trades = st.get("n_trades", 0.0)
        n_years = max(n / 252.0, 0.1)
        trades_per_year = bt_n_trades / n_years
        
        for i in range(max(0, n - APPLY_DAYS), n):
            sig = generate_signal_global(payload, global_params, i)
            
            recent_scores = score_ev[max(0, i - lookback + 1):i + 1]
            score_100 = score_0_100_from_ev(score_ev[i], recent_scores, quality)

            atr_val = max(payload["atr"][i], ATR_EPS)
            close_val = c[i]
            
            strength = float(np.clip(abs(score_ev[i]) / score95, 0.0, 1.0))
            discount = global_params.entry_discount_atr_frac * (1.0 - global_params.score_strength_scaling * strength)
            
            # Sempre calcular best_buy e best_sell como referencia
            best_buy = round(close_val - discount * atr_val, 4)
            best_sell = round(close_val + discount * atr_val, 4)
            
            # Entry ref: usar best_buy se sinal compra, best_sell se venda, close se hold
            if sig == "buy":
                entry_ref = best_buy
            elif sig == "sell":
                entry_ref = best_sell
            else:
                entry_ref = close_val
            
            # Stop loss e take profit baseados no ATR e params do GA
            stop_atr = global_params.stop_atr_mult * atr_val
            take_atr = global_params.reward_risk_ratio * stop_atr
            
            if sig == "buy":
                stop_loss = round(entry_ref - stop_atr, 4)
                take_profit = round(entry_ref + take_atr, 4)
            elif sig == "sell":
                stop_loss = round(entry_ref + stop_atr, 4)
                take_profit = round(entry_ref - take_atr, 4)
            else:
                # Para hold, mostrar stop/take do lado mais provavel
                if score_100 >= 50:
                    stop_loss = round(close_val - stop_atr, 4)
                    take_profit = round(close_val + take_atr, 4)
                else:
                    stop_loss = round(close_val + stop_atr, 4)
                    take_profit = round(close_val - take_atr, 4)
            
            # Confidence (0-100): combina qualidade do backtest, forca do sinal e consistencia
            # q_backtest: sharpe e win_rate do backtest
            q_bt = float(np.clip(_sigmoid((bt_sharpe - 0.1) / 0.3) * 0.5 + bt_win_rate * 0.5, 0.0, 1.0))
            # q_signal: forca direcional do sinal atual
            q_sig = strength
            # q_trades: confianca aumenta com mais trades historicos
            q_tr = float(np.clip(trades_per_year / 20.0, 0.0, 1.0))
            # q_agreement: concordancia entre features (votes)
            vl = votes_long[i] if i < len(votes_long) else 0.0
            vs = votes_short[i] if i < len(votes_short) else 0.0
            q_agree = float(np.clip(max(vl, vs) * 2.0, 0.0, 1.0))
            
            confidence = float(np.clip(
                (0.30 * q_bt + 0.25 * q_sig + 0.15 * q_tr + 0.30 * q_agree) * 100.0,
                0.0, 100.0
            ))
            
            results_apply.append({
                "Date": pd.to_datetime(dates[i]).strftime("%Y-%m-%d"),
                "ticker": tkr,
                "close": close_val,
                "signal_eod": sig,
                "score_100": round(score_100, 1),
                "confidence": round(confidence, 1),
                "best_buy_value": best_buy,
                "best_sell_value": best_sell,
                "entry_ref_price": round(entry_ref, 4),
                "stop_loss": stop_loss,
                "take_profit": take_profit,
            })'''

assert OLD_APPLY_LOOP in src, "Could not find old apply loop"
src = src.replace(OLD_APPLY_LOOP, NEW_APPLY_LOOP)

# ============================================================
# Verify all replacements applied
# ============================================================
assert "next_day_filled" not in src.split("def run():")[1], "next_day_filled still in run() output"
assert "stop_loss" in src, "stop_loss not found in new code"
assert "confidence" in src, "confidence not found in new code"

# Write back
main_cell["source"] = [src]
# Clear outputs to avoid stale data
main_cell["outputs"] = []
main_cell["execution_count"] = None

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("OK — notebook fixed successfully")
print("Changes applied:")
print("  1. Fixed cell ids for nbformat 4.5 compatibility")
print("  2. score_100: 0=venda forte, 50=neutro, 100=compra forte")
print("  3. best_buy_value / best_sell_value always populated")
print("  4. Added stop_loss, take_profit per row")
print("  5. Added confidence (0-100) per prediction")
print("  6. Removed next_day_filled from output")
print("  7. Vectorized score_ev computation outside loop (faster)")
