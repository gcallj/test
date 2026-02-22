import re

with open('main_cell_v2.py', 'r') as f:
    content = f.read()

# Replace precompute_global_payloads
old_precompute = """def precompute_global_payloads(windows: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]], full_df: pd.DataFrame, feature_cols: List[str]) -> List[Dict[str, Dict[str, np.ndarray]]]:
    payloads_by_window = []
    for _, _, te_start, te_end in windows:
        df_te = full_df[(full_df[DATE_COL] >= te_start) & (full_df[DATE_COL] <= te_end)]
        payloads = {}
        if not df_te.empty:
            for tk, g in df_te.groupby(TICKER_COL, sort=False):
                gg = g.sort_values(DATE_COL)
                c = gg[CLOSE_COL].to_numpy(np.float64)
                atr = gg["atr"].to_numpy(np.float64)
                payloads[tk] = {
                    "open": gg[OPEN_COL].to_numpy(np.float64),
                    "high": gg[HIGH_COL].to_numpy(np.float64),
                    "low": gg[LOW_COL].to_numpy(np.float64),
                    "close": c,
                    "atr": atr,
                    "score_matrix": gg[feature_cols].to_numpy(np.float64),
                    "vol_rank": rolling_percentile_rank(atr / np.maximum(c, ATR_EPS), 252),
                }
        payloads_by_window.append(payloads)
    return payloads_by_window"""

new_precompute = """def precompute_global_payloads(windows: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]], full_df: pd.DataFrame, feature_cols: List[str]) -> List[Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]]]]:
    payloads_by_window = []
    for tr_start, tr_end, te_start, te_end in windows:
        df_tr = full_df[(full_df[DATE_COL] >= tr_start) & (full_df[DATE_COL] <= tr_end)]
        df_te = full_df[(full_df[DATE_COL] >= te_start) & (full_df[DATE_COL] <= te_end)]
        
        def _build_payload(df_sub):
            payloads = {}
            if not df_sub.empty:
                for tk, g in df_sub.groupby(TICKER_COL, sort=False):
                    gg = g.sort_values(DATE_COL)
                    c = gg[CLOSE_COL].to_numpy(np.float64)
                    atr = gg["atr"].to_numpy(np.float64)
                    payloads[tk] = {
                        "open": gg[OPEN_COL].to_numpy(np.float64),
                        "high": gg[HIGH_COL].to_numpy(np.float64),
                        "low": gg[LOW_COL].to_numpy(np.float64),
                        "close": c,
                        "atr": atr,
                        "score_matrix": gg[feature_cols].to_numpy(np.float64),
                        "vol_rank": rolling_percentile_rank(atr / np.maximum(c, ATR_EPS), 252),
                    }
            return payloads
            
        payloads_by_window.append((_build_payload(df_tr), _build_payload(df_te)))
    return payloads_by_window"""

content = content.replace(old_precompute, new_precompute)

# Replace evaluate_global_walkforward
old_eval_wf = """def evaluate_global_walkforward(genome: List[float], payloads_by_window: List[Dict[str, Dict[str, np.ndarray]]]) -> float:
    vals = []
    for payloads in payloads_by_window:
        if payloads:
            vals.append(evaluate_global_genome(genome, payloads))
    return float(np.mean(vals)) if vals else -1e9"""

new_eval_wf = """def evaluate_global_walkforward(genome: List[float], payloads_by_window: List[Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]]]]) -> Tuple[float, float, float]:
    # Returns (fitness, sharpe_train, sharpe_oos)
    train_vals = []
    oos_vals = []
    for tr_payloads, te_payloads in payloads_by_window:
        if tr_payloads and te_payloads:
            train_vals.append(evaluate_global_genome(genome, tr_payloads))
            oos_vals.append(evaluate_global_genome(genome, te_payloads))
            
    if not oos_vals:
        return -1e9, -1e9, -1e9
        
    mean_train = float(np.mean(train_vals))
    mean_oos = float(np.mean(oos_vals))
    
    # Penalize difference between train and test
    alpha = 0.5
    fitness = mean_oos - alpha * abs(mean_train - mean_oos)
    
    return fitness, mean_train, mean_oos"""

content = content.replace(old_eval_wf, new_eval_wf)

with open('main_cell_v3.py', 'w') as f:
    f.write(content)
