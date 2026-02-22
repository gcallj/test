import json

with open('/workspaces/test/GA_stock.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        new_source = []
        skip = False
        for line in source:
            if "# Generate apply signals for last APPLY_DAYS" in line:
                skip = True
                new_source.append(line)
                new_source.append("        for i in range(max(0, n - APPLY_DAYS), n):\n")
                new_source.append("            sig = generate_signal_global(payload, global_params, i)\n")
                new_source.append("            \n")
                new_source.append("            # Calculate score_ev for the day\n")
                new_source.append("            x = np.asarray(payload[\"score_matrix\"], dtype=np.float64)\n")
                new_source.append("            feat_n = max(1, x.shape[1])\n")
                new_source.append("            votes_long = (x > global_params.z_threshold).sum(axis=1) / feat_n\n")
                new_source.append("            votes_short = (x < -global_params.z_threshold).sum(axis=1) / feat_n\n")
                new_source.append("            score_raw = votes_long - votes_short\n")
                new_source.append("            score_ev = pd.Series(score_raw).ewm(span=int(global_params.signal_ema_span), adjust=False).mean().to_numpy()\n")
                new_source.append("            \n")
                new_source.append("            # Calculate quality factor (confidence)\n")
                new_source.append("            quality = compute_quality_factor(st[\"sharpe\"], st[\"total_return\"], st[\"n_trades\"] / (n / 252))\n")
                new_source.append("            confidence_pct = float(np.clip(quality * 100.0, 0.0, 100.0))\n")
                new_source.append("            \n")
                new_source.append("            # Calculate 0-100 score\n")
                new_source.append("            score95 = np.nanpercentile(np.abs(score_ev), 95) if np.isfinite(np.nanmax(np.abs(score_ev))) else 1.0\n")
                new_source.append("            score95 = max(score95, ATR_EPS)\n")
                new_source.append("            score_100 = (np.clip(score_ev[i] / score95, -1.0, 1.0) + 1.0) / 2.0 * 100.0\n")
                new_source.append("\n")
                new_source.append("            # Calculate next day entry levels\n")
                new_source.append("            atr_val = payload[\"atr\"][i]\n")
                new_source.append("            close_val = c[i]\n")
                new_source.append("            \n")
                new_source.append("            # Calculate strength\n")
                new_source.append("            strength = float(np.clip(abs(score_ev[i]) / score95, 0.0, 1.0))\n")
                new_source.append("            discount = global_params.entry_discount_atr_frac * (1.0 - global_params.score_strength_scaling * strength)\n")
                new_source.append("            \n")
                new_source.append("            best_buy = close_val - discount * atr_val\n")
                new_source.append("            best_sell = close_val + discount * atr_val\n")
                new_source.append("            \n")
                new_source.append("            stop_loss_buy = best_buy - global_params.stop_atr_mult * atr_val\n")
                new_source.append("            stop_loss_sell = best_sell + global_params.stop_atr_mult * atr_val\n")
                new_source.append("            \n")
                new_source.append("            results_apply.append({\n")
                new_source.append("                \"Date\": pd.to_datetime(dates[i]).strftime(\"%Y-%m-%d\"),\n")
                new_source.append("                \"ticker\": tkr,\n")
                new_source.append("                \"close\": close_val,\n")
                new_source.append("                \"signal_eod\": sig,\n")
                new_source.append("                \"score_100\": score_100,\n")
                new_source.append("                \"confidence_pct\": confidence_pct,\n")
                new_source.append("                \"best_buy_value\": best_buy,\n")
                new_source.append("                \"stop_loss_buy\": stop_loss_buy,\n")
                new_source.append("                \"best_sell_value\": best_sell,\n")
                new_source.append("                \"stop_loss_sell\": stop_loss_sell,\n")
                new_source.append("                \"position_risk_pct\": global_params.max_loss_per_trade_pct,\n")
                new_source.append("            })\n")
                continue
            
            if skip:
                if "results_summary.append({" in line:
                    skip = False
                    new_source.append("\n")
                    new_source.append(line)
            else:
                new_source.append(line)
        
        cell['source'] = new_source

with open('/workspaces/test/GA_stock.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

