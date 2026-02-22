import json

with open('GA_stock.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'if hard_loss > gp.max_loss_per_trade_pct:' in ''.join(cell['source']):
        source = ''.join(cell['source'])
        
        old_str = """            if hard_loss > gp.max_loss_per_trade_pct:
                ret = (o[i] / entry_px - 1.0) * pos
                equity *= (1.0 + ret)
                trade_rets.append(ret)
                consec_stops += 1 if ret < 0 else 0
                if ret > 0:
                    consec_stops = 0
                pos = 0
                continue

            fav = ((h[i] - entry_px) if pos > 0 else (entry_px - l[i]))
            adv = ((entry_px - l[i]) if pos > 0 else (h[i] - entry_px))

            if gp.partial_take_pct > 0 and (not partial_taken) and fav >= gp.partial_take_level * stop_abs:
                part_ret = gp.partial_take_pct * gp.partial_take_level * stop_abs / max(entry_px, ATR_EPS)
                equity *= (1.0 + part_ret)
                partial_taken = True

            stop_hit = adv >= stop_abs
            take_hit = fav >= take_abs
            time_stop = (bars >= gp.time_stop_bars) and (fav < 0.5 * stop_abs)

            if stop_hit or take_hit or time_stop:
                exit_px = c[i]
                if stop_hit:
                    exit_px = entry_px - pos * stop_abs
                elif take_hit:
                    exit_px = entry_px + pos * take_abs
                ret = (exit_px / entry_px - 1.0) * pos
                equity *= (1.0 + ret)
                trade_rets.append(ret)
                if stop_hit and ret < 0:
                    consec_stops += 1
                elif ret > 0:
                    consec_stops = 0
                pos = 0
                bars = 0
                partial_taken = False
                if gp.consecutive_loss_cooldown > 0 and consec_stops >= 2:
                    cooldown = gp.consecutive_loss_cooldown
                continue"""
                
        new_str = """            if hard_loss > gp.max_loss_per_trade_pct:
                ret = (o[i] / entry_px - 1.0) * pos - 0.0020
                equity *= (1.0 + ret)
                trade_rets.append(ret)
                consec_stops += 1 if ret < 0 else 0
                if ret > 0:
                    consec_stops = 0
                pos = 0
                continue

            fav = ((h[i] - entry_px) if pos > 0 else (entry_px - l[i]))
            adv = ((entry_px - l[i]) if pos > 0 else (h[i] - entry_px))

            if gp.partial_take_pct > 0 and (not partial_taken) and fav >= gp.partial_take_level * stop_abs:
                part_ret = gp.partial_take_pct * gp.partial_take_level * stop_abs / max(entry_px, ATR_EPS)
                equity *= (1.0 + part_ret - 0.0010)
                pos = pos * (1.0 - gp.partial_take_pct)
                partial_taken = True

            stop_hit = adv >= stop_abs
            take_hit = fav >= take_abs
            time_stop = (bars >= gp.time_stop_bars) and (fav < 0.5 * stop_abs)

            if stop_hit or take_hit or time_stop:
                exit_px = c[i]
                if stop_hit:
                    exit_px = entry_px - np.sign(pos) * stop_abs
                elif take_hit:
                    exit_px = entry_px + np.sign(pos) * take_abs
                
                cost = 0.0010 if partial_taken else 0.0020
                ret = (exit_px / entry_px - 1.0) * pos - cost
                equity *= (1.0 + ret)
                trade_rets.append(ret)
                if stop_hit and ret < 0:
                    consec_stops += 1
                elif ret > 0:
                    consec_stops = 0
                pos = 0
                bars = 0
                partial_taken = False
                if gp.consecutive_loss_cooldown > 0 and consec_stops >= 2:
                    cooldown = gp.consecutive_loss_cooldown
                continue"""
                
        source = source.replace(old_str, new_str)
        
        # Also fix the fitness function to penalize low trades more aggressively
        old_fit = """def global_fitness_from_stats(per_ticker_stats: List[Dict[str, float]]) -> float:
    if len(per_ticker_stats) == 0:
        return -1e9
    sharpe = np.array([s.get("sharpe", 0.0) for s in per_ticker_stats], dtype=np.float64)
    ret = np.array([s.get("total_return", 0.0) for s in per_ticker_stats], dtype=np.float64)
    ntr = np.array([s.get("n_trades", 0.0) for s in per_ticker_stats], dtype=np.float64)
    med_sharpe = float(np.median(sharpe))
    k = max(1, int(np.ceil(0.10 * len(ret))))
    worst_10 = float(np.mean(np.sort(ret)[:k]))
    med_trades = float(np.median(ntr))
    cross_std = float(np.std(ret))
    if med_trades < 4:
        return -1e9
    return float(0.40 * med_sharpe + 0.25 * worst_10 - 0.20 * cross_std + 0.15 * np.log(max(med_trades, 1.0)))"""

        new_fit = """def global_fitness_from_stats(per_ticker_stats: List[Dict[str, float]]) -> float:
    if len(per_ticker_stats) == 0:
        return -1e9
    sharpe = np.array([s.get("sharpe", 0.0) for s in per_ticker_stats], dtype=np.float64)
    ret = np.array([s.get("total_return", 0.0) for s in per_ticker_stats], dtype=np.float64)
    ntr = np.array([s.get("n_trades", 0.0) for s in per_ticker_stats], dtype=np.float64)
    med_sharpe = float(np.median(sharpe))
    k = max(1, int(np.ceil(0.10 * len(ret))))
    worst_10 = float(np.mean(np.sort(ret)[:k]))
    med_trades = float(np.median(ntr))
    cross_std = float(np.std(ret))
    
    # Penalize low trades heavily
    if med_trades < 15:
        return -1e9
        
    trade_penalty = 0.0
    if med_trades < 30:
        trade_penalty = (30 - med_trades) * 0.1
        
    return float(0.40 * med_sharpe + 0.25 * worst_10 - 0.20 * cross_std + 0.15 * np.log(max(med_trades, 1.0)) - trade_penalty)"""
        
        source = source.replace(old_fit, new_fit)
        
        # Also fix evaluate_global_walkforward
        old_eval = """            if te_trades < 30:
                penalty += (30 - te_trades) * 0.05"""
                
        new_eval = """            if te_trades < 50:
                penalty += (50 - te_trades) * 0.1"""
                
        source = source.replace(old_eval, new_eval)
        
        cell['source'] = [line + '\n' for line in source.split('\n')]
        if cell['source']:
            cell['source'][-1] = cell['source'][-1].rstrip('\n')

with open('GA_stock.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
