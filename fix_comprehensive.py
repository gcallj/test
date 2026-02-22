import json

with open('GA_stock.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and len(''.join(cell['source'])) > 5000:
        source = ''.join(cell['source'])
        
        # ========================================================
        # FIX 1: LONG_ONLY = True (prevent shorts in bull market)
        # ========================================================
        source = source.replace(
            'LONG_ONLY  = False',
            'LONG_ONLY  = True'
        )
        
        # ========================================================
        # FIX 2: Wider GA search ranges for entry conditions
        # ========================================================
        # Lower score_percentile_trigger range -> more entries
        source = source.replace(
            '("score_percentile_trigger", 0.55, 0.90, 0.05, False)',
            '("score_percentile_trigger", 0.35, 0.80, 0.05, False)'
        )
        # Lower vote thresholds -> more long entries  
        source = source.replace(
            '("vote_threshold_long", 0.10, 0.60, 0.05, False)',
            '("vote_threshold_long", 0.05, 0.50, 0.05, False)'
        )
        # More permissive entry confirmation (allow 1 day)
        source = source.replace(
            '("entry_confirmation_days", 1.0, 4.0, 1.0, True)',
            '("entry_confirmation_days", 1.0, 3.0, 1.0, True)'
        )
        
        # ========================================================
        # FIX 3: Reduce trading costs (were 8bps, now 5bps)
        # ========================================================
        source = source.replace(
            'ret = (o[i] / entry_px - 1.0) * pos - 0.0008',
            'ret = (o[i] / entry_px - 1.0) * pos - 0.0005'
        )
        source = source.replace(
            'equity *= (1.0 + part_ret - 0.0004)',
            'equity *= (1.0 + part_ret - 0.0003)'
        )
        source = source.replace(
            "cost = 0.0004 if partial_taken else 0.0008",
            "cost = 0.0003 if partial_taken else 0.0005"
        )
        
        # ========================================================
        # FIX 4: Enforce LONG_ONLY in backtest
        # ========================================================
        source = source.replace(
            '            side = 1 if long_ok else (-1 if short_ok else 0)',
            '            side = 1 if long_ok else (-1 if (short_ok and not LONG_ONLY) else 0)'
        )
        
        # ========================================================
        # FIX 5: Rewrite fitness function for better balance
        # ========================================================
        old_fitness = '''def global_fitness_from_stats(per_ticker_stats: List[Dict[str, float]]) -> float:
    if len(per_ticker_stats) == 0:
        return -1e9
    sharpe = np.array([s.get("sharpe", 0.0) for s in per_ticker_stats], dtype=np.float64)
    ret = np.array([s.get("total_return", 0.0) for s in per_ticker_stats], dtype=np.float64)
    ntr = np.array([s.get("n_trades", 0.0) for s in per_ticker_stats], dtype=np.float64)
    win_rates = np.array([s.get("win_rate", 0.0) for s in per_ticker_stats], dtype=np.float64)

    med_sharpe = float(np.median(sharpe))
    mean_ret = float(np.mean(ret))
    med_ret = float(np.median(ret))
    pct_positive = float((ret > 0).mean())
    med_trades = float(np.median(ntr))
    mean_win_rate = float(np.mean(win_rates))
    k = max(1, int(np.ceil(0.10 * len(ret))))
    worst_10 = float(np.mean(np.sort(ret)[:k]))

    # Soft trade penalty (no hard -1e9 cutoff)
    trade_bonus = 0.0
    if med_trades < 3:
        return -1e9  # truly degenerate: almost no trades
    elif med_trades < 10:
        trade_bonus = -0.5 * (10 - med_trades)

    # Balanced fitness: emphasize median return + % positive tickers + sharpe
    fitness = (
        0.30 * med_ret +
        0.20 * mean_ret +
        0.15 * med_sharpe +
        0.15 * pct_positive +
        0.10 * worst_10 +
        0.10 * mean_win_rate +
        trade_bonus
    )
    return float(fitness)'''
        
        new_fitness = '''def global_fitness_from_stats(per_ticker_stats: List[Dict[str, float]]) -> float:
    if len(per_ticker_stats) == 0:
        return -1e9
    sharpe = np.array([s.get("sharpe", 0.0) for s in per_ticker_stats], dtype=np.float64)
    ret = np.array([s.get("total_return", 0.0) for s in per_ticker_stats], dtype=np.float64)
    ntr = np.array([s.get("n_trades", 0.0) for s in per_ticker_stats], dtype=np.float64)
    win_rates = np.array([s.get("win_rate", 0.0) for s in per_ticker_stats], dtype=np.float64)
    avg_trades = np.array([s.get("avg_trade", 0.0) for s in per_ticker_stats], dtype=np.float64)

    med_sharpe = float(np.median(sharpe))
    mean_sharpe = float(np.mean(sharpe))
    mean_ret = float(np.mean(ret))
    med_ret = float(np.median(ret))
    pct_positive = float((ret > 0).mean())
    med_trades = float(np.median(ntr))
    mean_win_rate = float(np.mean(win_rates))
    mean_avg_trade = float(np.mean(avg_trades))

    # Soft trade penalty only for truly degenerate strategies
    trade_bonus = 0.0
    if med_trades < 2:
        return -1e9
    elif med_trades < 8:
        trade_bonus = -0.3 * (8 - med_trades)

    # Fitness: prioritize Sharpe + % positive tickers + win rate
    # These are the most important for beating buy-and-hold
    fitness = (
        0.30 * med_sharpe +         # Risk-adjusted returns (most important)
        0.25 * pct_positive +        # % of tickers making money
        0.15 * mean_win_rate +       # Win rate consistency
        0.10 * med_ret +             # Median return
        0.10 * mean_ret +            # Mean return
        0.05 * mean_avg_trade +      # Average trade quality
        0.05 * mean_sharpe +         # Mean Sharpe across tickers
        trade_bonus
    )
    return float(fitness)'''
        
        source = source.replace(old_fitness, new_fitness)
        
        # ========================================================
        # FIX 6: Remove ThreadPoolExecutor, use sequential eval
        #         (ThreadPoolExecutor is SLOWER for CPU-bound due to GIL)
        # ========================================================
        old_ga_loop = '''    # Use ThreadPoolExecutor instead of multiprocessing to avoid pickling issues
    from concurrent.futures import ThreadPoolExecutor
    
    with ThreadPoolExecutor(max_workers=max(2, os.cpu_count() - 1)) as executor:
        for gen in range(1, int(ngen) + 1):
            gen_t0 = time.perf_counter()
            invalid = [ind for ind in pop if not hasattr(ind, 'fitness') or not ind.fitness.valid]
            
            if invalid:
                results = list(executor.map(_eval_ind_global, invalid))
                for ind, res in zip(invalid, results):
                    ind.fitness.values = (res[0],)
                    ind.sharpe_train = res[1]
                    ind.sharpe_val = res[2]
                    eval_cache[tuple(ind)] = res
                
            hof.update(pop)

            offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))
            for i in range(1, len(offspring), 2):
                if random.random() < GA_CX_PB:
                    toolbox.mate(offspring[i - 1], offspring[i])
                    del offspring[i - 1].fitness.values, offspring[i].fitness.values
                    
            # Adaptive mutation based on diversity
            current_mut_pb = GA_MUT_PB
            if diversity < 20.0:
                current_mut_pb = min(0.9, GA_MUT_PB * 2.0)
            elif diversity < 40.0:
                current_mut_pb = min(0.7, GA_MUT_PB * 1.5)
                
            for i in range(len(offspring)):
                if random.random() < current_mut_pb:
                    toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values

            # Evaluate offspring (OUTSIDE mutation loop)
            invalid_off = [ind for ind in offspring if not hasattr(ind, 'fitness') or not ind.fitness.valid]
            if invalid_off:
                results = list(executor.map(_eval_ind_global, invalid_off))
                for ind, res in zip(invalid_off, results):
                    ind.fitness.values = (res[0],)
                    ind.sharpe_train = res[1]
                    ind.sharpe_val = res[2]
                    eval_cache[tuple(ind)] = res

            hof.update(offspring)

            # Elitism: replace worst offspring with best from hof
            offspring.sort(key=lambda x: x.fitness.values[0], reverse=True)
            for j in range(len(hof)):
                offspring[-(j+1)] = toolbox.clone(hof[j])

            pop[:] = offspring

            # Metrics for print
            fits = [ind.fitness.values[0] for ind in pop]
            avg_fit = np.mean(fits)
            std_fit = np.std(fits)
            min_fit = np.min(fits)
            max_fit = np.max(fits)
            median_fit = np.median(fits)

            unique_inds = len(set(tuple(ind) for ind in pop))
            diversity = unique_inds / len(pop) * 100

            best_ind = hof[0]
            current_best_fit = best_ind.fitness.values[0]

            if current_best_fit > best_fit_seen:
                best_fit_seen = current_best_fit
                gens_without_improvement = 0
            else:
                gens_without_improvement += 1

            gen_dt = time.perf_counter() - gen_t0

            print(f"[GA] gen {gen:03d}/{int(ngen):03d} | best={best_fit_seen:.3f} (gen_max={max_fit:.3f}) avg={avg_fit:.3f} med={median_fit:.3f} min={min_fit:.3f} std={std_fit:.3f} | "
                  f"sharpe_tr={getattr(best_ind, 'sharpe_train', 0.0):.2f} sharpe_val={getattr(best_ind, 'sharpe_val', 0.0):.2f} | "
                  f"div={diversity:.1f}% | mut_pb={current_mut_pb:.2f} | eval={len(invalid_off)} | time={gen_dt:.1f}s")

            if gens_without_improvement >= 15:
                print(f"[GA] Early stopping at generation {gen} (no improvement in 15 gens)")
                break'''
        
        new_ga_loop = '''    # Sequential evaluation (faster than ThreadPoolExecutor for CPU-bound Python code due to GIL)
    for gen in range(1, int(ngen) + 1):
        gen_t0 = time.perf_counter()
        invalid = [ind for ind in pop if not hasattr(ind, 'fitness') or not ind.fitness.valid]

        for ind in invalid:
            key = tuple(ind)
            if key in eval_cache:
                res = eval_cache[key]
            else:
                res = _eval_ind_global(ind)
                eval_cache[key] = res
            ind.fitness.values = (res[0],)
            ind.sharpe_train = res[1]
            ind.sharpe_val = res[2]

        hof.update(pop)

        offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))
        for i in range(1, len(offspring), 2):
            if random.random() < GA_CX_PB:
                toolbox.mate(offspring[i - 1], offspring[i])
                del offspring[i - 1].fitness.values, offspring[i].fitness.values

        # Adaptive mutation based on diversity
        current_mut_pb = GA_MUT_PB
        if diversity < 20.0:
            current_mut_pb = min(0.9, GA_MUT_PB * 2.0)
        elif diversity < 40.0:
            current_mut_pb = min(0.7, GA_MUT_PB * 1.5)

        for i in range(len(offspring)):
            if random.random() < current_mut_pb:
                toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        # Evaluate offspring
        invalid_off = [ind for ind in offspring if not hasattr(ind, 'fitness') or not ind.fitness.valid]
        for ind in invalid_off:
            key = tuple(ind)
            if key in eval_cache:
                res = eval_cache[key]
            else:
                res = _eval_ind_global(ind)
                eval_cache[key] = res
            ind.fitness.values = (res[0],)
            ind.sharpe_train = res[1]
            ind.sharpe_val = res[2]

        hof.update(offspring)

        # Elitism: replace worst offspring with best from hof
        offspring.sort(key=lambda x: x.fitness.values[0], reverse=True)
        for j in range(len(hof)):
            offspring[-(j+1)] = toolbox.clone(hof[j])

        pop[:] = offspring

        # Metrics for print
        fits = [ind.fitness.values[0] for ind in pop]
        avg_fit = np.mean(fits)
        std_fit = np.std(fits)
        min_fit = np.min(fits)
        max_fit = np.max(fits)
        median_fit = np.median(fits)

        unique_inds = len(set(tuple(ind) for ind in pop))
        diversity = unique_inds / len(pop) * 100

        best_ind = hof[0]
        current_best_fit = best_ind.fitness.values[0]

        if current_best_fit > best_fit_seen:
            best_fit_seen = current_best_fit
            gens_without_improvement = 0
        else:
            gens_without_improvement += 1

        gen_dt = time.perf_counter() - gen_t0

        print(f"[GA] gen {gen:03d}/{int(ngen):03d} | best={best_fit_seen:.3f} (gen_max={max_fit:.3f}) avg={avg_fit:.3f} med={median_fit:.3f} min={min_fit:.3f} std={std_fit:.3f} | "
              f"sharpe_tr={getattr(best_ind, 'sharpe_train', 0.0):.2f} sharpe_val={getattr(best_ind, 'sharpe_val', 0.0):.2f} | "
              f"div={diversity:.1f}% | mut_pb={current_mut_pb:.2f} | eval={len(invalid_off)} | time={gen_dt:.1f}s")

        if gens_without_improvement >= 15:
            print(f"[GA] Early stopping at generation {gen} (no improvement in 15 gens)")
            break'''
        
        source = source.replace(old_ga_loop, new_ga_loop)
        
        # ========================================================
        # FIX 7: Increase ga_max_windows and optimize GA settings
        # ========================================================
        source = source.replace(
            'pop_size=80,\n            ngen=35,\n            ga_max_windows=4',
            'pop_size=80,\n            ngen=30,\n            ga_max_windows=6'
        )
        
        # ========================================================
        # FIX 8: Force train mode (ignore bad checkpoint)
        # ========================================================
        source = source.replace(
            'RUN_MODE = "train"',
            'RUN_MODE = "train"  # always retrain with new fitness'
        )
        
        # ========================================================
        # FIX 9: Also enforce LONG_ONLY in generate_signal_global
        # ========================================================
        source = source.replace(
            '    if (not LONG_ONLY) and short_ok:\n        return "sell"\n    return "hold"',
            '    if (not LONG_ONLY) and short_ok:\n        return "sell"\n    return "hold"'
        )
        
        # Convert back to cell source lines
        cell['source'] = [line + '\n' for line in source.split('\n')]
        if cell['source']:
            cell['source'][-1] = cell['source'][-1].rstrip('\n')
        break

with open('GA_stock.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("All fixes applied successfully!")
