import json

with open('GA_stock.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'if hard_loss > gp.max_loss_per_trade_pct:' in ''.join(cell['source']):
        source = ''.join(cell['source'])
        
        # Fix the indentation issue
        old_str = """        hof.update(offspring)
        
        # Elitism: replace worst offspring with best from hof
        offspring.sort(key=lambda x: x.fitness.values[0], reverse=True)
        for i in range(len(hof)):
            offspring[-(i+1)] = toolbox.clone(hof[i])
            
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
            break"""
                
        new_str = """            hof.update(offspring)
            
            # Elitism: replace worst offspring with best from hof
            offspring.sort(key=lambda x: x.fitness.values[0], reverse=True)
            for i in range(len(hof)):
                offspring[-(i+1)] = toolbox.clone(hof[i])
                
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
                break"""
                
        source = source.replace(old_str, new_str)
        
        cell['source'] = [line + '\n' for line in source.split('\n')]
        if cell['source']:
            cell['source'][-1] = cell['source'][-1].rstrip('\n')

with open('GA_stock.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
