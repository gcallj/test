import json

with open('GA_stock.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'if hard_loss > gp.max_loss_per_trade_pct:' in ''.join(cell['source']):
        source = ''.join(cell['source'])
        
        old_str = """                print(f"[GA] gen {gen:03d}/{int(ngen):03d} | best={best_fit_seen:.3f} (gen_max={max_fit:.3f}) avg={avg_fit:.3f} med={median_fit:.3f} min={min_fit:.3f} std={std_fit:.3f} | "
                      f"sharpe_tr={getattr(best_ind, 'sharpe_train', 0.0):.2f} sharpe_val={getattr(best_ind, 'sharpe_val', 0.0):.2f} | "
                      f"div={diversity:.1f}% | mut_pb={current_mut_pb:.2f} | eval={len(invalid_off)} | time={gen_dt:.1f}s")
                      
                    if gens_without_improvement >= 15:
                        print(f"[GA] Early stopping at generation {gen} (no improvement in 15 gens)")
                        break"""
                
        new_str = """                print(f"[GA] gen {gen:03d}/{int(ngen):03d} | best={best_fit_seen:.3f} (gen_max={max_fit:.3f}) avg={avg_fit:.3f} med={median_fit:.3f} min={min_fit:.3f} std={std_fit:.3f} | "
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
