import json

with open('GA_stock.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'if hard_loss > gp.max_loss_per_trade_pct:' in ''.join(cell['source']):
        source = ''.join(cell['source'])
        
        # Fix the indentation issue
        old_str = """        invalid_off = [ind for ind in offspring if not hasattr(ind, 'fitness') or not ind.fitness.valid]
            if invalid_off:
                results = list(executor.map(_eval_ind_global, invalid_off))
                for ind, res in zip(invalid_off, results):
                    ind.fitness.values = (res[0],)
                    ind.sharpe_train = res[1]
                    ind.sharpe_val = res[2]
                    eval_cache[tuple(ind)] = res"""
                    
        new_str = """            invalid_off = [ind for ind in offspring if not hasattr(ind, 'fitness') or not ind.fitness.valid]
            if invalid_off:
                results = list(executor.map(_eval_ind_global, invalid_off))
                for ind, res in zip(invalid_off, results):
                    ind.fitness.values = (res[0],)
                    ind.sharpe_train = res[1]
                    ind.sharpe_val = res[2]
                    eval_cache[tuple(ind)] = res"""
                    
        source = source.replace(old_str, new_str)
        
        cell['source'] = [line + '\n' for line in source.split('\n')]
        if cell['source']:
            cell['source'][-1] = cell['source'][-1].rstrip('\n')

with open('GA_stock.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
