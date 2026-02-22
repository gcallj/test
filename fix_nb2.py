import json

with open('GA_stock.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'if hard_loss > gp.max_loss_per_trade_pct:' in ''.join(cell['source']):
        source = ''.join(cell['source'])
        
        # Fix the multiprocessing issue by using a simpler approach
        old_mp = """    pool = multiprocessing.Pool(processes=max(2, multiprocessing.cpu_count() - 1))
    
    for gen in range(1, int(ngen) + 1):
        gen_t0 = time.perf_counter()
        invalid = [ind for ind in pop if not hasattr(ind, 'fitness') or not ind.fitness.valid]
        
        # Paralelizar com multiprocessing
        if invalid:
            results = pool.map(_eval_ind_global, invalid)
            for ind, res in zip(invalid, results):
                ind.fitness.values = (res[0],)
                ind.sharpe_train = res[1]
                ind.sharpe_val = res[2]
                eval_cache[tuple(ind)] = res"""
                
        new_mp = """    # Use ThreadPoolExecutor instead of multiprocessing to avoid pickling issues
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
                    eval_cache[tuple(ind)] = res"""
                    
        source = source.replace(old_mp, new_mp)
        
        old_mp2 = """        invalid_off = [ind for ind in offspring if not hasattr(ind, 'fitness') or not ind.fitness.valid]
        if invalid_off:
            results = pool.map(_eval_ind_global, invalid_off)
            for ind, res in zip(invalid_off, results):
                ind.fitness.values = (res[0],)
                ind.sharpe_train = res[1]
                ind.sharpe_val = res[2]
                eval_cache[tuple(ind)] = res"""
                
        new_mp2 = """        invalid_off = [ind for ind in offspring if not hasattr(ind, 'fitness') or not ind.fitness.valid]
            if invalid_off:
                results = list(executor.map(_eval_ind_global, invalid_off))
                for ind, res in zip(invalid_off, results):
                    ind.fitness.values = (res[0],)
                    ind.sharpe_train = res[1]
                    ind.sharpe_val = res[2]
                    eval_cache[tuple(ind)] = res"""
                    
        source = source.replace(old_mp2, new_mp2)
        
        old_mp3 = """        if gens_without_improvement >= 15:
            print(f"[GA] Early stopping at generation {gen} (no improvement in 10 gens)")
            break
            
    pool.close()
    pool.join()"""
    
        new_mp3 = """            if gens_without_improvement >= 15:
                print(f"[GA] Early stopping at generation {gen} (no improvement in 15 gens)")
                break"""
                
        source = source.replace(old_mp3, new_mp3)
        
        # Fix indentation for the rest of the loop
        lines = source.split('\n')
        in_loop = False
        for i, line in enumerate(lines):
            if 'hof.update(pop)' in line:
                in_loop = True
            if 'print("\\n[GA] Top 5 cromossomos:")' in line:
                in_loop = False
                
            if in_loop and line.startswith('        '):
                lines[i] = '    ' + line
                
        source = '\n'.join(lines)
        
        cell['source'] = [line + '\n' for line in source.split('\n')]
        if cell['source']:
            cell['source'][-1] = cell['source'][-1].rstrip('\n')

with open('GA_stock.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
