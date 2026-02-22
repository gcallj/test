import json

with open('GA_stock.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'if hard_loss > gp.max_loss_per_trade_pct:' in ''.join(cell['source']):
        source = ''.join(cell['source'])
        
        old_str = """            offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))
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
                        del offspring[i].fitness.values"""
                
        new_str = """            offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))
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
                    del offspring[i].fitness.values"""
                
        source = source.replace(old_str, new_str)
        
        cell['source'] = [line + '\n' for line in source.split('\n')]
        if cell['source']:
            cell['source'][-1] = cell['source'][-1].rstrip('\n')

with open('GA_stock.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
