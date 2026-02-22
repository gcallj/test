import json

with open('GA_stock.ipynb', 'r') as f:
    nb = json.load(f)

with open('main_cell_restored.py', 'r') as f:
    content = f.read()

for cell in nb['cells']:
    if cell.get('metadata', {}).get('id') == '#VSC-04f7d421' or (cell['cell_type'] == 'code' and 'if hard_loss > gp.max_loss_per_trade_pct:' in ''.join(cell['source'])):
        cell['source'] = [line + '\n' for line in content.split('\n')]
        # remove last newline
        if cell['source']:
            cell['source'][-1] = cell['source'][-1].rstrip('\n')
        break

with open('GA_stock.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
