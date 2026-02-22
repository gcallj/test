import json

with open('GA_stock.ipynb', 'r') as f:
    nb = json.load(f)

# The cell we want to replace is the one with the long code.
# Let's find it.
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and len(cell['source']) > 100:
        # This is the main cell.
        pass
