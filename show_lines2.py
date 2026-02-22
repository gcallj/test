import json

with open('GA_stock.ipynb', 'r') as f:
    nb = json.load(f)

cell = nb['cells'][2]
source = ''.join(cell['source'])
lines = source.split('\n')
for i in range(840, 870):
    if i < len(lines):
        print(f"{i+1}: {lines[i]}")
