import json
import ast

with open('GA_stock.ipynb', 'r') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        try:
            ast.parse(source)
        except SyntaxError as e:
            print(f"Syntax error in cell {i}: {e}")
            print(f"Line {e.lineno}: {e.text}")
