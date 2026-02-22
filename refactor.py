import re

with open('main_cell.py', 'r') as f:
    content = f.read()

# 16. FIX XLSXWRITER
content = content.replace('import math', '!pip install xlsxwriter -q\nimport math')

# 6. REDUZIR POPULAÇÃO E GERAÇÕES COM EARLY STOPPING
content = content.replace('GA_POP_SIZE = 320', 'GA_POP_SIZE = 100')
content = content.replace('GA_NGEN     = 55', 'GA_NGEN     = 40')
content = content.replace('EARLY_STOP  = 10', 'EARLY_STOP  = 10')

# 9. REDUZIR WINDOWS
content = content.replace('ga_max_windows: int = 6', 'ga_max_windows: int = 4')

with open('main_cell_v2.py', 'w') as f:
    f.write(content)
