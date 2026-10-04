from pathlib import Path
import json
root=Path('/content/hydraulic_ai_alt')
p=root/'develop.log'
print(p.read_text()[-6500:] if p.exists() else 'not started')
print('selection_locked',(root/'runs/pca_20261003_cpu/selection_lock.json').exists())
