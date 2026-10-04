from pathlib import Path
root=Path('/content/hydraulic_ai_alt')
p=root/'final.log';print(p.read_text()[-5000:] if p.exists() else 'not started')
print('final_complete',(root/'runs/pca_20261003_cpu/final_summary.json').exists())
