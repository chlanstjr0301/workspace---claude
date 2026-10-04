from pathlib import Path
import os,json
root=Path('/content/hydraulic_ai/runs/experiment_20261003_01')
print((root/'driver.log').read_text()[-9000:])
print('Models saved:',len(list((root/'models').glob('*.joblib'))))
print('Selection locked:',(root/'selection_lock.json').exists())
print('Final complete:',(root/'final_complete.json').exists())
if (root/'process.json').exists():
    p=json.loads((root/'process.json').read_text());
    try: print('Process status:',Path(f"/proc/{p['pid']}/status").read_text().splitlines()[:3])
    except FileNotFoundError: print('Process ended')
