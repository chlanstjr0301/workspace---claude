from pathlib import Path
import subprocess,sys,zipfile,json,os
root=Path('/content/hydraulic_ai_alt');root.mkdir(exist_ok=True)
with zipfile.ZipFile('/content/hydraulic_alt_input.zip') as z:z.extractall(root)
script='''set -eu
cd /content/hydraulic_ai_alt
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python scripts/checks.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python src/experiment.py --out runs/pca_20261003_cpu --stage develop
'''
(root/'develop.sh').write_text(script)
log=open(root/'develop.log','w')
p=subprocess.Popen(['bash',str(root/'develop.sh')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'develop_pid.json').write_text(json.dumps({'pid':p.pid,'run':'pca_20261003_cpu'}))
print(json.dumps({'pid':p.pid,'root':str(root),'stage':'develop','test_scoring':False}))
