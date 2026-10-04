from pathlib import Path
import subprocess,json
root=Path('/content/hydraulic_ai_alt')
(root/'develop.log').rename(root/'environment_initial_failure.log')
script='''set -eu
cd /content/hydraulic_ai_alt
python3 -m venv --without-pip .venv
python3 -m pip --python .venv/bin/python install -r requirements.lock.txt
.venv/bin/python scripts/checks.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python src/experiment.py --out runs/pca_20261003_cpu --stage develop
'''
(root/'develop.sh').write_text(script)
p=subprocess.Popen(['bash',str(root/'develop.sh')],stdout=open(root/'develop.log','w'),stderr=subprocess.STDOUT,start_new_session=True)
(root/'develop_pid.json').write_text(json.dumps({'pid':p.pid,'run':'pca_20261003_cpu'}))
print(json.dumps({'pid':p.pid,'change':'venv without ensurepip, pip --python targets private env','test_scores_seen':False}))
