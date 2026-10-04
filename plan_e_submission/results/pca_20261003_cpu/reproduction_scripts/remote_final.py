from pathlib import Path
import subprocess,os,json,shutil
root=Path('/content/hydraulic_ai_alt');run=root/'runs/pca_20261003_cpu'
shutil.copytree(root/'src',run/'source_final',ignore=shutil.ignore_patterns('__pycache__'))
code='''import sys,json
from pathlib import Path
sys.path.insert(0,'src')
from experiment import Experiment
p=Path('runs/pca_20261003_cpu')
e=Experiment(p);e.load('develop')
lock=json.loads((p/'selection_lock.json').read_text())
for name in ['primary','alternative','if_primary']:
 c=lock[name];e.error_analysis(c['key'],'selection',c['target_fpr'])
e.final()
'''
(root/'final_stage.py').write_text(code)
env=os.environ.copy();env.update(MPLBACKEND='Agg',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
p=subprocess.Popen([str(root/'.venv/bin/python'),'final_stage.py'],cwd=root,env=env,stdout=open(root/'final.log','w'),stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'pid':p.pid,'stage':'locked final evaluation'}))
