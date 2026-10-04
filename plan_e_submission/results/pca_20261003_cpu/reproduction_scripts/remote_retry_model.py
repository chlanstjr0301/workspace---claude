from pathlib import Path
import subprocess,json,os
root=Path('/content/hydraulic_ai_alt');(root/'develop.log').rename(root/'plotting_initial_failure.log')
env=os.environ.copy();env.update(MPLBACKEND='Agg',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
p=subprocess.Popen([str(root/'.venv/bin/python'),'src/experiment.py','--out','runs/pca_20261003_cpu','--stage','develop'],cwd=root,env=env,stdout=open(root/'develop.log','w'),stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'pid':p.pid,'stage':'develop','test_scoring':False}))
