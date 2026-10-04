from pathlib import Path
import zipfile, subprocess, sys, json, hashlib
root=Path('/content/hydraulic_ai')
if not root.exists():
    with zipfile.ZipFile('/content/hydraulic_input.zip') as z: z.extractall('/content')
run=root/'runs'/'experiment_20261003_01'
run.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(root/'src'))
from evaluation import metrics,postprocess
import pandas as pd
assert metrics([0,0,1,1],[.1,.9,.8,.2],[0,1,1,0])['F1']==.5
assert metrics([0,0,1],[.1,.2,.3],[0,0,0])['Precision']==0
f=pd.DataFrame({'burst_id':['a','a','b','b']})
assert postprocess(f,[1,1,1,1],'consecutive2').tolist()==[0,1,0,1]
assert postprocess(f,[1,0,1,1],'two_of_three').tolist()==[0,0,0,1]
print('Metric and causal boundary checks passed')
log=(run/'driver.log').open('a')
p=subprocess.Popen([sys.executable,'-u','src/experiment.py','--out',str(run),'--stage','develop'],cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(run/'process.json').write_text(json.dumps({'pid':p.pid,'stage':'develop'}))
print('Started CPU development experiment PID',p.pid)
