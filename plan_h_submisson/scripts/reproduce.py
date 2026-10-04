"""Portable fixed-model reproduction; never searches or reselects a candidate."""
from pathlib import Path
import argparse,shutil,subprocess,sys,os,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def prepare(out):
 out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
 for source,dest in [('code/runtime','src'),('data/raw','data'),('data/analysis/inputs','inputs'),('configs/manifests','manifests'),('models','models')]:shutil.copytree(ROOT/source,out/dest)
 for p in (ROOT/'results/frozen').iterdir():
  if p.is_file() and p.suffix in ['.json','.sha256','.txt','.md']:shutil.copy2(p,out/p.name)
 for sub in ['logs','figures','predictions']:(out/sub).mkdir(exist_ok=True)
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--check',action='store_true');a=p.parse_args();out=prepare(a.out)
 env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',MPLBACKEND='Agg')
 for script in ['review.py']+(['audit_data.py','validate_frozen.py'] if a.check else []):
  with (out/'logs'/script.replace('.py','.log')).open('w') as log:subprocess.run([sys.executable,str(out/'src'/script)],check=True,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=out)
 import pandas as pd,numpy as np
 checks=[]
 for f in sorted((ROOT/'results/frozen/predictions').glob('*.csv')):
  x=pd.read_csv(f);y=pd.read_csv(out/'predictions'/f.name)
  assert x.row_id.tolist()==y.row_id.tolist() and np.array_equal(x.prediction,y.prediction)
  for col in ['score','R5_score','integrated_score']:
   if col in x:assert np.allclose(x[col],y[col],atol=1e-9,rtol=1e-9,equal_nan=True)
  checks.append(dict(file=f.name,rows=len(x),exact_alarms=True,numeric_tolerance='atol=rtol=1e-9'))
 (out/'portable_inference_check.json').write_text(json.dumps(dict(passed=True,predictions=checks,new_fit_calls=0),indent=2))
 print('Reproduced fixed predictions:',out)
