"""Portable complete replay, always into a newly created output folder."""
from pathlib import Path
import os,sys,shutil,json,subprocess,datetime,time
ROOT=Path(__file__).resolve().parents[1]
out=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else ROOT/'replays'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
out.mkdir(parents=True,exist_ok=False)
for dirname in ['src','inputs','data','manifests']:
 shutil.copytree(ROOT/dirname,out/dirname,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for dirname in ['models','predictions','figures','tables','logs','references']:(out/dirname).mkdir()
shutil.copy2(ROOT/'models/baseline.joblib',out/'models/baseline.joblib')
for name in ['preregistration.json','preregistration.sha256','inventory.json','requirements.lock.txt','hypotheses.md','RUN_EXPERIMENT.sh']:
 shutil.copy2(ROOT/name,out/name)
shutil.copy2(ROOT/'references/literature_notes_ko.md',out/'references/literature_notes_ko.md')
for name in ['archive_audit.json','references/source_manifest.json']:
 if (ROOT/name).exists():shutil.copy2(ROOT/name,out/name)
env=os.environ.copy();env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',MPLBACKEND='Agg');start=time.perf_counter()
for script,args in [('reproduce.py',[]),('experiment.py',['select']),('experiment.py',['evaluate']),('validate.py',[]),('audit_artifacts.py',[]),('report.py',[])]:
 log=out/'logs'/(script.removesuffix('.py')+('_'+args[0] if args else '')+'.log');cmd=[sys.executable,str(out/'src'/script)]+args
 with log.open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
 if r.returncode:print(log.read_text());raise RuntimeError(f'Failed: {script}; logs preserved in {out}')
 print('Completed',script,*args,flush=True)
# Compare all scores/decisions, not model pickle bytes or wall-clock metadata.
import numpy as np,pandas as pd
lk=json.loads((ROOT/'selected_config.json').read_text());new=json.loads((out/'selected_config.json').read_text());assert (lk['selected'],lk['threshold_control'])==(new['selected'],new['threshold_control'])
checks=[]
for src in sorted((ROOT/'predictions').glob('*.csv')):
 a=pd.read_csv(src);b=pd.read_csv(out/'predictions'/src.name);assert a.row_id.tolist()==b.row_id.tolist();assert np.array_equal(a.prediction,b.prediction)
 for col in ['score','Q','T2','aux_score','integrated_score']:
  if col in a:assert np.allclose(a[col],b[col],equal_nan=True,rtol=1e-9,atol=1e-10),(src.name,col)
 checks.append(dict(file=src.name,rows=len(a),equal_predictions=True))
a=pd.read_csv(ROOT/'all_trials.csv');b=pd.read_csv(out/'all_trials.csv');assert a.candidate.tolist()==b.candidate.tolist()
for col in ['TP','FN','FP','TN','F1','F2','FPR','AP','PR_AUC_trapezoid','ROC_AUC','threshold']:assert np.allclose(a[col],b[col],equal_nan=True,rtol=1e-9,atol=1e-10)
result=dict(passed=True,output=str(out),complete_retraining_of_auxiliaries=True,baseline_stored_inference_not_refit=True,selection_equal=True,files=checks,seconds=time.perf_counter()-start,not_independent_statistical_replication=True)
(out/'end_to_end_replay.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(output=str(out),passed=True,seconds=result['seconds'])),flush=True)
