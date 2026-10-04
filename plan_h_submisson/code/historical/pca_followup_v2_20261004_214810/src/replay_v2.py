"""Replay the registered PCA-only study in a new directory, preserving prior artifacts."""
from pathlib import Path
import sys,os,shutil,json,subprocess,datetime,time
P=Path(__file__).resolve().parents[1];out=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else P/'replays'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');out.mkdir(parents=True,exist_ok=False)
for folder in ['src','inputs','models','data','manifests','references']:shutil.copytree(P/folder,out/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for folder in ['predictions','tables','figures','logs']:(out/folder).mkdir()
for f in ['inventory.json','requirements.lock.txt','diagnostic_plan.json','protocol_v2.json','protocol_v2.sha256','RUN_EXPERIMENT.sh']:shutil.copy2(P/f,out/f)
env=os.environ.copy();env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',MPLBACKEND='Agg');start=time.perf_counter();stages=[('diagnose.py',[]),('experiment_v2.py',['select']),('experiment_v2.py',['evaluate']),('validate_v2.py',[]),('analyze.py',[]),('report_v2.py',[])]
# Recompute calibration scores and verify both registered higher-quantile values before any S results.
sys.path.insert(0,str(out/'src'))
from v2 import context,aux_score
import numpy as np,pandas as pd
d,e,b0,r5=context();ci=d.index[(d.split=='calibration')&(d.label==0)].to_numpy();scores=aux_score(r5,e,b0,ci);available=scores[np.isfinite(scores)];proto=json.loads((out/'protocol_v2.json').read_text())
for cfg in proto['candidates']:assert float(np.quantile(available,cfg['q'],method='higher'))==cfg['theta']
pd.DataFrame(proto['calibration']['tail_audit']).to_csv(out/'tables/calibration_quantiles.csv',index=False)
cal=d.loc[ci,['row_id','source_file','source_row','TimeStamp','burst_id']].copy();cal['R5_score']=scores;cal.to_csv(out/'tables/calibration_scores.csv',index=False)
for name,args in stages:
 log=out/'logs'/(name.removesuffix('.py')+('_'+args[0] if args else '')+'.log')
 with log.open('w') as f:r=subprocess.run([sys.executable,str(out/'src'/name)]+args,stdout=f,stderr=subprocess.STDOUT,env=env)
 if r.returncode:print(log.read_text());raise RuntimeError(f'{name} failed; logs saved in {out}')
 print('Completed',name,*args,flush=True)
old=json.loads((P/'selected_config.json').read_text());new=json.loads((out/'selected_config.json').read_text());assert old['selected_config']==new['selected_config'];files=[]
for path in sorted((P/'predictions').glob('*.csv')):
 a=pd.read_csv(path);b=pd.read_csv(out/'predictions'/path.name);assert a.row_id.tolist()==b.row_id.tolist();assert np.array_equal(a.prediction,b.prediction),(path.name,'prediction')
 for k in ['score','R5_score','integrated_score','aux_score']:
  if k in a:assert np.allclose(a[k],b[k],equal_nan=True,atol=1e-10,rtol=1e-9),(path.name,k)
 files.append(dict(file=path.name,rows=len(a),matched=True))
for name in ['all_trials.csv','block_metrics.csv','followup_metrics.csv']:
 a=pd.read_csv(P/name);b=pd.read_csv(out/name)
 for k in ['TP','FN','FP','TN','F1','F2','FPR']:assert np.allclose(a[k],b[k],equal_nan=True)
result=dict(passed=True,output=str(out),selected=new['selected'],fixed_model_stored_inference=True,new_model_training=False,C_quantiles_recomputed_equal=True,all_row_predictions_match=files,seconds=time.perf_counter()-start,not_independent_statistical_validation=True);(out/'end_to_end_replay.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(dict(output=str(out),passed=True,seconds=result['seconds'])),flush=True)
