"""Post-evaluation artifact checks/descriptive summaries. Never selects or fits."""
import argparse,json,hashlib,sys,tempfile
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from infer import infer
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=Path(a.run)
lock=json.loads((run/'selection_lock.json').read_text());df=pd.read_csv(run/'manifests/clean_rows.csv');key=lock['primary']['key'];checks={}
# Verify portable inference against already frozen final predictions, separately by source/partition.
record=pd.read_csv(run/'predictions'/f'operational_{key}_target0p01_test.csv')
with tempfile.TemporaryDirectory(dir=run,prefix='inference_check_') as tmp:
 for file,g in df[df.split=='test'].groupby('source_file',sort=False):
  csvpath=Path(tmp)/file;g[['TimeStamp','AI0_Vibration','AI1_Vibration','AI2_Current']].to_csv(csvpath,index=False)
  out=infer(run,csvpath);expected=record[record.source_file==file]
  assert np.allclose(out.score,expected.score,rtol=1e-10,atol=1e-10)
  assert np.array_equal(out.prediction,expected.prediction)
  assert np.array_equal(out.fallback,expected.fallback)
  checks[file]={'rows':len(g),'max_score_difference':float(np.max(np.abs(out.score.to_numpy()-expected.score.to_numpy()))),'predictions_identical':True,'no_labels_in_input':True}
# Add explicit denominators in derivative copies, preserving all original tables.
for sp in ['calibration','selection','test']:
 r=pd.read_csv(run/'tables'/f'metrics_{sp}.csv');r['normal']=r.FP+r.TN;r['anomaly']=r.TP+r.FN
 r.to_csv(run/'tables'/f'metrics_with_counts_{sp}.csv',index=False)
 # Verify common and operational scores have the exact same row order for every candidate.
 nat=pd.read_csv(run/'predictions'/f'all_native_{sp}.csv.gz');op=pd.read_csv(run/'predictions'/f'all_operational_{sp}.csv.gz')
 assert nat.row_id.equals(op.row_id)
 assert not op.drop(columns='row_id').isna().any().any()
 meta=json.loads((run/'models/score_metadata.json').read_text())
 routes={'row_id':op.row_id}
 for k in meta:
  missing=nat[k].isna();m=meta[k]
  fb=lock['fallback'] if k.startswith('P') else 'I0_W1_ensemble' if str(m['seed'])=='ensemble' else f'I0_W1_S{m["seed"]}'
  routes[k]=np.where(missing,fb,k)
  if k.startswith('P'):
   expect=np.where(missing,nat[fb],nat[k]);assert np.allclose(expect,op[k],rtol=1e-10,atol=1e-10)
 pd.DataFrame(routes).to_csv(run/'predictions'/f'all_routes_{sp}.csv.gz',index=False)
checks['all_pca_operational_scores_exact_Q95_fallback']=True
checks['all_models_have_full_operational_coverage']=True
checks['all_candidate_rowwise_routes_saved']=True
# Timing summaries separate window preparation/elapsed observation time from computation.
times=[]
for f in (run/'logs').glob('score_*.json'):
 d=json.loads(f.read_text());times.append({'artifact':f.name,**d,'microseconds_per_row':d['seconds']/d['rows']*1e6})
pd.DataFrame(times).to_csv(run/'tables/score_runtime.csv',index=False)
# Pair first-alarm delay changes; missing detections stay explicit.
for sp in ['selection','test']:
 b=pd.read_csv(run/'tables'/f'burst_alarms_{sp}.csv');base=b[b.policy=='none'][['burst_id','first_alarm_since_observed_burst_start_seconds']].rename(columns={'first_alarm_since_observed_burst_start_seconds':'raw_first_alarm_seconds'})
 out=b.merge(base,on='burst_id',validate='many_to_one');out['added_observed_delay_seconds']=out.first_alarm_since_observed_burst_start_seconds-out.raw_first_alarm_seconds
 out['raw_detected_now_missed']=out.raw_first_alarm_seconds.notna()&out.first_alarm_since_observed_burst_start_seconds.isna()
 out.to_csv(run/'tables'/f'postprocess_paired_delays_{sp}.csv',index=False)
(run/'manifests/final_artifact_validation.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
# Export each candidate's FP/FN at the prespecified primary calibration target.
# The chosen alternative threshold already has its own detailed prediction file.
meta=json.loads((run/'models/score_metadata.json').read_text());basecache={}
for sp in ['selection','test']:
 ids=df.index[df.split==sp].to_numpy();frame=df.loc[ids].reset_index(drop=True)
 op=pd.read_csv(run/'predictions'/f'all_operational_{sp}.csv.gz');nat=pd.read_csv(run/'predictions'/f'all_native_{sp}.csv.gz')
 r=pd.read_csv(run/'tables'/f'metrics_{sp}.csv');r=r[(r.scope=='operational')&(r.target_fpr==.01)].set_index('key')
 out=[]
 for key,m in meta.items():
  s=op[key].to_numpy();pred=s>r.loc[key,'threshold'];bad=pred!=frame.label.to_numpy();idx=np.flatnonzero(bad)
  if not len(idx):continue
  a=frame.iloc[idx].copy();a['candidate']=key;a['score']=s[idx];a['threshold']=r.loc[key,'threshold'];a['prediction']=pred[idx].astype(int);a['error']=np.where(pred[idx],'FP','FN')
  fallback=nat[key].isna().to_numpy()[idx];fb=lock['fallback'] if key.startswith('P') else 'I0_W1_ensemble' if m['seed']=='ensemble' else f'I0_W1_S{m["seed"]}'
  a['fallback']=fallback;a['used_model']=np.where(fallback,fb,key)
  w=m['window'];start=ids[idx]-w+1;start[fallback]=ids[idx][fallback]
  a['window_start']=df.TimeStamp.to_numpy()[start];a['window_end']=a.TimeStamp;a['window_rows']=ids[idx]-start+1
  a['elapsed_seconds']=(pd.to_datetime(a.TimeStamp).to_numpy()-pd.to_datetime(a.window_start).to_numpy())/np.timedelta64(1,'s')
  for col in ['Q','T2','QT']:a[col]=np.nan
  if key.startswith('P'):
   for usekey,mask in [(key,~fallback),(fb,fallback)]:
    if not mask.any():continue
    base=meta[usekey]['base'];cachekey=(base,sp)
    if cachekey not in basecache:
     raw=dict(np.load(run/'predictions'/f'raw_{base}_{"test" if sp=="test" else "develop"}.npz'));norm=json.loads((run/'models'/f'norm_{base}.json').read_text())
     if 'Q' in norm and 'T2' in norm:raw['QT']=np.maximum(raw['Q']/norm['Q'],raw['T2']/norm['T2'])
     basecache[cachekey]=raw
    for col in ['Q','T2','QT']:
     if col in basecache[cachekey]:a.loc[a.index[mask],col]=basecache[cachekey][col][ids[idx][mask]]
  out.append(a)
 pd.concat(out,ignore_index=True).to_csv(run/'predictions'/f'all_candidate_errors_{sp}.csv.gz',index=False)
print('All candidate FP/FN details exported at target FPR .01')
from evaluation import postprocess,metrics
for sp in ['selection','test']:
 frame=pd.read_csv(run/'predictions'/f'operational_{lock["primary"]["key"]}_target0p01_{sp}.csv')
 w=lock['primary']['window'];records=[]
 for policy in ['none','consecutive2','two_of_three']:
  pp=postprocess(frame,frame.prediction.to_numpy(),policy)
  for name,mask in [('short_burst',frame.burst_length<w),('long_burst',frame.burst_length>=w),('start',frame.burst_pos<w),('later',frame.burst_pos>=w)]:
   if not mask.any():continue
   records.append(dict(policy=policy,condition=name,normal=int((frame.loc[mask,'label']==0).sum()),anomaly=int((frame.loc[mask,'label']==1).sum()),**metrics(frame.loc[mask,'label'],frame.loc[mask,'score'],pp[np.asarray(mask)])))
 pd.DataFrame(records).to_csv(run/'tables'/f'postprocess_conditions_{sp}.csv',index=False)
print('Postprocessing conditional denominators saved')
