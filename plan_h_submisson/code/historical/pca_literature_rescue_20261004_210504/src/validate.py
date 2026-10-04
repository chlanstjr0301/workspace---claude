from auxiliary import *
import sys,subprocess,tempfile
from experiment import context

def validate():
 d,e,base,p=context();lock=json.loads((R/'selected_config.json').read_text());checks={};assert len(p['candidates'])==18;assert lock['selected'] is None or lock['selected'] in lock['configs'];checks['18_registered_slots']=True
 assert sha(R/'selected_config.json')==(R/'selected_config.sha256').read_text().strip();assert sha(R/'preregistration.json')==(R/'preregistration.sha256').read_text().strip();checks['locks_preserved']=True
 inv=json.loads((R/'inventory.json').read_text());preserved=[]
 for item in inv['files']:
  assert sha(R/item['path'])==item['sha256'];old=Path(item['source'])
  if old.exists():assert sha(old)==item['sha256']
  preserved.append(item['path'])
 checks['input_and_existing_source_hashes']=preserved
 assert d.row_id.is_unique;assert d.groupby('burst_id').split.nunique().max()==1
 s1,s2,v=[set(p['blocks'][k]) for k in ['S1','S2','V']];assert not (s1&s2 or s1&v or s2&v);assert s1|s2|v==set(d.index[d.split=='selection']);assert set(d.loc[list(s1),'burst_id']).isdisjoint(d.loc[list(s2),'burst_id']);assert set(d.loc[list(s2),'burst_id']).isdisjoint(d.loc[list(v),'burst_id']);checks['split_and_block_disjoint']=True
 # Audit every eligible window, not only endpoints, for source and role boundary violations.
 for w,aware in [(1,False),(20,False),(2,True),(3,True)]:
  st=e.starts_for(w,aware);ii=np.flatnonzero(st>=0)
  for lag in range(w):
   for col in ['source_file','split','train_role']+(['burst_id'] if aware else []):assert np.array_equal(d[col].to_numpy()[ii],d[col].to_numpy()[ii-lag])
 checks['no_cross_file_split_fit_role_or_future_windows']=True
 trainids=d.index[d.split=='train'].to_numpy();assert (d.loc[trainids,'label']==0).all();fitroles={}
 for key in ['R1','R2','R3','R4','R5']:
  if key=='R1':ids=trainids[e.starts_for(20)[trainids]>=0]
  elif key in ['R2','R3']:ids=d.index[d.row_id.isin(pd.read_csv(R/'manifests/svdd_train_sample.csv').row_id)].to_numpy()
  else:ids=trainids[e.starts_for(2 if key=='R4' else 3,True)[trainids]>=0]
  assert (d.loc[ids,'split']=='train').all() and (d.loc[ids,'label']==0).all();d.loc[ids,['row_id','source_file','source_row','burst_id','train_role']].to_csv(R/'manifests'/f'fit_rows_{key}.csv',index=False);fitroles[key]=len(ids)
 checks['normal_only_fit_row_counts']=fitroles
 # Separate interpreter reconstructs bursts/windows from unlabeled CSV rather than frozen metadata.
 ii=np.array(sorted(s1|s2));b,_=evaluated(e.predict(base,ii),base['threshold']);joined=[]
 with tempfile.TemporaryDirectory(dir=R/'logs') as temp:
  for name,g in d.loc[ii].groupby('source_file',sort=False):
   ip=Path(temp)/name;op=Path(temp)/(name+'.out');g[['row_id','TimeStamp']+SENSORS].to_csv(ip,index=False);z=subprocess.run([sys.executable,str(R/'src/infer_aux.py'),'--csv',str(ip),'--out',str(op)],capture_output=True,text=True);assert z.returncode==0,z.stderr;joined.append(pd.read_csv(op))
  r=pd.concat(joined);assert r.row_id.tolist()==b.row_id.tolist();assert np.allclose(r.base_score,b.score,atol=1e-10,rtol=1e-10);assert np.array_equal(r.base_prediction,b.prediction)
  errors={}
  for key in ['R0','R1','R2','R3','R4','R5']:
   m=joblib.load(R/'models'/f'{key}.joblib');expected=aux_score(m,e,base,ii,b.score);assert np.allclose(r[key],expected,equal_nan=True,atol=1e-10,rtol=1e-10);errors[key]=float(np.nanmax(abs(r[key].to_numpy()-expected)))
 checks['separate_process_unlabeled_inference_max_error']=errors
 # Mutation is exclusively inside S1/S2; no unselected auxiliary ever scored on V/test.
 changed=d.copy();past=[]
 for _,g in d.loc[ii].groupby('source_file',sort=False):
  ids=g.index.to_numpy();cut=len(ids)//2;past.extend(ids[:cut]);changed.loc[ids[cut:],SENSORS]=changed.loc[ids[cut:],SENSORS]*(-13)+777
 ee=Engine(changed);past=np.array(sorted(past));bp=e.predict(base,past);cp=ee.predict(base,past);assert np.allclose(bp.score,cp.score,atol=0,rtol=0)
 for key in ['R0','R1','R2','R3','R4','R5']:
  m=joblib.load(R/'models'/f'{key}.joblib');a=aux_score(m,e,base,past,bp.score);c=aux_score(m,ee,base,past,cp.score);assert np.allclose(a,c,equal_nan=True,atol=0,rtol=0)
 checks['future_sensor_mutation_past_scores_and_decisions_unchanged']=True
 # Labels are never estimator inputs; modifying them does not alter inference.
 dd=d.copy();dd['label']=1-dd.label;ed=Engine(dd);assert np.allclose(e.predict(base,ii).score,ed.predict(base,ii).score)
 for key in ['R0','R1','R2','R3','R4','R5']:
  m=joblib.load(R/'models'/f'{key}.joblib');assert np.allclose(aux_score(m,e,base,ii,b.score),aux_score(m,ed,base,ii,b.score),equal_nan=True)
 checks['label_permutation_no_score_effect']=True
 # Boundary guard: short predictor cannot activate at first L observations of a burst/partition.
 for key,L in [('R4',1),('R5',2)]:
  ss=aux_score(joblib.load(R/'models'/f'{key}.joblib'),e,base,ii,b.score);assert not np.isfinite(ss[d.loc[ii,'burst_pos'].to_numpy()<=L]).any()
 checks['gap_state_reset']=True
 # Calibration order-statistic and ties, including the no-available case.
 tau,td=cal_threshold(np.zeros(5),np.array([3.,3.,2.,1.,np.nan]),.2);assert tau==3 and td['actual_new_FP']==0
 tau,td=cal_threshold(np.zeros(5),np.full(5,np.nan),0);assert np.isinf(tau) and td['actual_new_FP']==0
 checks['strict_greater_ties_and_unavailable_denominator']=True
 validation_rows=[]
 for f in (R/'predictions').glob('*.csv'):
  r=pd.read_csv(f);nm=f.stem
  if 'base_prediction' in r:
   y=r.label.to_numpy();bp=r.base_prediction.to_numpy();pred=r.prediction.to_numpy();assert not ((bp==1)&(pred==0)).any();assert np.array_equal(r.integrated_score>1,pred.astype(bool))
  score=r.integrated_score if 'integrated_score' in r else r.score;mm=metrics(r.label,score,r.prediction);validation_rows.append(dict(file=f.name,rows=len(r),TP=mm['TP'],FN=mm['FN'],FP=mm['FP'],TN=mm['TN']))
 pd.DataFrame(validation_rows).to_csv(R/'tables/verified_confusion_matrices.csv',index=False);checks['OR_no_lost_TP_no_removed_FP_and_score_threshold_agreement']=True
 trials=pd.read_csv(R/'all_trials.csv');assert len(trials)==18;assert set(trials.status)=={'completed'};checks['all_candidates_and_statuses_disclosed']=True
 implementation=json.loads((R/'implementation_lock.json').read_text());assert all(sha(R/'src'/f)==h for f,h in implementation['sources'].items());checks['selection_implementation_unmodified']=True
 checks['models_match_selection_lock']=all(sha(R/'models'/f)==h for f,h in lock['models'].items());assert checks['models_match_selection_lock']
 hist=json.loads((R/'historical_evaluation.json').read_text());assert set(hist['candidates'])<=set(['baseline',lock['selected'],lock['threshold_control']]);checks['restricted_historical_evaluation']=hist['candidates'];checks['bootstrap']='not performed: four anomaly selection bursts and date/state confounding'
 js(R/'validation_report.json',dict(passed=True,checks=checks,created=pd.Timestamp.now(tz='UTC').isoformat()));print('Validation passed')
if __name__=='__main__':validate()
