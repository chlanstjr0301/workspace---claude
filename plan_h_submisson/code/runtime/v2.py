"""Fixed B0 + fixed lag-2 R5; causal alarm gates and separated effect/maintenance criteria."""
from auxiliary import *
from collections import deque
TOL=1e-12

def judge(base,cand,mode='maintenance'):
 assert mode in ['maintenance','effect'];n0=int(base['TN']+base['FP']);n1=int(base['TP']+base['FN']);assert n0==cand['TN']+cand['FP'] and n1==cand['TP']+cand['FN']
 rules=dict(FP_budget=int(cand['FP'])-int(base['FP'])<=allowed_fp(n0,.001),absolute_FPR=int(cand['FP'])<=allowed_fp(n0,.01))
 fn_maint=bool(cand['FN']<=base['FN']) if n1 else None
 reduction=bool(cand['FN']<base['FN']) if n1 and base['FN']>0 else 'not_assessable'
 if n1:
  rules.update(FN=bool(cand['FN']<base['FN']) if mode=='effect' else fn_maint,F2=bool(cand['F2']>base['F2']+TOL) if mode=='effect' else bool(cand['F2']>=base['F2']-TOL),F1=bool(cand['F1']>=base['F1']-TOL))
 passed=all(rules.values());return dict(passes=passed,mode=mode,stability_pass=passed if mode=='maintenance' else None,effect_pass=passed if mode=='effect' else None,fn_maintenance_pass=fn_maint,fn_reduction_reproduced=reduction,anomaly_metrics_assessable=n1>0,allowed_additional_FP=allowed_fp(n0,.001),absolute_FP_cap=allowed_fp(n0,.01),failed_rules=';'.join(k for k,v in rules.items() if not v))

def shifted(values,frame,lag):
 out=np.zeros(len(frame));v=np.asarray(values)
 for _,positions in frame.groupby(['source_file','split','train_role','burst_id'],sort=False,dropna=False).indices.items():
  pp=np.asarray(positions)
  if len(pp)>lag:out[pp[lag:]]=v[pp[:-lag]]
 return out

def gate(frame,s,theta,rule):
 ratio=np.where(np.isfinite(s),np.asarray(s)/theta,0.);h=ratio>1;p1=shifted(ratio,frame,1);p2=shifted(ratio,frame,2)
 if rule=='G0':g=ratio
 elif rule=='G1':g=np.minimum(ratio,p1)
 elif rule=='G2':g=np.minimum(ratio,np.maximum(p1,p2))
 else:raise ValueError(rule)
 return g>1,g,h,p1>1,p2>1

def generate(e,b0,r5,ids,theta,rule):
 b,_=evaluated(e.predict(b0,ids),b0['threshold']);s=aux_score(r5,e,b0,ids,b.score);extra,g,h,hm1,hm2=gate(b,s,theta,rule);out=b.copy();out['B0_prediction']=b.prediction;out['R5_score']=s;out['R5_available']=np.isfinite(s);out['aux_threshold']=theta;out['h']=h;out['h_previous1']=hm1;out['h_previous2']=hm2;out['aux_alarm']=extra;out['prediction']=(b.prediction.astype(bool)|extra).astype(int);out['integrated_score']=np.maximum(b.score/b0['threshold'],g);out['rule']=rule;out['alarm_time']=out.TimeStamp;out['error']=np.where((out.label==1)&(out.prediction==0),'FN',np.where((out.label==0)&(out.prediction==1),'FP','correct'));assert np.array_equal(out.prediction.astype(bool),out.integrated_score>1);assert not ((b.prediction==1)&(out.prediction==0)).any();return out

def measure(r):return metrics(r.label,r.integrated_score if 'integrated_score' in r else r.score,r.prediction)
def base_predictions(e,b0,ids):
 r,_=evaluated(e.predict(b0,ids),b0['threshold']);r['B0_prediction']=r.prediction;r['integrated_score']=r.score/b0['threshold'];return r

def pair_counts(a,b):
 assert a.row_id.tolist()==b.row_id.tolist();y=a.label.to_numpy();x=a.prediction.to_numpy();z=b.prediction.to_numpy();return dict(new_TP=int(((y==1)&(x==0)&(z==1)).sum()),new_FN=int(((y==1)&(x==1)&(z==0)).sum()),additional_FP=int(((y==0)&(x==0)&(z==1)).sum()),removed_FP=int(((y==0)&(x==1)&(z==0)).sum()))

def context():
 d=loadrows(R/'inputs/frozen_rows.csv');return d,Engine(d),joblib.load(R/'models/baseline.joblib'),joblib.load(R/'models/R5.joblib')

def online_scores(frame,b0,r5,configs):
 """Independent single-row state machine; features use only observations received so far.
 Missing raw values are unavailable to R5 and reset its lag/h state; supplied dataset has no missing values.
 """
 records=[];main=deque(maxlen=20);past=deque(maxlen=2);hs={c['id']:deque(maxlen=2) for c in configs};lastpartition=None;lasttime=None
 for row in frame.itertuples(index=False):
  partition=(row.source_file,row.split,row.train_role);newpartition=partition!=lastpartition;gap=newpartition or (row.TimeStamp-lasttime).total_seconds()>.5
  if newpartition:main.clear()
  if gap:
   past.clear()
   for a in hs.values():a.clear()
  raw=np.array([getattr(row,s) for s in SENSORS],float);main.append(raw);w=20 if len(main)>=20 else 1;entry=b0['entries'][w];bu=entry['bundle'];x=features(np.array(main)[None,:,:])[0] if w==20 else raw;z=(x[bu['keep']]-bu['scaler'].mean_)/bu['scaler'].scale_;pc=bu['model'];center=z-pc.mean_;t=center@pc.components_.T;res=center-t@pc.components_;bs=float(np.sum(res**2)/entry['reference']);bp=bs>b0['threshold'];aux=np.nan
  if len(past)==2 and np.isfinite(raw).all():
   zs=(raw-r5['scaler'].mean_)/r5['scaler'].scale_;u=np.r_[(past[-1]-r5['scaler'].mean_)/r5['scaler'].scale_,(past[-2]-r5['scaler'].mean_)/r5['scaler'].scale_];prediction=(u-r5['u_mean'])@r5['coef']+r5['z_mean'];aux=float(distance(r5['covariance'],(zs-prediction)[None,:])[0])
  out=dict(row_id=row.row_id,base_score=bs,base_prediction=int(bp),R5_score=aux)
  for c in configs:
   h=bool(np.isfinite(aux) and aux>c['theta']);history=hs[c['id']];p1=history[-1] if len(history) else False;p2=history[-2] if len(history)>1 else False;extra=h if c['rule']=='G0' else h and p1 if c['rule']=='G1' else h and (p1 or p2);out[c['id']]=int(bp or extra);history.append(h)
  records.append(out)
  if np.isfinite(raw).all():past.append(raw)
  else:past.clear()
  lastpartition=partition;lasttime=row.TimeStamp
 return pd.DataFrame(records)
