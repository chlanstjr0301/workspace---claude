import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='2'
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import json,hashlib,time,math
from decimal import Decimal,ROUND_FLOOR
from functools import cmp_to_key
import numpy as np,pandas as pd,joblib
from models import SENSORS,FEATURES,features,fit_pca,score,reference
from evaluation import metrics,threshold,alarm_summary
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(o):
 if isinstance(o,dict):return {str(k):clean(v) for k,v in o.items()}
 if isinstance(o,(list,tuple)):return [clean(v) for v in o]
 if isinstance(o,np.ndarray):return clean(o.tolist())
 if isinstance(o,np.generic):return clean(o.item())
 if isinstance(o,float) and not np.isfinite(o):return None
 if isinstance(o,Path):return str(o)
 return o
def js(p,o):Path(p).write_text(json.dumps(clean(o),ensure_ascii=False,indent=2,allow_nan=False))
def loadrows(p):
 d=pd.read_csv(p,keep_default_na=False);d.TimeStamp=pd.to_datetime(d.TimeStamp);return d

def allowed_fp(n,delta):return int((Decimal(str(delta))*Decimal(int(n))).to_integral_value(rounding=ROUND_FLOOR))
def compare(base,candidate,delta,tol=1e-12):
 n=int(base['TN']+base['FP']);extra=allowed_fp(n,delta);limit=min(int(base['FP'])+extra,allowed_fp(n,.01));rules=dict(recall_increased=candidate['Recall']>base['Recall']+tol,F2_increased=candidate['F2']>base['F2']+tol,F1_not_worse=candidate['F1']>=base['F1']-tol,incremental_FP_limit=int(candidate['FP'])<=int(base['FP'])+extra,absolute_FPR_limit=int(candidate['FP'])<=allowed_fp(n,.01))
 return dict(eligible=all(rules.values()),allowed_additional_FP=extra,maximum_FP=limit,normal_n=n,**rules)
def pick(rows):
 def cmp(a,b):
  for key,direction in [('F2',-1),('FPR',1),('Recall',-1),('F1',-1),('complexity',1)]:
   dif=a[key]-b[key]
   if abs(dif)>1e-12:return (-1 if dif<0 else 1)*direction
  return (a['candidate']>b['candidate'])-(a['candidate']<b['candidate'])
 return sorted(rows,key=cmp_to_key(cmp))[0] if rows else None

class Engine:
 def __init__(self,df):self.df=df;self.raw=df[SENSORS].to_numpy(float);self.starts={}
 def starts_for(self,w,aware=False):
  key=(w,aware)
  if key in self.starts:return self.starts[key]
  st=np.full(len(self.df),-1,int);cols=['source_file','split','train_role']+(['burst_id'] if aware else [])
  for _,g in self.df.groupby(cols,sort=False,dropna=False):
   ids=g.index.to_numpy()
   if len(ids)>=w:st[ids[w-1:]]=ids[:len(ids)-w+1]
  ii=np.flatnonzero(st>=0);assert (ii-st[ii]==w-1).all()
  for c in cols:assert np.array_equal(self.df[c].to_numpy()[ii],self.df[c].to_numpy()[st[ii]])
  self.starts[key]=st;return st
 def x(self,w,idx):
  st=self.starts_for(w)[idx];assert (st>=0).all();a=self.raw[st[:,None]+np.arange(w)];return a[:,0,:] if w==1 else features(a)
 def predict(self,system,idx):
  idx=np.asarray(idx);routes=np.ones(len(idx),int);routes[self.starts_for(20)[idx]>=0]=20
  if system.get('short_window'):
   w=system['short_window'];routes[(routes==1)&(self.starts_for(w)[idx]>=0)]=w
  s=np.empty(len(idx));q=s.copy();t2=s.copy();starts=np.empty(len(idx),int);contrib=np.zeros((len(idx),3))
  for w,entry in system['entries'].items():
   loc=np.flatnonzero(routes==w)
   if not len(loc):continue
   ids=idx[loc];b=entry['bundle'];v=score(b,self.x(w,ids));q[loc]=v['Q'];t2[loc]=v['T2'];s[loc]=v['Q']/entry['reference'];starts[loc]=self.starts_for(w)[ids];names=np.array(b['names'])[b['keep']]
   for j,sensor in enumerate(SENSORS):contrib[loc,j]=v['contributions'][:,[n.startswith(sensor) for n in names]].sum(1)
  out=self.df.loc[idx].copy();out['score']=s;out['Q']=q;out['T2']=t2;out['route']=routes;out['used_model']=['P1_W20_K2_Q' if w==20 else 'P0_W1_K2_Q' if w==1 else f'P1_W{w}_K2_Q' for w in routes];out['fallback']=routes==1;out['short_complement']=np.isin(routes,[3,5]);out['native_W20']=routes==20;out['window_rows']=routes;out['window_start']=self.df.TimeStamp.to_numpy()[starts];out['window_end']=out.TimeStamp;out['elapsed_seconds']=(out.TimeStamp.to_numpy()-out.window_start.to_numpy())/np.timedelta64(1,'s');out['crosses_gap']=self.df.burst_id.to_numpy()[idx]!=self.df.burst_id.to_numpy()[starts]
  for j,sensor in enumerate(SENSORS):out[sensor+'__Q_contribution']=contrib[:,j]
  assert np.isfinite(s).all();return out

def evaluated(r,h):
 r=r.copy();r['threshold']=h;r['prediction']=(r.score>h).astype(int);r['error']=np.where((r.label==1)&(r.prediction==0),'FN',np.where((r.label==0)&(r.prediction==1),'FP','correct'));return r,metrics(r.label,r.score,r.prediction)
def paired(base,r):
 assert base.row_id.tolist()==r.row_id.tolist();y=r.label.to_numpy();bp=base.prediction.to_numpy();p=r.prediction.to_numpy();return dict(new_TP=int(((y==1)&(bp==0)&(p==1)).sum()),lost_TP=int(((y==1)&(bp==1)&(p==0)).sum()),removed_FP=int(((y==0)&(bp==1)&(p==0)).sum()),new_FP=int(((y==0)&(bp==0)&(p==1)).sum()),delta_TP=int(((y==1)&(p==1)).sum()-((y==1)&(bp==1)).sum()),decreased_FN=int(((y==1)&(bp==0)).sum()-((y==1)&(p==0)).sum()),delta_FP=int(((y==0)&(p==1)).sum()-((y==0)&(bp==1)).sum()))
