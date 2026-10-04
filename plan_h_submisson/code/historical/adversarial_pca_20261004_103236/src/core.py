"""Causal, normal-only PCA/IF systems; all policies are serialized, no labels at inference."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import json,hashlib,time
import numpy as np,pandas as pd,joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.ensemble import IsolationForest
from threadpoolctl import threadpool_limits
from models import SENSORS,features
from evaluation import metrics,threshold,postprocess,alarm_summary

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,o):
 Path(p).parent.mkdir(parents=True,exist_ok=True)
 Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist() if isinstance(x,np.ndarray) else str(x)))
def loadrows(p):
 d=pd.read_csv(p,keep_default_na=False);d.TimeStamp=pd.to_datetime(d.TimeStamp);return d

def stat_features(a,mode):
 if a.shape[1]==1:return a[:,0,:],list(SENSORS)
 if mode=='full':return features(a),[s+'__'+t for t in ['mean','std','rms','min','max'] for s in SENSORS]
 if mode=='compact':stats=[a.mean(1),a.std(1),a.min(1),a.max(1)];names=['mean','std','min','max']
 elif mode=='meanstd':stats=[a.mean(1),a.std(1)];names=['mean','std']
 elif mode=='medianiqr':stats=[np.median(a,axis=1),np.quantile(a,.75,axis=1)-np.quantile(a,.25,axis=1)];names=['median','iqr']
 elif mode=='delta':stats=[a.mean(1),a.std(1),(a[:,-1]-a[:,0])/(a.shape[1]-1)];names=['mean','std','mean_change']
 else:raise ValueError(mode)
 return np.concatenate(stats,axis=1),[s+'__'+t for t in names for s in SENSORS]

class Engine:
 def __init__(self,df,out,partition='review_partition'):
  self.df=df.copy();self.out=Path(out);self.raw=df[SENSORS].to_numpy(float);self.partition=partition;self.starts={};self.cache={};self.fitlog=[];self.t0=time.monotonic()
 def window(self,w,aware):
  key=(w,aware)
  if key in self.starts:return self.starts[key]
  st=np.full(len(self.df),-1,int)
  cols=['source_file',self.partition,'train_role']+(['burst_id'] if aware else [])
  for _,g in self.df.groupby(cols,sort=False,dropna=False):
   idx=g.index.to_numpy()
   if len(idx)>=w:st[idx[w-1:]]=idx[:-w+1] if w>1 else idx
  end=np.flatnonzero(st>=0);assert np.all(end-st[end]==w-1)
  for c in cols:assert np.array_equal(self.df[c].to_numpy()[end],self.df[c].to_numpy()[st[end]])
  self.starts[key]=st;return st
 def x(self,ids,w,aware,feat):
  st=self.window(w,aware)[ids];assert (st>=0).all();return stat_features(self.raw[st[:,None]+np.arange(w)],feat)
 def model(self,w,aware,feat,scaler,k,kind='PCA',seed=42):
  key=f'{kind}_w{w}_b{int(aware)}_{feat}_{scaler}_k{k}_s{seed if kind=="IF" else 0}'
  if key in self.cache:return self.cache[key]
  if time.monotonic()-self.t0>7200:raise TimeoutError('2-hour training budget')
  ids=np.flatnonzero((self.df.split=='train').to_numpy()&(self.window(w,aware)>=0));assert not self.df.loc[ids,'label'].any();assert len(ids)>=100
  t=time.perf_counter();x,names=self.x(ids,w,aware,feat);keep=x.std(0)>1e-12*np.maximum(1,np.abs(x.mean(0)))
  scale=(StandardScaler() if scaler=='standard' else RobustScaler()).fit(x[:,keep]);z=scale.transform(x[:,keep]);spec=PCA(svd_solver='full',whiten=False).fit(z);tol=max(1e-12,spec.explained_variance_[0]*1e-10);rank=int((spec.explained_variance_>tol).sum())
  with threadpool_limits(2):
   if kind=='PCA':
    assert k<rank,(key,k,rank);m=PCA(n_components=k,svd_solver='full',whiten=False).fit(z);assert (m.explained_variance_>tol).all()
   else:m=IsolationForest(n_estimators=300,max_samples=256,contamination='auto',random_state=seed,n_jobs=2).fit(z)
  b=dict(key=key,kind=kind,w=w,aware=aware,feat=feat,scaler_name=scaler,k=k,keep=keep,scaler=scale,model=m,rank=rank,tolerance=tol,names=names,train_rows=self.df.loc[ids,'row_id'].tolist(),eigenvalues=spec.explained_variance_,explained_variance_ratio=spec.explained_variance_ratio_,seconds=time.perf_counter()-t,seed=seed if kind=='IF' else None)
  self.cache[key]=b;joblib.dump(b,self.out/'models'/f'{key}.joblib',compress=3)
  self.fitlog.append({a:b[a] for a in ['key','rank','tolerance','seconds','seed']}|dict(n_train=len(ids),removed=[n for n,v in zip(names,keep) if not v],min_eigenvalue=float(spec.explained_variance_[-1]),condition=float(spec.explained_variance_[0]/max(spec.explained_variance_[-1],1e-300))))
  pd.DataFrame(self.fitlog).to_csv(self.out/'tables/model_fit_log.csv',index=False);return b
 def rawscore(self,b,ids):
  x,_=self.x(ids,b['w'],b['aware'],b['feat']);z=b['scaler'].transform(x[:,b['keep']])
  if b['kind']=='IF':return {'IF':-b['model'].score_samples(z)}
  t=b['model'].transform(z);r=(z-b['model'].inverse_transform(t))**2
  return dict(Q=r.sum(1),T2=(t*t/b['model'].explained_variance_).sum(1),contributions=r)
 def routes(self,spec,ids):
  route=np.ones(len(ids),int);st1=self.window(1,False)
  if spec['mode']=='route':
   for w in [3,5,10,20]:route[self.window(w,True)[ids]>=0]=w
  elif spec['mode']!='raw':
   st=self.window(20,spec['mode']=='aware');ok=st[ids]>=0
   if spec['mode']=='cap':
    elapsed=np.full(len(ids),np.inf);ii=ids[ok];elapsed[ok]=(self.df.TimeStamp.to_numpy()[ii]-self.df.TimeStamp.to_numpy()[st[ii]])/np.timedelta64(1,'s');ok&=elapsed<=2.5
   route[ok]=20
  return route
 def fit_system(self,spec,cal,seed=None):
  assert set(self.df.loc[cal,'label'])=={0};assert not (self.df.loc[cal,'split']=='test').any()
  seeds=[seed] if seed is not None else ([42,43,44] if spec['kind']=='IF' else [None]);members=[]
  routes=self.routes(spec,np.asarray(cal));cal=np.asarray(cal)
  for sd in seeds:
   member={}
   for w in ([1,3,5,10,20] if spec['mode']=='route' else [1,20] if spec['mode']!='raw' else [1]):
    aware=(spec['mode'] in ['route','aware']) and w>1
    # Delta differences confined to bursts, the delta candidate uses mode aware.
    b=self.model(w,aware,spec['feat'] if w>1 else 'full',spec['scaler'] if w>1 else 'standard',spec['k'] if w>1 else 2,spec['kind'],sd or 42)
    avail=cal[self.window(w,aware)[cal]>=0];routed=cal[routes==w]
    use_route=spec['mode']=='route' and len(routed)>=50
    pool=routed if use_route else avail
    if len(pool)<50:raise ValueError(f'normal calibration <50 for {b["key"]}')
    raw=self.rawscore(b,pool);scoretype=spec['score'] if w>1 else ('IF' if spec['kind']=='IF' else 'Q')
    refs={s:float(np.quantile(raw[s],.95,method='higher')) for s in raw if s!='contributions'}
    assert min(refs.values())>1e-12
    a=np.maximum(raw['Q']/refs['Q'],raw['T2']/refs['T2']) if scoretype=='QT' else raw[scoretype]
    ref=float(np.quantile(a,.95,method='higher'));assert ref>1e-12
    member[w]=dict(bundle=b,scoretype=scoretype,refs=refs,reference=ref,ecdf=np.sort(a) if spec['align']=='ecdf' else None,pool_n=len(pool),routed_n=len(routed),pool_policy='routed' if use_route else 'all_available_calibration',pool_ids=self.df.loc[pool,'row_id'].tolist())
   members.append(member)
  system=dict(spec=spec,members=members,cal_ids=self.df.loc[cal,'row_id'].tolist())
  scores=self.predict(system,cal)['score'];system['thresholds']={str(t):threshold(scores,t) for t in [.001,.005,.01]};system['calibration_fpr']={str(t):float(np.mean(scores>h)) for t,h in system['thresholds'].items()};return system
 def predict(self,system,ids):
  ids=np.asarray(ids);s=system['spec'];route=self.routes(s,ids);out=[];q=np.full(len(ids),np.nan);tt=q.copy();qt=q.copy();contrib=np.full((len(ids),3),np.nan);starts=np.empty(len(ids),int)
  for member in system['members']:
   scores=np.full(len(ids),np.nan)
   for w,m in member.items():
    loc=np.flatnonzero(route==w)
    if not len(loc):continue
    ii=ids[loc];b=m['bundle'];raw=self.rawscore(b,ii);typ=m['scoretype'];starts[loc]=self.window(w,b['aware'])[ii]
    a=np.maximum(raw['Q']/m['refs']['Q'],raw['T2']/m['refs']['T2']) if typ=='QT' else raw[typ]
    if m['ecdf'] is not None:
     # Right-continuous smoothed empirical CDF: ties share percentile, tails saturate; not probability.
     scores[loc]=(np.searchsorted(m['ecdf'],a,side='right')+.5)/(len(m['ecdf'])+1)
    else:scores[loc]=a/m['reference']
    if b['kind']=='PCA':
     q[loc]=raw['Q'];tt[loc]=raw['T2'];qt[loc]=np.maximum(raw['Q']/m['refs']['Q'],raw['T2']/m['refs']['T2']);names=np.array(b['names'])[b['keep']]
     for j,sensor in enumerate(SENSORS):contrib[loc,j]=raw['contributions'][:,[n.startswith(sensor) for n in names]].sum(1)
   assert np.isfinite(scores).all();out.append(scores)
  r=self.df.loc[ids].copy();r['score']=np.mean(out,axis=0);r['route']=route;r['fallback']=route==1;r['native_available']=route==20 if s['mode']!='raw' else True;r['window_rows']=route;r['window_start']=self.df.TimeStamp.to_numpy()[starts];r['window_end']=r.TimeStamp;r['elapsed_seconds']=(r.TimeStamp.to_numpy()-r.window_start.to_numpy())/np.timedelta64(1,'s');r['crosses_gap']=self.df.burst_id.to_numpy()[ids]!=self.df.burst_id.to_numpy()[starts];r['used_model']=s['id']+':W'+r.route.astype(str);r['Q']=q;r['T2']=tt;r['QT']=qt
  for j,sensor in enumerate(SENSORS):r[sensor+'__Q_contribution']=contrib[:,j]
  return r

def evaluate(r,th,policy='none'):
 r=r.copy();r['threshold']=th;r['prediction_raw']=(r.score>th).astype(int);r['prediction']=postprocess(r,r.prediction_raw,policy);r['policy']=policy;r['error']=np.where((r.label==1)&(r.prediction==0),'FN',np.where((r.label==0)&(r.prediction==1),'FP','correct'));return r,metrics(r.label,r.score,r.prediction)
