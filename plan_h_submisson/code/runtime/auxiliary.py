"""Normal-only auxiliary estimators. All inference accepts sensor-only Engine inputs."""
from common import *
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import pdist, cdist
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

def residual(e, base, ids):
 b=base['entries'][20]['bundle'];assert np.all(b['keep']) and len(b['keep'])==15
 z=b['scaler'].transform(e.x(20,np.asarray(ids))[:,b['keep']]);p=b['model']
 r=z-p.inverse_transform(p.transform(z))
 assert np.allclose(r,(z-p.mean_)-(z-p.mean_)@p.components_.T@p.components_,atol=1e-10)
 return r

def covariance(x):
 lw=LedoitWolf(assume_centered=False).fit(x);tr=float(np.trace(lw.covariance_));d=x.shape[1]
 if tr<=0:raise ValueError('zero covariance trace: auxiliary disabled')
 eps=1e-10*tr/d;cov=lw.covariance_+eps*np.eye(d);ev=np.linalg.eigvalsh(cov)
 return dict(mean=lw.location_,cov=cov,factor=cho_factor(cov,lower=True),diagnostics=dict(n=len(x),dimension=d,empirical_rank=int(np.linalg.matrix_rank(x-x.mean(0))),empirical_variance=np.var(x,axis=0),min_eigenvalue=float(ev.min()),max_eigenvalue=float(ev.max()),condition=float(ev.max()/ev.min()),shrinkage=float(lw.shrinkage_),epsilon=eps))

def distance(c,x):
 u=x-c['mean'];v=np.einsum('ij,ij->i',u,cho_solve(c['factor'],u.T).T);assert (v>=-1e-8).all();return np.maximum(v,0)

def lagdata(e,scaler,ids,L):
 ids=np.asarray(ids);valid=e.starts_for(L+1,True)[ids]>=0;ii=ids[valid]
 z=scaler.transform(e.raw);u=np.concatenate([z[ii-j] for j in range(1,L+1)],axis=1)
 return valid,u,z[ii]

def svdd_fit(x,s2):
 m=OneClassSVM(kernel='rbf',gamma=1/s2,nu=.01,tol=1e-7,max_iter=200000,cache_size=128).fit(x)
 if m.fit_status_!=0:raise RuntimeError('SVDD did not converge')
 a=m.dual_coef_.ravel()/(.01*len(x));sv=m.support_vectors_;k=np.exp(-cdist(sv,sv,'sqeuclidean')/s2)
 assert abs(a.sum()-1)<1e-9;assert a.min()>=0 and a.max()<=1/(.01*len(x))+1e-10
 return dict(kind='svdd',model=m,alpha=a,sv=sv,s2=s2,center_norm=float(a@k@a),training_n=len(x),C=1/(.01*len(x)))

def svdd_distance(m,x):
 parts=[]
 for i in range(0,len(x),512):
  k=np.exp(-cdist(x[i:i+512],m['sv'],'sqeuclidean')/m['s2']);parts.append(1-2*k@m['alpha']+m['center_norm'])
 return np.concatenate(parts) if parts else np.array([])

def verify_dual(x,s2):
 x=x[np.linspace(0,len(x)-1,128,dtype=int)];m=svdd_fit(x,s2);k=np.exp(-cdist(x,x,'sqeuclidean')/s2);a0=np.full(len(x),1/len(x))
 result=minimize(lambda a:float(a@k@a),a0,jac=lambda a:2*k@a,bounds=[(0,m['C'])]*len(x),constraints={'type':'eq','fun':lambda a:a.sum()-1,'jac':lambda a:np.ones(len(a))},method='SLSQP',options={'maxiter':2000,'ftol':1e-12})
 aa=np.zeros(len(x));aa[m['model'].support_]=m['alpha'];objdiff=abs(result.fun-aa@k@aa);direct=1-2*k@result.x+result.x@k@result.x;score=svdd_distance(m,x)
 # score_samples = sum(beta_i*K), beta = nu*N*alpha; normalize to obtain exact center distance.
 relation=1-2*m['model'].score_samples(x)/(.01*len(x))+m['center_norm']
 assert result.success and objdiff<=1e-6;assert np.allclose(score,relation,atol=1e-10)
 return dict(normal_subset_n=len(x),QP_success=bool(result.success),objective_difference=float(objdiff),distance_max_difference=float(np.max(abs(score-direct))),monotonic_relation_max_error=float(np.max(abs(score-relation))),alpha_sum=float(m['alpha'].sum()),alpha_max=float(m['alpha'].max()),C=m['C'],passed=True)

def fit_all(e,base,out):
 start=time.perf_counter();ids=e.df.index[e.df.split=='train'].to_numpy();assert (e.df.loc[ids,'label']==0).all()
 wi=ids[e.starts_for(20)[ids]>=0];r=residual(e,base,wi);sample=pd.read_csv(R/'manifests/svdd_train_sample.csv');lookup={v:i for i,v in enumerate(e.df.row_id)};si=np.array([lookup[v] for v in sample.row_id]);sr=residual(e,base,si);s2=float(np.median(pdist(sr,'sqeuclidean')))
 out.mkdir(exist_ok=True);diagnostics={};models={};scaler=StandardScaler().fit(e.raw[ids])
 for key in ['R0','R1','R2','R3','R4','R5']:
  t=time.perf_counter()
  try:
   if key=='R0':m=dict(kind='original')
   elif key=='R1':
    c=covariance(r);m=dict(kind='residual_LW',covariance=c);ss=distance(c,r);q=np.sum(r*r,axis=1);diagnostics[key]=dict(**c['diagnostics'],spearman_Q=float(spearmanr(q,ss).statistic),exact_same_ranking=bool(np.array_equal(np.argsort(q),np.argsort(ss))),raw_Q_residual_check=True)
   elif key in ['R2','R3']:
    if s2<=0:raise ValueError('median pair distance zero: disabled without alternative bandwidth')
    mult=1 if key=='R2' else 2;m=svdd_fit(sr,s2*mult);diagnostics[key]=dict(sample_n=len(sr),s0_squared=s2,s_squared=s2*mult,support_vectors=len(m['alpha']),C=m['C'],alpha_sum=float(m['alpha'].sum()),dual_check=verify_dual(sr,s2*mult))
   else:
    L=1 if key=='R4' else 2;valid,u,z=lagdata(e,scaler,ids,L);um=u.mean(0);zm=z.mean(0);uc=u-um;zc=z-zm;gram=uc.T@uc/len(u);lam=1e-3*np.trace(gram)/u.shape[1]
    if lam<=0:raise ValueError('zero predictor variance')
    coef=np.linalg.solve(gram+lam*np.eye(u.shape[1]),uc.T@zc/len(u));err=z-((u-um)@coef+zm);c=covariance(err);m=dict(kind='ridge',L=L,scaler=scaler,u_mean=um,z_mean=zm,coef=coef,lambda_=float(lam),covariance=c);diagnostics[key]=dict(**c['diagnostics'],lag=L,lambda_=float(lam),fit_n=len(u))
   m['active']=True;m['score_id']=key;models[key]=m;diagnostics.setdefault(key,{}).update(status='completed',fit_seconds=time.perf_counter()-t)
  except Exception as ex:
   import traceback
   (out.parent/'logs'/f'{key}_fit_failure.txt').write_text(traceback.format_exc());models[key]=dict(active=False,score_id=key,reason=str(ex));diagnostics[key]=dict(status='disabled',reason=str(ex),fit_seconds=time.perf_counter()-t)
  joblib.dump(models[key],out/f'{key}.joblib');print(key,diagnostics[key],flush=True)
  assert time.perf_counter()-start<5400,'training budget exhausted'
 js(out.parent/'model_diagnostics.json',diagnostics);return models,diagnostics

def aux_score(m,e,base,ids,base_scores=None):
 ids=np.asarray(ids);s=np.full(len(ids),np.nan)
 if not m['active']:return s
 if m['kind']=='original':return np.asarray(base_scores if base_scores is not None else e.predict(base,ids).score)
 if m['kind'] in ['residual_LW','svdd']:
  valid=e.starts_for(20)[ids]>=0;r=residual(e,base,ids[valid]);s[valid]=distance(m['covariance'],r) if m['kind']=='residual_LW' else svdd_distance(m,r)
 elif m['kind']=='ridge':
  valid,u,z=lagdata(e,m['scaler'],ids,m['L']);err=z-((u-m['u_mean'])@m['coef']+m['z_mean']);s[valid]=distance(m['covariance'],err)
 return s

def cal_threshold(base_pred,aux,a):
 k=allowed_fp(len(aux),a);scores=np.sort(aux[(np.asarray(base_pred)==0)&np.isfinite(aux)])[::-1]
 tau=float(scores[k]) if len(scores)>k else float(np.nextafter(scores[-1],-np.inf)) if len(scores) else float('inf')
 extra=int(((np.asarray(base_pred)==0)&np.isfinite(aux)&(aux>tau)).sum());assert extra<=k
 return tau,dict(normal_denominator=len(aux),allowed_new_FP=k,actual_new_FP=extra,eligible_available_n=len(scores),ties_at_threshold=int((scores==tau).sum()))

def combine(b,aux,tau,score_id,h):
 out=b.copy();out['aux_score']=aux;out['aux_available']=np.isfinite(aux);out['aux_threshold']=tau;out['base_prediction']=b.prediction;out['prediction']=(b.prediction.astype(bool)|(np.isfinite(aux)&(aux>tau))).astype(int)
 if score_id=='R0':out['integrated_score']=b.score/min(h,tau)
 elif 0<tau<np.inf:out['integrated_score']=np.maximum(b.score/h,np.where(np.isfinite(aux),aux/tau,0))
 else:out['integrated_score']=b.score/h
 assert np.array_equal(out.integrated_score>1,out.prediction.astype(bool))
 out['error']=np.where((out.label==1)&(out.prediction==0),'FN',np.where((out.label==0)&(out.prediction==1),'FP','correct'))
 pa=paired(b,out);assert pa['lost_TP']==0 and pa['removed_FP']==0
 return out,dict(**metrics(out.label,out.integrated_score,out.prediction),**pa,aux_available=int(out.aux_available.sum()),aux_available_normal=int(out.loc[out.label==0,'aux_available'].sum()),aux_available_anomaly=int(out.loc[out.label==1,'aux_available'].sum()))
