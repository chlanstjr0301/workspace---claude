"""CPU PCA/IF; input arrays contain sensor features only."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from scipy.optimize import minimize
from scipy.special import expit

SENSORS=['AI0_Vibration','AI1_Vibration','AI2_Current']
STATS=['mean','std','rms','min','max']
FEATURES=[s+'__'+stat for stat in STATS for s in SENSORS]

def features(x):
    return np.concatenate([x.mean(1),x.std(1,ddof=0),np.sqrt((x*x).mean(1)),x.min(1),x.max(1)],axis=1)

def fit_pca(x,k,names):
    keep=x.std(0)>1e-12*np.maximum(1,np.abs(x.mean(0)))
    if not keep.any(): raise ValueError('all constant features')
    scaler=StandardScaler().fit(x[:,keep]); z=scaler.transform(x[:,keep])
    spectrum=PCA(svd_solver='full',whiten=False).fit(z)
    tolerance=max(1e-12,float(spectrum.explained_variance_[0])*1e-10)
    rank=int((spectrum.explained_variance_>tolerance).sum())
    if k>rank: raise ValueError(f'k={k} exceeds numerical rank={rank}')
    model=PCA(n_components=k,svd_solver='full',whiten=False).fit(z)
    assert np.all(model.explained_variance_>tolerance)
    return dict(kind='PCA',scaler=scaler,model=model,keep=keep,k=k,rank=rank,tolerance=tolerance,
                names=list(names),removed=[n for n,yes in zip(names,keep) if not yes],
                eigenvalues=spectrum.explained_variance_.tolist(),cumulative_variance=np.cumsum(spectrum.explained_variance_ratio_).tolist(),q_valid=k<rank)

def fit_if(x,seed,names):
    scaler=StandardScaler().fit(x)
    model=IsolationForest(n_estimators=300,max_samples=256,random_state=seed,contamination='auto',n_jobs=2).fit(scaler.transform(x))
    return dict(kind='IF',scaler=scaler,model=model,names=list(names),seed=seed)

def score(bundle,x):
    if bundle['kind']=='IF': return {'IF':-bundle['model'].score_samples(bundle['scaler'].transform(x))}
    z=bundle['scaler'].transform(x[:,bundle['keep']]); t=bundle['model'].transform(z)
    residual=(z-bundle['model'].inverse_transform(t))**2
    q=residual.sum(1); t2=(t*t/bundle['model'].explained_variance_).sum(1)
    assert np.isfinite(t2).all() and np.all(q>=0) and np.all(t2>=0)
    return {'Q':q,'T2':t2,'contributions':residual}

def reference(x):
    value=float(np.quantile(x,.95,method='higher'))
    if not np.isfinite(value) or value<=1e-12: raise ValueError('normal calibration Q95 reference <= 1e-12')
    return value

def fit_sigmoid(s,y):
    x=np.log1p(s); mean=float(x.mean()); std=max(float(x.std()),1e-12); z=(x-mean)/std
    def objective(ab):
        h=ab[0]*z+ab[1]
        return float(np.mean(np.logaddexp(0,h)-y*h)+1e-4*ab[0]**2)
    opt=minimize(objective,[1.,float(np.log((y.mean()+1e-8)/(1-y.mean()+1e-8)))],method='L-BFGS-B',bounds=[(0,None),(None,None)])
    if not opt.success: raise RuntimeError('sigmoid optimizer: '+opt.message)
    return dict(a=float(opt.x[0]),b=float(opt.x[1]),mean=mean,std=std,regularization=1e-4,success=bool(opt.success),input='standardized log1p(score)')

def probability(obj,s):
    return expit(obj['a']*((np.log1p(s)-obj['mean'])/obj['std'])+obj['b'])
