"""Meaningful invariants on synthetic inputs; never reads final evaluation scores."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,pandas as pd
from models import features,fit_pca,score
from evaluation import metrics,threshold,postprocess
rng=np.random.default_rng(42);x=rng.normal(size=(300,3));x[:,2]=x[:,0]+.3*x[:,1]
b=fit_pca(x,1,['a','b','c']);assert b['rank']==2 and b['q_valid']
b2=fit_pca(x,2,['a','b','c']);assert not b2['q_valid']
assert np.max(score(b2,x)['Q'])<1e-20
x2=np.c_[x,np.ones(len(x))];b3=fit_pca(x2,1,['a','b','c','constant']);assert b3['removed']==['constant']
a=features(np.array([[[1.,2.,3.],[3.,4.,5.]]]))
assert np.allclose(a[0,:3],[2,3,4]) and np.allclose(a[0,3:6],1)
th=threshold(np.ones(100),.01);assert not np.any(np.ones(100)>th)
frame=pd.DataFrame({'burst_id':['a','a','b','b']});p=np.array([1,1,1,1])
assert postprocess(frame,p,'consecutive2').tolist()==[0,1,0,1]
r=metrics([0,0,1,1],[.1,.8,.7,.9],[0,1,0,1]);assert (r['TP'],r['FN'],r['FP'],r['TN'])==(1,1,1,1)
assert r['AP']!=r['PR_AUC_trapezoid']
print('PASS: rank/Q validity, constants, feature ddof, thresholds/ties, burst reset, confusion/AP separation')
