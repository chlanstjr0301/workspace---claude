import copy, random, time
import numpy as np
import torch
from torch import nn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

SENSORS=['AI0_Vibration','AI1_Vibration','AI2_Current']
STATS=['mean','std','rms','min','max']
FEATURES=[s+'__'+stat for stat in STATS for s in SENSORS]

class LSTMAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1=nn.LSTM(3,64,batch_first=True); self.e2=nn.LSTM(64,32,batch_first=True)
        self.d1=nn.LSTM(32,32,batch_first=True); self.d2=nn.LSTM(32,64,batch_first=True)
        self.out=nn.Linear(64,3)
    def forward(self,x):
        a,_=self.e1(x); _,(h,_)=self.e2(a)
        a=h[-1].unsqueeze(1).repeat(1,x.shape[1],1)
        a,_=self.d1(a); a,_=self.d2(a)
        return self.out(a)

def features(x):
    return np.concatenate([x.mean(1),x.std(1,ddof=0),np.sqrt((x*x).mean(1)),x.min(1),x.max(1)],axis=1)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)

def train_if(x,seed):
    scaler=StandardScaler().fit(x)
    model=IsolationForest(n_estimators=300,max_samples=256,contamination='auto',random_state=seed,n_jobs=2)
    model.fit(scaler.transform(x))
    return dict(kind='IF',scaler=scaler,model=model)

def train_ae(xfit,xearly,scale_rows,seed,config):
    seed_all(seed)
    scaler=MinMaxScaler().fit(np.abs(scale_rows))
    def tr(x): return torch.from_numpy(scaler.transform(np.abs(x).reshape(-1,3)).reshape(x.shape).astype('float32'))
    xf,xe=tr(xfit),tr(xearly)
    model=LSTMAE(); opt=torch.optim.Adam(model.parameters(),lr=config['lr'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=config['lr_factor'],patience=config['lr_patience'])
    best=float('inf'); best_state=None; stale=0; history=[]
    for epoch in range(config['max_epochs']):
        started=time.perf_counter(); model.train(); total=0
        for idx in torch.randperm(len(xf)).split(config['batch_size']):
            x=xf[idx]; opt.zero_grad(); loss=((model(x)-x)**2).mean(); loss.backward(); opt.step(); total+=float(loss.detach())*len(x)
        model.eval()
        with torch.no_grad(): val=sum(float(((model(x)-x)**2).mean())*len(x) for x in xe.split(512))/len(xe)
        history.append(dict(epoch=epoch+1,train_loss=total/len(xf),early_loss=val,seconds=time.perf_counter()-started,lr=opt.param_groups[0]['lr']))
        if val<best-config['min_delta']:
            best=val; best_state=copy.deepcopy(model.state_dict()); stale=0
        else: stale+=1
        scheduler.step(val)
        if stale>=config['early_stopping_patience']: break
    return dict(kind='AE',scaler=scaler,state=best_state,history=history)

def score(bundle,x):
    if bundle['kind']=='IF': return -bundle['model'].score_samples(bundle['scaler'].transform(x))
    model=LSTMAE(); model.load_state_dict(bundle['state']); model.eval()
    a=bundle['scaler'].transform(np.abs(x).reshape(-1,3)).reshape(x.shape).astype('float32')
    with torch.no_grad():
        return np.concatenate([((model(t)[:,-1,:]-t[:,-1,:])**2).mean(1).numpy() for t in torch.from_numpy(a).split(512)])

def norm_fit(x):
    from scipy.special import ndtri
    q=np.array([.001,.005,.01,.025,.05,.1,.25,.5,.75,.9,.95,.975,.99,.995,.999])
    xs=np.quantile(x,q); xs,idx=np.unique(xs,return_index=True); zs=ndtri(q[idx])
    if len(xs)<2: return dict(x=np.array([xs[0]-1,xs[0]+1]),z=np.array([-1.,1.]))
    return dict(x=xs,z=zs)

def norm_apply(obj,x):
    xs,z=obj['x'],obj['z']; v=np.interp(x,xs,z)
    # Strictly increasing piecewise-linear extrapolation retains anomaly-tail ranking.
    lo=x<xs[0]; hi=x>xs[-1]
    v[lo]=z[0]+(x[lo]-xs[0])*(z[1]-z[0])/(xs[1]-xs[0])
    v[hi]=z[-1]+(x[hi]-xs[-1])*(z[-1]-z[-2])/(xs[-1]-xs[-2])
    return v
