"""Causal offline inference for ONE chronological stream, no label input to models."""
import argparse,json,time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from models import SENSORS,features,score,norm_apply,seed_all
from evaluation import postprocess

def infer(frame,run,role='main'):
    run=Path(run); lock=json.loads((run/'selection_lock.json').read_text()); c=lock[role]; key=c['key']; m=key.split('_')[0]; w=int(key.split('_L')[1])
    frame=frame.copy().reset_index(drop=True); frame.TimeStamp=pd.to_datetime(frame.TimeStamp,errors='raise')
    a=frame[SENSORS].apply(pd.to_numeric,errors='raise').to_numpy(dtype=float)
    if not np.isfinite(a).all(): raise ValueError('Missing/nonfinite sensors: cannot infer; no imputation permitted')
    gaps=frame.TimeStamp.diff().dt.total_seconds()
    if (gaps.iloc[1:]<=0).any(): raise ValueError('Input must be deduplicated and strictly chronological')
    frame['burst_id']=(gaps.isna()|(gaps>.5)).cumsum().astype(str)
    frame['data_gap_before']=gaps>.5
    groups=frame.groupby('burst_id',sort=False).indices.values() if m in ['M1','M4'] else [np.arange(len(frame))]
    starts=np.full(len(frame),-1,dtype=int)
    for ids in groups:
        if len(ids)>=w: starts[ids[w-1:]]=ids[:len(ids)-w+1]
    ids=np.flatnonzero(starts>=0); warm=starts<0; ensemble=[]
    seed_all(42); tic=time.perf_counter()
    for seed in [42,43,44]:
        b2=joblib.load(run/'models'/f'M2_L1_s{seed}.joblib'); norm2=joblib.load(run/'models'/f'norm_M2_L1_s{seed}.joblib')
        s=norm_apply(norm2,score(b2,a))
        if key!='M2_L1':
            bundle=joblib.load(run/'models'/f'{key}_s{seed}.joblib'); norm=joblib.load(run/'models'/f'norm_{key}_s{seed}.joblib')
            if len(ids):
                x=a[starts[ids,None]+np.arange(w)];x=features(x) if m in ['M3','M4'] else x
                s[ids]=norm_apply(norm,score(bundle,x))
        ensemble.append(s)
    frame['score']=np.mean(ensemble,axis=0);frame['threshold']=c['threshold'];frame['raw_prediction']=(frame.score>c['threshold']).astype(int)
    frame['prediction']=postprocess(frame,frame.raw_prediction,c['policy']);frame['actual_model']=np.where(warm,'M2_L1',key)
    effective=np.where(warm,np.arange(len(frame)),starts)
    frame['window_start_time']=frame.TimeStamp.to_numpy()[effective];frame['window_rows']=np.where(warm,1,w)
    frame['window_elapsed_seconds']=(frame.TimeStamp-frame.window_start_time).dt.total_seconds()
    obj=joblib.load(run/'models'/f'probability_{role}.joblib')
    frame['calibrated_probability_experimental']=expit(obj['a']*(frame.score-obj['center'])/obj['scale']+obj['b'])
    return frame,dict(rows=len(frame),compute_seconds=time.perf_counter()-tic,model=key,policy=c['policy'],data_gaps=int(frame.data_gap_before.sum()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    target=Path(args.output)
    if target.exists():raise SystemExit('Refuse to overwrite existing output')
    df=pd.read_csv(args.input); subset=['TimeStamp']+SENSORS+(['Equipment_state'] if 'Equipment_state' in df else [])
    df=df.drop_duplicates(subset=subset,keep='first')
    result,info=infer(df,args.run);result.to_csv(target,index=False);print(json.dumps(info))
