"""Causal inference on one new file; no labels used, no refitting."""
import argparse,json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from models import SENSORS,features,score,probability
from evaluation import postprocess

def infer(run,path,policy='none'):
    run=Path(run);lock=json.loads((run/'selection_lock.json').read_text());meta=json.loads((run/'models/score_metadata.json').read_text())
    df=pd.read_csv(path);df['source_row']=np.arange(1,len(df)+1);df.TimeStamp=pd.to_datetime(df.TimeStamp,errors='raise')
    # For live observations with no Equipment_state, sensor/time duplicate policy is used.
    duplicate_keys=['TimeStamp']+SENSORS+(['Equipment_state'] if 'Equipment_state' in df else [])
    df=df.drop_duplicates(duplicate_keys,keep='first').reset_index(drop=True)
    gap=df.TimeStamp.diff().dt.total_seconds();assert (gap.dropna()>0).all()
    df['burst_id']=(gap.isna()|(gap>.5)).cumsum();raw=df[SENSORS].to_numpy(float);assert np.isfinite(raw).all()
    def apply(key):
        m=meta[key];b=joblib.load(run/'models'/f'{m["base"]}.joblib');w=m['window'];f=m['feature'];starts=np.full(len(df),-1,int)
        groups=[g.index.to_numpy() for _,g in df.groupby('burst_id')] if f.startswith('B') else [np.arange(len(df))]
        for ids in groups:
            if len(ids)>=w:starts[ids[w-1:]]=ids[:len(ids)-w+1]
        ends=np.flatnonzero(starts>=0);x=raw[starts[ends,None]+np.arange(w)]
        x=x[:,0,:] if w==1 else features(x)
        result=score(b,x);norm=json.loads((run/'models'/f'norm_{m["base"]}.json').read_text())
        result['QT']=np.maximum(result['Q']/norm['Q'],result['T2']/norm['T2'])
        out=np.full(len(df),np.nan);out[ends]=result[m['score']]/m['reference']
        return out,starts
    key=lock['primary']['key'];a,st=apply(key);fb,unused=apply(lock['fallback']);use=~np.isfinite(a);a[use]=fb[use];st[use]=np.flatnonzero(use)
    out=df[['source_row','TimeStamp']+SENSORS+['burst_id']].copy();out['score']=a;out['threshold']=lock['primary']['threshold'];out['fallback']=use;out['used_model']=np.where(use,lock['fallback'],key)
    out['window_start']=df.TimeStamp.to_numpy()[st];out['elapsed_seconds']=(df.TimeStamp.to_numpy()-out.window_start.to_numpy())/np.timedelta64(1,'s')
    p=a>lock['primary']['threshold'];out['prediction_raw']=p.astype(int);out['prediction']=postprocess(df,p,policy)
    out['probability']=probability(joblib.load(run/'models/sigmoid.joblib'),a)
    return out
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--csv',required=True);p.add_argument('--out',required=True);p.add_argument('--postprocess',choices=['none','consecutive2','two_of_three'],default='none');a=p.parse_args()
    out=Path(a.out)
    if out.exists():raise SystemExit('Refuse to overwrite output')
    infer(a.run,a.csv,a.postprocess).to_csv(out,index=False)
