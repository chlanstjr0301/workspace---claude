import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, average_precision_score, roc_auc_score, auc

def metrics(y,s,p):
    y=np.asarray(y); p=np.asarray(p).astype(int); s=np.asarray(s)
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    div=lambda a,b: float(a/b) if b else float('nan')
    precision=div(tp,tp+fp) if tp+fp else 0.
    recall=div(tp,tp+fn)
    if len(np.unique(y))==2:
        pr,rc,_=precision_recall_curve(y,s); ap=average_precision_score(y,s); pr_auc=auc(rc,pr); roc=roc_auc_score(y,s)
    else: ap=pr_auc=roc=float('nan')
    return dict(TP=int(tp),FN=int(fn),FP=int(fp),TN=int(tn),N=len(y),positive_rate=float(np.mean(y)),Recall=recall,Precision=precision,F1=div(2*tp,2*tp+fp+fn) if 2*tp+fp+fn else 0.,F2=div(5*tp,5*tp+4*fn+fp) if 5*tp+4*fn+fp else 0.,FPR=div(fp,fp+tn),FNR=div(fn,tp+fn),AP=float(ap),PR_AUC_trapezoid=float(pr_auc),ROC_AUC=float(roc),Accuracy=div(tp+tn,len(y)),FP_per_1000=1000*div(fp,fp+tn))

def threshold(normal_scores,target): return float(np.quantile(normal_scores,1-target,method='higher'))

def postprocess(frame,p,policy):
    p=np.asarray(p,dtype=int); out=np.zeros(len(p),dtype=int)
    for positions in frame.groupby('burst_id',sort=False).indices.values():
        hist=[]
        for i in positions:
            hist.append(p[i])
            out[i]=p[i] if policy=='none' else (len(hist)>=2 and hist[-1] and hist[-2]) if policy=='consecutive2' else (sum(hist[-3:])>=2)
    return out

def alarm_summary(frame,p):
    work=frame.copy(); work['alarm']=np.asarray(p); records=[]
    for bid,g in work.groupby('burst_id',sort=False):
        a=g.alarm.to_numpy(); rising=a.astype(bool)&~np.r_[False,a[:-1].astype(bool)]
        detected=np.flatnonzero(a)
        records.append(dict(burst_id=bid,label=int(g.label.iloc[0]),rows=len(g),alarm_rows=int(a.sum()),alarm_episodes=int(rising.sum()),duplicate_episodes=max(0,int(rising.sum())-1),no_alarm=not len(detected),first_alarm_since_observed_burst_start_seconds=float((g.TimeStamp.iloc[detected[0]]-g.TimeStamp.iloc[0]).total_seconds()) if len(detected) else np.nan))
    return pd.DataFrame(records)
