"""Generate descriptive analyses and plots without changing any model decision."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import precision_recall_curve,roc_curve,confusion_matrix
from models import SENSORS
from evaluation import metrics,postprocess,alarm_summary

ROOT=Path(__file__).resolve().parents[1]

def save(fig,path):
    fig.tight_layout(); fig.savefig(path,dpi=150,bbox_inches='tight'); plt.close(fig)

def error_conditions(out,rows,split):
    p=pd.read_csv(out/'predictions'/f'main_{split}.csv',parse_dates=['TimeStamp'])
    train=rows[rows.split=='train'].copy()
    train['current_change']=train.groupby('burst_id').AI2_Current.diff().abs()
    p['current_change']=p.groupby('burst_id').AI2_Current.diff().abs()
    w=int(json.loads((out/'selection_lock.json').read_text())['main']['key'].split('_L')[1])
    conditions={'burst_start_first_9_rows':np.where(p.burst_pos<=9,'start','later'), 'short_burst_under_10_rows':np.where(p.burst_length<10,'short','long'), 'fallback':p.fallback.astype(str)}
    bounds={}
    for sensor in SENSORS+['current_change']:
        vals=train[sensor].abs(); edges=np.unique(vals.dropna().quantile([.25,.5,.75]).to_numpy()); bounds[sensor]=edges.tolist()
        bins=np.searchsorted(edges,p[sensor].abs().fillna(-1).to_numpy(),side='right')+1
        labels=np.array(['Q'+str(b) for b in bins],dtype=object); labels[p[sensor].isna()]='unknown_at_burst_start'
        conditions[sensor+'_absolute_train_quartile']=labels
    h0=p.AI0_Vibration.abs()>bounds['AI0_Vibration'][-1]; hc=p.AI2_Current.abs()>bounds['AI2_Current'][-1]
    conditions['AI0_abs_high_x_current_abs_high']=['vibration_'+('high' if a else 'low')+'_current_'+('high' if b else 'low') for a,b in zip(h0,hc)]
    records=[]
    for name,values in conditions.items():
        for group in sorted(set(values)):
            sub=p[np.asarray(values)==group]; n0=int(sum(sub.label==0)); n1=int(sum(sub.label==1)); fp=int(sum((sub.label==0)&(sub.prediction==1)));fn=int(sum((sub.label==1)&(sub.prediction==0)))
            records.append(dict(condition=name,group=group,normal_n=n0,anomaly_n=n1,FP=fp,FN=fn,FPR=fp/n0 if n0 else np.nan,FNR=fn/n1 if n1 else np.nan))
    pd.DataFrame(records).to_csv(out/'tables'/f'error_conditions_{split}.csv',index=False)
    (out/'tables/error_bin_boundaries.json').write_text(json.dumps(bounds,indent=2))
    # Descriptive sensor correlations cannot establish physical causes.
    corr=[]
    for (part,y),g in rows[rows.split.isin(['train','calibration','selection'])].groupby(['split','label']):
        for kind in ['signed','absolute']:
            x=g[SENSORS].abs() if kind=='absolute' else g[SENSORS]
            c=x.corr()
            for i in range(3):
                for j in range(i+1,3):corr.append(dict(split=part,label=y,kind=kind,sensor1=SENSORS[i],sensor2=SENSORS[j],correlation=c.iloc[i,j],n=len(g)))
    pd.DataFrame(corr).to_csv(out/'tables/development_correlations.csv',index=False)

def plots(out,rows,split):
    lock=json.loads((out/'selection_lock.json').read_text()); c=lock['main']
    p=pd.read_csv(out/'predictions'/f'main_{split}.csv',parse_dates=['TimeStamp'])
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for m in ['M0','M1','M2','M3','M4']:
        key=f'{m}_L'+('1' if m=='M2' else '10')
        a=pd.read_csv(out/'predictions'/f'all_{key}_sensemble_{split}.csv.gz')
        common=pd.read_csv(out/'manifests'/f'common_{split}.csv').row_id
        a=a[a.row_id.isin(common)]
        pr,rc,_=precision_recall_curve(a.label,a.native_score); fpr,tpr,_=roc_curve(a.label,a.native_score)
        ax[0].plot(rc,pr,label=key); ax[1].plot(fpr,tpr,label=key)
        pd.DataFrame({'recall':rc,'precision':pr}).to_csv(out/'tables'/f'curve_PR_{key}_{split}.csv',index=False)
        pd.DataFrame({'FPR':fpr,'TPR':tpr}).to_csv(out/'tables'/f'curve_ROC_{key}_{split}.csv',index=False)
    ax[0].set(xlabel='Recall',ylabel='Precision',title=f'{split}: default L10, common rows');ax[1].set(xlabel='FPR',ylabel='TPR',title='ROC: continuous scores')
    for a in ax:a.legend(fontsize=8);a.grid(alpha=.2)
    save(fig,out/'figures'/f'curves_{split}.png')
    fig,ax=plt.subplots(figsize=(4,4)); cm=confusion_matrix(p.label,p.prediction,labels=[0,1]); ax.imshow(cm,cmap='Blues')
    for i in range(2):
        for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=18,color='red')
    ax.set(xticks=[0,1],yticks=[0,1],xlabel='Prediction (1=anomaly)',ylabel='True label',title=f'Main {split}');save(fig,out/'figures'/f'confusion_{split}.png')
    for label,g in p.groupby('label'):
        fig,ax=plt.subplots(5,1,figsize=(15,10),sharex=True)
        # Draw each observed burst separately. Never bridge unobserved gaps.
        for _,b in g.groupby('burst_id',sort=False):
            for j,s in enumerate(SENSORS):ax[j].plot(b.TimeStamp,b[s],'.-',ms=2,lw=.6,color='tab:blue' if label==0 else 'tab:orange')
            ax[3].plot(b.TimeStamp,b.score,'.-',ms=2,lw=.7,color='tab:purple');ax[4].step(b.TimeStamp,b.prediction,where='post',lw=1,color='tab:red')
        for j,s in enumerate(SENSORS):ax[j].set_ylabel(s)
        ax[3].axhline(c['threshold'],color='red',ls='--');ax[3].set_ylabel('Normalized score');ax[4].set_ylabel('Alarm');ax[4].set_ylim(-.1,1.1)
        ax[0].set_title(f'{split} | '+('normal 2022-07-12' if label==0 else 'anomaly 2022-07-17')+f" | {c['key']} + M2 fallback | {c['policy']}")
        ax[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'));save(fig,out/'figures'/f'timeline_{split}_label{label}.png')
    # Reliability: label counts per bin accompany the curve.
    fig,ax=plt.subplots(figsize=(5,5)); rel=[]
    for role in ['main','f1_alternative']:
        a=pd.read_csv(out/'predictions'/f'probability_{role}_{split}.csv'); a['bin']=pd.cut(a.probability,np.linspace(0,1,11),include_lowest=True)
        r=a.groupby('bin',observed=True).agg(n=('label','size'),normal_n=('label',lambda x:int(sum(x==0))),anomaly_n=('label','sum'),mean_probability=('probability','mean'),observed_fraction=('label','mean')).reset_index();r['role']=role;rel.append(r)
        ax.plot(r.mean_probability,r.observed_fraction,'o-',label=role)
    ax.plot([0,1],[0,1],'k--');ax.set(xlabel='Mean calibrated probability',ylabel='Observed anomaly fraction',title=f'Reliability: {split}');ax.legend();save(fig,out/'figures'/f'reliability_{split}.png')
    pd.concat(rel).to_csv(out/'tables'/f'reliability_bins_{split}.csv',index=False)
    # All three postprocessing choices at the selected model/threshold, with censored cases explicit.
    delays=[]; base=p.raw_prediction.to_numpy(); reference=alarm_summary(p,base).set_index('burst_id')
    for policy in ['none','consecutive2','two_of_three']:
        alarms=alarm_summary(p,postprocess(p,base,policy)); alarms['policy']=policy
        alarms['raw_first_alarm_seconds']=alarms.burst_id.map(reference.first_alarm_since_observed_burst_start_seconds)
        alarms['additional_observed_delay_seconds']=alarms.first_alarm_since_observed_burst_start_seconds-alarms.raw_first_alarm_seconds
        alarms['lost_observed_burst_alarm']=alarms.no_alarm & alarms.raw_first_alarm_seconds.notna()
        delays.append(alarms)
    pd.concat(delays).to_csv(out/'tables'/f'postprocessing_delays_{split}.csv',index=False)
    losses=[]
    for policy in ['none','consecutive2','two_of_three']:
        pred=postprocess(p,base,policy);y=p.label.to_numpy()
        losses.append(dict(policy=policy,newly_missed_anomaly_rows=int(sum((y==1)&(base==1)&(pred==0))),newly_detected_anomaly_rows=int(sum((y==1)&(base==0)&(pred==1))),suppressed_false_positive_rows=int(sum((y==0)&(base==1)&(pred==0))),new_false_positive_rows=int(sum((y==0)&(base==0)&(pred==1)))))
    pd.DataFrame(losses).to_csv(out/'tables'/f'postprocessing_row_changes_{split}.csv',index=False)
    calibration_metrics=[]
    for role in ['main','f1_alternative']:
        a=pd.read_csv(out/'predictions'/f'probability_{role}_{split}.csv')
        for version in ['raw','probability']:
            calibration_metrics.append(dict(role=role,version=version,**metrics(a.label,a.score if version=='raw' else a.probability,a[version+'_prediction'])))
    pd.DataFrame(calibration_metrics).to_csv(out/'tables'/f'probability_classification_{split}.csv',index=False)
    # Explicit FP/FN extracts for every base model/seed at the preregistered 1% target.
    for path in (out/'predictions').glob(f'all_*_{split}.csv.gz'):
        a=pd.read_csv(path); pred=a['prediction_0.01']
        a['error_type']=np.where((a.label==0)&(pred==1),'FP',np.where((a.label==1)&(pred==0),'FN','correct'))
        a[a.error_type!='correct'].to_csv(out/'predictions'/path.name.replace('all_','errors_target01_',1),index=False)

def general(out,rows):
    bursts=pd.read_csv(ROOT/'configs/frozen/bursts.csv'); gaps=pd.read_csv(ROOT/'configs/frozen/time_gaps.csv')
    summary=[]
    for (label,split),g in bursts.groupby(['label','split']):
        for w in [5,10,20]:summary.append(dict(label=label,split=split,window=w,bursts=len(g),short_bursts=int(sum(g.rows<w)),short_burst_fraction=float(np.mean(g.rows<w)),short_burst_rows=int(g.loc[g.rows<w,'rows'].sum()),observed_duration_seconds=float(g.duration_seconds.sum()),wall_span_seconds=(pd.to_datetime(g.end).max()-pd.to_datetime(g.start).min()).total_seconds()))
    pd.DataFrame(summary).to_csv(out/'tables/burst_summary.csv',index=False)
    bursts.groupby(['label','split'])[['rows','duration_seconds']].describe().to_csv(out/'tables/burst_distributions.csv')
    gaps.groupby('file').gap_seconds.describe(percentiles=[.5,.9,.95,.99]).to_csv(out/'tables/gap_distribution.csv')
    pd.concat([g.nlargest(10,'gap_seconds') for _,g in gaps.groupby('file')]).to_csv(out/'tables/largest_time_jumps.csv',index=False)
    rows.groupby('label')[SENSORS].agg(['min','median','max','std']).to_csv(out/'tables/sensor_descriptives_deduplicated.csv')
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for label,g in bursts.groupby('label'):ax[0].hist(g.rows,bins=25,alpha=.5,label=f'label={label}')
    for file,g in gaps.groupby('file'):ax[1].hist(g.gap_seconds[g.gap_seconds>.5],bins=25,alpha=.5,label=file)
    ax[0].set(xlabel='Rows per observed burst',ylabel='Count');ax[1].set(xlabel='Gap >0.5 s',ylabel='Count')
    for a in ax:a.legend(fontsize=8)
    save(fig,out/'figures/burst_gap_distribution.png')
    # Sensor scatter uses selection only, and states the observed labels/dates.
    g=rows[rows.split=='selection']; fig,ax=plt.subplots(1,3,figsize=(14,4))
    for a,(s1,s2) in zip(ax,[(SENSORS[0],SENSORS[1]),(SENSORS[0],SENSORS[2]),(SENSORS[1],SENSORS[2])]):
        for label,b in g.groupby('label'):a.scatter(b[s1],b[s2],s=5,alpha=.4,label=f'label={label}')
        a.set(xlabel=s1,ylabel=s2,title='Selection split');a.legend()
    save(fig,out/'figures/sensor_interactions_selection.png')
    for label in [0,1]:
        fig,ax=plt.subplots(3,1,figsize=(14,7),sharex=True)
        colors={'train':'tab:blue','calibration':'tab:green','selection':'tab:orange','test':'tab:red'}
        for (split,bid),g in rows[rows.label==label].groupby(['split','burst_id'],sort=False):
            for j,s in enumerate(SENSORS):ax[j].plot(g.TimeStamp,g[s],'.-',lw=.4,ms=1,color=colors[split])
        for split,color in colors.items():ax[0].plot([],[],color=color,label=split)
        ax[0].legend();ax[0].set_title(f'Observed data: label={label}, date={rows[rows.label==label].TimeStamp.iloc[0].date()}')
        for j,s in enumerate(SENSORS):ax[j].set_ylabel(s)
        ax[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'));save(fig,out/'figures'/f'splits_sensors_label{label}.png')
    logs=[]
    for f in (out/'logs').glob('score_*.json'):
        r=json.loads(f.read_text());r['file']=f.name;logs.append(r)
    pd.DataFrame(logs).to_csv(out/'tables/computation_time.csv',index=False)
    # Warmup is elapsed timestamp between first and Lth row, not L * sampling period.
    warm=[]
    for (split,bid),g in rows.groupby(['split','burst_id'],sort=False):
        for w in [5,10,20]:
            warm.append(dict(split=split,burst_id=bid,label=int(g.label.iloc[0]),rows=len(g),window=w,window_possible=len(g)>=w,warmup_seconds=(g.TimeStamp.iloc[w-1]-g.TimeStamp.iloc[0]).total_seconds() if len(g)>=w else np.nan))
    pd.DataFrame(warm).to_csv(out/'tables/window_warmup.csv',index=False)
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for m in ['M0','M1']:
        for seed in [42,43,44]:
            g=pd.read_csv(out/'logs'/f'history_{m}_L10_s{seed}.csv');ax[0].plot(g.epoch,g.train_loss,label=f'{m} s{seed}');ax[1].plot(g.epoch,g.early_loss,label=f'{m} s{seed}')
    ax[0].set(title='Normal internal fit MSE',xlabel='Epoch');ax[1].set(title='Normal internal early-stop MSE',xlabel='Epoch')
    for a in ax:a.legend(fontsize=7)
    save(fig,out/'figures/lstm_learning_curves.png')

def run(out):
    out=Path(out); rows=pd.read_csv(ROOT/'configs/frozen/rows.csv',parse_dates=['TimeStamp'])
    general(out,rows)
    for split in ['selection','test']:
        if (out/'predictions'/f'main_{split}.csv').exists():error_conditions(out,rows,split);plots(out,rows,split)
    print('Analyses and figures generated',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);run(p.parse_args().out)
