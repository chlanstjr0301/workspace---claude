"""Reproduce manuscript statistics and figures from saved OOF predictions.

No classifier fitting, hyperparameter/feature selection or threshold tuning.
StandardScaler.fit recovers original-fold statistics for centroid diagnostics.
"""
import hashlib
import json
import os
import platform
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.special import logit, expit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (auc, precision_recall_curve, average_precision_score, precision_score,
    recall_score, f1_score, balanced_accuracy_score, roc_auc_score)

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
WEIGHTS=['None','{0:1,1:3}','{0:1,1:5}','{0:1,1:10}','balanced']
FEATURES=['AI0_RMS','AI0_MaxAbs','AI1_RMS','AI1_PeakToPeak','AI1_Kurtosis']
EPS=1e-15


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def save(frame,name): frame.to_csv(OUT/'tables'/f'{name}.csv',index=False,encoding='utf-8-sig')


def tex_table(frame,name,caption,label,fmt=None):
    # Numeric values are serialized directly from the same audited dataframes.
    text=frame.to_latex(index=False,escape=False,float_format=fmt or (lambda v:f'{v:.4f}'))
    (OUT/'tables'/f'{name}.tex').write_text('\\begin{table*}[t]\n\\centering\n\\caption{'+caption+'}\n\\label{'+label+'}\n\\small\n'+text+'\\end{table*}\n',encoding='utf-8')


def main():
    for folder in ['figures','tables','source_results']: (OUT/folder).mkdir(parents=True,exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(OUT/'.matplotlib_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,
        'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    def figsave(fig,name):
        fig.tight_layout(); fig.savefig(OUT/'figures'/f'{name}.png',dpi=300,bbox_inches='tight')
        fig.savefig(OUT/'figures'/f'{name}.pdf',bbox_inches='tight')
        fig.savefig(OUT/'figures'/f'{name}.svg',bbox_inches='tight'); plt.close(fig)
    cw=ROOT/'results/logistic_class_weight_ablation'; length=ROOT/'results/logistic_burst_sample_count'
    inputs=[cw/'predictions.csv',cw/'cv_fold_metrics.csv',cw/'metrics.csv',cw/'cv_split_indices.json',cw/'experiment_metadata.json',
        cw/'feature_dataset.csv',cw/'effective_class_weights.csv',
        ROOT/'results/logistic_current_acf_ablation/burst_feature_dataset.csv',
        ROOT/'results/logistic_fn_burst_tracking/fold_coefficients.csv',ROOT/'results/logistic_fn_burst_tracking/validation_predictions.csv',
        ROOT/'results/logistic_fn_feature_analysis/recovered_fold_intercepts.csv',ROOT/'results/logistic_fn_feature_analysis/abnormal_fold_distances.csv',
        length/'metrics_by_sample_count.csv',length/'eligible_bursts_by_sample_count.csv',length/'tp_to_fn_by_sample_count.csv',
        length/'tp_to_fn_summary.csv',length/'feature_drift_by_sample_count.csv',length/'burst_619_620_predictions.csv',length/'experiment_metadata.json',
        ROOT/'reference/소성가공.ipynb',ROOT/'data/raw/press_data_normal.csv',ROOT/'data/raw/outlier_data.csv',
        ROOT/'src/LSTM-AutoEncoder/baseline_original.py',ROOT/'src/LSTM-AutoEncoder/baseline_original.sha256']
    manifest=[]
    for source in inputs:
        relative=source.relative_to(ROOT); target=OUT/'source_results'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            # Preserve exact copies (including the copied immutable file's read-only attribute).
            assert sha(source)==sha(target), 'Existing provenance copy differs; use a fresh output directory.'
        else:
            shutil.copy2(source,target)
        assert sha(source)==sha(target)
        manifest.append(dict(source=str(relative),copy=str(target.relative_to(OUT)),sha256=sha(source),bytes=source.stat().st_size))
    (OUT/'source_results/manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    baseline=ROOT/'src/LSTM-AutoEncoder/baseline_original.py'
    assert sha(baseline)==baseline.with_suffix('.sha256').read_text().split()[0].lower()=='3b0855d29c0e47e9856d3c293083d7f430ebe7cbe6f24ef85f247348e504a0a8'
    meta=json.loads((cw/'experiment_metadata.json').read_text(encoding='utf-8'))
    assert meta['features']==FEATURES and meta['decision_threshold']==.5 and meta['split_reused'] and meta['baseline_reproduced']
    assert meta['CV']==dict(n_splits=5,n_repeats=10,random_state=42,std_ddof=0)
    clean=pd.read_csv(cw/'feature_dataset.csv',index_col='notebook_feature_index')
    fullfeatures=pd.read_csv(inputs[7],index_col='notebook_feature_index')
    pd.testing.assert_frame_equal(clean,fullfeatures[clean.columns])
    assert clean.Equipment_state.value_counts().to_dict()=={0:575,1:20}
    folds=json.loads((cw/'cv_split_indices.json').read_text(encoding='utf-8'))
    tracking_folds=json.loads((ROOT/'results/logistic_fn_burst_tracking/cv_split_indices.json').read_text(encoding='utf-8'))
    assert all(a['train_feature_indices']==b['train_feature_indices'] and a['test_feature_indices']==b['test_feature_indices'] for a,b in zip(folds,tracking_folds)) and len(folds)==len(tracking_folds)==50
    pred=pd.read_csv(cw/'predictions.csv',keep_default_na=False); pred=pred.loc[pred.weight.isin(WEIGHTS)].copy()
    assert set(pred.weight)==set(WEIGHTS)
    np.testing.assert_array_equal(pred.prediction,(pred.probability>.5).astype(int))
    for weight,g in pred.groupby('weight'):
        assert len(g)==5950 and g.groupby('burst_id').size().eq(10).all()
        for i,fold in enumerate(folds,start=1):
            assert set(g.loc[g.fold.eq(i),'burst_id'])==set(clean.loc[fold['test_feature_indices'],'burst_id'])
    # PR-AUC is trapezoidal integration of precision_recall_curve; AP is separate.
    saved_fold=pd.read_csv(cw/'cv_fold_metrics.csv',keep_default_na=False).set_index(['weight','fold'])
    saved_means=pd.read_csv(cw/'metrics.csv',keep_default_na=False).set_index('weight')
    fold_rows=[]
    for (weight,fold),g in pred.groupby(['weight','fold'],sort=False):
        y,p,h=g.label.to_numpy(),g.probability.to_numpy(),g.prediction.to_numpy(); precision,recall,_=precision_recall_curve(y,p)
        row=dict(weight=weight,fold=fold,Precision=precision_score(y,h,zero_division=0),Recall=recall_score(y,h),F1=f1_score(y,h),
            Balanced_Accuracy=balanced_accuracy_score(y,h),ROC_AUC=roc_auc_score(y,p),PR_AUC=auc(recall,precision),AP=average_precision_score(y,p),
            FP=int(((y==0)&(h==1)).sum()),FN=int(((y==1)&(h==0)).sum()))
        for field in ['Precision','Recall','F1','Balanced_Accuracy','ROC_AUC','FP','FN']:
            assert np.isclose(row[field],saved_fold.loc[(weight,fold),field],atol=1e-12)
        fold_rows.append(row)
    fold_metrics=pd.DataFrame(fold_rows); save(fold_metrics,'class_weight_fold_metrics')
    metric_rows=[]
    for weight in WEIGHTS:
        g=fold_metrics.loc[fold_metrics.weight.eq(weight)]; row={'weight':weight}
        for field in ['Precision','Recall','F1','Balanced_Accuracy','ROC_AUC','PR_AUC','AP','FP','FN']:
            row[field+'_mean']=g[field].mean(); row[field+'_std']=g[field].std(ddof=0)
            if field not in ['PR_AUC','AP']:
                assert np.isclose(row[field+'_mean'],saved_means.loc[weight,field+'_mean'],atol=1e-12)
                assert np.isclose(row[field+'_std'],saved_means.loc[weight,field+'_std'],atol=1e-12)
        metric_rows.append(row)
    metrics=pd.DataFrame(metric_rows); save(metrics,'class_weights')
    display=pd.DataFrame({'Weight':['None','1:3','1:5','1:10','Balanced']})
    for col,title in [('Precision','P'),('Recall','R'),('F1','F1'),('Balanced_Accuracy','BA'),('ROC_AUC','ROC-AUC'),('PR_AUC','PR-AUC')]:
        display[title]=[f'${r[col+"_mean"]:.3f}\\pm{r[col+"_std"]:.3f}$' for _,r in metrics.iterrows()]
    display['FP/fold']=[f'${r.FP_mean:.2f}\\pm{r.FP_std:.2f}$' for _,r in metrics.iterrows()]
    display['FN/fold']=[f'${r.FN_mean:.2f}\\pm{r.FN_std:.2f}$' for _,r in metrics.iterrows()]
    tex_table(display,'class_weights','Class-weight comparison on identical 50 folds, mean $\\pm$ population standard deviation (ddof 0). P/R are precision/recall; BA is balanced accuracy. PR-AUC uses trapezoidal integration, not average precision. FP/FN are counts per validation fold.','tab:weights')
    pred['repeat']=(pred.fold-1)//5+1; pred['group']=np.where(pred.label.eq(0),'Normal',np.where(pred.burst_id.isin([619,620]),'FN abnormal','Detected abnormal'))
    pred['score_clipped']=((pred.probability<EPS)|(pred.probability>1-EPS)).astype(int)
    pred['score']=logit(np.clip(pred.probability.to_numpy(),EPS,1-EPS))
    saturation=pred.groupby('weight').agg(exact_zero=('probability',lambda x:int(x.eq(0).sum())),exact_one=('probability',lambda x:int(x.eq(1).sum())),clipped_count=('score_clipped','sum')).reset_index()
    save(saturation,'score_clipping')
    none=pred.loc[pred.weight.eq('None')].copy(); balanced=pred.loc[pred.weight.eq('balanced')]
    matched=none.merge(balanced,on=['fold','burst_id','label','repeat','group'],suffixes=('_none','_balanced'),validate='one_to_one')
    matched['delta_probability']=matched.probability_balanced-matched.probability_none
    matched['delta_score']=matched.score_balanced-matched.score_none
    matched['crossing']=np.select([matched.prediction_none.eq(0)&matched.prediction_balanced.eq(1),matched.prediction_none.eq(1)&matched.prediction_balanced.eq(0)],['0_to_1','1_to_0'],default='unchanged')
    save(matched,'none_balanced_matched_shifts')
    shifts=matched.groupby('group').agg(n_unique_bursts=('burst_id','nunique'),n_predictions=('burst_id','size'),
        mean_probability_none=('probability_none','mean'),mean_probability_balanced=('probability_balanced','mean'),
        mean_delta_probability=('delta_probability','mean'),median_delta_probability=('delta_probability','median'),
        mean_score_none=('score_none','mean'),mean_score_balanced=('score_balanced','mean'),mean_delta_score=('delta_score','mean'),
        median_delta_score=('delta_score','median'),positive_score_shifts=('delta_score',lambda x:int(x.gt(0).sum())),negative_score_shifts=('delta_score',lambda x:int(x.lt(0).sum()))).reset_index()
    save(shifts,'none_balanced_shift_summary')
    crossings=matched.groupby(['group','crossing']).agg(n_predictions=('burst_id','size'),n_unique_bursts=('burst_id','nunique')).reset_index(); save(crossings,'threshold_crossings')
    scorestats=none.groupby('group').score.agg(['count','mean','std','median','min','max']).reset_index()
    scorestats['Q1']=scorestats.group.map(none.groupby('group').score.quantile(.25)); scorestats['Q3']=scorestats.group.map(none.groupby('group').score.quantile(.75))
    all_abnormal=none.loc[none.label.eq(1),'score']
    scorestats=pd.concat([scorestats,pd.DataFrame([dict(group='All abnormal',count=len(all_abnormal),mean=all_abnormal.mean(),std=all_abnormal.std(ddof=1),median=all_abnormal.median(),min=all_abnormal.min(),max=all_abnormal.max(),Q1=all_abnormal.quantile(.25),Q3=all_abnormal.quantile(.75))])],ignore_index=True)
    save(scorestats,'none_score_distribution')
    robust=pred.loc[pred.label.eq(1)].groupby(['weight','burst_id']).agg(validation_count=('prediction','size'),TP_count=('prediction','sum'),
        probability_mean=('probability','mean'),probability_std=('probability',lambda x:x.std(ddof=0)),probability_min=('probability','min'),probability_max=('probability','max'),score_mean=('score','mean')).reset_index()
    robust['FN_count']=robust.validation_count-robust.TP_count
    robust['stability']=np.select([robust.TP_count.eq(10),robust.TP_count.eq(0)],['stable_TP','persistent_FN'],default='unstable')
    robust['margin_stratum']=np.select([robust.score_mean.ge(1),robust.score_mean.le(-1)],['positive_margin','negative_margin'],default='near_boundary')
    save(robust,'abnormal_robustness_per_weight')
    strata=robust.groupby(['weight','margin_stratum','stability']).size().reset_index(name='n_unique_abnormal_bursts'); save(strata,'score_strata')
    feature_rows=[]
    for f in FEATURES:
        n=clean.loc[clean.Equipment_state.eq(0),f]; a=clean.loc[clean.Equipment_state.eq(1),f]
        pooled=np.sqrt(((len(n)-1)*n.var(ddof=1)+(len(a)-1)*a.var(ddof=1))/(len(n)+len(a)-2))
        for label,x in [('Normal',n),('Abnormal',a)]:
            feature_rows.append(dict(feature=f,group=label,n=len(x),mean=x.mean(),std=x.std(ddof=1),median=x.median(),Q1=x.quantile(.25),Q3=x.quantile(.75),
                signed_cohen_d_abnormal_minus_normal=(a.mean()-n.mean())/pooled))
    separability=pd.DataFrame(feature_rows); save(separability,'feature_separability')
    # Training-only centroids, in each frozen fold's original scaled space.
    coeff=pd.read_csv(inputs[8]); intercepts=pd.read_csv(inputs[10]).set_index('cv_index'); centroid_rows=[]; centroid_vectors=[]; max_error=0
    for i,record in enumerate(folds,start=1):
        tr,va=pd.Index(record['train_feature_indices']),pd.Index(record['test_feature_indices'])
        assert not set(tr)&set(va)
        scaler=StandardScaler().fit(clean.loc[tr,FEATURES])
        train=scaler.transform(clean.loc[tr,FEATURES]); valid=scaler.transform(clean.loc[va,FEATURES]); ytr=clean.loc[tr,'Equipment_state'].to_numpy()
        centroid0=train[ytr==0].mean(axis=0); centroid1=train[ytr==1].mean(axis=0)
        for label,centroid in [(0,centroid0),(1,centroid1)]:
            centroid_vectors.append(dict(fold=i,repeat=(i-1)//5+1,label=label,n_training_bursts=int((ytr==label).sum()),**{f:float(v) for f,v in zip(FEATURES,centroid)}))
        coefficient=coeff.loc[coeff.cv_index.eq(i)].set_index('feature').loc[FEATURES,'coefficient'].to_numpy()
        reconstructed=expit(valid@coefficient+intercepts.loc[i,'intercept_recovered'])
        p=none.loc[none.fold.eq(i)].set_index('burst_id').loc[clean.loc[va,'burst_id']]
        error=float(np.max(abs(reconstructed-p.probability.to_numpy()))); max_error=max(max_error,error); assert error<1e-10
        for j,(idx,parent) in enumerate(clean.loc[va].iterrows()):
            centroid_rows.append(dict(fold=i,repeat=(i-1)//5+1,burst_id=int(parent.burst_id),label=int(parent.Equipment_state),
                group='Normal' if parent.Equipment_state==0 else ('FN abnormal' if parent.burst_id in [619,620] else 'Detected abnormal'),
                distance_normal_centroid=float(np.linalg.norm(valid[j]-centroid0)),distance_abnormal_centroid=float(np.linalg.norm(valid[j]-centroid1))))
    centroids=pd.DataFrame(centroid_rows); save(centroids,'fold_centroid_distances'); save(pd.DataFrame(centroid_vectors),'fold_centroids')
    abnormal_centroids=centroids.loc[centroids.label.eq(1)].groupby(['burst_id','group']).agg(
        distance_normal_mean=('distance_normal_centroid','mean'),distance_normal_std=('distance_normal_centroid',lambda x:x.std(ddof=0)),
        distance_abnormal_mean=('distance_abnormal_centroid','mean'),distance_abnormal_std=('distance_abnormal_centroid',lambda x:x.std(ddof=0))).reset_index(); save(abnormal_centroids,'abnormal_centroid_distances')
    previous=pd.read_csv(inputs[11]).sort_values(['cv_index','burst_id'])
    current=centroids.loc[centroids.label.eq(1)].sort_values(['fold','burst_id'])
    np.testing.assert_array_equal(previous.burst_id,current.burst_id)
    np.testing.assert_allclose(previous.distance,current.distance_normal_centroid,atol=1e-12,rtol=0)
    fn=robust.loc[robust.weight.eq('None')&robust.burst_id.isin([619,620])].merge(clean[['burst_id','n_samples']],on='burst_id').merge(abnormal_centroids,on='burst_id')
    save(fn,'fn_619_620')
    fn_display=fn[['burst_id','n_samples','FN_count','probability_mean','distance_normal_mean','distance_abnormal_mean']].copy()
    fn_display.columns=['Burst','Length','FN / 10','Mean $p$','$d_N$','$d_A$']
    tex_table(fn_display,'fn_619_620','Persistent unweighted false negatives. Distances are means over 10 original OOF appearances to training-only centroids in standardized five-dimensional feature space. Probability standard deviation and per-weight stability are supplied in source tables.','tab:fn',lambda x:f'{x:.6f}')
    normal_raw=pd.read_csv(inputs[20]); abnormal_raw=pd.read_csv(inputs[21]); raw=pd.concat([normal_raw,abnormal_raw],ignore_index=True)
    raw['TimeStamp']=pd.to_datetime(raw.TimeStamp); raw=raw.sort_values('TimeStamp').reset_index(drop=True)
    raw['gap']=raw.TimeStamp.diff().dt.total_seconds(); raw['state_change']=raw.Equipment_state.ne(raw.Equipment_state.shift())
    raw['burst_id']=(raw.gap.isna()|raw.gap.gt(1)|raw.state_change).cumsum()
    assert raw.groupby('burst_id').size().loc[clean.burst_id].to_numpy().tolist()==clean.n_samples.tolist()
    ds=[]
    for label,name,source in [(0,'Normal',normal_raw),(1,'Abnormal',abnormal_raw)]:
        c=clean.loc[clean.Equipment_state.eq(label)]; stamps=pd.to_datetime(source.TimeStamp)
        within=raw.loc[raw.Equipment_state.eq(label)&~raw.state_change&raw.gap.le(1)&raw.gap.gt(0),'gap']
        ds.append(dict(group=name,raw_rows=len(source),raw_bursts=raw.loc[raw.Equipment_state.eq(label),'burst_id'].nunique(),eligible_bursts=len(c),
            excluded_lt4=int(raw.loc[raw.Equipment_state.eq(label),'burst_id'].nunique()-len(c)),
            calendar_dates=stamps.dt.date.nunique(),date_min=str(stamps.min().date()),date_max=str(stamps.max().date()),
            samples_min=int(c.n_samples.min()),samples_median=c.n_samples.median(),samples_max=int(c.n_samples.max()),median_within_burst_gap_seconds=within.median()))
    dataset=pd.DataFrame(ds); save(dataset,'dataset_statistics')
    dsdisplay=dataset[['group','raw_rows','raw_bursts','eligible_bursts','calendar_dates','samples_min','samples_median','samples_max']].copy()
    dsdisplay.columns=['Class','Raw rows','Raw bursts','Retained','Dates','$N_{min}$','$N_{med}$','$N_{max}$']
    tex_table(dsdisplay,'dataset','Dataset coverage after the original minimum-four-sample filter. Calendar dates are observable acquisition dates, not verified independent sessions. Counts concern bursts, not sensor rows.','tab:dataset',lambda x:f'{x:g}')
    length_metrics=pd.read_csv(length/'metrics_by_sample_count.csv'); coverage=pd.read_csv(length/'eligible_bursts_by_sample_count.csv')
    transitions=pd.read_csv(length/'tp_to_fn_summary.csv'); drift=pd.read_csv(length/'feature_drift_by_sample_count.csv')
    primary=length_metrics.loc[length_metrics.cohort.eq('eligible_N_ge_target')&length_metrics.source.eq('crop')&length_metrics.aggregation.eq('crop_OOF')].copy()
    lc=primary.merge(coverage,on='sample_count').merge(transitions.loc[transitions.cohort.eq('eligible_N_ge_target')],on='sample_count',suffixes=('','_transition'))
    save(lc,'sample_count_results')
    ldisplay=pd.DataFrame({'$L$':lc.sample_count,'Normal / abnormal':[f'{n}/{a}' for n,a in zip(lc.eligible_normal,lc.eligible_abnormal)],
        'Recall':[f'${m:.3f}\\pm{s:.3f}$' for m,s in zip(lc.Recall_mean,lc.Recall_std)],
        'F1':[f'${m:.3f}\\pm{s:.3f}$' for m,s in zip(lc.F1_mean,lc.F1_std)],
        'Mean abnormal $p$':lc.abnormal_probability_mean,
        'TP $\\rightarrow$ FN':[f'{n}/{d}' for n,d in zip(lc.any_FN_parents,lc.eligible_evaluated_original_TP)],'Normal FP':lc.FP_total})
    tex_table(ldisplay,'sample_count','Frozen unweighted crop inference; eligibility is $N\\geq L$. Recall/F1 mean $\\pm$ population standard deviation across 10 repeats pooled over crop positions within each repeat, unlike Table~\\ref{tab:weights}. Transitions count original-TP parents with any crop/fold FN; the denominator excludes original FNs. Normal FP counts pool repeated crop predictions.','tab:length')
    save(coverage,'sample_count_coverage'); save(transitions,'sample_count_transitions'); save(drift,'sample_count_drift')
    # Figure 1: vector pipeline with explicit frozen reuse and fit scope.
    save(clean[['burst_id','Equipment_state']+FEATURES].reset_index(),'figure_feature_distribution_data')
    save(none[['fold','burst_id','label','group','probability']],'figure_none_probability_data')
    save(length_metrics.loc[length_metrics.aggregation.isin(['crop_OOF','parent_OOF'])],'figure_sample_count_recall_data')
    save(drift.loc[drift.cohort.eq('common_N_ge30')&drift.true_label.eq(1)],'figure_feature_drift_data')
    fig,ax=plt.subplots(figsize=(7.1,2.6)); ax.axis('off')
    boxes=[(.01,.58,.21,.29,'Signed raw signals\nSort timestamp\nGap >1 s / state change'),(.27,.58,.20,.29,'Original bursts\nN >=4\nFive fixed features'),(.52,.58,.20,.29,'Saved CV folds\nTrain-only scaling\nFrozen LR weights'),(.78,.58,.21,.29,'OOF probability\np >0.5\nMetrics / margins')]
    for x,y,w,h,text in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.008',facecolor='#edf3f8',edgecolor='#254763',transform=ax.transAxes)); ax.text(x+w/2,y+h/2,text,ha='center',va='center',transform=ax.transAxes,fontsize=8)
    for x in [.235,.485,.735]: ax.annotate('',xy=(x+.026,.725),xytext=(x-.008,.725),xycoords='axes fraction',arrowprops={'arrowstyle':'->','color':'#254763'})
    ax.text(.5,.27,'Diagnostic branch: deduplicated start / middle / end crops\nExact same features + parent validation models; no classifier refit',ha='center',va='center',transform=ax.transAxes,fontsize=9)
    ax.annotate('',xy=(.53,.40),xytext=(.37,.56),xycoords='axes fraction',arrowprops={'arrowstyle':'->','color':'#254763'})
    figsave(fig,'pipeline')
    fig,axes=plt.subplots(1,5,figsize=(7.2,2.6))
    for ax,f in zip(axes,FEATURES):
        n=clean.loc[clean.Equipment_state.eq(0),f]; a=clean.loc[clean.Equipment_state.eq(1),f]
        ax.boxplot([n,a],tick_labels=['N','A'],showfliers=True,flierprops={'markersize':2}); ax.set_title(f.replace('_','\n'),fontsize=8)
        for bid,color,marker in [(619,'#b2182b','x'),(620,'#7b3294','+')]: ax.scatter(2,clean.loc[clean.burst_id.eq(bid),f],c=color,marker=marker,s=24,zorder=4)
        ax.tick_params(labelsize=7); ax.grid(axis='y',alpha=.2)
    fig.suptitle('Normal (N=575), abnormal (A=20); x:619, +:620',fontsize=10,y=1.06); figsave(fig,'feature_distributions')
    fig,ax=plt.subplots(figsize=(3.5,2.6)); x=np.arange(5)
    for field,color,offset in [('Precision','#2166ac',-.06),('Recall','#1b7837',0),('F1','#b2182b',.06)]:
        ax.errorbar(x+offset,metrics[field+'_mean'],yerr=metrics[field+'_std'],marker='o',ms=4,capsize=2,label=field,color=color)
    ax.set_xticks(x,['None','1:3','1:5','1:10','Bal.']); ax.set(ylabel='Mean CV metric',ylim=(.48,1.04)); ax.legend(); ax.grid(alpha=.2); figsave(fig,'class_weight_metrics')
    fig,ax=plt.subplots(figsize=(3.5,2.6))
    # Histogram probabilities, retaining exact saved values (no clipping for metrics).
    for group,color in [('Normal','#2166ac'),('Detected abnormal','#1b7837')]:
        ax.hist(none.loc[none.group.eq(group),'probability'],bins=np.linspace(0,1,31),histtype='step',density=True,label=group,color=color,lw=1.4)
    for bid,color,marker in [(619,'#b2182b','x'),(620,'#7b3294','+')]:
        v=none.loc[none.burst_id.eq(bid),'probability'].mean(); ax.scatter(v,.055 if bid==619 else .08,marker=marker,c=color,s=35,label=f'FN {bid}',zorder=5)
    ax.axvline(.5,ls='--',color='gray',lw=1); ax.set(xlabel='OOF probability of abnormal',ylabel='Density (log scale)',yscale='log',ylim=(.03,70)); ax.legend(fontsize=7); figsave(fig,'none_probability_distribution')
    fig,ax=plt.subplots(figsize=(3.5,2.7))
    for cohort,color,short in [('eligible_N_ge_target','#2166ac','Eligible'),('common_N_ge30','#1b7837','Fixed N>=30')]:
        for source,unit,style in [('crop','crop_OOF','-o'),('original','parent_OOF','--')]:
            g=length_metrics.loc[length_metrics.cohort.eq(cohort)&length_metrics.source.eq(source)&length_metrics.aggregation.eq(unit)].sort_values('sample_count')
            ax.plot(g.sample_count,g.Recall_mean,style,color=color,ms=3,label=short+(' crop' if source=='crop' else ' full'))
    ax.set(xlabel='Crop sample count',ylabel='Mean within-repeat recall',ylim=(.32,1.04)); ax.legend(fontsize=7); ax.grid(alpha=.2); figsave(fig,'sample_count_recall')
    fig,ax=plt.subplots(figsize=(3.5,2.7))
    for f,g in drift.loc[drift.cohort.eq('common_N_ge30')&drift.true_label.eq(1)].groupby('feature'):
        ax.plot(g.sample_count,g.mean_absolute_standardized_delta,'-o',ms=3,label=f)
    ax.set(xlabel='Crop sample count',ylabel='Mean absolute drift / normal SD'); ax.legend(fontsize=6.5); ax.grid(alpha=.2); figsave(fig,'sample_count_feature_drift')
    # Generated prose uses same exact data, while paper rounded text is audited separately.
    summary=dict(dataset=dataset.to_dict('records'),metrics=metrics.to_dict('records'),shifts=shifts.to_dict('records'),crossings=crossings.to_dict('records'),
        saturation=saturation.to_dict('records'),fn=fn.to_dict('records'),none_stability=robust.loc[robust.weight.eq('None')].groupby('stability').size().to_dict(),
        balanced_stability=robust.loc[robust.weight.eq('balanced')].groupby('stability').size().to_dict(),
        detected_abnormal_distance_normal_mean=float(abnormal_centroids.loc[abnormal_centroids.group.eq('Detected abnormal'),'distance_normal_mean'].mean()),
        detected_abnormal_distance_normal_min=float(abnormal_centroids.loc[abnormal_centroids.group.eq('Detected abnormal'),'distance_normal_mean'].min()),
        max_probability_error=max_error)
    (OUT/'tables/analysis_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    audit=dict(classifier_fits=0,threshold=.5,features=FEATURES,weights=WEIGHTS,source_copy_hashes_verified=True,original_hash_verified=True,
        folds_identical_across_weights_and_tracking=True,validation_membership_verified=True,all_predictions_strict_threshold_verified=True,
        published_saved_metrics_reproduced=True,pr_auc_definition='auc(recall,precision) of sklearn.precision_recall_curve, per fold',
        average_precision_saved_separately=True,score_logit_clip_epsilon=EPS,score_clipping=saturation.to_dict('records'),
        centroid_training_rows_only=True,previous_centroid_distances_reproduced=True,max_none_probability_reconstruction_error=max_error,
        csv_and_tex_tables_same_dataframes=True,figure_png_dpi=300,figure_vectors=['pdf','svg'],
        Python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__)
    for record in manifest: assert sha(ROOT/record['source'])==record['sha256']
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
    print(metrics[['weight','PR_AUC_mean','PR_AUC_std','AP_mean']].to_string(index=False)); print(crossings.to_string(index=False)); print(shifts.to_string(index=False)); print(dataset.to_string(index=False)); print('stability',summary['none_stability'],summary['balanced_stability']); print('max error',max_error)


if __name__=='__main__': main()
