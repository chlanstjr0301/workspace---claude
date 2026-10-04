"""Sample-count diagnostic with frozen original OOF models, no LR training.

Only inference input length/position changes; exact signed notebook features,
saved validation folds, coefficients, intercepts and strict >.5 stay fixed.
"""
import json
import os
import platform
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import stats
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from logistic_current_acf_ablation import sha
from logistic_burst_length_sensitivity import ROOT, TRACK, FEATURES

OUT = ROOT/'results/logistic_burst_sample_count'
LENGTHS = [4, 6, 8, 10, 12, 15, 20, 25, 30]


def repeat_metrics(frame):
    rows = []
    for repeat, group in frame.groupby('repeat'):
        y = group.true_label.to_numpy(); p = group.probability_y1.to_numpy(); pred = p > .5
        rows.append(dict(repeat=repeat, Accuracy=accuracy_score(y, pred), Precision=precision_score(y, pred, zero_division=0),
            Recall=recall_score(y, pred, zero_division=0), F1=f1_score(y, pred, zero_division=0),
            Balanced_Accuracy=balanced_accuracy_score(y, pred),
            abnormal_probability=float(p[y==1].mean()), normal_probability=float(p[y==0].mean()),
            FP=int(((y==0)&pred).sum()), FN=int(((y==1)&~pred).sum())))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True); plots=OUT/'plots'; plots.mkdir(exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(OUT/'.matplotlib_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    nbpath=ROOT/'reference/소성가공.ipynb'; baseline=ROOT/'src/LSTM-AutoEncoder/baseline_original.py'
    assert sha(baseline)==baseline.with_suffix('.sha256').read_text().split()[0].lower()=='3b0855d29c0e47e9856d3c293083d7f430ebe7cbe6f24ef85f247348e504a0a8'
    paths=[nbpath, baseline, ROOT/'data/raw/press_data_normal.csv', ROOT/'data/raw/outlier_data.csv',
        TRACK/'cv_split_indices.json', TRACK/'fold_coefficients.csv', TRACK/'validation_predictions.csv',
        ROOT/'results/logistic_fn_feature_analysis/recovered_fold_intercepts.csv',
        ROOT/'results/logistic_current_acf_ablation/burst_feature_dataset.csv']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    nb=json.loads(nbpath.read_text(encoding='utf-8'))
    ns=dict(np=np,pd=pd,stats=stats,data=pd.concat([pd.read_csv(paths[2]),pd.read_csv(paths[3])],ignore_index=True))
    with warnings.catch_warnings(record=True) as warns:
        warnings.simplefilter('always')
        for i in [6,7,8,9,27]:
            exec(compile(''.join(nb['cells'][i]['source']),f'notebook_cell_{i}','exec'),ns)
    clean=ns['featureData_clean']; X=clean[FEATURES]
    saved=pd.read_csv(paths[8],index_col='notebook_feature_index')
    pd.testing.assert_frame_equal(clean,saved.drop(columns='Current_ACF_PeakStrength'),check_names=False)
    original=pd.read_csv(paths[6]); coef_table=pd.read_csv(paths[5]); folds=json.loads(paths[4].read_text(encoding='utf-8'))
    intercepts=pd.read_csv(paths[7]).set_index('cv_index')
    assert len(folds)==50 and original.groupby('burst_id').size().eq(10).all()
    tp_ids=original.groupby('burst_id').filter(lambda g:g.true_label.eq(1).all() and g.predicted_label.eq(1).all()).burst_id.unique()
    assert len(tp_ids)==18
    normal_std=clean.loc[clean.Equipment_state.eq(0),FEATURES].std(ddof=1)
    normal_std.to_frame('normal_original_std_ddof1').to_csv(OUT/'feature_reference_scales.csv')
    physical=[]
    with warnings.catch_warnings(record=True) as crop_warns:
        warnings.simplefilter('always')
        for idx,parent in clean.iterrows():
            raw=ns['data'].loc[ns['data'].burst_id.eq(parent.burst_id)]
            assert len(raw)==parent.n_samples
            for length in LENGTHS:
                if len(raw)<length: continue
                positions={}
                for name,start in [('start',0),('middle',(len(raw)-length)//2),('end',len(raw)-length)]:
                    positions.setdefault(start,[]).append(name)
                for start,names in positions.items():
                    crop=raw.iloc[start:start+length]; values={}
                    for channel in ['AI0','AI1']:
                        values.update({f'{channel}_{key}':value for key,value in ns['get_vibration_features'](crop[f'{channel}_Vibration']).items()})
                    valid=all(np.isfinite(values[f]) for f in FEATURES)
                    physical.append(dict(crop_id=f'{int(parent.burst_id)}_{length}_{start}',notebook_feature_index=int(idx),
                        burst_id=int(parent.burst_id),true_label=int(parent.Equipment_state),parent_length=int(parent.n_samples),
                        sample_count=length,start_index=start,position='/'.join(names),computable=valid,
                        uncomputable_features=','.join(f for f in FEATURES if not np.isfinite(values[f])),
                        **{f:float(values[f]) for f in FEATURES}))
    crops=pd.DataFrame(physical); crops.to_csv(OUT/'physical_crop_features.csv',index=False,encoding='utf-8-sig')
    predictions=[]; checks=[]
    for i,fold in enumerate(folds,start=1):
        tr,va=pd.Index(fold['train_feature_indices']),pd.Index(fold['test_feature_indices'])
        assert not set(tr)&set(va)
        scaler=StandardScaler().fit(X.loc[tr])  # Statistics from original train rows only; never fits LR.
        coef=coef_table.loc[coef_table.cv_index.eq(i)].set_index('feature').loc[FEATURES,'coefficient'].to_numpy()
        intercept=intercepts.loc[i,'intercept_recovered']
        old=original.loc[original.cv_index.eq(i)].set_index('notebook_feature_index').loc[va]
        reconstructed=expit(scaler.transform(X.loc[va])@coef+intercept)
        error=float(np.max(abs(reconstructed-old.probability_y1.to_numpy())))
        assert error<1e-10; np.testing.assert_array_equal(reconstructed>.5,old.predicted_label)
        checks.append(dict(cv_index=i,max_probability_error=error))
        subset=crops.loc[crops.notebook_feature_index.isin(va)].copy()
        subset['probability_y1']=np.nan; subset['predicted_label']=np.nan
        valid=subset.computable
        subset.loc[valid,'probability_y1']=expit(scaler.transform(subset.loc[valid,FEATURES])@coef+intercept)
        subset.loc[valid,'predicted_label']=(subset.loc[valid,'probability_y1']>.5).astype(int)
        subset['cv_index']=i; subset['repeat']=(i-1)//5+1; subset['fold']=(i-1)%5+1
        subset['parent_probability_y1']=subset.notebook_feature_index.map(old.probability_y1)
        predictions.append(subset)
    predictions=pd.concat(predictions,ignore_index=True)
    assert predictions.groupby('crop_id').size().eq(10).all()
    predictions.to_csv(OUT/'crop_predictions.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(checks).to_csv(OUT/'probability_reproduction_checks.csv',index=False)
    common_ids=clean.loc[clean.n_samples.ge(30),'burst_id']; counts=[]; metric_rows=[]; repeat_rows=[]; transitions=[]; drift=[]
    for length in LENGTHS:
        eligible=clean.loc[clean.n_samples.ge(length)]
        physical_length=crops.loc[crops.sample_count.eq(length)]
        count=dict(sample_count=length,eligible_normal=int(eligible.Equipment_state.eq(0).sum()),
            eligible_abnormal=int(eligible.Equipment_state.eq(1).sum()),eligible_original_TP=int(eligible.burst_id.isin(tp_ids).sum()),
            excluded_normal=575-int(eligible.Equipment_state.eq(0).sum()),excluded_abnormal=20-int(eligible.Equipment_state.eq(1).sum()),
            n_physical_crops=len(physical_length),n_OOF_predictions=len(physical_length)*10,
            uncomputable_physical_crops=int((~physical_length.computable).sum()),
            evaluated_physical_crops=int(physical_length.computable.sum()),evaluated_OOF_predictions=int(physical_length.computable.sum())*10,
            evaluated_unique_normal=physical_length.loc[physical_length.computable&physical_length.true_label.eq(0),'burst_id'].nunique(),
            evaluated_unique_abnormal=physical_length.loc[physical_length.computable&physical_length.true_label.eq(1),'burst_id'].nunique())
        counts.append(count)
        for cohort,ids in [('eligible_N_ge_target',eligible.burst_id),('common_N_ge30',common_ids)]:
            crop_predictions=predictions.loc[predictions.sample_count.eq(length)&predictions.burst_id.isin(ids)&predictions.computable].copy()
            # For computability failures, matched original reference covers evaluated parent IDs.
            evaluated_ids=crop_predictions.burst_id.unique()
            full=original.loc[original.burst_id.isin(evaluated_ids)]
            parent=crop_predictions.groupby(['burst_id','repeat']).agg(true_label=('true_label','first'),probability_y1=('probability_y1','mean')).reset_index()
            for source,unit,frame in [('crop','crop_OOF',crop_predictions),('original','parent_OOF',full),('crop','parent_repeat_mean_position_probability',parent)]:
                repeats=repeat_metrics(frame); repeats['sample_count']=length; repeats['cohort']=cohort; repeats['source']=source; repeats['aggregation']=unit
                repeat_rows.append(repeats)
                row=dict(sample_count=length,cohort=cohort,source=source,aggregation=unit,n_unique_bursts=frame.burst_id.nunique(),
                    n_unique_normal_bursts=frame.loc[frame.true_label.eq(0),'burst_id'].nunique(),n_unique_abnormal_bursts=frame.loc[frame.true_label.eq(1),'burst_id'].nunique(),
                    n_predictions=len(frame),n_repeats=len(repeats))
                for name in ['Accuracy','Precision','Recall','F1','Balanced_Accuracy','abnormal_probability','normal_probability','FP','FN']:
                    row[name+'_mean']=repeats[name].mean(); row[name+'_std']=repeats[name].std(ddof=0)
                row['FP_total']=int(repeats.FP.sum()); row['FN_total']=int(repeats.FN.sum()); metric_rows.append(row)
            selected=crop_predictions.loc[crop_predictions.burst_id.isin(tp_ids)]
            for bid,g in selected.groupby('burst_id'):
                error=g.predicted_label.eq(0)
                transitions.append(dict(sample_count=length,cohort=cohort,burst_id=int(bid),parent_length=int(g.parent_length.iloc[0]),
                    n_physical_crops=g.crop_id.nunique(),n_predictions=len(g),FN_count=int(error.sum()),FN_fraction=error.mean(),
                    any_FN=bool(error.any()),persistent_all_crop_fold_FN=bool(error.all())))
            selected_crops=physical_length.loc[physical_length.burst_id.isin(ids)]
            parent_values=clean.set_index('burst_id')[FEATURES]
            for label in [0,1]:
                c=selected_crops.loc[selected_crops.true_label.eq(label)]
                for feature in FEATURES:
                    delta=c[feature].to_numpy()-parent_values.loc[c.burst_id,feature].to_numpy()
                    finite=np.isfinite(delta)
                    drift.append(dict(sample_count=length,cohort=cohort,true_label=label,feature=feature,n_unique_bursts=c.burst_id.nunique(),
                        n_physical_crops=len(c),n_finite_feature_crops=int(finite.sum()),mean_absolute_raw_delta=float(np.mean(abs(delta[finite]))),
                        mean_absolute_standardized_delta=float(np.mean(abs(delta[finite])/normal_std[feature]))))
    counts=pd.DataFrame(counts); metrics=pd.DataFrame(metric_rows); transitions=pd.DataFrame(transitions); drift=pd.DataFrame(drift)
    for filename,frame in [('eligible_bursts_by_sample_count',counts),('metrics_by_sample_count',metrics),('repeat_metrics',pd.concat(repeat_rows)),
        ('tp_to_fn_by_sample_count',transitions),('feature_drift_by_sample_count',drift)]:
        frame.to_csv(OUT/f'{filename}.csv',index=False,encoding='utf-8-sig')
    transition_summary=transitions.groupby(['cohort','sample_count']).agg(eligible_evaluated_original_TP=('burst_id','size'),
        any_FN_parents=('any_FN','sum'),persistent_FN_parents=('persistent_all_crop_fold_FN','sum'),FN_predictions=('FN_count','sum'),n_predictions=('n_predictions','sum')).reset_index()
    transition_summary['any_FN_parent_rate']=transition_summary.any_FN_parents/transition_summary.eligible_evaluated_original_TP
    transition_summary.to_csv(OUT/'tp_to_fn_summary.csv',index=False,encoding='utf-8-sig')
    fn_rows=[]
    for bid in [619,620]:
        N=int(clean.loc[clean.burst_id.eq(bid),'n_samples'].iloc[0])
        for length in LENGTHS:
            p=predictions.loc[predictions.burst_id.eq(bid)&predictions.sample_count.eq(length)&predictions.computable]
            fn_rows.append(dict(burst_id=bid,parent_length=N,sample_count=length,eligible=N>=length,
                status='excluded_N_less_than_target' if N<length else ('uncomputable' if p.empty else 'evaluated'),
                n_physical_crops=p.crop_id.nunique(),n_predictions=len(p),probability_mean=p.probability_y1.mean(),
                probability_std=p.probability_y1.std(ddof=0),predicted_abnormal_fraction=p.predicted_label.mean()))
    fn_summary=pd.DataFrame(fn_rows); fn_summary.to_csv(OUT/'burst_619_620_predictions.csv',index=False,encoding='utf-8-sig')
    predictions.loc[predictions.burst_id.isin([619,620])].to_csv(OUT/'burst_619_620_crop_predictions.csv',index=False,encoding='utf-8-sig')
    for metric in ['Recall','F1','abnormal_probability']:
        fig,ax=plt.subplots(figsize=(8,5))
        for cohort,label in [('eligible_N_ge_target','Eligible crop'),('common_N_ge30','Common N>=30 crop')]:
            for source,unit,style in [('crop','crop_OOF','-o'),('original','parent_OOF','--')]:
                m=metrics.loc[metrics.cohort.eq(cohort)&metrics.source.eq(source)&metrics.aggregation.eq(unit)].sort_values('sample_count')
                ax.plot(m.sample_count,m[metric+'_mean'],style,label=label if source=='crop' else label+' full parent')
        ax.set(xlabel='Sample count',ylabel=metric,title='Frozen OOF models; original reference matched to eligible parents'); ax.legend(fontsize=8); ax.grid(alpha=.25)
        fig.tight_layout(); fig.savefig(plots/f'sample_count_{metric}.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for cohort,g in transition_summary.groupby('cohort'): ax.plot(g.sample_count,g.any_FN_parent_rate,'-o',label=cohort)
    ax.set(xlabel='Sample count',ylabel='Any-FN parent fraction'); ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(plots/'sample_count_tp_to_fn_rate.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    for f,g in drift.loc[drift.cohort.eq('common_N_ge30')&drift.true_label.eq(1)].groupby('feature'):
        ax.plot(g.sample_count,g.mean_absolute_standardized_delta,'-o',label=f)
    ax.set(xlabel='Sample count',ylabel='Mean absolute drift / original normal SD',title='Fixed common abnormal cohort'); ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(plots/'sample_count_feature_drift.png',dpi=160); plt.close(fig)
    for p in paths: assert sha(p)==hashes[str(p.relative_to(ROOT))]
    metadata=dict(input_sha256=hashes,features=FEATURES,target_lengths=LENGTHS,class_weight=None,threshold=.5,decision_rule='strict P>0.5',
        classifier_fits=0,scaler_recovery='original saved training rows only',max_original_probability_error=max(c['max_probability_error'] for c in checks),
        crop_positions='start0,middle floor((N-L)/2),end N-L; duplicate starts collapsed; N=L one crop',
        metric_mean_std='10 within-repeat pooled diagnostics; population std ddof0; dependent crops/repeats',
        parent_aggregation='mean position probability per parent/repeat before strict threshold; diagnostic only',
        uncomputable_policy='retain crop logs; NA probability and prediction; excluded explicitly from evaluated metrics; no imputation',
        matched_original_reference='same evaluated eligible parent IDs',drift_reference='original575 normal SD ddof1; physical crop counted once',
        Python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,
        warnings=list(set(str(w.message) for w in list(warns)+list(crop_warns))))
    (OUT/'experiment_metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
    short=metrics.loc[metrics.cohort.eq('eligible_N_ge_target')&metrics.source.eq('crop')&metrics.aggregation.eq('crop_OOF')]
    fixed=metrics.loc[metrics.cohort.eq('common_N_ge30')&metrics.source.eq('crop')&metrics.aggregation.eq('crop_OOF')]
    def md(frame):
        return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+''.join('| '+' | '.join(str(v) for v in row)+' |\n' for row in frame.itertuples(index=False,name=None))
    report='# 표본 수에 따른 frozen-model 진단\n\n분류기를 다시 학습하지 않고 기존50개 fold의 계수·복원 절편과 원본 train의 StandardScaler 통계를 재사용했습니다. 원본 예측 확률 최대 오차는 '+str(metadata['max_original_probability_error'])+'입니다. 원본 전체575정상·20이상의 Recall은0.9이며619·620은 원래FN입니다.\n\n'
    report+='## 길이별 결과\n\n평균±표준편차는 위치별 crop을 repeat 안에서 모은10회 반복 기준(ddof=0)입니다. crop과 반복은 독립 표본이 아닙니다. 길이별 대상 수가 달라지므로 동일 대상의 원본 비교 및 고정 N>=30 코호트를 함께 저장했습니다.\n\n'
    table=short[['sample_count','n_unique_normal_bursts','n_unique_abnormal_bursts','Recall_mean','Recall_std','F1_mean','abnormal_probability_mean','FP_total','FN_total']].round(5)
    report+=md(table)+'\n\n고정 N>=30 코호트:\n\n'+md(fixed[['sample_count','n_unique_normal_bursts','n_unique_abnormal_bursts','Recall_mean','F1_mean','abnormal_probability_mean']].round(5))
    report+='\n## TP→FN\n\n분모는 원래18개TP 중 해당 길이에서 평가 가능한 부모 burst입니다. any_FN은 한 위치·fold 이상FN, persistent_FN은 모든 위치·fold에서FN입니다.\n\n'+md(transition_summary)
    report+='\n## 619·620\n\n길이가 부족한 경우 NA로 남겼고 padding하지 않았습니다.\n\n'+md(fn_summary.round(6))
    ranking=drift.loc[drift.cohort.eq('common_N_ge30')&drift.true_label.eq(1)].groupby('feature').mean_absolute_standardized_delta.mean().sort_values(ascending=False)
    report+='\n## Feature 변화\n\n고정 코호트 이상 부모에서 물리 crop을 한 번씩 계산한 절대 변화량 / 원본 정상575개 표준편차입니다. 아래 순위는9개 길이의 평균을 같은 가중치로 평균했습니다. 원시 단위는 feature 간 직접 비교할 수 없습니다.\n\n'+md(ranking.reset_index().round(5))
    report+='\n계산 불가능한 crop은 '+str(int(counts.uncomputable_physical_crops.sum()))+'개입니다. 평가 가능·불가능 수를 별도 기록했으며 보간·대체·새로운 feature·threshold 조정은 하지 않았습니다. 위치 평균 후 threshold를 적용한 부모 단위 집계도 별도 저장했으며 공식 평가 방식을 바꾼 것이 아닙니다. 길이와 구간 내용이 함께 변하므로 결과만으로619·620의 길이 단독 인과나 label 오류를 주장할 수 없습니다.\n'
    report+='\n## 결론\n\n- 가장 큰 인접 길이 하락은15→12개입니다. 길이별 eligible Recall은68.18%→56.25%(−11.93%p), 동일 N>=30 부모에서는60%→50%(−10%p)입니다. 다만30개에서도 crop Recall이82.07%로 원본100%보다 낮아,15개를 유일한 안전 경계로 볼 수는 없습니다.\n'
    report+='- 10~15개는 탐지 누락 위험이 큽니다. eligible crop Recall은10개52.86%,15개68.18%; 원래 TP 중 하나 이상의 crop/fold가FN인 부모는 각각12/15개와9/15개입니다. 이번15개 조건은 N>=15라서 원본 길이15인 TP614를 포함하며, 이전 N>15 분석의9/14와 분모가 다릅니다. 고정 코호트의 TP10개 중10개에서는10/10,15개에서는8/10개가 일부FN입니다. 두 길이에서 모든 위치·fold가FN인 부모는603 하나입니다.\n'
    report+='- 표준화 변화량의9개 길이 평균은AI0_RMS가1위(7.802 정상SD), AI0_MaxAbs가2위(7.331)입니다. 다만 극단적으로 짧은4·6·8·10개에서는AI0_MaxAbs가AI0_RMS보다 조금 큽니다. AI1_Kurtosis의 원시 숫자 변화가 크다는 사실과 단위 보정 순위를 구분해야 합니다.\n'
    report+='- 619·620만의 특수 현상이라고 볼 수는 없습니다. 탐지되던 다른 이상 부모도 단축하면 많이FN으로 바뀝니다. 그러나619·620은 사용 가능한 모든 crop에서 계속FN이며 원래 길이보다 긴 입력을 실제 데이터로 만들 수 없으므로, 두 사례가 길이 때문에FN이었는지는 아직 확정할 수 없습니다. padding이나 label 오류 가정은 하지 않았습니다.\n'
    report+='- 위 수치는 위치별 crop OOF 기준입니다. 위치 확률을 먼저 평균하는 부모 단위 진단은 별도 CSV에 있으며, 예를 들어 고정 코호트30개 Recall98%,15개62%,10개57%입니다. 두 집계 방식의 수치를 혼합하지 않습니다.\n'
    (OUT/'analysis_report.md').write_text(report,encoding='utf-8')
    print(table.to_string(index=False)); print('COMMON'); print(fixed[['sample_count','Recall_mean','F1_mean']].to_string(index=False)); print(transition_summary.to_string(index=False)); print(ranking.to_string()); print(fn_summary.to_string(index=False)); print('uncomputable',int(counts.uncomputable_physical_crops.sum()))


if __name__=='__main__': main()
