"""Frozen-model diagnostic of burst crops; never fits a classifier.

Only diagnostic input duration/position changes. Notebook features, signed raw
signals, saved validation membership, coefficients, intercepts and threshold
remain identical. Scaling statistics are recovered from original training rows.
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
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score
from logistic_current_acf_ablation import sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/logistic_burst_length_sensitivity'
TRACK = ROOT / 'results/logistic_fn_burst_tracking'
FEATURES = ['AI0_RMS', 'AI0_MaxAbs', 'AI1_RMS', 'AI1_PeakToPeak', 'AI1_Kurtosis']


def summarize(frame, cohort, length, unit):
    y, p = frame.true_label.to_numpy(), frame.probability_y1.to_numpy()
    pred = p > .5
    abnormal, normal = y == 1, y == 0
    return dict(cohort=cohort, length=length, aggregation=unit,
                n_unique_bursts=frame.burst_id.nunique(), n_predictions=len(frame),
                n_unique_abnormal_bursts=frame.loc[abnormal, 'burst_id'].nunique(),
                n_unique_normal_bursts=frame.loc[normal, 'burst_id'].nunique(),
                n_abnormal_predictions=int(abnormal.sum()), n_normal_predictions=int(normal.sum()),
                abnormal_detection_rate=float(pred[abnormal].mean()),
                abnormal_probability_mean=float(p[abnormal].mean()),
                abnormal_probability_median=float(np.median(p[abnormal])),
                normal_FP_rate=float(pred[normal].mean()), normal_probability_mean=float(p[normal].mean()),
                Precision=precision_score(y, pred, zero_division=0), Recall=recall_score(y, pred),
                F1=f1_score(y, pred), Balanced_Accuracy=balanced_accuracy_score(y, pred))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plots = OUT / 'plots'; plots.mkdir(exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(OUT / '.matplotlib_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    notebook = ROOT / 'reference/소성가공.ipynb'
    baseline = ROOT / 'src/LSTM-AutoEncoder/baseline_original.py'
    assert sha(baseline) == baseline.with_suffix('.sha256').read_text().split()[0].lower() == '3b0855d29c0e47e9856d3c293083d7f430ebe7cbe6f24ef85f247348e504a0a8'
    paths = [notebook, baseline, ROOT/'data/raw/press_data_normal.csv', ROOT/'data/raw/outlier_data.csv',
             TRACK/'cv_split_indices.json', TRACK/'fold_coefficients.csv', TRACK/'validation_predictions.csv',
             ROOT/'results/logistic_fn_feature_analysis/recovered_fold_intercepts.csv',
             ROOT/'results/logistic_current_acf_ablation/burst_feature_dataset.csv']
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    nb = json.loads(notebook.read_text(encoding='utf-8'))
    ns = dict(np=np, pd=pd, stats=stats, data=pd.concat([pd.read_csv(paths[2]), pd.read_csv(paths[3])], ignore_index=True))
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        for i in [6, 7, 8, 9, 27]:
            exec(compile(''.join(nb['cells'][i]['source']), f'notebook_cell_{i}', 'exec'), ns)
    clean = ns['featureData_clean']
    saved = pd.read_csv(paths[8], index_col='notebook_feature_index')
    pd.testing.assert_frame_equal(clean, saved.drop(columns='Current_ACF_PeakStrength'), check_names=False)
    original = pd.read_csv(paths[6]); coefficients = pd.read_csv(paths[5])
    intercepts = pd.read_csv(paths[7]).set_index('cv_index')
    folds = json.loads(paths[4].read_text(encoding='utf-8'))
    X = clean[FEATURES]
    assert len(folds) == 50 and original.groupby('burst_id').size().eq(10).all()
    normal_std = clean.loc[clean.Equipment_state.eq(0), FEATURES].std(ddof=1)
    normal_std.to_frame('normal_original_std_ddof1').to_csv(OUT/'feature_reference_scales.csv', encoding='utf-8-sig')
    physical, drift = [], []
    for idx, parent in clean.iterrows():
        raw = ns['data'].loc[ns['data'].burst_id.eq(parent.burst_id)]
        assert len(raw) == parent.n_samples
        for length in [10, 15, 20]:
            if len(raw) < length:
                continue
            positions = {}
            for name, start in [('start', 0), ('middle', (len(raw)-length)//2), ('end', len(raw)-length)]:
                positions.setdefault(start, []).append(name)
            for start, names in positions.items():
                crop = raw.iloc[start:start+length]
                feature_values = {}
                for channel in ['AI0', 'AI1']:
                    values = ns['get_vibration_features'](crop[f'{channel}_Vibration'])
                    feature_values.update({f'{channel}_{key}': value for key, value in values.items()})
                row = dict(crop_id=f'{int(parent.burst_id)}_{length}_{start}', notebook_feature_index=int(idx),
                           burst_id=int(parent.burst_id), true_label=int(parent.Equipment_state),
                           parent_length=int(parent.n_samples), length=length, start_index=start,
                           position='/'.join(names), actually_shorter=int(parent.n_samples > length),
                           start_timestamp=str(crop.TimeStamp.iloc[0]), end_timestamp=str(crop.TimeStamp.iloc[-1]),
                           **{f: float(feature_values[f]) for f in FEATURES})
                physical.append(row)
                for f in FEATURES:
                    delta = row[f]-parent[f]
                    drift.append(dict(crop_id=row['crop_id'], burst_id=row['burst_id'], true_label=row['true_label'],
                                      parent_length=row['parent_length'], length=length, feature=f,
                                      raw_delta=delta, raw_absolute_delta=abs(delta),
                                      standardized_absolute_delta=abs(delta)/normal_std[f]))
    crops = pd.DataFrame(physical)
    assert np.isfinite(crops[FEATURES].to_numpy()).all()
    predicted, verification = [], []
    for i, fold in enumerate(folds, start=1):
        tr, va = pd.Index(fold['train_feature_indices']), pd.Index(fold['test_feature_indices'])
        assert not set(tr)&set(va)
        scaler = StandardScaler().fit(X.loc[tr])  # Original rows only: statistics recovery, no classifier fit.
        coef = coefficients.loc[coefficients.cv_index.eq(i)].set_index('feature').loc[FEATURES, 'coefficient'].to_numpy()
        intercept = intercepts.loc[i, 'intercept_recovered']
        oof = original.loc[original.cv_index.eq(i)].set_index('notebook_feature_index').loc[va]
        reconstructed = expit(scaler.transform(X.loc[va]) @ coef + intercept)
        error = float(np.max(np.abs(reconstructed-oof.probability_y1.to_numpy())))
        assert error < 1e-10
        np.testing.assert_array_equal(reconstructed > .5, oof.predicted_label.to_numpy())
        verification.append(dict(cv_index=i, max_probability_error=error))
        subset = crops.loc[crops.notebook_feature_index.isin(va)].copy()
        subset['probability_y1'] = expit(scaler.transform(subset[FEATURES]) @ coef + intercept)
        subset['predicted_label'] = (subset.probability_y1 > .5).astype(int)
        subset['cv_index'] = i; subset['repeat'] = (i-1)//5+1; subset['fold'] = (i-1)%5+1
        subset['parent_probability_y1'] = subset.notebook_feature_index.map(oof.probability_y1)
        subset['parent_predicted_label'] = subset.notebook_feature_index.map(oof.predicted_label)
        predicted.append(subset)
    predictions = pd.concat(predicted, ignore_index=True)
    assert predictions.groupby('crop_id').size().eq(10).all()
    predictions.to_csv(OUT/'crop_predictions.csv', index=False, encoding='utf-8-sig')
    crops.to_csv(OUT/'physical_crop_features.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(verification).to_csv(OUT/'probability_reproduction_checks.csv', index=False)
    common_ids = clean.loc[clean.n_samples.ge(20), 'burst_id']
    metric_rows = []
    aggregated_rows = []
    def append_metrics(frame, cohort, length):
        metric_rows.append(summarize(frame, cohort, length, 'crop_OOF'))
        aggregate = frame.groupby(['burst_id', 'repeat']).agg(true_label=('true_label', 'first'),
                         probability_y1=('probability_y1', 'mean')).reset_index()
        metric_rows.append(summarize(aggregate, cohort, length, 'parent_repeat_mean_position_probability'))
        aggregate['cohort'] = cohort; aggregate['length'] = length; aggregated_rows.append(aggregate)
        unique = aggregate.groupby('burst_id').agg(true_label=('true_label', 'first'),
                                                   probability_y1=('probability_y1', 'mean')).reset_index()
        metric_rows.append(summarize(unique, cohort, length, 'parent_allrepeat_mean_probability'))
    append_metrics(original, 'all_original_reference', 'original')
    append_metrics(original.loc[original.burst_id.isin(common_ids)], 'common_N_ge20', 'original')
    for length in [20, 15, 10]:
        append_metrics(predictions.loc[predictions.length.eq(length)&predictions.burst_id.isin(common_ids)], 'common_N_ge20', str(length))
        eligible_ids = clean.loc[clean.n_samples.gt(length), 'burst_id']
        append_metrics(original.loc[original.burst_id.isin(eligible_ids)], f'actually_shorter_N_gt{length}', 'original')
        append_metrics(predictions.loc[predictions.length.eq(length)&predictions.actually_shorter.eq(1)], f'actually_shorter_N_gt{length}', str(length))
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT/'length_metrics.csv', index=False, encoding='utf-8-sig')
    pd.concat(aggregated_rows).to_csv(OUT/'parent_repeat_diagnostic_aggregation.csv', index=False, encoding='utf-8-sig')
    transition_rows = []
    for cohort in ['common_N_ge20', 'actually_shorter']:
        for length in [10, 15, 20]:
            selected = predictions.loc[predictions.length.eq(length)]
            selected = selected.loc[selected.burst_id.isin(common_ids)] if cohort == 'common_N_ge20' else selected.loc[selected.actually_shorter.eq(1)]
            for bid, group in selected.groupby('burst_id'):
                label = int(group.true_label.iloc[0]); before = int(group.parent_predicted_label.iloc[0])
                if before != label:  # FN 619/620 are not originally TP.
                    continue
                errors = group.predicted_label.ne(label)
                transition_rows.append(dict(cohort=cohort, length=length, burst_id=int(bid), true_label=label,
                    transition='TP_to_FN' if label else 'TN_to_FP', parent_length=int(group.parent_length.iloc[0]),
                    n_physical_crops=group.crop_id.nunique(), n_predictions=len(group), error_count=int(errors.sum()),
                    error_fraction=float(errors.mean()), any_error=bool(errors.any()), persistent_error=bool(errors.all()),
                    parent_probability_mean=group.parent_probability_y1.mean(), crop_probability_mean=group.probability_y1.mean(),
                    crop_probability_min=group.probability_y1.min(), crop_probability_max=group.probability_y1.max()))
    transitions = pd.DataFrame(transition_rows)
    transitions.to_csv(OUT/'tp_to_fn_transitions.csv', index=False, encoding='utf-8-sig')
    drift = pd.DataFrame(drift)
    drift.to_csv(OUT/'physical_crop_feature_drift.csv', index=False, encoding='utf-8-sig')
    drift['cohort'] = 'all_available'
    common_drift = drift.loc[drift.burst_id.isin(common_ids)].copy(); common_drift['cohort'] = 'common_N_ge20'
    drift_summary = pd.concat([drift, common_drift]).groupby(['cohort', 'length', 'true_label', 'feature']).agg(
        n_unique_bursts=('burst_id', 'nunique'), n_physical_crops=('crop_id', 'size'),
        mean_raw_delta=('raw_delta', 'mean'), mean_absolute_raw_delta=('raw_absolute_delta', 'mean'),
        mean_absolute_standardized_delta=('standardized_absolute_delta', 'mean')).reset_index()
    drift_summary.to_csv(OUT/'feature_drift_by_length.csv', index=False, encoding='utf-8-sig')
    # Descriptive distance only: one common normal-original standard deviation per feature.
    distance_rows = []
    crop_probs = predictions.groupby('crop_id').probability_y1.mean()
    for bid in [619, 620]:
        target = clean.loc[clean.burst_id.eq(bid), FEATURES].iloc[0]
        for length in [10, 15, 20]:
            candidates = crops.loc[crops.true_label.eq(1)&~crops.burst_id.isin([619, 620])&crops.length.eq(length)]
            distances = np.linalg.norm((candidates[FEATURES]-target).to_numpy()/normal_std.to_numpy(), axis=1)
            for j, (_, candidate) in enumerate(candidates.iterrows()):
                parent = clean.loc[clean.burst_id.eq(candidate.burst_id), FEATURES].iloc[0]
                distance_rows.append(dict(fn_burst_id=bid, length=length, crop_id=candidate.crop_id,
                    parent_burst_id=int(candidate.burst_id), position=candidate.position, crop_distance=distances[j],
                    parent_distance=float(np.linalg.norm((parent-target)/normal_std)),
                    crop_probability_mean=crop_probs.loc[candidate.crop_id],
                    parent_probability_mean=original.loc[original.burst_id.eq(candidate.burst_id), 'probability_y1'].mean()))
    distances = pd.DataFrame(distance_rows)
    distances.to_csv(OUT/'fn_vs_abnormal_crop_distances.csv', index=False, encoding='utf-8-sig')
    nearest = distances.sort_values('crop_distance').groupby(['fn_burst_id', 'length']).head(1)
    nearest.to_csv(OUT/'fn_nearest_abnormal_crops.csv', index=False, encoding='utf-8-sig')
    primary = metrics.loc[metrics.cohort.eq('common_N_ge20')&metrics.aggregation.eq('crop_OOF')].set_index('length').loc[['original', '20', '15', '10']]
    for column, name, ylabel in [('Recall', 'length_vs_recall', 'Abnormal recall'), ('abnormal_probability_mean', 'length_vs_mean_probability', 'Mean abnormal probability')]:
        fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(primary.index, primary[column], marker='o')
        if column != 'Recall':
            ax.plot(primary.index, primary.normal_probability_mean, marker='o', label='Normal'); ax.legend()
        ax.set(xlabel='Crop samples (original = full parent)', ylabel=ylabel, title='Same eligible parent cohort N >= 20')
        ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(plots/f'{name}.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for feature in FEATURES:
        values = drift_summary.loc[drift_summary.cohort.eq('common_N_ge20')&drift_summary.true_label.eq(1)&drift_summary.feature.eq(feature)].sort_values('length')
        ax.plot(values.length, values.mean_absolute_standardized_delta, marker='o', label=feature)
    ax.set(xlabel='Crop samples', ylabel='Mean absolute drift / normal original SD', title='Abnormal feature sensitivity'); ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(plots/'feature_drift_by_length.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    shifts = transitions.loc[transitions.cohort.eq('common_N_ge20')&transitions.true_label.eq(1)]
    for bid, values in shifts.groupby('burst_id'):
        values = values.sort_values('length'); ax.plot(values.length, values.crop_probability_mean, marker='o', label=str(bid))
    ax.axhline(.5, color='black', linestyle='--'); ax.set(xlabel='Crop samples', ylabel='Mean P(abnormal)', title='Originally TP parents: crop probability')
    ax.legend(ncol=4, fontsize=8); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(plots/'tp_to_fn_parent_probability_shift.png', dpi=180); plt.close(fig)
    for p in paths:
        assert sha(p) == hashes[str(p.relative_to(ROOT))]
    affected = transitions.loc[transitions.true_label.eq(1)].groupby(['cohort', 'length']).agg(
        eligible_original_TP_parents=('burst_id', 'size'), affected_any_FN=('any_error', 'sum'),
        persistent_all_crop_fold_FN=('persistent_error', 'sum'), FN_predictions=('error_count', 'sum'), n_predictions=('n_predictions', 'sum')).reset_index()
    affected.to_csv(OUT/'transition_summary.csv', index=False)
    ranking = drift_summary.loc[drift_summary.cohort.eq('common_N_ge20')&drift_summary.true_label.eq(1)].groupby('feature').mean_absolute_standardized_delta.mean().sort_values(ascending=False)
    metadata = dict(input_sha256=hashes, features=FEATURES, class_weight=None, threshold=.5, decision_rule='strict P(y=1)>0.5',
        classifier_fits=0, scaler_recovery='StandardScaler.fit(original saved training indices), no crop fitting',
        saved_folds=50, repeats=10, max_original_probability_error=max(x['max_probability_error'] for x in verification),
        crop_positions='start=0; middle=floor((N-L)/2); end=N-L; identical offsets deduplicated',
        primary_cohort='same original parents N>=20 at original/20/15/10',
        diagnostic_parent_aggregation='mean probability over unique positions within parent/repeat, then strict >.5; optional mean over repeats',
        dependency_warning='crops and repeated OOF predictions are dependent; no new official evaluation protocol',
        any_FN='at least one crop/fold prediction is FN', persistent_FN='all crop/fold predictions are FN',
        feature_drift_reference='original normal feature SD ddof=1, descriptive only',
        Python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
        preprocessing_warnings=list(set(str(w.message) for w in recorded)))
    (OUT/'experiment_metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    report = '# Frozen-model burst length diagnostic\n\nNo classifier was fitted. All 50 saved coefficients and recovered intercepts were reused with original fold scalers; original probabilities reproduced to maximum error '+str(metadata['max_original_probability_error'])+'. Signed notebook features and strict threshold >0.5 are unchanged. No ACF, imputation, threshold tuning or label changes.\n\n'
    report += 'Original full dataset: 575 normal + 20 abnormal; recall 0.9 (619/620 always FN), normal FP 0. Common N>=20 has 13 originally detected abnormal parents; the other five originally TP parents (600,604,609,610,614) are ineligible for length20. Original common-cohort recall is 1.0.\n\n'
    report += 'Primary metrics (same N>=20 parents; dependent crop OOF observations):\n\n'+primary.to_csv()+'\n\nTransition counts (honest eligible-original-TP denominators, not all 18):\n\n'+affected.to_csv(index=False)
    report += '\n\nAny FN means at least one position/fold FN; persistent FN means every position/fold FN. Normal TN->FP uses identical definitions. Each physical crop contributes once to drift summaries, rather than ten repeated fold observations. Standardized feature sensitivity ranking:\n\n'+ranking.to_string()
    report += '\n\nNearest originally detected abnormal crops to FN619/620 (five-feature Euclidean distance divided by original normal SD):\n\n'+nearest.to_csv(index=False)
    report += '\n\nParent-repeat results average probabilities across deduplicated positions before applying >0.5. Parent-allrepeat results additionally average repeats. These are diagnostic aggregations, not a change to the official evaluation protocol. Physical crops and repeated validation measurements are dependent; counts are not independent sample counts. Distances describe representation similarity and cannot establish causation or label error.\n'
    report += '\n## Interpretation\n\nShortening worsens detection in the same 13 abnormal / 452 normal parent cohort: crop OOF recall 100% -> 74.87% -> 66.67% -> 56.15% for original/20/15/10. Normal FP remains zero. Parent-repeat mean-position diagnostic recall also falls (100%, 76.92%, 70.77%, 59.23%).\n\n'
    report += 'Among actually shortened originally TP parents, any-FN transitions affect 12/15 eligible at length10 and 9/14 at length15 (out of 18 original TPs, three/four respectively are ineligible). Only burst603 is persistent FN across every position and all 10 repeats at both lengths; many parents lose detection at some positions rather than all positions.\n\n'
    report += 'AI0_RMS has the largest mean absolute drift standardized by original normal SD at every length, followed by AI0_MaxAbs. Raw-unit drift cannot fairly rank unlike feature units; AI1_Kurtosis has large raw numerical changes but is not the largest standardized drift.\n\n'
    report += 'The short-window explanation for 619/620 gains descriptive support: originally detected long abnormal bursts can yield short low-probability crops close to their five-feature vectors. At matching lengths, nearest crops are 603/end/10 for619 (distance1.504 versus parent13.763, mean P=.00134) and615/start/15 for620 (distance.699 versus parent25.010, mean P=.00458). This supports a length-and-window-content effect, not proof that length alone caused their original FN. Their observed raw content remains distinct and no label error is inferred.\n'
    report += '\n## 한국어 결론\n\n동일한 길이20 이상 코호트(이상13개·정상452개)에서 짧게 자를수록 이상 탐지율이 원본100% → 20개74.87% → 15개66.67% → 10개56.15%로 하락했습니다. 정상 오탐은 모두0입니다.\n\n실제로 길이가 줄어든 원래 TP 중 일부 위치·fold에서 FN으로 바뀐 burst는 10개에서12/15개, 15개에서9/14개로 많습니다. 분모는 전체18개가 아니라 각 길이로 단축 가능한 TP이며, 모든 위치·10회 반복에서 FN인 경우는 두 길이 모두603 한 개입니다.\n\n단위 차이를 보정한 불안정성은 AI0_RMS가 가장 큽니다. 순위는 공통 코호트의 이상13개에서 각 물리 crop의 원본 대비 절대 변화량을 원본 정상575개의 feature 표준편차로 나눈 뒤, 길이10·15·20별 평균을 동일 가중치로 평균한 결과입니다. AI0_RMS는 세 길이 각각에서도1위입니다. 원시 수치로는 AI1_Kurtosis 변화량이 크지만 단위가 달라 직접 순위 비교할 수 없습니다.\n\n619·620의 짧은 구간 영향은 설명력을 얻었습니다. 탐지되던 긴 이상 burst를 자른 일부 구간이 두 FN의 feature 벡터와 가까워지고 확률도 매우 낮아졌습니다. 다만 길이와 구간 내용이 함께 변하므로 길이만의 인과관계가 입증된 것은 아니며, label 오류도 추정하지 않았습니다.\n'
    (OUT/'analysis_report.md').write_text(report, encoding='utf-8')
    print(primary[['n_unique_abnormal_bursts', 'Recall', 'abnormal_probability_mean', 'normal_FP_rate']].to_string())
    print(affected.to_string(index=False)); print(ranking.to_string()); print(nearest.to_string(index=False))
    print('Saved', OUT)


if __name__ == '__main__':
    main()
