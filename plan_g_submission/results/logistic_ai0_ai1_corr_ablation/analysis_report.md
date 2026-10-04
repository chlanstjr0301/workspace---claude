# Logistic Regression AI0–AI1 signed correlation ablation

Baseline A: `['AI0_RMS', 'AI0_MaxAbs', 'AI1_RMS', 'AI1_PeakToPeak', 'AI1_Kurtosis']`. AI0_Kurtosis는 제외한다. B는 `AI0_AI1_Corr` 하나만 추가한다.

Notebook cell6–9,27의 기존 preprocessing/feature 생성 코드를 그대로 실행했다. CSV concat 후 timestamp 정렬, gap>1초 또는 Equipment_state 변경에 따른 burst 분할, 길이4 미만 제외. 575 정상/20 이상 burst와 기존 feature 값이 이전 실험과 완전히 동일함을 확인했다.

추가 feature는 cell51과 동일하게 각 burst의 signed 원신호 AI0_Vibration과 AI1_Vibration 사이 Pearson correlation이다. NaN이면0, abs(corr)는 사용하지 않는다. 다른 feature·행·전처리는 변경하지 않았다.

StandardScaler + LogisticRegression(class_weight='balanced',random_state=42)의 다른 defaults 유지. RepeatedStratifiedKFold(5 folds,10 repeats,random_state=42)의 기존 저장 index를 재사용하고 notebook 생성 결과와 대조했다. 각 fold train에서만 scaler fit, 두 조건의 기존5개 feature scaler mean/scale도 동일함을 assert했다. 판정은 원래 predict(), positive class1, ROC-AUC는 확률 기준. Threshold/hyperparameter tuning은 없다. baseline 5개 CV 지표를 1e-12 이내 재현했으며 notebook/CSV/공식 baseline hash는 실행 전후 동일하다.

## 동일 50-fold CV 평균

| condition | Accuracy | Precision | Recall | F1 | Balanced_Accuracy | ROC_AUC | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_Baseline | 0.980840 | 0.725450 | 0.900000 | 0.776431 | 0.941826 | 0.947826 | 113.120000 | 1.880000 | 0.400000 | 3.600000 |
| B_Plus_Corr | 0.973109 | 0.621400 | 0.900000 | 0.709996 | 0.937826 | 0.941087 | 112.200000 | 2.800000 | 0.400000 | 3.600000 |

TN/FP/FN/TP는 fold별 평균이며 총 독립 sample 수가 아니다.

### B−A

| comparison | Accuracy | Precision | Recall | F1 | Balanced_Accuracy | ROC_AUC |
| --- | --- | --- | --- | --- | --- | --- |
| B-A | -0.007731 | -0.104051 | 0.000000 | -0.066436 | -0.004000 | -0.006739 |

### CV std (ddof=0)

| condition | Accuracy | Precision | Recall | F1 | Balanced_Accuracy | ROC_AUC |
| --- | --- | --- | --- | --- | --- | --- |
| A_Baseline | 0.013344 | 0.197147 | 0.132288 | 0.112947 | 0.062795 | 0.080256 |
| B_Plus_Corr | 0.015219 | 0.178555 | 0.132288 | 0.109600 | 0.061832 | 0.086901 |

## 표준화 coefficient

| condition | feature | coefficient | coefficient_std | mean_absolute_coefficient | mean_fold_absolute_rank | absolute_coefficient | absolute_rank |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A_Baseline | AI0_RMS | 1.483630 | 0.501100 | 1.483630 | 1.220000 | 1.483630 | 1 |
| A_Baseline | AI0_MaxAbs | 0.860838 | 0.321676 | 0.860838 | 2.900000 | 0.860838 | 2 |
| A_Baseline | AI1_Kurtosis | 0.796146 | 0.128333 | 0.796146 | 2.780000 | 0.796146 | 3 |
| A_Baseline | AI1_PeakToPeak | -0.618303 | 0.351549 | 0.641905 | 3.860000 | 0.618303 | 4 |
| A_Baseline | AI1_RMS | -0.508958 | 0.550106 | 0.583228 | 4.240000 | 0.508958 | 5 |
| B_Plus_Corr | AI0_RMS | 1.758110 | 0.389563 | 1.758110 | 1.060000 | 1.758110 | 1 |
| B_Plus_Corr | AI0_MaxAbs | 1.037397 | 0.266890 | 1.037397 | 2.820000 | 1.037397 | 2 |
| B_Plus_Corr | AI1_Kurtosis | 0.874762 | 0.169181 | 0.874762 | 3.360000 | 0.874762 | 3 |
| B_Plus_Corr | AI1_RMS | -0.753004 | 0.470331 | 0.768665 | 4.140000 | 0.753004 | 4 |
| B_Plus_Corr | AI1_PeakToPeak | -0.620303 | 0.353634 | 0.644646 | 4.660000 | 0.620303 | 5 |
| B_Plus_Corr | AI0_AI1_Corr | 0.513882 | 0.439592 | 0.634983 | 4.960000 | 0.513882 | 6 |

추가 correlation의 CV 평균 coefficient=0.513882, std=0.439592, 평균계수의 절댓값 순위=6/6. 평균 coefficient는 fold 모델을 요약한 값이며 단일 fit 모델 계수는 아니다. absolute_rank는 |mean(coefficient)| 기준, mean_absolute_coefficient와 fold별 rank도 별도로 저장했다.

## 클래스별 correlation 통계 (std ddof=1)

| class | count | mean | median | std |
| --- | --- | --- | --- | --- |
| normal | 575 | 0.232119 | 0.280143 | 0.298134 |
| abnormal | 20 | -0.295471 | -0.367582 | 0.330322 |

## 해석

- Accuracy: B−A=-0.007731.
- Precision: B−A=-0.104051.
- Recall: B−A=+0.000000.
- F1: B−A=-0.066436.
- Balanced_Accuracy: B−A=-0.004000.
- ROC_AUC: B−A=-0.006739.

CV 평균 F1은 개선되지 않았다. 각 지표의 방향과 변동을 함께 고려하며 coefficient 크기만으로 추가 효용을 결론 내리지 않는다. 반복 fold는 독립이 아니고 이상 burst는20개이므로 일반화·통계적 유의성으로 확대 해석하지 않는다.
