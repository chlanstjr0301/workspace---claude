# Logistic Regression Current ACF feature ablation

본 비교의 baseline은 notebook cell 38의 반복 CV 평균(Recall=0.9, F1=0.7764314574314574, Balanced Accuracy=0.9418260869565219, ROC-AUC=0.9478260869565218, Precision=0.7254502164502165)이다. 추가 모델은 같은 5개 입력에 Current_ACF_PeakStrength만 추가하며, 동일한 50개 fold의 CV 평균을 주 결과로 비교한다. 고정 test는 부록 참고 결과다.

## 고정 조건

소성가공.ipynb cell 38의 best 5개 feature를 A로 사용한다. AI0_Kurtosis는 제외되어 있다. B는 Current_ACF_PeakStrength 하나만 추가한다. 기존 notebook 및 LSTM 원본 SHA256이 실행 전후 동일함을 확인했다.

A feature: `['AI0_RMS', 'AI0_MaxAbs', 'AI1_RMS', 'AI1_PeakToPeak', 'AI1_Kurtosis']`

정상/이상 CSV를 concat한 뒤 TimeStamp로 정렬하고, gap > 1초 또는 Equipment_state 변경으로 burst를 나눈다. 기존과 동일하게 길이 4 미만 burst를 제외한다. 진동 feature는 notebook 원래 signed 신호의 계산 함수를 그대로 실행했다. 새 abs 변환, 결측 대체, row 제거, feature 추가는 하지 않았다.

595 burst(정상 575, 이상 20), train 476(460/16), test 119(115/4). 노트북의 stratified test_size=0.2, random_state=42 split을 재현하고 A/B에 같은 burst index 및 순서를 적용했다. StandardScaler는 각 train에만 fit. LogisticRegression(class_weight='balanced', random_state=42)의 나머지 기본값을 유지한다. predict() 판정, positive label=1, 확률 기반 ROC-AUC. Test에 의한 threshold 또는 hyperparameter 선택은 없다.

## ACF 정의

전체 burst의 signed AI2_Current에서 평균을 제거한다. ACF(k)=sum(x_centered[t]*x_centered[t+k])/sum(x_centered[t]^2); lag별 overlap 보정은 하지 않는다. scipy.signal.find_peaks로 전체 ACF의 local peak를 구한 뒤 1≤lag≤floor(N/2), ACF>0인 후보의 최대값만 사용한다. 범위 끝에서도 실제 오른쪽 이웃과 비교한다. Constant/nonfinite/no positive peak는 0. PeakLag는 진단 파일에만 저장되며 입력에서 제외한다.

## 주 평가: 사용자 지정 baseline과 동일한 50-fold paired CV

RepeatedStratifiedKFold(5 folds, 10 repeats, random_state=42)를 전체 595 burst에서 재현한다. 두 조건이 동일 fold를 사용하고 각 fold train에서만 scaler를 fit한다. 노트북의 CV는 전체 데이터 대상이므로 고정 test와 독립된 추가 검증은 아니다. CV std는 notebook과 동일한 ddof=0.

| condition | metric | mean | std |
| --- | --- | --- | --- |
| A_Baseline | Accuracy | 0.980840 | 0.013344 |
| A_Baseline | Precision | 0.725450 | 0.197147 |
| A_Baseline | Recall | 0.900000 | 0.132288 |
| A_Baseline | F1 | 0.776431 | 0.112947 |
| A_Baseline | Balanced_Accuracy | 0.941826 | 0.062795 |
| A_Baseline | ROC_AUC | 0.947826 | 0.080256 |
| B_Plus_ACF | Accuracy | 0.979328 | 0.013881 |
| B_Plus_ACF | Precision | 0.705794 | 0.195901 |
| B_Plus_ACF | Recall | 0.900000 | 0.132288 |
| B_Plus_ACF | F1 | 0.763051 | 0.109383 |
| B_Plus_ACF | Balanced_Accuracy | 0.941043 | 0.062390 |
| B_Plus_ACF | ROC_AUC | 0.949348 | 0.078984 |

| metric | mean_delta_B_minus_A |
| --- | --- |
| Accuracy | -0.001513 |
| Precision | -0.019657 |
| Recall | 0.000000 |
| F1 | -0.013381 |
| Balanced_Accuracy | -0.000783 |
| ROC_AUC | 0.001522 |

### Notebook 저장 baseline CV 재현 확인

| metric | notebook | current | delta |
| --- | --- | --- | --- |
| Precision | 0.725450 | 0.725450 | 0.000000 |
| Recall | 0.900000 | 0.900000 | 0.000000 |
| F1 | 0.776431 | 0.776431 | 0.000000 |
| Balanced_Accuracy | 0.941826 | 0.941826 | 0.000000 |
| ROC_AUC | 0.947826 | 0.947826 | 0.000000 |

## 부록: 고정 test 결과

| condition | Accuracy | Precision | Recall | F1 | Balanced_Accuracy | ROC_AUC | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_Baseline | 0.991597 | 0.800000 | 1.000000 | 0.888889 | 0.995652 | 1.000000 | 114 | 1 | 0 | 4 |
| B_Plus_ACF | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 115 | 0 | 0 | 4 |

### 변화량 B−A

| comparison | Accuracy | Precision | Recall | F1 | Balanced_Accuracy | ROC_AUC |
| --- | --- | --- | --- | --- | --- | --- |
| B-A | 0.008403 | 0.200000 | 0.000000 | 0.111111 | 0.004348 | 0.000000 |

## 계수 및 기술통계

| condition | feature | coefficient | absolute_coefficient | absolute_rank |
| --- | --- | --- | --- | --- |
| A_Baseline | AI0_RMS | 1.365344 | 1.365344 | 1 |
| A_Baseline | AI1_RMS | -0.841472 | 0.841472 | 2 |
| A_Baseline | AI1_Kurtosis | 0.777585 | 0.777585 | 3 |
| A_Baseline | AI0_MaxAbs | 0.662383 | 0.662383 | 4 |
| A_Baseline | AI1_PeakToPeak | -0.273045 | 0.273045 | 5 |
| B_Plus_ACF | AI0_RMS | 1.334603 | 1.334603 | 1 |
| B_Plus_ACF | AI1_RMS | -0.965856 | 0.965856 | 2 |
| B_Plus_ACF | AI0_MaxAbs | 0.732772 | 0.732772 | 3 |
| B_Plus_ACF | AI1_Kurtosis | 0.694576 | 0.694576 | 4 |
| B_Plus_ACF | Current_ACF_PeakStrength | -0.453717 | 0.453717 | 5 |
| B_Plus_ACF | AI1_PeakToPeak | -0.239525 | 0.239525 | 6 |

| class | count | mean | median | std |
| --- | --- | --- | --- | --- |
| normal | 575 | 0.374491 | 0.554834 | 0.309125 |
| abnormal | 20 | 0.238693 | 0.244440 | 0.205560 |

ACF 표준화 계수 -0.453717, 절댓값 순위 5/6. 계수 크기는 표준화 입력 기준의 상대적 영향이며 독립적 또는 인과적 기여의 증명은 아니다.

## 해석

고정 test에서는 기존 정상 오탐 1건이 사라져 F1이 0.888889→1.000000으로 개선됐다. Recall과 ROC-AUC는 이미 1.0이므로 변화가 없다. 그러나 기존 best를 평가한 반복 CV에서는 F1과 Precision, Balanced Accuracy가 소폭 하락하고 Recall은 동일하며 ROC-AUC만 소폭 상승했다. 따라서 단일 test에서는 도움이 됐지만, 전반적이고 일관된 개선이 확인됐다고 결론 내릴 수 없다.

- F1: 고정 test Δ=+0.111111, CV 평균 Δ=-0.013381.
- Recall: 고정 test Δ=+0.000000, CV 평균 Δ=+0.000000.
- Balanced_Accuracy: 고정 test Δ=+0.004348, CV 평균 Δ=-0.000783.
- ROC_AUC: 고정 test Δ=+0.000000, CV 평균 Δ=+0.001522.

고정 test의 이상 burst가 4개뿐이므로 Recall은 한 건당 0.25 변한다. 50개 CV fold는 반복에 따른 중복 표본으로 서로 독립이 아니다. 전체 지표와 CV에서 나타난 변화를 함께 해석하고, 계수만으로 성능 개선을 주장하지 않는다. 기존 best feature 선택 이력은 그대로 유지했으며 별도의 tuning이나 모델 선택은 수행하지 않았다.
