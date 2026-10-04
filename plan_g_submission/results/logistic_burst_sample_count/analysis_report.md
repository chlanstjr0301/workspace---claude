# 표본 수에 따른 frozen-model 진단

분류기를 다시 학습하지 않고 기존50개 fold의 계수·복원 절편과 원본 train의 StandardScaler 통계를 재사용했습니다. 원본 예측 확률 최대 오차는 4.218847493575595e-15입니다. 원본 전체575정상·20이상의 Recall은0.9이며619·620은 원래FN입니다.

## 길이별 결과

평균±표준편차는 위치별 crop을 repeat 안에서 모은10회 반복 기준(ddof=0)입니다. crop과 반복은 독립 표본이 아닙니다. 길이별 대상 수가 달라지므로 동일 대상의 원본 비교 및 고정 N>=30 코호트를 함께 저장했습니다.

| sample_count | n_unique_normal_bursts | n_unique_abnormal_bursts | Recall_mean | Recall_std | F1_mean | abnormal_probability_mean | FP_total | FN_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 575 | 20 | 0.46071 | 0.00714 | 0.63077 | 0.45729 | 0 | 302 |
| 6 | 565 | 18 | 0.43519 | 0.00926 | 0.60639 | 0.42197 | 0 | 305 |
| 8 | 550 | 18 | 0.46731 | 0.00881 | 0.63691 | 0.44954 | 0 | 277 |
| 10 | 530 | 17 | 0.52857 | 0.00612 | 0.69157 | 0.50916 | 0 | 231 |
| 12 | 511 | 16 | 0.5625 | 0.0 | 0.72 | 0.54979 | 0 | 210 |
| 15 | 480 | 16 | 0.68182 | 0.0 | 0.81081 | 0.66044 | 0 | 140 |
| 20 | 452 | 13 | 0.74872 | 0.01538 | 0.85622 | 0.72009 | 0 | 98 |
| 25 | 402 | 12 | 0.76471 | 0.0 | 0.86667 | 0.72351 | 0 | 80 |
| 30 | 360 | 10 | 0.82069 | 0.01379 | 0.90145 | 0.77971 | 0 | 52 |


고정 N>=30 코호트:

| sample_count | n_unique_normal_bursts | n_unique_abnormal_bursts | Recall_mean | F1_mean | abnormal_probability_mean |
| --- | --- | --- | --- | --- | --- |
| 4 | 360 | 10 | 0.4 | 0.57143 | 0.42571 |
| 6 | 360 | 10 | 0.40333 | 0.57475 | 0.39188 |
| 8 | 360 | 10 | 0.46667 | 0.63636 | 0.4395 |
| 10 | 360 | 10 | 0.49667 | 0.66364 | 0.46646 |
| 12 | 360 | 10 | 0.5 | 0.66667 | 0.48326 |
| 15 | 360 | 10 | 0.6 | 0.75 | 0.57491 |
| 20 | 360 | 10 | 0.67333 | 0.80461 | 0.63915 |
| 25 | 360 | 10 | 0.73333 | 0.84615 | 0.6886 |
| 30 | 360 | 10 | 0.82069 | 0.90145 | 0.77971 |

## TP→FN

분모는 원래18개TP 중 해당 길이에서 평가 가능한 부모 burst입니다. any_FN은 한 위치·fold 이상FN, persistent_FN은 모든 위치·fold에서FN입니다.

| cohort | sample_count | eligible_evaluated_original_TP | any_FN_parents | persistent_FN_parents | FN_predictions | n_predictions | any_FN_parent_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| common_N_ge30 | 4 | 10 | 10 | 1 | 180 | 300 | 1.0 |
| common_N_ge30 | 6 | 10 | 10 | 1 | 179 | 300 | 1.0 |
| common_N_ge30 | 8 | 10 | 10 | 1 | 160 | 300 | 1.0 |
| common_N_ge30 | 10 | 10 | 10 | 1 | 151 | 300 | 1.0 |
| common_N_ge30 | 12 | 10 | 10 | 1 | 150 | 300 | 1.0 |
| common_N_ge30 | 15 | 10 | 8 | 1 | 120 | 300 | 0.8 |
| common_N_ge30 | 20 | 10 | 8 | 0 | 98 | 300 | 0.8 |
| common_N_ge30 | 25 | 10 | 6 | 0 | 80 | 300 | 0.6 |
| common_N_ge30 | 30 | 10 | 5 | 0 | 52 | 290 | 0.5 |
| eligible_N_ge_target | 4 | 18 | 14 | 1 | 242 | 500 | 0.7777777777777778 |
| eligible_N_ge_target | 6 | 16 | 14 | 1 | 245 | 480 | 0.875 |
| eligible_N_ge_target | 8 | 16 | 13 | 1 | 217 | 460 | 0.8125 |
| eligible_N_ge_target | 10 | 15 | 12 | 1 | 191 | 450 | 0.8 |
| eligible_N_ge_target | 12 | 15 | 12 | 1 | 180 | 450 | 0.8 |
| eligible_N_ge_target | 15 | 15 | 9 | 1 | 130 | 430 | 0.6 |
| eligible_N_ge_target | 20 | 13 | 8 | 0 | 98 | 390 | 0.6153846153846154 |
| eligible_N_ge_target | 25 | 12 | 6 | 0 | 80 | 340 | 0.5 |
| eligible_N_ge_target | 30 | 10 | 5 | 0 | 52 | 290 | 0.5 |

## 619·620

길이가 부족한 경우 NA로 남겼고 padding하지 않았습니다.

| burst_id | parent_length | sample_count | eligible | status | n_physical_crops | n_predictions | probability_mean | probability_std | predicted_abnormal_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 619 | 10 | 4 | True | evaluated | 3 | 30 | 0.000371 | 0.000296 | 0.0 |
| 619 | 10 | 6 | True | evaluated | 3 | 30 | 0.00108 | 0.000467 | 0.0 |
| 619 | 10 | 8 | True | evaluated | 3 | 30 | 0.00195 | 0.001645 | 0.0 |
| 619 | 10 | 10 | True | evaluated | 1 | 10 | 0.001333 | 0.000387 | 0.0 |
| 619 | 10 | 12 | False | excluded_N_less_than_target | 0 | 0 | nan | nan | nan |
| 619 | 10 | 15 | False | excluded_N_less_than_target | 0 | 0 | nan | nan | nan |
| 619 | 10 | 20 | False | excluded_N_less_than_target | 0 | 0 | nan | nan | nan |
| 619 | 10 | 25 | False | excluded_N_less_than_target | 0 | 0 | nan | nan | nan |
| 619 | 10 | 30 | False | excluded_N_less_than_target | 0 | 0 | nan | nan | nan |
| 620 | 15 | 4 | True | evaluated | 3 | 30 | 0.008345 | 0.011372 | 0.0 |
| 620 | 15 | 6 | True | evaluated | 3 | 30 | 0.001348 | 0.001276 | 0.0 |
| 620 | 15 | 8 | True | evaluated | 3 | 30 | 0.004817 | 0.004111 | 0.0 |
| 620 | 15 | 10 | True | evaluated | 3 | 30 | 0.004685 | 0.006086 | 0.0 |
| 620 | 15 | 12 | True | evaluated | 3 | 30 | 0.001124 | 0.000572 | 0.0 |
| 620 | 15 | 15 | True | evaluated | 1 | 10 | 0.001028 | 0.000314 | 0.0 |
| 620 | 15 | 20 | False | excluded_N_less_than_target | 0 | 0 | nan | nan | nan |
| 620 | 15 | 25 | False | excluded_N_less_than_target | 0 | 0 | nan | nan | nan |
| 620 | 15 | 30 | False | excluded_N_less_than_target | 0 | 0 | nan | nan | nan |

## Feature 변화

고정 코호트 이상 부모에서 물리 crop을 한 번씩 계산한 절대 변화량 / 원본 정상575개 표준편차입니다. 아래 순위는9개 길이의 평균을 같은 가중치로 평균했습니다. 원시 단위는 feature 간 직접 비교할 수 없습니다.

| feature | mean_absolute_standardized_delta |
| --- | --- |
| AI0_RMS | 7.80186 |
| AI0_MaxAbs | 7.33072 |
| AI1_Kurtosis | 3.45183 |
| AI1_PeakToPeak | 2.79866 |
| AI1_RMS | 1.35589 |

계산 불가능한 crop은 0개입니다. 평가 가능·불가능 수를 별도 기록했으며 보간·대체·새로운 feature·threshold 조정은 하지 않았습니다. 위치 평균 후 threshold를 적용한 부모 단위 집계도 별도 저장했으며 공식 평가 방식을 바꾼 것이 아닙니다. 길이와 구간 내용이 함께 변하므로 결과만으로619·620의 길이 단독 인과나 label 오류를 주장할 수 없습니다.
- 10~15개는 탐지 누락 위험이 큽니다. eligible crop Recall은10개52.86%,15개68.18%; 원래 TP 중 하나 이상의 crop/fold가FN인 부모는 각각12/15개와9/15개입니다. 이번15개 조건은 N>=15라서 원본 길이15인 TP614를 포함하며, 이전 N>15 분석의9/14와 분모가 다릅니다. 고정 코호트의 TP10개 중10개에서는10/10,15개에서는8/10개가 일부FN입니다. 두 길이에서 모든 위치·fold가FN인 부모는603 하나입니다.
- 표준화 변화량의9개 길이 평균은AI0_RMS가1위(7.802 정상SD), AI0_MaxAbs가2위(7.331)입니다. 다만 극단적으로 짧은4·6·8·10개에서는AI0_MaxAbs가AI0_RMS보다 조금 큽니다. AI1_Kurtosis의 원시 숫자 변화가 크다는 사실과 단위 보정 순위를 구분해야 합니다.
- 619·620만의 특수 현상이라고 볼 수는 없습니다. 탐지되던 다른 이상 부모도 단축하면 많이FN으로 바뀝니다. 그러나619·620은 사용 가능한 모든 crop에서 계속FN이며 원래 길이보다 긴 입력을 실제 데이터로 만들 수 없으므로, 두 사례가 길이 때문에FN이었는지는 아직 확정할 수 없습니다. padding이나 label 오류 가정은 하지 않았습니다.
- 위 수치는 위치별 crop OOF 기준입니다. 위치 확률을 먼저 평균하는 부모 단위 진단은 별도 CSV에 있으며, 예를 들어 고정 코호트30개 Recall98%,15개62%,10개57%입니다. 두 집계 방식의 수치를 혼합하지 않습니다.
