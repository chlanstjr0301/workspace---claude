# 기존 최우수 PCA의 미탐 감소를 위한 문헌 기반 제한 실험

작성일 2026-10-04 · 실행 pca_literature_rescue_20261004_210504 · 로컬 CPU 2 threads

## 결론

**선정 후보 없음. 기존 PCA 유지.** 6개 보조 점수 × 3개 보정 예산, 총18개 운영 구성을 실행했다. 원모델의 경보를 보존하는 OR만 허용했다. 선택에서 모든 조건을 통과한 후보는0개이며 후반/과거 결과로 탈락 후보를 다시 선정하지 않았다.

기존 원모델은 이미 노출된 과거 평가의 **실제 이상357개 중329개 탐지,28개 미탐; 정상3990개 중1개 오경보**로 재현됐다. 정상1000관측당 오경보 0.250627개다. Recall 0.921569, Precision 0.996970, F1 0.9577874818, F2 0.9357224118, FPR 0.00025063, AP 0.98709814. 배포하거나 원모델을 덮어쓰지 않았다.

선택 pooled만 보면 R5_A0/R5_A1(2시점 예측잔차)이 FN3→1, FP8→9, F1 0.765957→0.800000, F2 0.818182→0.884956이었다. **이는 적격 개선이 아니다.** S2의 정상616개 중 FP6→7, FPR0.974026%→1.136364%로 바뀌어, 추가FP예산0과 절대1%를 모두 위반했다. S2의 F1 0.500000→0.461538, F2 0.714286→0.681818도 악화했다. R4의 추가TP1 역시 같은 문제로 탈락했다. 이를 단순히 F2 개선 성공이라고 보고하면 시간 구간 검증 조건을 빠뜨리게 된다.

| 목표 | 판정 | 근거 |
| --- | --- | --- |
| (a) 같은 행에서 FN 감소 | 선택 pooled 일부 후보에서 관찰 | R5 FN3→1, R4 FN3→2; 기존 TP 손실0 |
| (b) 사전 추가FP예산 | 주 개선 후보 없음 | R4/R5 S2 추가FP1 > 허용0; 절대FPR1%도 초과 |
| (c) F2 증가와 F1 비하락 | pooled 일부 통과, 필수 블록 조건 실패 | S2 F1/F2 하락 |
| (d) 선택하지 않은 후반 시간 검증 | 통과 후보 없음 / 개선 확인 불가 | 선정 후보가 없고 V 원모델 FN=0. R0 대조군도 개선0 |

## 원모델 재현과 반복 개발 이력

P1은 같은 파일·split·train_role 안에서 긴 공백을 넘어 과거20행을 잇는다. 센서별 mean/std(ddof=0)/RMS/min/max, 총15특징을 정상 학습으로 표준화하고 full-SVD PCA의2개 성분을 유지한다. Q는 표준화 공간 복원오차의 제곱합이다. 부족한 창은 원래 P0 W1 k2 Q로 보완한다. 각각 정상calibration Q95로 나눈 점수를 연결하고 원 임계값1.3078792257682357보다 클 때 경보, 후처리 없음이다. 날짜·라벨·파일명·행번호·burst_id는 센서 특징에 포함하지 않았다.

이번에는 저장 모델을 별도 프로세스에서 라벨 없는 입력으로 추론해4347행의 ID·점수·판정을 대조했다. 원설정 재학습은 직전 회차에서 수행한 기록과 구분한다. 이번 새 학습은 보조 모델에 한정했다. 근거는 reproduction.json, reproduction_report_ko.md, models/baseline.joblib, inputs/original_expected_test.csv다.

직전 median/IQR은 원모델보다 Recall이 높아 선정된 것이 아니었다. 당시 내부99이상에서83TP/8FP였고 원15특징 target1%는85TP/31FP로 FPR 제약 미달이었다. 당시 적격 후보군에서는 선정됐지만, 고정 뒤 개발 후반 및 과거 평가에서 악화했다. 이후 3행 보완은 개발109TP/2FN/0FP였으나 과거323TP/34FN/0FP로 원모델보다 미탐이 늘었다. 두 변경은 이번에 반복하지 않았다. inputs/previous_*_report.md와 all_trials 사본, archive_audit.json에 근거를 남겼다.

원본 정상20000행에서 TimeStamp·센서·Equipment_state가 같은 추가중복1행만 제거해19999행, 이상600행이다. 행 번호를 보존했다. 정상/이상 날짜는2022-07-12/2022-07-17로 달라 날짜와 상태가 얽혀 있다. 주 간격은0.1초지만 >0.5초 공백은 정상598·이상20개이고 최대16.643/8.572초다. 큰 값과 음의 전류는 제거하지 않았다. 단위·AI0/AI1의 상하 대응·운전 부하는 확인되지 않았다.

## 데이터 역할과 사전 등록

정상 train12012행/360버스트에서만 원 전처리와 새 보조 모델을 적합했다. 정상calibration2011행/63버스트는 임계값에만 썼다. 이상calibration132행은 새 학습/보정에 사용하지 않았다. 원 selection 정상1986행/58버스트·이상111행/4버스트를 각 클래스의 시간순 burst 개수로 나누었다. 서로 다른 두 날짜를 동시에 관측한 정상/이상처럼 묶지 않았다.

| block | label | rows | bursts | first | last |
| --- | --- | --- | --- | --- | --- |
| S1 | 0 | 639 | 19 | 2022-07-12 00:53:19.759000 | 2022-07-12 00:55:46.351000 |
| S2 | 0 | 616 | 19 | 2022-07-12 00:55:52.785000 | 2022-07-12 00:58:22.487000 |
| V | 0 | 731 | 20 | 2022-07-12 00:58:24.593000 | 2022-07-12 01:01:01.732000 |
| S1 | 1 | 18 | 1 | 2022-07-17 10:51:39.999000 | 2022-07-17 10:51:41.699000 |
| S2 | 1 | 3 | 1 | 2022-07-17 10:51:48.093000 | 2022-07-17 10:51:48.293000 |
| V | 1 | 90 | 2 | 2022-07-17 10:51:56.684000 | 2022-07-17 10:52:08.307000 |


S1+S2는 정상1255·이상21, 양성비율 1.645768%이다. V는 정상731·이상90이다. 정상train/calibration/selection/test 원분할은 유지했다. S/V는 원selection 안의 보고·선택 역할 분리다. 원모델의 기존 윈도 상태는 reporting block에서 임의 초기화하지 않는다. 새 예측잔차는 파일·split·train_role·gap>.5초마다 초기화한다. S2/V 첫 버스트의 과거행을 갖는 원 P1의 인과적 상태 이월은 원정책 보존이며 fit 또는 임계값 재적합이 아니다. 이 평가를 독립 iid fold 또는 새 독립 테스트라고 부르지 않는다.

preregistration.json은 보조 점수 계산 전 기록했고 SHA-256은 `9397ad82d3cd18d2de5690b0a0250ffc04b6fd998e3e49fc772422e94a49d7e7`이다. 구현 해시는 fit 전에 implementation_lock.json으로, 선택은 V 개봉 전에 selected_config.json으로 고정했다. 과거 연구에서 개발·과거평가·FN 조건을 이미 보았고 이번도 반복 개발이다. 검출기 학습은 정상 전용이지만 운영 구성 선택은 개발 이상 라벨을 사용했다.

## 방법과 예산

원경보 A0 OR (aux_available AND aux_score > tau_aux)만 사용했다. 원 TP를 잃지 않지만 원 FP도 줄일 수 없다. 사용 불가능한 보조 경로는 원경보로 돌아가며 모든 원행이 평가에 남는다. 후처리·억제·다수결·point adjustment는 없다.

| block | normal | anomaly | baseline_FP | extra_FP_main | extra_FP_strict | extra_FP_exploratory | absolute_FP_cap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | 639 | 18 | 2 | 0 | 0 | 1 | 6 |
| S2 | 616 | 3 | 6 | 0 | 0 | 1 | 6 |
| S_pooled | 1255 | 21 | 8 | 1 | 0 | 2 | 12 |
| V | 731 | 90 | 0 | 0 | 0 | 1 | 7 |
| historical | 3990 | 357 | 1 | 3 | 0 | 7 | 39 |


주 예산은 ΔFPR≤0.001(0.1%p), 절대FPR≤1%다. 정수로 내림하며 작은 블록에 최소1개를 허용하지 않았다. S1/S2 각각 예산·F1/F2 비하락을 요구하고 pooled FN/F2 엄격 개선·F1 비하락을 요구했다. 동률1e-12, pooled F2→추가FP→FN→사전 계산량 순서→등록ID다. Δ0은 엄격 보조, Δ.002는 탈락 후보의 탐색적 표시에 불과하다. 현재18개는 엄격/탐색 예산에서도 완전 적격이 없다.

정상calibration 분모2011을 유지한 추가FP 예산0/0.0005/0.001은 정수0/1/2개다. A0가0이고 보조 점수가 있는 정상값을 내림차순 정렬해 k번째(0시작) 값을 임계값으로 삼아 strict > 판정했다. 동점은 경보하지 않는다. 없는 점수는 분모에 유지한다. 실제 threshold·동점 수는 thresholds.csv다. 이번 임계값은 모두 양수여서 통합 연속점수 max(base/h0,aux/tau)>1과 OR 판정이 일치한다. R0은 score/min(h0,tau)로 순위가 동일하다. 점수는 확률이 아니다.

- R0: 기존 점수의 경보 추가. 세 후보의 AP/사다리꼴PR-AUC/ROC-AUC가 원점수와 모두 동일함을 검사했다.
- R1: 원15차원 잔차 r=(I−PPᵀ)(z−mu), 정상train11974개에 LedoitWolf. 잔차 rank13, shrinkage 0.00083921155, 최소고유값 0.00025039862, 조건수 6340.7661, epsilon 2.984e-11. Cholesky 선형해법으로 거리 계산. Q와 Spearman 0.800151, 순위는 달랐으나 선택에서 추가TP는 없었다. PCA 고유벡터 재학습의 효과라고 주장하지 않는다.
- R2/R3: 정상train에서 모든 eligible burst를 덮는 결정적4096행을 먼저 고정했다. 잔차 축을 추가 표준화하지 않았고 중앙 쌍거리²는 7.158131318; 두 대역폭만 썼다. C=0.0244140625, ν=.01. libsvm 계수 β를 νN으로 나눠 α의 합1/상한C를 확인했다. 중심거리 1−2Kα+αᵀKα를 직접 계산했다. 128정상행의 직접 제약QP와 목적함수 차이는 4.78e-14/7.92e-14, 거리 최대차이는 2.35e-07/2.13e-07. default predict()를 사용하지 않았다. PCA+SVDD 하이브리드다.
- R4/R5: 원센서3개를 정상train으로 표준화하고 동일burst 내부 lag1/2로 현재 센서를 예측한다. 중심화 ridge와 비정규화 절편, lambda=1e-3 trace(UᵀU/n)/dimU. 학습행11652/11296, lambda 0.0010002378/0.0010006425. 예측오차의 정상 LW 거리를 사용한다. 현재까지 L개가 연속할 때만 활성화한다. CVA의 Rs/Rr가 아닌 PCA+인과적 예측잔차 하이브리드다.

## 선택 결과: 실패 후보도 모두 공개

다음은 동일1276행에서 계산한 결과다. 임계값 단위는 보조 점수별로 다르므로 크기를 서로 비교하지 않는다. 모델 학습·계산 시간과 disabled/중단 여부는 all_trials.csv/model_diagnostics.json/logs에 있다.18개 모두 실행 완료했으며 실패/중단한 fit은 없었다. 선택 단계 경과시간은 3.336초로90분 예산보다 짧았다. PCA seed를 바꾼 가짜 반복을 하지 않았다.

| candidate | threshold | TP | FN | FP | new_TP | new_FP | F1 | F2 | FPR | eligible | failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R0_A0 | 1.307879 | 18 | 3 | 8 | 0 | 0 | 0.765957 | 0.818182 | 0.006375 | False | S_pooled:FN_decreased;S_pooled:F2 |
| R0_A1 | 1.307875 | 18 | 3 | 8 | 0 | 0 | 0.765957 | 0.818182 | 0.006375 | False | S_pooled:FN_decreased;S_pooled:F2 |
| R0_A2 | 1.301088 | 18 | 3 | 8 | 0 | 0 | 0.765957 | 0.818182 | 0.006375 | False | S_pooled:FN_decreased;S_pooled:F2 |
| R1_A0 | 100.366610 | 18 | 3 | 8 | 0 | 0 | 0.765957 | 0.818182 | 0.006375 | False | S_pooled:FN_decreased;S_pooled:F2 |
| R1_A1 | 93.438998 | 18 | 3 | 8 | 0 | 0 | 0.765957 | 0.818182 | 0.006375 | False | S_pooled:FN_decreased;S_pooled:F2 |
| R1_A2 | 84.661460 | 18 | 3 | 9 | 0 | 1 | 0.750000 | 0.810811 | 0.007171 | False | S2:F2;S2:F1;S2:FP_budget;S2:absolute_FPR;S_pooled:FN_decreased;S_pooled:F2;S_pooled:F1 |
| R2_A0 | 0.982680 | 18 | 3 | 9 | 0 | 1 | 0.750000 | 0.810811 | 0.007171 | False | S1:F2;S1:F1;S1:FP_budget;S_pooled:FN_decreased;S_pooled:F2;S_pooled:F1 |
| R2_A1 | 0.980841 | 18 | 3 | 9 | 0 | 1 | 0.750000 | 0.810811 | 0.007171 | False | S1:F2;S1:F1;S1:FP_budget;S_pooled:FN_decreased;S_pooled:F2;S_pooled:F1 |
| R2_A2 | 0.980242 | 18 | 3 | 10 | 0 | 2 | 0.734694 | 0.803571 | 0.007968 | False | S1:F2;S1:F1;S1:FP_budget;S_pooled:FN_decreased;S_pooled:F2;S_pooled:F1;S_pooled:FP_budget |
| R3_A0 | 0.885533 | 18 | 3 | 10 | 0 | 2 | 0.734694 | 0.803571 | 0.007968 | False | S1:F2;S1:F1;S1:FP_budget;S_pooled:FN_decreased;S_pooled:F2;S_pooled:F1;S_pooled:FP_budget |
| R3_A1 | 0.866717 | 18 | 3 | 17 | 0 | 9 | 0.642857 | 0.756303 | 0.013546 | False | S1:F2;S1:F1;S1:FP_budget;S2:F2;S2:F1;S2:FP_budget;S2:absolute_FPR;S_pooled:FN_decreased;S_pooled:F2;S_pooled:F1;S_pooled:FP_budget;S_pooled:absolute_FPR |
| R3_A2 | 0.865233 | 18 | 3 | 17 | 0 | 9 | 0.642857 | 0.756303 | 0.013546 | False | S1:F2;S1:F1;S1:FP_budget;S2:F2;S2:F1;S2:FP_budget;S2:absolute_FPR;S_pooled:FN_decreased;S_pooled:F2;S_pooled:F1;S_pooled:FP_budget;S_pooled:absolute_FPR |
| R4_A0 | 21.147751 | 19 | 2 | 9 | 1 | 1 | 0.775510 | 0.848214 | 0.007171 | False | S2:F2;S2:F1;S2:FP_budget;S2:absolute_FPR |
| R4_A1 | 20.725790 | 19 | 2 | 9 | 1 | 1 | 0.775510 | 0.848214 | 0.007171 | False | S2:F2;S2:F1;S2:FP_budget;S2:absolute_FPR |
| R4_A2 | 15.142193 | 19 | 2 | 11 | 1 | 3 | 0.745098 | 0.833333 | 0.008765 | False | S1:FP_budget;S2:F2;S2:F1;S2:FP_budget;S2:absolute_FPR;S_pooled:F1;S_pooled:FP_budget |
| R5_A0 | 19.486921 | 20 | 1 | 9 | 2 | 1 | 0.800000 | 0.884956 | 0.007171 | False | S2:F2;S2:F1;S2:FP_budget;S2:absolute_FPR |
| R5_A1 | 17.543122 | 20 | 1 | 9 | 2 | 1 | 0.800000 | 0.884956 | 0.007171 | False | S2:F2;S2:F1;S2:FP_budget;S2:absolute_FPR |
| R5_A2 | 15.904632 | 20 | 1 | 10 | 2 | 2 | 0.784314 | 0.877193 | 0.007968 | False | S2:F2;S2:F1;S2:FP_budget;S2:absolute_FPR;S_pooled:FP_budget |


| candidate | block | TP | FN | FP | TN | Recall | Precision | F1 | F2 | FPR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | S1 | 15 | 3 | 2 | 637 | 0.833333 | 0.882353 | 0.857143 | 0.842697 | 0.003130 |
| baseline | S2 | 3 | 0 | 6 | 610 | 1.000000 | 0.333333 | 0.500000 | 0.714286 | 0.009740 |
| baseline | S_pooled | 18 | 3 | 8 | 1247 | 0.857143 | 0.692308 | 0.765957 | 0.818182 | 0.006375 |
| R4_A0 | S1 | 16 | 2 | 2 | 637 | 0.888889 | 0.888889 | 0.888889 | 0.888889 | 0.003130 |
| R4_A0 | S2 | 3 | 0 | 7 | 609 | 1.000000 | 0.300000 | 0.461538 | 0.681818 | 0.011364 |
| R4_A0 | S_pooled | 19 | 2 | 9 | 1246 | 0.904762 | 0.678571 | 0.775510 | 0.848214 | 0.007171 |
| R5_A0 | S1 | 17 | 1 | 2 | 637 | 0.944444 | 0.894737 | 0.918919 | 0.934066 | 0.003130 |
| R5_A0 | S2 | 3 | 0 | 7 | 609 | 1.000000 | 0.300000 | 0.461538 | 0.681818 | 0.011364 |
| R5_A0 | S_pooled | 20 | 1 | 9 | 1246 | 0.952381 | 0.689655 | 0.800000 | 0.884956 | 0.007171 |


![FN FP 교환](figures/fn_fp_tradeoff.png)
![F1 F2](figures/f1_f2_budgets.png)
![PR ROC](figures/selection_pr_roc.png)

R1/R2/R3은 원 W20 잔차가 있을 때만 활성화한다. 개발의 원FN3개는 P0 fallback에 있어 이 세 보조 모델은 해당 미탐에 점수를 낼 수 없었다. 차원을 달리하는 P0 잔차를 임의 혼합하지 않았다. R4/R5는 일부 초기 미탐을 포착했지만 정상구간에 추가 경보를 만들었다. 조건별 분모와 FN/FP는 tables/error_conditions.csv, burst별 분포와 가용성은 tables/burst_errors.csv 및 coverage.csv다. 미래의 burst_length는 진단 표를 만드는 데만 썼다.

![공백을 끊은 오류 시간축](figures/error_timeline_separate_dates.png)

## V와 제한된 과거 평가

선정 후보가 없어 R1–R5를 V/과거test에 추가 적용하지 않았다. 선택 자료에서 미리 고른 R0_A0은 임계값이 원모델과 정확히 같았다. 두 이름으로 대조 저장했지만 실제 서로 다른 운영 구성은1개다. 이 중복을 새 개선 또는 독립 반복으로 세지 않는다.

| candidate | block | TP | FN | FP | TN | Recall | Precision | F1 | F2 | FPR | AP | PR_AUC_trapezoid | ROC_AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | V | 90 | 0 | 0 | 731 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 |
| R0_A0 | V | 90 | 0 | 0 | 731 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 |
| baseline | historical | 329 | 28 | 1 | 3989 | 0.921569 | 0.996970 | 0.957787 | 0.935722 | 0.000251 | 0.987098 | 0.987086 | 0.996203 |
| R0_A0 | historical | 329 | 28 | 1 | 3989 | 0.921569 | 0.996970 | 0.957787 | 0.935722 | 0.000251 | 0.987098 | 0.987086 | 0.996203 |


V 원모델 FN0은 FN의 엄격 감소가 불가능한 천장 상태다. 새로 수집한 미노출 독립 검증 통과가 아니며, 개선 확인 불가로 기록한다. 탈락 R5를 과거test에서 평가하지 않았으므로 그 과거 성능 수치는 없다. 과거test 결과를 보고 후보나 임계값을 재선택하지 않았다.

![혼동행렬](figures/confusion_matrices.png)

## 검증과 한계

validation_report.json에서 원본/모델/코드 해시, 분할·창 원시행 경계, 정상전용 fit, 별도 프로세스 라벨 없는 추론, 원행 정렬, OR 불변조건, 미래 센서값 변경 시 과거 점수·판정 불변, 라벨 변경 시 점수 불변, gap 초기화, strict > 동점 처리와 비가용 분모를 검사했다. S1+S2의 모든18후보와 제한된 V/과거 예측을 저장했다. 원TP손실0·기존FP제거0이라는 OR 제약도 전부 맞았다.

추가TP/FP의 원본행은 paired_errors.csv다. 수집 공백을 정상 노출 시간에 넣지 않았으므로 FP/hour는 산출하지 않았다. alarm_episodes.csv는 행FP와 연속 경보 episode를 구분하고 버스트 경계에서 끊는다. 첫 관측부터 첫 경보까지의 실제 timestamp 차이를 기록하며 무경보 burst도 남긴다. 이 값은 실제 고장 시작이나 고장 전 예측 시간과 다르다.

이상selection은4개 burst뿐이고 두 클래스가 다른 날짜다. S2 이상3행은 한 burst로, 반복개발에서 매우 작은 평가 분모가 결정을 좌우한다. 이번에는 iid 행 bootstrap·p-value·PCA seed 반복을 하지 않았다. burst도 독립 고장 사건이 아니므로 미래 일반화를 보장하는 신뢰구간을 제시하지 않는다. 중첩창은 iid가 아니고 LW 원문의 가정을 그대로 충족하지 않는다. 10Hz만으로 앨리어싱 발생을 확정하거나 PCA가 면역이라고 주장하지 않는다.

(a) 일부 개발FN 감소와 (c) pooled F1/F2 이득은 관찰됐다. (b) 시간블록 오경보 예산은 실패했고 (d) 후반 개선은 확인되지 않았다. **오경보 증가 없는 적격 개선, 임계값만의 적격 개선, 최종 유지할 신규 후보 모두 없다. 기존 PCA를 유지한다.** 압력·불량·비가동·정비 이력이 없으므로 직접 예측하거나 감소시켰다고 할 수 없다. 다양한 날짜·부하·설비의 정상과 이상, 실제 상태 전환 및 정비 시각을 새로 모으기 전에는 현장 개선을 입증했다고 할 수 없다.

## 재현 및 파일 안내

이 폴더에서 `bash RUN_EXPERIMENT.sh`를 실행하면 별도의 새 재현 디렉터리에 원모델 추론 확인→보조 모델 재학습→선택 고정→V/과거 제한 평가→검증→보고서를 작성한다. 원본 결과를 덮어쓰지 않는다. Python3.14.4와 requirements.lock.txt의 버전이 실제 환경이다. 다른 환경에서는 프로젝트 전용 가상환경을 만들고 잠금 패키지를 설치한 뒤 `PYTHON_BIN=/path/to/python bash RUN_EXPERIMENT.sh`로 지정한다. 데이터·split·모델·코드가 함께 제공되며 인증값은 포함하지 않는다.

처음의 prepare.py는 기존 프로젝트 자료를 모아 등록한 출처 기록용이다. 전달 패키지의 재현은 replay.py가 동봉 입력·사전등록을 사용하므로 원래 과거 run 경로를 요구하지 않는다. source/model/data 해시는 inventory.json, implementation_lock.json, selected_config.json, 최종 MANIFEST_SHA256.json에 있다. 원문 재배포 대신 아래 확인기록과 URL을 제공한다.

## 전체 실행 재현 확인

완료 후 end_to_end_replay.json에 새 폴더의 재학습·후보 선택·행별 점수/판정 대조 결과를 저장한다. 단순 저장 모델 재추론과 보조 모델 전체 재학습 재현을 구분하며, 재실행은 독립 통계 반복이 아니다. 최종 전달 패키지의 해당 JSON에서 실제 통과 여부를 확인할 수 있다.

# 확인한 문헌과 이번 제안의 경계

확인일 2026-10-04. 원문 전체의 모든 명제를 검증했다고 주장하지 않는다. 아래 표의 지정 부분을 텍스트로 확인했고 P1 식14, P2 표4, P3 식13–14는 페이지 이미지도 확인했다. 원문 PDF/추출 텍스트는 로컬 검토용이며 전달 ZIP에는 재배포하지 않는다. 링크와 해시, 자체 작성한 이 요약을 제공한다.

| 문헌 | 실제 확인 부분 | 원문 근거와 한계 | 이번 실험과의 연결 |
| --- | --- | --- | --- |
| [P1] Ledoit, O.; Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. Journal of Multivariate Analysis 88, 365–411. [DOI](https://doi.org/10.1016/S0047-259X(03)00096-4), [공개 원문](https://www.econ.uzh.ch/dam/jcr:ffffffff-935a-b0d6-ffff-ffffceb83f14/wellCond.pdf) | Sections 2–3, 식14 인쇄380쪽/PDF16쪽, Section4, Conclusion388–389쪽 | 공분산과 구면행렬의 선형 수축으로 조건수를 개선한다. iid 관측을 전제로 한 공분산 위험의 논의이며 F1/F2 향상 정리가 아니다. | R1은 고정 PCA 잔차의 방향 가중치를 바꾸는 프로젝트 제안이다. 구면 수축은 고유벡터를 유지하므로 PCA Q만 재학습한 것을 새 개선이라 하지 않는다. 중첩 시계열 창의 의존성이 원 가정과 다르다. |
| [P2] Tax, D. M. J.; Duin, R. P. W. (2004). Support Vector Data Description. Machine Learning 54, 45–66. [DOI](https://doi.org/10.1023/B:MACH.0000008084.60811.49), [공개 원문](https://rduin.nl/papers/ML_SVDD_04.pdf) | Sections2.1,2.3–2.4; 식9–10/14/32, Section4 표3–4(61/63쪽), 결론64–65쪽 | 수중 펌프 진동의 주파수 특징 및 PCA 비교가 있으나 모든 차원에서 SVDD가 최상은 아니다. 매우 낮은 정상 거부율의 표본 한계를 기술한다. 상수 대각 Gaussian kernel에서 ν-SVC와 연결된다. | R2/R3은 기존 시간영역 PCA 잔차 위 정상전용 RBF-SVDD 하이브리드다. 정상화 dual과 중심거리 동등성을 작은 정상 표본의 QP로 확인한다. 원문의 주파수 특징·성능 수치를 전이하지 않는다. 이상 학습 표본은 쓰지 않는다. |
| [P3] Jiang, B.; Braatz, R. D. (2017). Fault detection of process correlation structure using canonical variate analysis-based correlation features. Journal of Process Control58,131–138. [DOI](https://doi.org/10.1016/j.jprocont.2017.09.003), [공개 원문](https://web.mit.edu/braatzgroup/Jiang_JProCon_2017.pdf) | 식8–14(132–133쪽), Sections3–5, 표2(PDF6쪽) | 모의 gene network, 500관측치 창이다. 명목1%와 독립 정상 평가에서 제안법 실제2.2% FPR는 다르다. CVA는 과거/미래 벡터의 관계를 사용한다. | R4/R5의 인과적 lag1/2 ridge 예측잔차는 상관구조 관점에 착안한 자체 제안이며 Rs/Rr CVA 재현이 아니다. t 이후 관측을 사용하지 않는다. 펌프 실증 또는 수백Hz 스펙트럼 복원을 주장하지 않는다. |
| [P4] Saito, T.; Rehmsmeier, M. (2015). The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. PLOS ONE10(3),e0118432. [DOI/원문](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432) | Table1, PRC 설명, simulation 분석, Conclusion; HTML 원문 | 불균형 자료에서 PR 곡선을 함께 보고 양성 예측의 신뢰도를 살펴야 한다. | 연속 점수 AP/PR 곡선을 보조 보고한다. F2 증가·F1 비하락·추가FPR0.001 조건을 이 논문이 정한 것은 아니다. |
| [P5] Cawley, G. C.; Talbot, N. L. C. (2010). On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation. JMLR11,2079–2107. [공식 페이지](https://jmlr.org/papers/v11/cawley10a.html), [원문](https://jmlr.org/papers/volume11/cawley10a/cawley10a.pdf) | Sections4–6; PDF6–7,17,19,24–25쪽 | 선택 기준 자체를 반복 최적화할 때 평가가 낙관적으로 될 수 있음을 다룬다. 공식 페이지에 확인되지 않는 DOI를 만들지 않았다. | 18개 제한·선택 잠금·후반 V 개봉 순서를 적용한다. 같은 이미 노출된 자료의 역할 분리는 새 독립 검증이 아니다. 무작위 iid CV를 시계열에 그대로 옮기지 않는다. |

공식 구현도 확인했다: [LedoitWolf](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html)의 구면 수축식과 중심화, [OneClassSVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html)의 RBF/ν 구현, [average_precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)의 비보간 AP와 사다리꼴 PR 면적의 차이. AP는 recall 증가량으로 precision을 가중하며 사다리꼴 PR-AUC와 별도로 계산한다.

6개 점수·3개 추가 보정 FP 예산·블록 분할·1% 상한·0.001 허용 증가·1e-12 동률·ridge 강도·lag·OR 보존은 이번 프로젝트의 사전 설계다. 문헌이나 대회가 필수로 정한 규칙이 아니다. F2의 가중치는 현장 비용 비율을 의미하지 않는다.
