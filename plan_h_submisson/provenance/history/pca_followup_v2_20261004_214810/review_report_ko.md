# 기존 PCA의 확인 경보 제한과 V 유지 판정 분리 실험

작성일2026-10-04 · pca_followup_v2_20261004_214810 · 로컬CPU2 threads

## 결과

**선정 후보 G1_Q999.** 원PCA와 R5 lag2 예측기는 그대로 두고, 보조 점수가 현재와 직전 관측에서 모두 임계값을 넘을 때만 보조 경보를 추가하는 G1을 선택했다. 정상calibration R5 점수의 higher99.9백분위수는 **17.54312199063481**, 비교는 strict >다. B0 경보는 삭제하거나 지연하지 않는다. 모델 객체·센서·PCA차원·특징·seed는 바뀌지 않았고, 보조 경보의 임계값과 확인 규칙만 바뀌었다. 신규 검출기를 장비에 배포하거나 기존 파일을 덮어쓰지 않았다.

이미 공개된 과거 평가에서 이상357개 중 344개 탐지·13개 미탐, 정상3990개 중 1개 오경보였다. 기존329TP/28FN/1FP 대비 새TP15·새FN0·추가FP0·제거FP0이다.

| 결론 구분 | 이번 결과 | 해석 |
| --- | --- | --- |
| 평가 규칙 수정 | 효과와 유지 조건 분리 완료 | V의0→0FN,1→1F2는 유지 통과; 예측 자체를 바꾸는 수정 아님 |
| 선택 구간 개선 | B0 FN3→2, FP8→8, F1/F2 상승 | 재사용 S에서 사전 조건 통과 |
| 기존 R5와의 교환 | C0 FN1→2, FP9→8 | 추가 오경보1개를 없애면서 기존R5의 추가 탐지1행을 잃음 |
| V 확인 | 90TP/0FN/0FP 유지 | stability_pass=True, FN 감소 재현=not_assessable |
| 과거 평가 개선 | B0 FN28→13, FP1→1 | 공개된 H에서 관찰된 개선; 사후 재선택 없음 |
| 독립 새 자료 검증 | 수행하지 못함 | 새로운 세션·날짜·설비 자료가 없음; 현장 개선 입증 아님 |

## 1. 원모델과 직전 실행 재현

실제 저장 출력 ID는 **baseline**, bundle name은 **original**이었다. B0는 이번 보고의 별칭이며 이전 A0(t)는 원경보 함수의 수학적 표기다. 다른 과거 최고 모델로 기준선을 바꾸지 않았다.

원P1은 센서3개, 과거20행, mean/std(ddof0)/RMS/min/max 총15특징, 정상학습StandardScaler, PCA2성분 full SVD, Q 잔차다. 같은 원분할 안에서는 시간 공백을 넘어 행 창을 잇고 부족한19행은 원P0 W1 k2 Q로 보완한다. 정상calibration Q95 정렬과 원임계값1.3078792257682357 및 후처리없음을 그대로 유지했다. R5는 정상train만으로 적합한 raw sensor StandardScaler, lag2 ridge, 예측잔차 Ledoit–Wolf 거리다.

저장 B0/R5의 S 판정과 점수를 직전 파일에 대조했다. B0의 H는 라벨 없는 CSV를 별도 프로세스로 추론해329TP/28FN/1FP를 재현했다. 검증에서 R5 회귀·공분산을 T만으로 재구성해 저장 파라미터와 일치함도 확인했다. 이는 같은 고정모델 재현 검사이며 새 후보모델 학습이 아니다.

직전18개는 모두 S 선택에서 탈락하고 selected=null이었다. R5_A0/A1은 S1+S2 FN3→1, FP8→9였지만 S2에서 정상616행 중FP6→7, FPR0.974026%→1.136364%, F1/F2 하락으로 탈락했다. 이번 V 정책 변경으로 이 탈락을 취소하지 않는다. 직전 R5 V는 미평가였으며0FN이라고 가정하지 않았다. R5_A1은 S 판정은 A0와 같지만 임계값이 다르다. C0는 요청대로 기존 R5_A0 하나만 둔다.

원 코드·설정·모델·분할·예측 사본은 inputs/previous, models, manifests에 있다. inventory.json에 원경로와 SHA-256을 남겼으며 원파일 무변경을 검증했다.

## 2. 추가 FP 진단을 먼저 수행

diagnostic_plan.json에서 표시 범위를 과거19행·미래2행으로 고정하고 diagnosis_ko.md를 쓴 다음 protocol_v2.json을 잠갔다. 미래2행은 진단 그림에만 표시했다. 모델과 판정은 t까지의 정보만 쓴다.

| row_id | TimeStamp | label | burst_pos | AI0_Vibration | AI1_Vibration | AI2_Current | R5_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0:15270 | 2022-07-12 00:58:21.587 | 0 | 41 | 0.332216 | 0.328809 | -250.527050 | 22.680453 |
| 1:135 | 2022-07-17 10:51:40.199 | 1 | 3 | 0.006157 | -0.110532 | 85.830699 | 56.410560 |
| 1:140 | 2022-07-17 10:51:40.699 | 1 | 8 | -0.291867 | -0.456679 | -70.333489 | 175.810919 |


| row_id | change | block | previous_h1 | previous_h2 | next_h_diagnostic_only | isolated_both_neighbors |
| --- | --- | --- | --- | --- | --- | --- |
| 0:15270 | new_FP | S2 | False | False | False | True |
| 1:135 | new_TP | S1 | False | False | True | False |
| 1:140 | new_TP | S1 | True | True | True | False |


정상 추가FP는 **0:15270**, 2022-07-12 00:58:21.587, burst0:461의41번째 행이었다. R5 점수22.680453은 기존 임계값19.486921을 넘지만 이전2회와 다음1회는 넘지 않았다. 창 준비 부족이나 세션 경계의 첫 행이 아니다. 그 시점 관측/예측 전류는 -250.527050/-238.443724였고, AI0는0.332216/0.011453, AI1은0.328809/0.132089였다. 이는 관측과 예측의 불일치이며 실제 부하·기계 원인을 확인한 것이 아니다.

새 이상 탐지1:135와1:140은 동일 burst1:5의3번째/8번째 행이다. 3번째 행은 lag2 보조 점수가 최초로 가용한 시점이라 직전 h가 없다. 8번째 행은 과거 h가 있어 확인 경보를 유지할 가능성이 있었다. 독립 고장2건으로 세지 않는다.

다른 정상T/C에서 [t-2,t-1,t] 센서 문맥이 가까운 별도burst3개씩을 비교했다. 추가FP 주변과 가장 가까운 정상 사례0:12969/0:4584/0:10493의 R5점수는9.707519/10.612595/13.271161로 기존 보조 경보가 없었다. 단 한 FP에서 일반적 원인을 단정하지 않고 이웃과 임계값의 차이를 남겼다. 정상 센서 부호나 라벨을 바꾸지 않았다.

R5는 t-2,t-1에서 t를 예측하며 잔차는 **실제 t 관측 뒤 t에서 경보 가능**하다. 미래 관측을 기다리거나 과거로 경보를 돌리는 구현은 없다. 원자료/특징/예측값/잔차/입력행목록/목표·발행·경보시각은 tables/diagnostic_changed_rows.csv, diagnostic_neighborhoods.csv에 있다. 이번 조사에서 시점 정렬·strict>·경계 초기화의 구현 결함은 발견되지 않았다. 수정한 것은 V의 판정 정책과 별도로 지정된 보조 확인 규칙이다.

![추가FP 문맥](figures/diagnosis_0_15270.png)
![최초 가용 시점에서 탐지한 이상](figures/diagnosis_1_135.png)
![지속 보조경보에서 탐지한 이상](figures/diagnosis_1_140.png)

## 3. 역할·행 집합·원자료 의존 범위

T는 정상12012행, C는 정상2011행이다. 원이상calibration132행은 모델/임계값 적합에 쓰지 않았다. 모델과 scaler는 T에만, 새 두 임계값은 C의 가용 R5 점수에만 의존한다. S1/S2는 후보 선택, V는 고정 후 유지 확인, H는 공개된 과거 평가다. 정상/이상은2022-07-12/2022-07-17로 다른 날짜이며 같은 시간의 두 운전상태처럼 해석하지 않는다.

| block | label | rows | bursts | first | last |
| --- | --- | --- | --- | --- | --- |
| S1 | 0 | 639 | 19 | 2022-07-12 00:53:19.759000 | 2022-07-12 00:55:46.351000 |
| S2 | 0 | 616 | 19 | 2022-07-12 00:55:52.785000 | 2022-07-12 00:58:22.487000 |
| V | 0 | 731 | 20 | 2022-07-12 00:58:24.593000 | 2022-07-12 01:01:01.732000 |
| S1 | 1 | 18 | 1 | 2022-07-17 10:51:39.999000 | 2022-07-17 10:51:41.699000 |
| S2 | 1 | 3 | 1 | 2022-07-17 10:51:48.093000 | 2022-07-17 10:51:48.293000 |
| V | 1 | 90 | 2 | 2022-07-17 10:51:56.684000 | 2022-07-17 10:52:08.307000 |


어떤 구성도 시작 행이나 FP행을 제외하지 않았다. 원행 평가 분모는 S=1276(정상1255/이상21), V=821(731/90), H=4347(3990/357)로 동일하다. 공통 warm-up은 모든 행을 B0로 판정하면서 R5 첫2행은 불가용, 과거 h는0으로 취급하는 정책이다. G1/G2의 첫 보조 확인 가능 위치는 burst4번째 관측이다. native coverage와 fallback/불가용 수는 tables/coverage.csv에 있다.

최대 원자료 범위는 B0 t-19..t(또는P0의t), G0 t-2..t, G1 t-3..t, G2 t-4..t다. 파일·원split·train_role 경계를 넘어 창이나 이력을 잇지 않는다. R5 lag와 확인 이력은 gap>.5초마다 초기화한다. B0만 원정의대로 gap를 넘어 P1창을 유지한다.

**S1/S2/V는 원selection 안의 시간 보고 블록이다. 원B0 정의를 보존했으므로 S2에서21행, V에서38행의 B0 창이 앞 보고 블록의 이미 관측한 행을 포함한다.** 이 의존은 별도 표로 공개한다. G0/G1/G2의 보조 이력은 보고 블록 경계가 온전한burst 경계이므로 이전블록에서 이어지지 않는다. T/C와 평가 사이 또는 H 사이에는 원자료 공유가 없었다. 이 설계는 purge된 독립fold가 아니며 과거 관측을 인과적으로 이용하는 고정모델의 시간 경과 점검이다.

| block | kind | evaluation_rows | unique_raw_dependency_rows | rows_using_previous_reporting_block | available_lag2_rows | max_past_rows |
| --- | --- | --- | --- | --- | --- | --- |
| S2 | B0 | 619 | 656 | 21 | 619 | 19 |
| V | B0 | 821 | 859 | 38 | 821 | 19 |


## 4. 효과와 유지 판정을 분리한 정책

S pooled 효과는 FN엄격감소·F2엄격상승·F1비하락과 FP한도를 모두 요구한다. S1/S2/V 유지는 FN비증가·F2/F1비하락과 FP한도를 요구한다. 이상이0개인 블록은 FP/FPR만 판단하고 FN/F1/F2 조건은 적용하지 않는다. 비교 허용오차1e-12, 정수FP는 내림이며 최소1개를 허용하지 않는다.

| block | N0 | N1 | FN | FP | FPR | FP_absolute_cap | extra_FP_budget | baseline_feasible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | 639 | 18 | 3 | 2 | 0.003130 | 6 | 0 | True |
| S2 | 616 | 3 | 0 | 6 | 0.009740 | 6 | 0 | True |
| S_pooled | 1255 | 21 | 3 | 8 | 0.006375 | 12 | 1 | True |
| V | 731 | 90 | 0 | 0 | 0.000000 | 7 | 0 | True |


B0 자체는 모든 필수 블록의 절대1% 한도를 지켰다. 추가FP 한도는 S1/S2/V 각0, S pooled1, H3이다. 특히 S2 FP6→7은 이번에도 실패다.

0→0FN이면 fn_maintenance_pass=True, fn_reduction_reproduced=not_assessable다. 다른 유지조건까지 만족할 때 stability_pass=True다. F2 1→1도 유지 통과다. 이 상태값 분리는 FN이 개선됐다는 새 주장이 아니다. C1은 B0와 같은 예측에 새 판정 함수를 적용한 정책 검사이며 후보 수에 포함하지 않았다.

| candidate | block | old_pass | old_failed_rules | new_pass | fn_maintenance_pass | fn_reduction_reproduced | predictions_changed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | S_pooled | False | FN_decreased;F2 | False | True | False | False |
| C1 | V | False | FN_decreased;F2 | True | True | not_assessable | False |
| C0 | S2 | False | F2;F1;FP_budget;absolute_FPR | False | True | not_assessable | False |


프로토콜 해시: `507bc28e8ee0faf42f96e40aff7166a3ea5470eb1d3ffb0a77b6ae90eebe49b9`. 설정과 코드 해시를 후보의 S 결과 계산 전에 잠갔고 선택 후 V/H 결과를 보고 후보를 바꾸지 않았다. S1/S2 분석도 반복 개발이며 사전 고정이 과거 노출을 지워주지는 않는다.

## 5. 지정한 여섯 구성과 선택

G0 = B0 OR h_t. G1 = B0 OR(h_t AND h_(t-1)). G2 = B0 OR(h_t AND(h_(t-1) OR h_(t-2))). 현재 h_t=0이면 이전 h만으로 보조 경보를 연장하지 않는다. B0는 억제하거나 지연하지 않는다. 연속 통합점수는 max(B0점수/원임계값, 보조확인점수)로 정의했다. 보조확인점수는 G0에서 s_t/theta, G1에서 min(s_t,s_(t-1))/theta, G2에서 min(s_t,max(s_(t-1),s_(t-2)))/theta이며 불가용 과거값은0이다. 이번 양수 임계값에서 통합점수>1이 실제 경보와 일치함을 검사했다. AP/PR은 이 연속점수로 계산했고 경보0/1값을 순위점수로 쓰지 않았다.

| q | threshold | normal_C_total | available_C | unavailable_C | quantile_higher_zero_based_index | ties | strict_tail_count | available_resolution | total_resolution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.995000 | 14.676684 | 2011 | 1889 | 122 | 1879 | 1 | 9 | 0.000529 | 0.000497 |
| 0.999000 | 17.543122 | 2011 | 1889 | 122 | 1887 | 1 | 1 | 0.000529 | 0.000497 |


C의2011행 중1889개에 lag2 점수가 있었고122행은 비가용이었다. 경험 분위수는 가용 점수에 대해 higher로 계산했으며 판정·FP분모에서는2011행을 유지했다. 점수표본 해상도는1/1889≈0.0529%p이며 동점은 strict>에서 경보하지 않는다. q=.995/.999의 꼬리 관측 수는9/1개로 작다. 이를 미래 FPR보장으로 해석하지 않는다.

| candidate | theta | TP | FN | FP | F1 | F2 | eligible | status | duplicate_of | failed_rules |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G0_Q995 | 14.676684 | 20 | 1 | 14 | 0.727273 | 0.847458 | False | completed |  | S1:FP_budget ; S2:FP_budget;absolute_FPR;F2;F1 ; S_pooled:FP_budget;absolute_FPR;F1 |
| G0_Q999 | 17.543122 | 20 | 1 | 9 | 0.800000 | 0.884956 | False | duplicate_excluded | previous:R5_A1 | S2:FP_budget;absolute_FPR;F2;F1 |
| G1_Q995 | 14.676684 | 19 | 2 | 9 | 0.775510 | 0.848214 | False | completed |  | S2:FP_budget;absolute_FPR;F2;F1 |
| G1_Q999 | 17.543122 | 19 | 2 | 8 | 0.791667 | 0.855856 | True | completed |  |  |
| G2_Q995 | 14.676684 | 19 | 2 | 9 | 0.775510 | 0.848214 | False | duplicate_excluded | G1_Q995 | S2:FP_budget;absolute_FPR;F2;F1 |
| G2_Q999 | 17.543122 | 19 | 2 | 8 | 0.791667 | 0.855856 | False | duplicate_excluded | G1_Q999 |  |


6개 등록 슬롯 중 G0_Q999는 기존 R5_A1과 같은모델·임계값·규칙으로 중복 제외했다. G2_Q995와G2_Q999는 각각 G1의 C+S 판정이 동일해 제외했다. 이는 C+S에서 관찰된 판정 동등성이지 모든 미래 자료에서 두 규칙이 같다는 수학적 증명은 아니다. 슬롯을 다른 후보로 채우지 않았고 V/H로 중복 여부를 고르지 않았다. 새 비중복 구성은3개다.

| candidate | block | TP | FN | FP | TN | Recall | Precision | F1 | F2 | FPR | passes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | S1 | 15 | 3 | 2 | 637 | 0.833333 | 0.882353 | 0.857143 | 0.842697 | 0.003130 | True |
| B0 | S2 | 3 | 0 | 6 | 610 | 1.000000 | 0.333333 | 0.500000 | 0.714286 | 0.009740 | True |
| B0 | S_pooled | 18 | 3 | 8 | 1247 | 0.857143 | 0.692308 | 0.765957 | 0.818182 | 0.006375 | False |
| C0 | S1 | 17 | 1 | 2 | 637 | 0.944444 | 0.894737 | 0.918919 | 0.934066 | 0.003130 | True |
| C0 | S2 | 3 | 0 | 7 | 609 | 1.000000 | 0.300000 | 0.461538 | 0.681818 | 0.011364 | False |
| C0 | S_pooled | 20 | 1 | 9 | 1246 | 0.952381 | 0.689655 | 0.800000 | 0.884956 | 0.007171 | True |
| G1_Q999 | S1 | 16 | 2 | 2 | 637 | 0.888889 | 0.888889 | 0.888889 | 0.888889 | 0.003130 | True |
| G1_Q999 | S2 | 3 | 0 | 6 | 610 | 1.000000 | 0.333333 | 0.500000 | 0.714286 | 0.009740 | True |
| G1_Q999 | S_pooled | 19 | 2 | 8 | 1247 | 0.904762 | 0.703704 | 0.791667 | 0.855856 | 0.006375 | True |


G1_Q999는 S1에서FN3→2, FP2→2; S2에서FN0→0, FP6→6이었다. 합산 FN3→2, FP8→8, F1 0.765957→0.791667, F2 0.818182→0.855856으로 적격이다. 기존 C0 대비 정상 추가FP를 제거하지만1:135의 탐지를 잃는다. 다른 추가탐지1:140은 유지한다. 원B0의 TP를 잃은 것은 아니다.

임계값만 바꾼 G0_Q999의 S 판정은 C0와 같아서 여전히 S2에서 실패했다. 같은17.543122 임계값에서 G1과 비교하면 확인 규칙이 제거한FP1과 새FN1을 분리해 볼 수 있다. 이를 PCA성분이나 새모델 학습의 이득으로 부르지 않는다.

![선택 절충](figures/selection_tradeoff.png)
![선택 PR](figures/PR_S_pooled.png)

## 6. 잠금 후 V와 H

선택된 G1_Q999 한 개와 B0만 V에서 평가했다. C0나 탈락 후보의 V/H 결과는 계산하지 않았다. V 유지 통과 뒤 같은 고정 후보와 B0만 H에 적용했다.

| candidate | block | TP | FN | FP | TN | Recall | Precision | F1 | F2 | FPR | AP | PR_AUC_trapezoid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | V | 90 | 0 | 0 | 731 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 | 1.000000 |
| G1_Q999 | V | 90 | 0 | 0 | 731 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 | 1.000000 |
| B0 | H | 329 | 28 | 1 | 3989 | 0.921569 | 0.996970 | 0.957787 | 0.935722 | 0.000251 | 0.987098 | 0.987086 |
| G1_Q999 | H | 344 | 13 | 1 | 3989 | 0.963585 | 0.997101 | 0.980057 | 0.970107 | 0.000251 | 0.995770 | 0.995766 |


V는 B0와 후보 모두90TP/0FN/0FP, F1=F2=1이다. **V 성능 유지 확인**이며 FN 감소 재현은not_assessable다. 과거처럼 엄격개선을 요구하면 같은 예측을 비통과로 처리하지만, 이번 유지 통과는 모델 성능 수치 상승과 구분된다.

H에서 기존329TP/28FN/1FP/3989TN → 선정344TP/13FN/1FP/3989TN이었다. Recall92.1569%→96.3585%, Precision99.6970%→99.7101%, F1 0.957787→0.980057, F2 0.935722→0.970107. 정상1000관측당FP는 양쪽0.250627개다. 사전FP 예산 안에서 FN감소·F2증가·F1비하락을 충족했다. **이미 노출된 과거 평가에서 관찰한 개선**이며 새로운독립시험이 아니다. H를 본 뒤 추가후보·임계값변경·재선택을 하지 않았다.

| block | comparator | candidate | new_TP | new_FN | additional_FP | removed_FP | new_TP_bursts | new_FN_bursts | independent_failure_events |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S_pooled | B0 | G1_Q999 | 1 | 0 | 0 | 0 | 1 | 0 | unknown; burst counts are observation groups |
| S_pooled | C0 | G1_Q999 | 0 | 1 | 0 | 1 | 0 | 1 | unknown; burst counts are observation groups |
| V | B0 | G1_Q999 | 0 | 0 | 0 | 0 | 0 | 0 | unknown; burst counts are observation groups |
| H | B0 | G1_Q999 | 15 | 0 | 0 | 0 | 3 | 0 | unknown; burst counts are observation groups |


![V 혼동행렬](figures/confusion_V.png)
![과거 평가 혼동행렬](figures/confusion_H.png)
![과거 평가 PR](figures/PR_H.png)

## 7. 경보 시각·구간·놓친 이상

S의 추가탐지1행은1개 burst, H의 추가탐지15행은3개 burst에 분포했다. 독립 고장 사건 수는 알 수 없다. 실제 고장 시작/상태전환/정비 정답이 없으므로 고장 전 예측시간이나 진짜 고장탐지지연을 계산하지 않는다.

S에서 C0의 이상1:135를 놓치지만 같은burst의 첫 전체경보 시각은 B0/C0/선정방식이 같았다. 첫보조 h에서 확인경보까지의 시간과 전체B0포함 첫경보는 다른 값이다. tables/confirmation_delay.csv와 first_observation_alarm_delays.csv에 모두 구분했다. H에서는 burst1:9의 첫 관측→첫경보가0.4초에서0.3초가 되었고 나머지 이상burst의 첫경보시각은 같았다. 이것은 관측 시작 기준의0.1초 차이이며 고장 전 조기예측 증거가 아니다.

H의 정상 행FP1은 정상 경보episode1개이고, 이상 행탐지329→344와 이상경보episode18→17은 다른 집계다. 버스트 사이에서 경보 연결을 끊었다. H의13개 이상burst 모두 한 번 이상 경보가 있었으나 이를 이상357행을 모두 맞혔다고 바꾸지 않았다. 무경보 구간도 tables/alarm_episodes.csv에 남겼다. 수집중단시간을 정상노출로 세지 않았고 FP/hour는 계산하지 않았다.

## 8. 검증·재현·한계

validation_report.json과 tables/regression_tests.csv의 **151개 검사**가 통과했다. 필수 FN0→0 유지/개선분리, FN0→1 실패, S2 FP6→7 실패, 정상만 있는 V, G1/G2 경계·현재h필수조건, 원B0 TP보존, 미래값변경불변, 점수와 판정의 일치, 원본해시, T전용모델 재구성, 별도프로세스 재로딩과 배치/온라인 일치를 포함한다. 원 B0는 새로운 정책으로 예측이 바뀌지 않았다.

온라인 구현은 과거20센서행·lag2·이전h2개만 버퍼에 보관한다. 라벨이나 미래버스트길이가 필요하지 않다. S1276행의모든등록구성과 selection전체2097행의선정구성, H4347행의선정구성을 독립 프로세스와 비교했다. 점수 최대오차는 부동소수점 수준이고 판정은 전부 동일했다. 미선정후보의 V/H는 이 검증에서도 계산하지 않았다.

새 독립 자료가 없고 날짜와 라벨이 겹쳐 있으며 S의 이상21행은단2개burst다. 선택의 이득이1행이므로 불확실성을 작게 보아서는 안 된다. 행iid bootstrap이나PCA seed반복으로 통계적 유의성을 주장하지 않았다. 실제압력·불량·비가동 감소, 다른설비 일반화, 앨리어싱 면역을 입증하지 않았다. 선택과검증결과가현장오경보율을보장하지 않는다.

## 파일과 재실행

`bash RUN_EXPERIMENT.sh`는 동봉 입력·저장모델로 새디렉터리에 진단재현→고정프로토콜 확인→선택→V/H조건부평가→회귀검증→분석·보고서를 수행한다. 모델/분할을재탐색하지 않으며 결과를 덮어쓰지 않는다. 실제 전체재실행 확인은 end_to_end_replay.json에 저장한다. Python및라이브러리버전은inventory.json/requirements.lock.txt에 있다. src/infer_online.py는한원분할의센서CSV를현재시점순서로처리하는별도추론코드다.

진단 diagnosis_ko.md, 프로토콜 protocol_v2.json, 선택 selected_config.json, 전체표 all_trials.csv/block_metrics.csv, 선택·과거변경행 paired_selected_rows.csv, 행별판정 predictions/, 모델 models/, 검증 validation_report.json, 근거문헌 references/를 제공한다. 전달 ZIP의 파일별SHA-256과CRC를검증한다.

# 문헌 확인과 해석 범위

확인일:2026-10-04.

- **Cawley, G. C.; Talbot, N. L. C. (2010). On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation. JMLR11,2079–2107.** [공식 페이지](https://jmlr.org/papers/v11/cawley10a.html), [원문](https://jmlr.org/papers/volume11/cawley10a/cawley10a.pdf). 공식 초록과 원문 Section6(인쇄2103쪽/PDF25쪽)의 선택 과적합·평가 편향 설명을 확인했다. 선택 절차도 모델 적합의 일부로 보아 평가와 분리해야 한다는 근거다. 공식 페이지에서 DOI를 확인하지 못했으므로 적지 않는다. 이번에는 후보 수 제한, S 선택 후 설정 잠금, V 실패 시 재선택 금지를 적용했다. 이미 노출된 S/V/H를 새 자료라고 바꾸어 부를 근거는 아니다.
- **Bergmeir, C.; Hyndman, R. J.; Koo, B. (2018). A note on the validity of cross-validation for evaluating autoregressive time series prediction. Computational Statistics & Data Analysis120,70–83.** [DOI](https://doi.org/10.1016/j.csda.2017.11.003), [공개 원문](https://robjhyndman.com/papers/cv-wp.pdf). 공개본은2017-07-23자 preprint다. Abstract, Sections1–3(정상성·ergodicity·추정 일치성·오차의 MDS 가정), Section7 결론을 확인했다. 순수 자기회귀 모형의 오차가 상관되지 않는 등 조건 아래 표준K-fold가 가능하다는 논의다. 모든 시계열에서 무작위CV를 금지했다는 논문이 아니다. 이번 자료에서 그 가정이 충족됐다고 검증하지 않았으며 원래 시간 분할을 유지하고 원행 의존범위를 감사하는 데 해석상 참고했다. DOI의 출판사 연결은 도구에서500 접근 오류였으나 저자 공개 원문은 확인했다.

FP≤1%, ΔFPR≤.001은 기존 프로젝트 정책을 유지한 것이다. 효과/유지 판정 분리, G0/G1/G2, 분위수 .995/.999, 최대6개, 동점1e-12는 이번 프로젝트 설계다. 문헌이 이 조합의 성능 향상을 보장한다고 주장하지 않는다. F2 가중치를 현장 고장/오경보 비용비로 해석하지 않는다.
