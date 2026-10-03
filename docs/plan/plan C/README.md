# Plan C — 논문 근거형 우승 계획

> **실행 완료(2026-10-04):** 구현·검증 결과는 [EXECUTION_RESULTS.md](EXECUTION_RESULTS.md),
> 재현 코드는 `plan_c_submission/`을 참조한다.

**과제 ③ · 진동·전류 시계열 기반 프레스 유압펌프 이상 조기탐지 및 오경보 분석**  
작성일 2026-10-04 · 문헌조사 기준일 2026-10-04 · 제출 마감 2026-10-08 23:59

> Plan C의 원칙은 **Paper → Risk → Protocol → Evidence**다. 최신 모델 이름을 나열하지 않고, 각 논문이 지적한 실패 위험을 검증 규칙과 모델 구조로 바꾼다.

---

## 0. 결론

### 제안 시스템

**SHIFT-Guard: Burst-aware, Shift-aware, Interpretable Fault Triage**

하나의 이상점수로 정상/고장을 바로 나누지 않는다. 다음 세 상태를 구분한다.

1. **Known normal**: 학습된 정상 운전상태 안에 있음
2. **New normal / measurement shift**: 부하·센서·수집조건 변화 가능성이 큼
3. **Fault evidence**: 이득·오프셋·극성 변화로 설명하기 어려운 형태·관계 이상이 반복됨

```text
CSV
 → timestamp audit
 → burst segmentation
 → gap-safe 10-sample windows
 → amplitude / shape / relation / acquisition features
 → Shift score + Fault score
 → block/burst conformal calibration
 → green / yellow / red
 → channel evidence + recommended inspection
```

### 논문조사가 바꾼 다섯 가지

| 문헌의 결론 | 이 프로젝트의 결정 |
|---|---|
| Point adjustment는 무작위 점수도 고성능처럼 만들 수 있음 | point-adjusted F1 금지. 원시 window F1과 range/burst 지표를 병기 |
| 시계열 이상은 점이 아니라 범위로 평가해야 함 | window, predicted alarm range, acquisition burst를 분리해 평가 |
| 단순 통계모델이 최신 딥러닝보다 강할 수 있음 | Dynamic PCA/MSPC를 주력 후보로 올리고 LSTM-AE는 공식 비교군으로 둠 |
| 정상성은 시간에 따라 변하는 “new normal” 문제를 가짐 | 저부하·센서 변화 점수를 고장 점수와 분리하고 정상 블록 CV를 수행 |
| conformal의 교환가능성 가정은 시계열에서 깨짐 | 중첩 window를 그대로 calibration하지 않고 burst/block 단위 민감도를 병기 |

### 심사위원용 한 문장

> **SHIFT-Guard는 고장처럼 보이는 모든 변화를 고장으로 부르지 않고, 계측·운전 변화와 설비 이상 증거를 분리해 오경보 원인과 다음 조치까지 제시한다.**

---

## 1. 문헌 기반 문제 재정의

### 1.1 이 과제는 일반적인 지도학습이 아니다

- 정상 20,000행은 2022-07-12 기록이다.
- 이상 600행은 2022-07-17의 고장 상태 기록이다.
- `Equipment_state`는 파일 안에서 상수다.
- 이상 window 수가 많아도 독립 고장 사건은 1건이다.
- 정상과 이상이 날짜·운전·센서 조건으로 완전히 분리돼 있다.

따라서 모델이 학습할 수 있는 것은 다음이 섞인 신호다.

```text
관측 차이 = 실제 고장 + 부하 변화 + 센서 이득/극성 + 수집일/DAQ 차이 + 잡음
```

Plan C의 목표는 완전한 인과 분리가 아니다. 데이터만으로 분리할 수 없는 부분을 밝히고, **교란에 약한 증거와 강한 증거를 서로 다른 경보 단계로 운영**하는 것이다.

### 1.2 연구 질문

| ID | 질문 | 판단 근거 |
|---|---|---|
| RQ1 | 공식 LSTM-AE가 단순 통계모델보다 실제로 우수한가 | 동일 split/window/calibration 비교 |
| RQ2 | 탐지력은 진폭, 시간형태, 채널관계 중 어디서 나오는가 | 특징군 ablation과 점수 기여도 |
| RQ3 | 정상 저부하 상태가 왜 오경보가 되는가 | leave-one-block-out 정상 CV |
| RQ4 | 전류 자기상관 차이는 센서·샘플링 섭동으로 재현 가능한가 | gain/offset/jitter stress test |
| RQ5 | 확률처럼 해석 가능한 출력이 가능한가 | block/burst conformal p-value와 coverage audit |
| RQ6 | 경보를 어떤 점검 행동으로 연결할 수 있는가 | shift/fault evidence 조합별 의사결정표 |

---

## 2. 데이터와 분석 단위

### 2.1 시간 구조

- 연속 샘플 간격 중앙값은 약 0.1초다.
- 0.5초 초과 간격에서 새 burst로 나눈다.
- 정상 599개, 이상 21개의 acquisition burst가 존재한다.
- 최대 burst 길이는 50샘플, 즉 약 5초다.
- 공백을 가로지르는 sliding window는 만들지 않는다.

### 2.2 평가 단위

| 단위 | 정의 | 역할 | 독립성 해석 |
|---|---|---|---|
| Point | 원본 행 | 품질 점검 | 인접 샘플에 종속 |
| Window | burst 내부 10샘플, stride 1 | 모델 입력 | 9/10이 겹쳐 강한 종속 |
| Alarm range | 연속 경보 window 병합 | 실제 경보 횟수 | 현장 단위 |
| Acquisition burst | 공백 사이 수집 묶음 | calibration·민감도 | 같은 고장 기록 안의 반복 측정 |
| Fault record | 이상 CSV 전체 | 독립 양성 사건 | 1건 |

### 2.3 라벨 한계

- 이상 CSV 전체가 이미 `Equipment_state=1`이므로 고장 전조 시작점이 없다.
- 조기예측 lead time을 측정할 수 없다.
- 각 burst 시작을 가상 진입점으로 둔 탐지지연은 **sensitivity analysis**로만 사용한다.
- 21개 burst를 21개 독립 고장으로 부르지 않는다.

---

## 3. 고정 검증 프로토콜

### 3.1 정상 데이터: blocked cross-validation

정상 데이터를 timestamp/burst 순서를 유지한 5개 연속 block으로 나눈다.

```text
fold k:
  train       = 3 blocks
  calibration = 1 block
  normal test = 1 block
```

- 모든 block이 한 번씩 normal test가 된다.
- burst를 fold 경계에서 자르지 않는다.
- scaler, PCA, covariance, threshold는 train/calibration만 사용한다.
- 평균 성능뿐 아니라 **worst-block false alarms**를 모델 선정에 사용한다.
- 16,000~17,999행 저부하 block을 포함하는 fold를 별도 강조한다.

### 3.2 이상 데이터: 고정 평가

- 모델과 threshold를 정상 데이터만으로 고정한 뒤 이상 CSV를 평가한다.
- 모든 후보는 같은 이상 window와 같은 alarm merging 규칙을 쓴다.
- 이미 Plan A/B에서 이상 데이터가 탐색됐다는 사실을 보고서 한계에 공개한다.
- Plan C 이후에는 이상 성능을 보고 feature/threshold를 다시 바꾸지 않는다.

### 3.3 경보 병합

기본값:

```text
alarm start = 같은 burst 안에서 3개 연속 window가 threshold 초과
alarm end   = 3개 연속 window가 threshold 이하
```

- `k = 1, 3, 5` 민감도를 함께 보고한다.
- FP window 수와 FP alarm range 수를 모두 보존한다.
- burst 사이 공백을 넘어 alarm을 연결하지 않는다.

---

## 4. 평가 지표

### 4.1 금지

- point adjustment 후 F1
- 이상 구간 한 점만 맞히면 전체 구간을 TP로 바꾸는 처리
- Accuracy 단독 보고
- stride-1 window를 독립 표본으로 간주한 신뢰구간
- 최고 F1 threshold를 이상 test에서 직접 선택

### 4.2 필수 지표

| 계층 | 지표 | 목적 |
|---|---|---|
| Window | Precision, Recall, F1, MCC | 공지 요구 충족 |
| Ranking | AP/AUC-PR, AUROC | threshold 독립 비교 |
| Range | range precision/recall 또는 VUS-PR | 시간범위 검출 품질 |
| Alarm | FP alarm count, FAR/h | 현장 오경보 부담 |
| Block | worst-block FAR, block별 p-value coverage | new normal 강건성 |
| Delay | burst-entry detection delay | 민감도 분석 |
| Robustness | perturbation 후 Recall/FAR 절대값 | 교란 강건성 |

### 4.3 보고 우선순위

1. Worst-block false alarms
2. AP/AUC-PR 또는 VUS-PR
3. Window Recall과 F1
4. Alarm FAR/h와 Poisson 95% 상한
5. Burst-entry detection delay

VUS-PR 구현이 마감 전 안정적으로 검증되지 않으면 range precision/recall과 AP를 사용하고, 새 지표를 억지로 넣지 않는다.

---

## 5. 특징: Shift evidence와 Fault evidence의 분리

### 5.1 특징군

| 군 | 특징 예 | 주된 의미 |
|---|---|---|
| A: amplitude | std, RMS, peak-to-peak, MAD | 부하·이득·진동 크기 |
| S: shape | lag-1/2 ACF, zero crossing, difference MAD | 시간형태·지속성 |
| R: relation | corr(AI0, AI1), abs-corr, std ratio | 상·하부 센서 관계 |
| O: offset | mean, median, absolute mean | 부하·오프셋 |
| Q: acquisition | dt mean/std, burst position/length | 수집조건 |

### 5.2 두 점수

**Shift score**

- A + O + Q 중심
- 정상 train의 부하·센서 수준에서 얼마나 이동했는가
- 높다고 바로 고장으로 판단하지 않는다.

**Fault score**

- S + scale-invariant R 중심
- 채널별 z-normalization 뒤 Dynamic PCA/MSPC 또는 shrinkage Mahalanobis
- 이득·오프셋 변화 후에도 유지되는지를 요구한다.

### 5.3 결정 규칙

| Shift | Fault | 상태 | 해석 |
|---|---|---|---|
| 낮음 | 낮음 | Green | known normal |
| 높음 | 낮음 | Yellow-S | 새 부하·센서·DAQ 상태 가능성 |
| 낮음 | 높음 | Yellow-F | 미세한 설비 이상 증거, 재측정 필요 |
| 높음 | 높음 | Red | 여러 증거가 일치하는 이상 |

이 구조는 전류 lag-1 ACF가 실제 고장인지 날짜/계측 차이인지 확정할 수 없는 문제를 숨기지 않고 운영 규칙으로 바꾼다.

---

## 6. 후보 모델

### 6.1 필수 비교

| ID | 모델 | 입력 | 문헌상 역할 |
|---|---|---|---|
| B0 | 정상 min/max 또는 robust quantile rule | 원신호/진폭 | 무학습 기준선 |
| B1 | Isolation Forest | A/S/R/O 특징 | 비선형·경량 비교군 |
| B2 | 공식 LSTM Encoder-Decoder | raw window | 가이드북·재구성 계열 기준선 |
| M1 | Shrinkage Mahalanobis | S + invariant R | 해석 가능한 fault score |
| M2 | PCA-MSPC | A/S/R/O | T² + SPE 이중 감시 |
| M3 | Dynamic PCA | lagged normalized channels | 시간동학을 직접 포함한 주력 후보 |
| F | SHIFT-Guard | Shift score + best fault score | 최종 운영안 |

### 6.2 Dynamic PCA 구현

10샘플 window를 다음처럼 펼친다.

```text
[AI0(t), AI1(t), AI2(t),
 AI0(t-1), AI1(t-1), AI2(t-1),
 ...]
```

- lag 후보는 1, 2, 4까지만 비교한다.
- PCA component 수는 정상 train 재구성 분산 90/95%와 normal CV를 함께 본다.
- `T²`는 주성분 공간 내 정상 변화, `SPE/Q`는 정상 부분공간 밖 이탈을 잡는다.
- control limit는 parametric 가정값과 정상 calibration empirical quantile을 둘 다 계산한다.
- 최종 threshold는 정상 calibration empirical quantile을 우선한다.

### 6.3 LSTM-AE 역할 제한

- 공식 구조 재현은 필수다.
- `naive/abs/offset/last-step MSE`와 `gap-safe/raw/no-offset/full-window MSE`를 분리한다.
- 정상 데이터만으로 학습·early stopping·threshold를 결정한다.
- deep model이 통계모델을 이기려면 F1뿐 아니라 worst-block FAR과 perturbation robustness도 개선해야 한다.
- 개선하지 못하면 “복잡한 모델이 항상 낫지 않다”는 문헌과 실험의 일치 사례로 활용한다.

### 6.4 사용하지 않을 모델

- Transformer, foundation model: 표본과 window가 너무 작고 재현 리스크가 큼
- 지도 LightGBM 최종모델: 양성 사건 1건과 날짜 교란을 학습할 위험
- FFT 기반 결함주파수 모델: 10 Hz에서 물리 해석 불가
- test-time adaptation 본 구현: 고장 데이터를 새 정상으로 흡수할 위험과 검증 표본 부족

Test-time adaptation 논문은 **향후 운영 제안**으로만 사용한다. 제출 모델은 자동 적응 대신 Yellow-S quarantine을 둔다.

---

## 7. Calibration과 확률 표현

### 7.1 기본 conformal score

```text
p_normal(x) = (1 + #{calibration score >= score(x)}) / (n + 1)
```

- `p_normal`은 정상성 p-value다.
- `1 - p_normal`은 anomaly risk rank다.
- 실제 고장 사후확률이라고 부르지 않는다.

### 7.2 시계열 종속성 대응

일반 split conformal의 정확한 유한표본 보장은 exchangeability에 의존한다. 중첩 window에는 이 가정이 맞지 않는다.

따라서 세 버전을 비교한다.

| 버전 | calibration unit | 장점 | 한계 |
|---|---|---|---|
| C0 | 모든 window | 해상도 높음 | 강한 중첩 종속성 |
| C1 | burst별 최대/상위 분위 점수 | 경보 단위와 일치 | calibration 표본 감소 |
| C2 | 시간 block별 empirical CDF | regime shift 확인 | 5개 block으로 거침 |

최종 보고에는 C0 점수와 C1/C2 coverage audit를 같이 둔다. “conformal이 FAR을 보장한다”고 단정하지 않고, 가정 위반과 실측 coverage를 함께 제시한다.

### 7.3 온라인 FDR

온라인 FDR 제어는 반복 경보에서 목표 precision을 관리하는 문헌적 근거가 있다. 다만 현재 데이터는 정상 관측시간과 독립 고장 수가 부족하므로:

- 본 제출에서는 시뮬레이션 또는 설계 제안으로만 제시
- 실제 모델 selection에는 사용하지 않음
- 향후 다일 연속 스트림이 확보되면 LORD/SAFFRON 계열과 비교

---

## 8. 원인 설명과 smearing 방지

PCA contribution plot은 공정 모니터링의 표준 도구지만, 상관된 변수 사이에서 fault contribution이 다른 변수로 퍼지는 smearing이 발생할 수 있다.

따라서 “원인”을 다음 3단계 증거로 제한한다.

1. **Detection contribution**: T²/SPE 또는 Mahalanobis 기여도
2. **Leave-one-group-out**: A/S/R/O 특징군을 제거했을 때 점수 감소
3. **Counterfactual channel repair**: 한 채널을 정상 중앙값/형태로 대체했을 때 점수 감소

세 방법이 일치할 때만 “주요 기여 채널”이라고 쓴다. 기계 고장 부품을 특정하는 표현은 정비 라벨이 없으므로 사용하지 않는다.

### 권장 표현

- 가능: “AI2 시간형태가 경보점수의 71%를 설명했다.”
- 가능: “AI1을 정상 형태로 대체했을 때 fault score가 43% 감소했다.”
- 금지: “AI2가 기어 마모의 원인이다.”
- 금지: “상·하부 역위상이 베어링 유격을 증명한다.”

---

## 9. 필수 실험

| ID | 실험 | 비교 | 성공 조건 |
|---|---|---|---|
| C-E01 | gap audit | naive vs burst-safe | 공백 횡단 window 0개 |
| C-E02 | metric audit | raw F1 vs range metric | point adjustment 미사용 확인 |
| C-E03 | model benchmark | B0/B1/B2/M1/M2/M3 | 동일 fold·threshold 규칙 |
| C-E04 | normal block CV | 5개 temporal block | worst-block FP 공개 |
| C-E05 | window sensitivity | 5/10/20 | 성능과 잔존율 동시 보고 |
| C-E06 | feature ablation | A/S/R/O/Q | 탐지·오경보 원인 분리 |
| C-E07 | sensor stress | gain/offset/polarity | fault score 유지 여부 |
| C-E08 | jitter stress | sampling jitter grid | AI2 ACF 재현 난이도 |
| C-E09 | calibration audit | C0/C1/C2 | 정상 coverage와 경보수 비교 |
| C-E10 | explanation audit | contribution/ablation/repair | 상위 채널 일치도 |
| C-E11 | alarm persistence | k=1/3/5 | delay–false alarm trade-off |
| C-E12 | clean rerun | 새 환경 | 한 명령으로 전 결과 생성 |

### 섭동 그리드

| 섭동 | 값 |
|---|---|
| gain | 0.5, 0.8, 1.2, 1.5, 2.0 |
| offset | train std의 -2, -1, +1, +2배 |
| polarity | 각 진동 채널 부호 반전 |
| Gaussian noise | train std의 1%, 5%, 10% |
| time jitter | 0.1, 0.5, 1, 2, 5 ms |
| channel dropout | 각 채널 마스킹 후 fallback 확인 |

성능 유지율만 쓰지 않고 perturbation 후의 절대 Recall, FPR, FAR를 보고한다.

---

## 10. 모델 선정 규칙

결과를 보기 전에 다음 순서로 고정한다.

1. 모든 정상 fold에서 실행되는가?
2. point adjustment 없이 B0보다 AP/F1 중 하나 이상 개선하는가?
3. worst-block alarm 수가 B0보다 작거나 같은가?
4. gain/offset/polarity 섭동 후 Recall 하락이 20%p 이하인가?
5. 설명 audit에서 상위 채널이 2개 이상의 방법으로 확인되는가?
6. inference가 CPU에서 실시간 요구보다 충분히 빠른가?

통과 모델이 여러 개면 다음 우선순위를 쓴다.

```text
worst-block FAR → perturbation absolute Recall → AP/VUS-PR → simplicity
```

F1이 가장 높은 모델을 자동 선택하지 않는다.

---

## 11. 보고서 스토리

### 제1장: 데이터 이해 및 진단

헤드 메시지:

> 이 데이터의 핵심 문제는 결측값이 아니라 독립 사건 부족, 버스트 수집, 날짜 교란, 정상상태 변화다.

필수 증거:

- 실제 timestamp timeline과 공백 분포
- burst 길이 분포
- 정상 block별 A/S/R/O 변화
- 정상/이상 파일 단위 라벨 구조
- sampling rate와 해석 가능한 주파수 한계

### 제2장: 모델 개발 및 성능

헤드 메시지:

> 공식 LSTM-AE부터 Dynamic PCA까지 같은 조건으로 비교하고, 최고 평균점수가 아니라 새 정상과 센서 섭동에 가장 강한 모델을 선택했다.

필수 증거:

- B0/B1/B2/M1/M2/M3 동일 조건 표
- raw F1, AP, range metric, worst-block FAR
- 모델 선정 waterfall
- 학습/추론 시간

### 제3장: 영향요인 및 오류분석

헤드 메시지:

> 탐지 증거와 오경보 증거를 분리했으며, 주요 기여 채널은 contribution 하나가 아니라 ablation과 counterfactual repair로 교차검증했다.

필수 증거:

- A/S/R/O ablation
- block × operating condition FP heatmap
- burst별 FN/탐지지연
- explanation agreement table

### 제4장: 현장 활용

헤드 메시지:

> SHIFT-Guard는 고장과 새 정상상태를 같은 경보로 처리하지 않는다.

| 상태 | 화면 | 조치 |
|---|---|---|
| Green | known normal | 운전 지속 |
| Yellow-S | shift evidence | 센서·DAQ·부하·금형 변경 확인 |
| Yellow-F | fault evidence | 동일 조건 재측정, 교대 내 설비점검 |
| Red | shift + fault evidence | 생산조정 및 즉시 정밀점검 |

### 제5장: 창의성·차별성

- 문헌이 지적한 evaluation inflation을 실제 검증 규칙으로 차단
- new normal과 fault evidence의 분리
- burst/block-aware conformal audit
- smearing을 고려한 3중 원인 검증
- 데이터가 허용하지 않는 예지보전 주장을 하지 않는 의사결정 설계

### 제6장: 코드 및 재현성

- `run_all.py` 단일 진입점
- 실험 ID와 config 고정
- 데이터·설정·코드 해시 기록
- 표·그림 원본 자동 생성
- quick/full 모드 제공

---

## 12. 구현 구조

```text
plan_c_submission/
├── README.md
├── requirements.txt
├── config/
│   ├── data.yaml
│   ├── models.yaml
│   └── experiments.yaml
├── run_all.py
├── src/
│   ├── audit.py
│   ├── bursts.py
│   ├── features.py
│   ├── metrics.py
│   ├── calibration.py
│   ├── models/
│   │   ├── baselines.py
│   │   ├── mspc.py
│   │   ├── dynamic_pca.py
│   │   └── lstm_ae.py
│   ├── stress.py
│   ├── explain.py
│   └── report.py
├── tests/
│   ├── test_gap_safety.py
│   ├── test_fold_leakage.py
│   ├── test_no_point_adjustment.py
│   └── test_output_schema.py
└── outputs/
    ├── manifest.json
    ├── predictions.csv
    ├── tables/
    └── figures/
```

### predictions.csv

| 필드 | 내용 |
|---|---|
| source, original_rows | 추적성 |
| burst_id, time_start, time_end | 시간 단위 |
| shift_score, fault_score | 두 증거 |
| p_shift, p_fault | 정상성 p-value |
| state | Green/Yellow-S/Yellow-F/Red |
| top_channel, top_feature_group | 설명 |
| alarm_id | 병합된 현장 경보 |
| label | 평가용 정답 |

---

## 13. 4일 실행 순서

### 10월 4일

- [ ] 논문 기반 protocol freeze
- [ ] 기존 Plan B 전처리를 `src/`로 이동
- [ ] 5-fold burst-safe split 구현
- [ ] raw/window/burst/alarm metric 구현
- [ ] B0, M1 기존 결과 재현

### 10월 5일

- [ ] PCA-MSPC와 Dynamic PCA 구현
- [ ] Isolation Forest 동일 조건 실행
- [ ] LSTM-AE 공식/수정 버전 실행
- [ ] C0/C1/C2 calibration audit

### 10월 6일

- [ ] feature ablation
- [ ] gain/offset/polarity/noise/jitter stress
- [ ] explanation audit
- [ ] SHIFT-Guard 최종 규칙 동결

### 10월 7일

- [ ] 보고서와 발표자료 작성
- [ ] 모든 표·그림 자동 생성
- [ ] README·requirements·예측 CSV 완성
- [ ] 블라인드 정보 검사

### 10월 8일

- [ ] clean environment quick/full 재실행
- [ ] 수치·그림·인용 대조
- [ ] 만족도 캡처 삽입
- [ ] PDF/PPT/zip 검수
- [ ] 18시 이전 1차 제출 목표

---

## 14. 성공 기준

Plan C가 완료됐다고 판단하려면 다음이 모두 필요하다.

- [ ] point adjustment 없는 동일 조건 모델 비교
- [ ] 5개 정상 block의 worst-case 오경보
- [ ] Dynamic PCA/MSPC와 LSTM-AE 비교
- [ ] Shift score와 Fault score의 분리
- [ ] block/burst conformal coverage audit
- [ ] 최소 5종 섭동 강건성 결과
- [ ] contribution–ablation–repair 설명 일치도
- [ ] Green/Yellow-S/Yellow-F/Red 현장 조치표
- [ ] 한 명령 clean rerun
- [ ] 논문 주장과 자체 실험 결과를 구분한 보고서

---

## 15. 최종 포지셔닝

Plan C는 “딥러닝을 더 크게 만들었다”는 계획이 아니다. 논문에서 반복적으로 지적된 세 가지 문제—**평가 부풀림, new normal, calibration 가정 위반**—를 이 데이터에 맞춰 직접 통제하는 계획이다.

우승 메시지는 다음과 같다.

> **우리는 고장 하나를 수백 개 샘플로 부풀려 자랑하지 않았다. 대신 정상상태 변화가 왜 오경보가 되는지, 어떤 증거가 센서 변화에도 살아남는지, 경보가 현장에서 어떤 행동으로 이어져야 하는지를 재현 가능한 실험으로 보였다.**

세부 논문 검토와 출처는 [LITERATURE_REVIEW.md](LITERATURE_REVIEW.md), 실행 순서는 [EXPERIMENT_REGISTER.md](EXPERIMENT_REGISTER.md)를 따른다.
