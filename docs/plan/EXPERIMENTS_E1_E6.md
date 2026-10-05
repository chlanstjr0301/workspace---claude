# 최종보고서 보강 실험 지시서 (E1–E6)

- **근거**: `docs/audit/FINAL_REPORT_ADVERSARIAL_REVIEW.md` (제출본 d11ed9d 적대적 검토)
- **대상 코드**: `plan_d_submission/`
- **목적**: 검토에서 문구 수정만으로는 해결되지 않는 지적에 근거 표를 만든다. **모든 실험은 진단용이다.**

---

## 0. 공통 규칙 (모든 실험에 적용)

### 0.1 바꾸면 안 되는 것

- 동결 구성은 그대로 둔다. 1단 M3 PCA-MSPC(특징 23개), 2단 M1 Mahalanobis(진동 특징 15개), conformal `p ≤ 0.01`, 빨강 = 같은 버스트 3 window 연속, 학습 블록 0–2, 보정 블록 3, 평가 블록 4.
- 바꾸지 않는 대상: `config.yaml`, `src/` 모델·특징·임계값 코드, `outputs/predictions.csv`, 기존 `outputs/tables/*.csv`.
- **실험 결과로 모델을 다시 선정하지 않는다.** 다른 후보가 더 좋게 나와도 "사후 비교 결과"로만 보고한다.

### 0.2 구현 형태

- 새 파일 하나에 모은다: `plan_d_submission/make_review_tables.py`
  - 함수: `e1()` … `e6()`, `main()`
  - 단독 실행: `python make_review_tables.py`, 일부만 실행: `python make_review_tables.py --only E1 E3`
- 출력 표 이름은 `outputs/tables/v{번호}{소번호}_*.csv` 형식이다(예: `v1a_acquisition_fingerprint.csv`). 기존 접두어 e/r/d/c/i와 겹치지 않는다.
- `run_all.py`의 보조 스크립트 루프 끝(`("상호작용 진단표 i1-i4", IT)` 다음)에 `("검토 대응 진단표 v1-v6", RV)`를 추가한다.
- `decision_log.md`에 **DL-018** 한 행을 추가한다. 내용: "검토 대응 진단표 v1–v6 추가. 모델·특징·임계값·경보 규칙 변경 없음. 진단 결과를 선정에 쓰지 않음."

### 0.3 재사용할 기존 코드

| 필요 | 위치 |
|---|---|
| 데이터·window·특징 준비 | `D.load_all(cfg, HERE)` → `R.prepare(cfg, normal, outlier, 10)`. 반환 dict 키: `Xn Xo mn mo Fn Fo names gof` |
| 동결 구성 재현 | `make_audit_tables.Frozen(cfg, P, stage1_cols=None)`, `.judge(F_full, burst_id)` → `p1 p2 s1 s2 red score1` |
| 연속 규칙 | `make_audit_tables.persist(flag, bid, n)` |
| 혼동행렬 | `make_audit_tables.confusion(y, pred)` |
| CP 구간, Poisson 상한 | `make_audit_tables.cp_interval`, `poisson_upper` |
| 경보 사건 묶기, 노출시간 | `src.evaluation.merge_alarms`, `false_alarm_profile` |
| CV fold 정의 | `src.windows.cv_folds(5)`: fold k = 평가 k, 보정 (k+1)%5, 나머지 학습 |
| 모델 생성 | `src.models.build(cfg, seed)` → 이름 `BL0 M1 M2 M3` |
| conformal p | `src.calibration.conformal_p(cal_scores, new_scores)` |
| 기여도·사유 문구 | `M3.contrib`, `M1.contrib`, `src.explain.top_contributions`, `reason_phrase` |
| 선정 로직 | `src.selection` (`SORT_KEYS`, `_pick`) |

### 0.4 시작 시 필수 검증

`main()` 첫 단계에서 `make_audit_tables.main()`과 같은 방식으로 동결 구성의 `p1`·`p2`가 `predictions.csv`와 일치하는지 `assert`한다. 일치하지 않으면 중단한다.

### 0.5 사전 판정 규칙

실험마다 아래에 **결과를 보기 전에 정한 해석 규칙**이 있다. 보고서 문구는 이 규칙대로 쓴다. 결과에 맞춰 규칙을 바꾸지 않는다. 바꿔야 하면 DL에 사유를 남긴다.

### 0.6 공통 정의

- **평가 표본**: 정상은 블록 4(2,929 window), 고장은 428 window(이상 버스트 17개).
- **AUROC**: 정상 0, 고장 1. 점수 방향이 반대인 단일 특징은 `max(AUC, 1−AUC)`로 쓰고, 방향을 별도 열에 적는다.
- **시드**: `cfg["seed"]`(20261004). 난수는 `np.random.default_rng(seed)`만 쓴다.
- **실행시간 목표**: 전체 3분 이내(CPU). BL-1(LSTM-AE)은 어떤 실험에도 포함하지 않는다.

---

## E1. 수집 경로 진단과 교란 분해 (1장·3.1 근거)

**확인할 질문**
1. 정상·고장 두 파일은 수집·내보내기 경로가 다른가?
2. 최종 시스템의 판정은 전류의 직류 성분 차이에서 오는가, 파형 형태에서 오는가?
3. 전류를 전혀 보지 않는 진동 특징만으로도 정상·고장이 갈리는가?

### E1a. 수집 지문 (원본 데이터)

- **입력**: `data/raw/press_data_normal.csv`, `press_data_outlier.csv`. 소수 자릿수를 세려면 **반드시 문자열로 읽는다**(`dtype=str`).
- **절차** (파일 × 채널 6개 조합 각각):
  1. `n`, `n_unique`, `unique_ratio = n_unique / n`
  2. `max_decimals`, `median_decimals`: 원문 문자열의 소수점 아래 자릿수. `-44.107442000000006` 같은 부동소수 표기 잔재는 자릿수가 15 이상이면 별도 플래그 `float_artifact`로 센다.
  3. `min_pos_diff`: 고유값을 정렬한 뒤 인접 차이 중 양수의 최솟값
  4. 격자 검사: 후보 간격 `q`는 ① `min_pos_diff`, ② 1.1920929(검토에서 발견한 값) 두 가지. 각각 `grid_resid_max = max|x/q − round(x/q)|`를 구하고, `on_grid = grid_resid_max < 1e-3`.
  5. 클리핑 의심: `clip_share_min`, `clip_share_max` = 최솟값·최댓값과 같은 행의 비율
- **출력**: `v1a_acquisition_fingerprint.csv`
  - 열: `source, channel, n, n_unique, unique_ratio, max_decimals, median_decimals, float_artifact, min_pos_diff, grid_q, grid_resid_max, on_grid, clip_share_min, clip_share_max`
- **검증**: 고장 전류는 `q=1.1920929`에서 `on_grid=True`, 정상 전류는 `False`(검토 시 확인값 5e-5 / 0.4999). 다르게 나오면 중단하고 원인을 보고한다.

### E1b. 1장 기초 통계 보강

- **출력**: `v1b_basic_stats.csv` (항목, 정상, 고장, 비고)
- **항목**:
  1. 불균형 비율: 행(600 / 20,600), window(428 / 15,299), 평가 표본 유병률(428 / 3,357)
  2. 중복 TimeStamp 1건의 처리 방식: `src/data.py`의 실제 처리(유지/제거)를 코드에서 확인해 문자열로 적는다
  3. 이상치: 채널별로 정상 학습 블록 0–2 원신호의 평균·표준편차 기준 `|z| > 4`인 행 비율
  4. 생산단위 식별 가능성: 버스트 길이(최대 4.9초), 샘플 간격 0.1초, Nyquist 5 Hz를 적는다. "10 Hz 버스트 수집에서 프레스 스트로크 단위 식별은 불가"를 판정값으로 기록한다

### E1c. 직류·양자화 반사실 시험 (동결 시스템)

- **방법**: 입력 원신호 `X`(window × 10 × 3)를 변환한 뒤 `FT.build`로 특징을 다시 만들고 `Frozen.judge`로 판정한다. 모델과 보정점수는 그대로 둔다.
- **조건**:

| ID | 변환 | 묻는 것 |
|---|---|---|
| C0 | 없음 (재현 확인) | 표 2-8과 같아야 함 |
| C1 | 고장 전류 전체에 상수 이동: `Xo[:,:,2] += mean(정상 학습 블록 전류) − mean(고장 전류)` | 파일 간 직류 차이를 없애도 판정이 유지되는가 |
| C2 | 정상(평가 블록) 전류를 고장 격자로 양자화: `round(x/q)*q`, `q=1.1920929` | 양자화 자체가 경보를 만드는가 |
| C3 | 정상·고장 모두 window별 전류 평균 제거 | window 단위 직류 정보의 역할 |
| C4 | C1 + C2 동시 | 수집 경로 차이 두 가지를 함께 없앤 경우 |

- **측정값** (조건마다): `stage1_auroc`(score1), `normal_stage1_rate`, `normal_red_rate`, `fault_stage1_recall`, `fault_red_recall`, `red_f1`
- **출력**: `v1c_dc_quantization_counterfactual.csv`
- **검증**: C0가 `r2_operational_performance.csv`의 1단·빨강 값과 소수 4자리까지 같아야 한다.
- **사전 판정 규칙**:
  - C1에서 `|Δ stage1_auroc| < 0.005`이고 `|Δ fault_red_recall| < 0.02` → 보고서에 "판정은 전류 직류 성분 차이에 의존하지 않는다"고 쓴다.
  - 그렇지 않으면 → "판정의 일부가 전류 직류 차이에 의존한다"고 쓰고, 변화량을 함께 적는다.
  - C2에서 `|Δ normal_stage1_rate| < 0.5%p` → "양자화 자체가 오경보를 만들지 않는다"고 쓴다.

### E1d. 채널·특징군별 분리력 (진단용 별도 적합)

- **방법**: 아래 특징 부분집합마다 M1과 M3를 학습 블록 0–2로 적합하고, 보정 블록 3으로 conformal p를 만든다. 평가는 평가 블록 4 정상과 고장 428 window로 한다. **동결 모델과는 별개의 진단 모델**이다.
- **부분집합** (특징 이름은 `P["names"]`에서 문자열 규칙으로 고른다):

| ID | 내용 | 규칙 |
|---|---|---|
| G-all | 전체 23 | `model_groups` A·S·R·O |
| G-vib | 진동 전체 15 | CUR 미포함이고 O_가 아님 (= 2단 입력) |
| G-vibA | 진동 진폭 6 | `A_*_AI0`, `A_*_AI1` |
| G-vibS | 진동 형태 6 | `S_*_AI0`, `S_*_AI1` |
| G-R | 상·하부 관계 3 | `R_*` |
| G-cur | 전류 전체 8 | CUR 포함 또는 O_ |
| G-curA | 전류 진폭 3 | `A_*_CUR` |
| G-curS | 전류 형태 3 | `S_*_CUR` |
| G-O | 전류 수준 2 | `O_*` |

- **측정값**: `auroc`, `ap`, `normal_alarm_rate`(p≤0.01), `fault_recall`(p≤0.01), `n_features`. M1·M3 각각 계산한다.
- **출력**: `v1d_channel_group_separability.csv`
- **단일 특징 순위**: 23개 특징 각각의 원값으로 평가 표본 AUROC(방향 무관 값과 방향)를 계산한다. 정상 평가 블록 중앙값과 고장 중앙값을 함께 적는다. → `v1e_single_feature_auroc.csv` (E6에서도 사용)
- **검증**: G-all·M3의 `auroc`가 `r11_negative_control_channels.csv` 제출본 행(1.0000)과, G-cur·M3가 전류 전용 행(0.9992)과 같아야 한다(같은 정의이므로).
- **사전 판정 규칙**:
  - G-vibA의 M1 또는 M3 `auroc ≥ 0.98`이고 `normal_alarm_rate ≤ 2%` → "전류를 전혀 보지 않는 진동 진폭만으로도 고장 기록이 갈린다"를 3.1과 발표 방어 문구로 쓴다.
  - 단일 특징 1위가 전류 형태(`S_*_CUR`)이면 → 1.5·3.1의 교란 서술을 "전류 파형 형태"로 바꾼다.
  - 어떤 결과든 동결 모델은 바꾸지 않는다고 명시한다.

**보고서 반영**: 1.2에 "두 파일의 수집 경로 차이" 표(v1a 요약) 추가. 1.5 교란 서술 교체. 3.1에 v1c·v1d 요약 표 1개. 발표 Q&A "날짜 탐지 아닌가" 답변.

---

## E2. 최종 시스템의 동일 조건 비교 (2장, 감점 −3~4점 항목)

**확인할 질문**: 후보 모델을 같은 2단 확인과 3연속 규칙 안에 넣으면 어떤 결과인가? 동결 규칙은 다른 블록 배치에서도 유지되는가?

### E2a. 동결 분할에서 후보별 전체 시스템

- **방법**: `Frozen`을 일반화한다. 1단 모델 이름을 인자로 받는 하위 클래스(`FrozenAny(cfg, P, stage1_name)`)를 만든다. 2단(M1 진동 15)·임계값·연속 규칙·블록은 동결 구성과 같다.
- **후보**: `BL0`, `M1`, `M2`, `M3`(BL-1 제외: 재학습 3.6시간)
- **규칙 3종**: ① 1단 단독, ② 1단 AND 2단, ③ 1단 AND 2단 + 3연속(빨강)
- **측정값** (후보 × 규칙): `TP FP FN TN precision recall f1 fpr`, `fp_events`(merge_alarms), `bursts_detected`(17개 중), `stage1_auroc`
- **출력**: `v2a_frozen_split_full_system.csv`
- **검증**: M3 행이 `r8_stage_decomposition.csv`의 해당 규칙 행과 일치해야 한다(FP 241 / 35 / 3).
- **주의**: BL-0는 학습범위 안의 점수가 모두 0이라 conformal p 동률이 많다. 결과 옆에 `p_ties_share`(보정점수 중 최솟값과 같은 비율)를 적는다.

### E2b. 동결 규칙의 5-fold 적용

- **방법**: `cv_folds(5)`의 각 fold에서 학습 블록으로 1단·2단을 적합하고, 보정 블록으로 p를 만든다. 평가 블록과 고장 428 window에 규칙 ①②③을 적용한다. 1단 후보 4개 모두 수행한다(20회 적합).
- **측정값** (후보 × fold × 규칙): E2a와 같은 값. 그리고 후보 × 규칙별 fold 평균, 표준편차, 최악값.
- **출력**: `v2b_cv_full_system.csv` (fold별), `v2c_cv_full_system_summary.csv` (요약)
- **검증**: 규칙 ①의 M3 fold별 `fp`가 `e3_normal_block_cv.csv`의 M3 행과 일치해야 한다(같은 모델·같은 fold 정의).
- **사전 판정 규칙**:
  - **주 지표는 빨강 F1과 빨강 FPR의 fold 평균**이다.
  - M3가 두 지표 모두 최상이 아니어도 그대로 보고하고 모델은 바꾸지 않는다. 문구: "사후 비교에서 X가 Y 지표에서 우수했으나, 선정은 사전 기준(섭동 강건성)에 따랐다."
  - fold 간 빨강 FPR 최댓값을 운영 계획의 상한 참고값으로 적는다.

**보고서 반영**: 2.4 비교표 아래에 "최종 시스템 구조로 본 비교" 표 1개(v2a 규칙 ③ + v2c 요약). 요약과 2장 첫머리 박스에 "같은 2단 구조 안에서의 비교" 한 줄.

---

## E3. 버스트 단위 부트스트랩 (불확실성)

**확인할 질문**: window를 독립으로 보지 않을 때 간판 수치의 불확실성은 얼마인가?

- **입력**: `predictions.csv` (`split ∈ {holdout, fault}`, `alarm_level`, `burst_id`, `p_normal_stage1`). 모델 재적합은 하지 않는다.
- **방법**:
  - 정상 평가 블록 버스트 106개와 고장 버스트 17개를 **각 군 안에서 복원추출**한다(층화 클러스터 부트스트랩).
  - 반복 B = 5,000, `rng = default_rng(seed)`
  - 회마다 뽑힌 버스트의 window를 모아 빨강과 1단의 지표를 계산한다: `fpr recall precision f1 fp_windows fp_events`
- **출력**:
  - `v3a_burst_bootstrap_ci.csv`: 지표별 점추정, 2.5%, 97.5%, 그리고 `share_fp_zero`(FP=0 회차 비율)
  - `v3b_posterior_bootstrap.csv`: 유병률 1%·0.1%·0.01%에서 P(고장 | 빨강)의 점추정, CP 보수값(r10 기존), **부트스트랩 보수값**(TPR 2.5% 분위와 FPR 97.5% 분위 사용)
- **검증**: 점추정이 표 2-8 값(빨강 F1 0.931, FPR 0.0010, Recall 0.876)과 같아야 한다.
- **해석상 주의** (보고서에 그대로 적는다): 고장 버스트 17개는 한 고장 기록의 조각이다. 이 구간은 기록 안의 변동만 반영하며, 고장 사건 간 변동은 반영하지 않는다.

**보고서 반영**: 표 2-8에 95% 구간 열 추가. 표 2-12에 부트스트랩 보수 열 추가. 4.3의 운영 기준 문구를 부트스트랩 보수값 기준으로 바꾼다.

---

## E4. 선정 민감도와 대조군의 임계값 무관 지표

### E4a. 선정 기준 민감도

- **입력**: `outputs/tables/e2_selection_gates.csv` (모델 재계산 없음)
- **방법**:
  1. 정정 기준(v2)의 나머지 게이트는 고정하고, 섭동 게이트 한도 `t`를 0~100%p 범위에서 0.1 간격으로 바꾼다. 통과 조건은 `worst_recall_drop_pp ≤ t`이고 `worst_shift_fp_increase_pp ≤ t`다. 통과 모델 중 `SORT_KEYS` 순서로 1위를 고른다. 같은 선정이 이어지는 구간으로 묶어 표로 만든다.
  2. gate3 허용오차(절대값)를 0~0.02 범위에서 0.001 간격으로 바꿔 같은 방식으로 구간을 만든다.
  3. 대안 원리 3가지의 선정 결과:
     - (a) minimax: 최악 섭동 오경보 증가 최소
     - (b) 섭동 4종 순위합 최소
     - (c) 원 기준 v0
- **출력**: `v4a_selection_sensitivity.csv` (열: `sweep, from, to, chosen, n_passed`), `v4b_selection_principles.csv`
- **검증**: `t = 20`에서 M3, 손계산 결과(17.0 미만이면 통과 모델 없음, 17.0 ≤ t < 89.7이면 M3, 89.7 이상이면 M1)와 일치해야 한다.
- **사전 판정 규칙**: M3가 선정되는 `t` 구간 폭이 50%p 이상이고 minimax에서도 M3이면, 2.6에 다음 문장을 넣는다. "섭동 한도를 17.0~89.7%p 어디에 두어도, 최악 증가 최소화 원리로도 M3가 선정된다. 다만 원 기준(v0)으로는 BL-0다."

### E4b. 지도학습 대조군·음성대조의 AUROC

- **방법**: `make_supervised_control.py`와 같은 분할·특징·분류기를 재현한다. 예측 레이블 대신 `decision_function` 점수로 `auroc`, `ap`를 계산한다. 같은 실험을 특징집합 두 가지로 한다: ① 기존(`groups`, 품질 특징 포함 25개), ② `model_groups` 23개.
- **추가 계산**: M4 fold별 실제 평가 유병률과 그 유병률에서의 동전던지기 F1(`2π·0.5/(π+0.5)`)·전부양성 F1을 구한다. 표 2-4의 0.203을 정정하는 근거다.
- **출력**: `v4c_control_threshold_free.csv` (열: `id, feature_set, f1, auroc, ap, test_prevalence, coin_f1, allpos_f1`)
- **검증**: 기존 특징집합의 F1이 `d7_supervised_control.csv`와 일치해야 한다.
- **사전 판정 규칙**:
  - NC의 AUROC가 0.6 이상인 쌍이 있으면 → "임계값 기준 F1은 우연 부근이나, 순위 기준으로는 정상 시간블록이 일부 구분된다(AUROC a~b)."
  - 모두 0.6 미만이면 → "순위 기준으로도 구분되지 않는다."

**보고서 반영**: 2.6에 민감도 한 문장과 작은 표. 2.5 음성대조 문구와 표 2-4 정정.

---

## E5. 현장 운영 지표 (4장)

### E5a. 교대 단위 경보 빈도

- **방법**:
  1. 수집 비율 `duty = 버스트 내부 관측시간 합 / 벽시계 기간`. 정상 전체와 평가 블록 4 각각 계산한다. 블록 4의 벽시계 기간 = 첫 window 시작부터 마지막 window 끝까지.
  2. 노랑·빨강 사건 수(`merge_alarms`)를 다음 세 단위로 환산하고 Poisson 95% 상한을 붙인다:
     - 수집시간 기준 시간당
     - 벽시계 기준 시간당(= 수집시간당 × duty)
     - 8시간 교대당(= 벽시계 시간당 × 8)
  3. 같은 계산을 정상 5블록 전체에도 한다. 블록 0–3은 학습·보정 구간이라 `in_sample=True`로 표시한다.
- **출력**: `v5a_alarm_per_shift.csv` (열: `scope, in_sample, level, events, collect_hours, wall_hours, duty, per_collect_h, per_wall_h, per_shift_8h, per_shift_8h_upper95`)
- **검증**: 평가 블록 빨강의 수집시간당 값이 `r7_alarm_event_rates.csv`(19.06)와 같아야 한다.
- **해석상 주의**: 현재 수집 방식(버스트 수집)이 현장에서도 같다고 가정한 환산이다. 연속 수집이면 duty = 1로 다시 계산해야 한다고 적는다.

### E5b. 정지 검토 규칙의 발동 횟수

- **규칙 정의** (4.3의 문구를 실행 가능한 형태로 고정. 결과를 보고 바꾸지 않는다):
  - **S1** "빨강 1건": 빨강 사건 1건 발생
  - **S2** "빨강 2회 연속": **연속한 두 버스트**(같은 source, 버스트 번호 순서상 인접)에서 각각 빨강 사건 발생
  - **S3** "10분 내 빨강 2건": 벽시계 10분 창 안에 빨강 사건 2건 이상
  - "2개 교대 연속"은 데이터 길이(정상 약 77분)로 평가할 수 없으므로 표에 "평가 불가"로 둔다
- **측정값**: 정상 블록별 발동 횟수(블록 0–3은 `in_sample`). 고장 기록에서는 첫 발동 시각(기록 시작 후 초)과 첫 발동까지의 버스트 수.
- **출력**: `v5b_stop_rule_triggers.csv`
- **사전 판정 규칙**:
  - 정상 평가 블록에서 S2 발동이 0이고 고장에서 S2가 발동하면 → 4.3의 "유병률 0.1% 가정 시 S2를 정지 검토 조건으로" 권고를 유지한다.
  - 정상에서 S2가 1회 이상 발동하면 → 그 횟수를 적고, 권고를 S3 또는 시범운영 확정으로 낮춘다.

### E5c. 경보 사유 분포와 빨강 권고 문구

- **방법**:
  1. `predictions.csv`의 `top_reason_1`을 4개 범주(전류 / 진동관계 R / 진동형태 S / 진동진폭 A)로 묶는다. 범주별 비율을 고장 빨강, 정상 빨강, 정상 노랑, 고장 노랑에서 각각 센다.
  2. 빨강 window에 대해 **2단 M1의 진동 기여 1위 특징**(`Frozen.m2.contrib`)을 계산하고 같은 방식으로 분포를 센다.
  3. 대안 권고 문구 열 `red_action_vib`를 만든다. 빨강이면 2단 진동 기여 1위로 `reason_phrase`를 적용한 결과다.
- **출력**: `v5c_reason_distribution.csv`. 그리고 window별 대안 문구는 `v5d_red_action_alternative.csv`(키: `source, burst_id, original_row_start`). **`predictions.csv`는 수정하지 않는다.**
- **검증**: 고장 빨강 중 전류 범주 비율이 검토 확인값(268/375 = 71.5%)과 같아야 한다.
- **후속 결정**: `predictions.csv`의 `recommended_action` 매핑을 실제로 바꿀지는 이 결과를 보고 사용자가 정한다. 바꾸면 경보 판정은 불변이고 문구만 바뀌며, 별도 DL로 기록한다.

**보고서 반영**: 4.2 표 4-2에 교대당 환산 열 추가. 4.3에 규칙 S1~S3 발동표. 4.4에 사유 분포와 "빨강은 진동 근거부터 해석" 문구.

---

## E6. 개별 영향변수 순위 (3.1)

**확인할 질문**: 23개 특징 중 어느 것이 판정을 주도하는가? 그것은 물리적으로 무엇인가?

- **방법** (특징 23개 각각):
  1. `top1_share_fault`, `top1_share_normal`: 1단 M3 기여도(`contrib` 절댓값) 1위가 그 특징인 비율. 고장 428 window와 정상 평가 블록 2,929 window 각각.
  2. `single_auroc`, `direction`: E1의 `v1e_single_feature_auroc.csv`에서 가져온다.
  3. `drop_one_d_normal_rate`, `drop_one_d_fault_recall`, `drop_one_d_ap`: 그 특징을 뺀 22개로 M3를 학습 블록 0–2에 재적합하고 블록 3으로 보정한다. 평가 표본의 1단 정상 경보율 변화, 고장 Recall 변화, AP 변화를 잰다(진단용 23회 적합).
  4. `group`, `channel`, `physical_meaning`: 아래 사전에서 가져온다.
- **물리 해석 사전** (코드에 상수로 넣는다):
  - `A_std/p2p/rms_*`: 진동·전류의 출렁임 크기
  - `S_ac1/ac2_*`: 0.1·0.2초 간격의 자기상관. 10 Hz 앨리어싱 아래의 리듬. 회전수·전원 주파수·수집 클록과 구분 불가
  - `S_zcr_*`: 부호 변화 빈도
  - `R_corr/abscorr`: 상·하부 진동의 동기성
  - `R_stdratio`: 상·하부 진폭비. 체결·정렬과 관련 가능
  - `O_mean/absmean_CUR`: 전류의 window 평균 수준
- **출력**: `v6_feature_importance.csv`. 정렬 순서: `top1_share_fault` 내림차순, 동률이면 `single_auroc` 내림차순(사전에 고정).
- **검증**: 채널별 `top1_share_fault` 합이 `r5_stage1_channel_share.csv`(전류 71.7%)와 일치해야 한다.
- **해석상 주의**: 탐지가 포화되어 drop-one의 AP 변화는 작을 것이다. 순위는 기여도 비중으로 정하고, drop-one은 "그 특징 없이도 유지되는가"를 보여주는 보조 열로만 쓴다.

**보고서 반영**: 3.1에 상위 10개 특징 표(특징, 물리 의미, 고장 기여 1위 비중, 단일 AUROC, 제거 시 변화).

---

## 7. 완료 조건과 반영 절차

1. `python make_review_tables.py`가 오류 없이 끝나고, 0.4와 각 실험의 **검증 항목이 모두 통과**한다.
2. `python run_all.py --mode quick`(새 환경, `outputs/` 삭제 후)에서 기존 표가 모두 변하지 않고 v 표가 추가된다. `pytest` 13건 통과.
3. 결과를 사전 판정 규칙에 따라 보고서에 반영한다. 이때 검토 문서의 P0·P1·P2 문구 수정도 함께 한다.
4. `report/check_numbers.py`에 새 수치(v1c C1의 AUROC 변화, v1d G-vibA AUROC, v2c 주 지표, v3a 구간, v4a 구간, v5a 교대당 값, v6 1위 특징)와 새 금지 표현을 추가하고 통과시킨다.
   - 새 금지 표현: "5개 중 3개가 Recall", "상호작용에서 나온다", "한 번도 보지 않은", "우연 수준이다", "80분의 1"(단독 사용 시)
5. PDF를 재빌드하고 `final_1/`을 동기화한다. 이후 재검토를 1회 한다.

## 8. 작업량 추정

| 실험 | 계산 | 구현 |
|---|---|---|
| E1 | 약 20초 (반사실 5조건 + 진단 적합 18회) | 1.5시간 |
| E2 | 약 30초 (적합 4 + 20회) | 2.5시간 |
| E3 | 약 10초 (5,000회, 모델 없음) | 1시간 |
| E4 | 약 10초 | 0.5시간 |
| E5 | 약 5초 | 1시간 |
| E6 | 약 20초 (적합 23회) | 0.5시간 |
| 반영·검사·재빌드 | — | 3시간 |
