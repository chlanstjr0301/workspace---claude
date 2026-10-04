# CARE-Press — 프레스 유압펌프 이상 조기탐지 및 오경보 분석

과제 ③ · 진동·전류 시계열 기반 이상탐지
**Condition-aware / Acquisition-gap-aware / Robust / Explainable**

---

## 1. 한 명령 실행

```bash
pip install -r requirements.txt
python run_all.py --config config.yaml --mode quick    # CPU, 약 25초
python run_all.py --config config.yaml --mode full     # + LSTM-AE 재현
```

`quick` 만으로 보고서의 모든 핵심 주장(표·그림·예측파일)이 재생성된다.
`full` 은 공식 LSTM-Autoencoder(BL-1) 재현을 추가한다. TensorFlow 가 없으면
PyTorch 로 같은 구조를 만들고, 둘 다 없으면 BL-1 만 건너뛴 채 나머지가 완주한다.

검증:

```bash
python -m pytest tests -q        # 13 passed
```

---

## 2. 이 데이터에서 먼저 확인해야 하는 것

| 사실 | 수치 | 근거 파일 |
|---|---|---|
| 연속 시계열이 아니라 **버스트 수집** | 정상 599버스트 / 중앙값 37샘플 / **최대 50샘플(4.9초)** | `outputs/tables/e0_burst_summary.csv` |
| 정상 20,000행의 실제 span | 4,616초 (10Hz 연속이면 2,000초) | `outputs/tables/e0_data_quality.csv` |
| 가이드북의 seq20+offset100(=120샘플)이 들어가는 버스트 | **0개** | `e0_burst_summary.csv` |
| gap-aware 로 제거되는 가짜 window | seq=10에서 25.6%, seq=20에서 50.1% | `e1_window_sensitivity.csv` |
| 독립 고장 사건 | **1건** (2022-07-17, window 428개는 그 조각) | `e0_data_quality.csv` |
| 정상/이상 수집일 | 2022-07-12 / 2022-07-17 (**날짜 교란**) | `e0_data_quality.csv` |

이 구조를 무시하면 window 가 수집 공백을 가로지르고, "10초 뒤 예측"이라는
서술이 성립하지 않는다.

---

## 3. 분석 프로토콜 (결과를 보기 전에 동결)

- window 는 **버스트 내부에서만** 생성한다 (길이 10샘플, stride 1).
- 정상 20,000행을 시간순 **5블록**으로 나눠 fold 마다 3블록 학습 / 1블록
  calibration / 1블록 holdout 으로 순환한다. 랜덤 K-fold 는 금지한다.
- 경계 버스트는 통째로 한 블록에 배정한다(= 버스트가 쪼개지지 않는다).
- **모든 스케일러·임계값은 해당 fold 의 정상 구간에서만** 적합한다.
- 이상 데이터는 임계값·특징·하이퍼파라미터 선택에 쓰지 않는다.
- 임계값은 정상성 conformal p-value 로 정의한다.
  `p_normal(x) = (1 + #{s_i >= s(x)}) / (n+1)`, 경보는 `p_normal <= 0.01`.
  `risk_score = 1 - p_normal` 은 **순위 기반 위험점수이지 고장 사후확률이 아니다.**

불변식은 테스트로 고정되어 있다.

| 테스트 | 검증 내용 |
|---|---|
| `tests/test_windows.py` | window 가 공백을 가로지르지 않음, 버스트가 블록에 쪼개지지 않음 |
| `tests/test_no_leakage.py` | 적합이 학습 블록만 사용, conformal p 가 calibration 에서 균등 |
| `tests/test_output_schema.py` | 예측 파일 필수 열, 표 존재, 해시 기록, 절대경로 없음 |

---

## 4. 모델과 선정 규칙

| ID | 모델 | 역할 |
|---|---|---|
| BL-0 | 학습 정상 범위 이탈 규칙 | 설명 가능한 최저 기준선 |
| BL-1 | 공식 LSTM-AE (G0 재현 / G1 동일조건) | 가이드북 기준선 |
| M1 | Shrinkage Mahalanobis | 화이트박스 후보 |
| M2 | Isolation Forest | 비선형 비교군 |
| M3 | PCA-MSPC (T²+SPE) | 제조현장 표준, 기여도 분해 |
| F | CARE-Press 2단 결합 | 최종 운영 모델 |

**F1 최고 모델을 자동 선택하지 않는다.** `src/selection.py` 의 게이트를
순서대로 적용한다: NaN → 섭동 강건성 → 최악 블록 오경보 → 버스트 탐지 →
conformal 교정도. 통과 모델 중 최악 블록 FAR 상한이 낮은 쪽을 고른다.

---

## 5. 최종 출력

`outputs/predictions.csv` 는 0/1 이 아니라 다음을 담는다.

| 열 | 의미 |
|---|---|
| `p_normal_stage1`, `p_normal_stage2` | 1단(전체 특징) / 2단(전류 제외 진동) 정상성 p-value |
| `risk_score` | `1 - p_normal_stage1` |
| `alarm_level` | `green` / `yellow` / `red` |
| `top_reason_1`, `top_reason_2` | 상위 기여 특징 |
| `recommended_action` | 현장 점검 문구 |

| 신호 | 조건 | 작업자 조치 |
|---|---|---|
| 초록 | 1단 정상 | 계속 운전 |
| 노랑 | 1단 이상, 2단(진동) 미확인 | **센서 체결·극성·이득·샘플링 확인** 후 교대 내 점검 |
| 빨강 | 1·2단 동시 이상이 연속 3 window | 생산 일정 조정 후 유압펌프 정밀점검 |

노랑과 빨강을 나누는 이유는 §6 의 섭동 결과 때문이다 — 계측조건 변화만으로도
모델이 고장을 외칠 수 있으므로, 그 경우를 설비 점검과 분리한다.

---

## 6. 결과 요약

모든 수치는 `outputs/tables/` 의 CSV 에서 생성된다.

- **탐지 성능으로는 모델을 구분할 수 없다.** BL-0·M1·M3 모두 AP=1.000,
  AUROC=1.000, Recall=1.000 (`e2_model_comparison.csv`). 정상/이상이
  완전분리되어 있다. 이는 날짜 교란과 분리할 수 없는 결과이며, 보고서에서
  "고장을 잘 맞혔다"가 아니라 **"이 데이터로는 성능을 변별할 수 없다"**로 쓴다.
- **계측조건 섭동에서는 전혀 다르다.** 정상 데이터에 이득·오프셋·극성·jitter 를
  주었을 때 오경보 증가폭(`e5_robustness_summary.csv`):

  | 모델 | 최악 오경보 증가 | 원인 섭동 |
  |---|---:|---|
  | M1 Mahalanobis | **+89.7 pp** | offset |
  | BL-0 범위 규칙 | **+61.3 pp** | gain |
  | M2 Isolation Forest | +22.3 pp | gain |
  | **M3 PCA-MSPC** | **+17.0 pp** | offset |

- **탐지력의 출처는 형태·진폭이다.** 특징군 단독 AP: S(형태) 0.9986,
  A(진폭) 0.9935, R(관계) 0.6923, O(전류 수준) 0.5052 (`e4_feature_ablation.csv`).
- **BL-0 는 임계값을 확률로 해석할 수 없다.** 학습 범위 내 점수가 전부 동률이라
  α=0.10 에서 실제 coverage 가 0.013 에 그친다 (`e8_conformal_coverage.csv`).

게이트를 통과한 모델은 **M3 PCA-MSPC** 단독이며, 2단 확인에는 전류를 제외한
진동 특징의 M1 을 쓴다 (`e2_selection_gates.csv`, `decision_log.md`).

---

## 7. 말하지 않는 것

- "고장 발생 10초 전에 예측했다" — 이상 데이터는 이미 `Equipment_state=1` 인
  고장 상태 기록이고, 고장 이전 구간이 없어 리드타임을 측정할 수 없다.
- "21건의 고장을 모두 잡았다" — 21개는 **수집 버스트**이지 고장 사건이 아니다.
- "오경보가 0건이므로 FAR=0" — Poisson 95% 상한을 함께 보고한다.
- "전류 변화의 원인은 기어 마모다" — 10Hz 데이터로 결함주파수를 해석하지 않는다.
- 정확도 97% 같은 불균형 착시를 전면에 내세우지 않는다.

---

## 8. 디렉터리

```
plan_d_submission/
├── config.yaml          분석 프로토콜 (동결)
├── decision_log.md      프로토콜 변경·발견 기록
├── run_all.py           전체 파이프라인
├── data/raw/            원본 CSV (미수정)
├── src/                 data windows features models calibration
│                        evaluation explain selection plotting deep bl1
├── outputs/
│   ├── predictions.csv  window 단위 최종 판정
│   ├── run_manifest.json 설정·데이터 해시·환경
│   ├── tables/          모든 표의 원본 CSV
│   └── figures/         보고서 그림
└── tests/               불변식 3종
```

재현성: 원본 데이터는 수정하지 않으며 SHA-256 을 `run_manifest.json` 에
기록한다. 절대경로를 쓰지 않고 시드를 고정한다. `outputs/` 를 지운 뒤
한 명령으로 전부 다시 생성된다.
