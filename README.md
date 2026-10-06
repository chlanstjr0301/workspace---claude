# CARE-Press — 프레스 유압펌프 이상 조기탐지 및 오경보 분석

제6회 K-인공지능 제조데이터 분석 경진대회 (일반국민/대학(원)생 부문) 문제 ③
「진동·전류 시계열 기반 프레스 유압펌프 이상 조기탐지 및 오경보 분석」 제출 코드

- 결과보고서: `report/결과보고서.pdf` (원고 `report/결과보고서.md`)
- 데이터: KAMP(인공지능 제조 플랫폼) 무료 제조AI데이터셋 「소성가공 예지보전 AI 데이터셋」
  (중소벤처기업부, KAMP, 스마트제조혁신추진단(㈜인터엑스), 2022.12.23., www.kamp-ai.kr)

---

## 1. 제출물 구성

| 대회 요구 항목 | 위치 |
|---|---|
| 소스코드 + requirements | `run_all.py`, `make_*.py`, `stream_replay.py`, `src/`, `requirements.txt` (정확한 버전: `requirements-lock.txt`) |
| 학습용 데이터 | `data/raw/press_data_normal.csv`, `data/raw/press_data_outlier.csv` (원본, 수정하지 않음) |
| README | 이 문서 |
| 테스트데이터 예측결과 파일 | `outputs/predictions.csv` (window 15,299행) |
| 결과보고서 | `report/결과보고서.pdf` |

## 2. 실행 방법

Python 3.11 이상 (3.11·3.13에서 검증). GPU 불필요.

```bash
pip install -r requirements.txt                          # 정확한 재현은 requirements-lock.txt
python run_all.py --config config.yaml --mode quick      # 약 3분: 표 84개·예측파일·그림·실시간 재생 검증
python -m pytest tests -q                                # 불변식·실시간 재생 테스트 14건
python report/check_numbers.py                           # 보고서 수치 ↔ CSV 대조 (124건)
python stream_replay.py                                  # (선택) 실시간 재생 검증만 단독 실행
python report/check_style.py                             # 보고서 작성 규약 검사
```

- `quick`이 데이터 점검 → window·특징 → 5-fold 비교 → 모델 선정 → 최종 적합 → 예측파일 → 진단표 → 그림 → 실시간 재생 검증을 순서대로 수행한다.
- `--mode full`은 가이드북 LSTM 오토인코더(BL-1) 재현을 추가한다(약 3.6시간, TensorFlow 또는 PyTorch). BL-1 결과 파일(`outputs/tables/e2_bl1_g*.csv`)이 동봉되어 있어 `quick`에서도 비교표에 들어간다.
- 보고서 PDF 재생성(선택): `python report/build_report_pdf.py` (markdown, playwright·Chromium, 명조 글꼴 필요). 공식 양식 글꼴인 휴먼명조가 없으면 나눔명조로 대체된다.
- `outputs/run_manifest_full.json`·`outputs/full_run.log`는 BL-1 재현(full 실행)의 환경·로그 기록이다.

## 3. 방법 요약

| 단계 | 내용 | 보고서 |
|---|---|---|
| 데이터 진단 | 정상 20,000행은 0.5초 초과 공백 598개로 끊긴 버스트 599개(최장 4.9초). 정상·고장의 수집일·수집 경로가 다르고 고장 전 구간은 0초 | 1장 |
| window | 버스트 안에서만 10샘플(1초) window 생성, 1샘플 간격 | 1.8, 2.1 |
| 검증 | 정상 기록을 버스트 단위 시간블록 5개로 나눈 교차검증. 적합·임계값은 정상 블록만 사용 | 2.1 |
| 임계값 | conformal p = (1 + #{보정 점수 ≥ 새 점수}) / (n + 1), p ≤ 0.01이면 경보 | 2.1 |
| 모델 비교 | BL-0 범위규칙, BL-1 LSTM-AE, M1 Mahalanobis, M2 Isolation Forest, M3 PCA-MSPC, 대조군 M4 | 2.2–2.5 |
| 선정 | 탐지 지표 포화 → 계측 섭동 시 오경보 증가로 1단 M3 선정 | 2.6 |
| 최종 시스템 | 1단 M3(특징 23개) 포착 → 2단 M1(전류를 뺀 진동 특징 15개) 확인 → 같은 버스트 3 window 연속이면 빨강 | 2.7 |
| 성능 (평가 블록) | 빨강 F1 0.931 (95% 0.892–0.955), 오경보 3 window, 고장 버스트 15/17 탐지 | 2.8 |
| 실시간 판정 | 샘플 도착마다 과거 10샘플로 판정. 기록을 한 샘플씩 재생한 판정이 일괄 예측 12,424 window와 같음, window당 처리 약 1.4 ms | 4.2 |

## 4. 출력

`outputs/predictions.csv` 주요 열:

| 열 | 의미 |
|---|---|
| `source` `block` `split` | 정상·고장 / 시간블록 / fit·calibration·holdout·fault |
| `burst_id` `original_row_start/end` `time_start/end` | 원본 행과 시각 추적 |
| `score_stage1/2` `p_normal_stage1/2` | 1·2단 점수와 conformal p |
| `risk_score` | 위험 순위 = 1 − p1 (확률 아님) |
| `anomaly_prob_cal` | 같은 고장 기록 안에서만 유효한 Platt 보정 확률 |
| `alarm_level` | green / yellow / red |
| `top_reason_1/2` `recommended_action` | 1단 기여 상위 특징, 권고 문구 (빨강은 진동 센서 확인 + 2단 기여 기준) |
| `label` | 평가용 정답 |

운영 성능은 `split`이 `holdout`·`fault`인 행만으로 다시 계산할 수 있다. 보고서의 모든 수치는 `outputs/tables/`의 CSV 84개에서 나온다.

## 실시간 운영

`stream_replay.py`의 `StreamJudge`가 현장 판정기의 참조 구현이다. 학습 후 고정한 모델·보정 점수를 갖고, 수집장치가 보내는 샘플(10 Hz)을 `push(시각, 채널값 3개, 행 번호)`로 1개씩 받아 window가 완성될 때마다 경보 상태를 돌려준다. 미래 샘플과 파일 전체 통계를 쓰지 않는다. 현장 수집장치와의 통신 연동은 시범운영 단계에서 구현한다.

## 5. 디렉터리

```
.
├─ README.md  requirements.txt  requirements-lock.txt  config.yaml
├─ decision_log.md            결과를 본 뒤의 결정 24건 기록
├─ run_all.py                 단일 진입점
├─ stream_replay.py           실시간 판정기(StreamJudge)와 재생 검증
├─ make_*.py                  보충·진단 표 스크립트 11개 (모델·임계값 불변)
├─ data/raw/                  원본 CSV 2개
├─ src/                       data windows features models calibration
│                             evaluation explain selection plotting deep bl1
├─ tests/                     window·누수·출력 형식·실시간 재생 테스트 14건
├─ outputs/                   predictions.csv  run_manifest.json  tables/  figures/
└─ report/                    결과보고서(md·pdf), 그림 생성, 수치·규약 검사, PDF 빌드
```

## 6. 재현성

- 시드 `config.yaml`의 `seed: 20261004`. 원본 데이터의 SHA-256을 `outputs/run_manifest.json`에 기록한다.
- `outputs/`를 지운 사본에서 1명령 실행 시 Python 3.13(numpy 2.5.3)과 Python 3.11(numpy 2.4.6) 모두 표 84개·예측파일이 저장본과 상대오차 1e-6 이내다.
- 코드는 파일 위치 기준 상대경로만 쓴다(절대경로 0건, 테스트로 확인).
