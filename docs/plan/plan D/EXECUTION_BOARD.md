# Plan D 실행 보드

이 파일은 매일 갱신하는 단일 작업판이다. 완료 증거가 없는 항목은 체크하지 않는다.

**최종 갱신 2026-10-04 01:40**

## 현재 상태

| 항목 | 상태 | 완료 증거 |
|---|---|---|
| 분석 프로토콜 동결 | **완료** | `submission/config.yaml`, `submission/decision_log.md` (PROTOCOL_FREEZE) |
| 제출 코드 골격 | **완료** | `submission/run_all.py` — quick 25초 완주, 테스트 13/13 통과 |
| 모델 동일조건 비교 | **완료(특징모델)** | `submission/outputs/tables/e2_model_comparison.csv` |
| 정상 블록 CV | **완료** | `submission/outputs/tables/e3_normal_block_cv.csv` (5 fold × 4 모델) |
| 강건성 섭동 | **완료** | `submission/outputs/tables/e5_robustness.csv`, `e6_shift_false_alarm.csv` |
| 오류분석 | **완료** | `submission/outputs/tables/e7_error_conditions.csv` |
| BL-1 공식 LSTM-AE 재현 | **중단·재실행 필요** | `submission/outputs/full_run.log` — Windows 네이티브 TF가 BL-1 시작 후 완료 기록 없이 종료됨 |
| 최종모델 동결 | **잠정** | decision_log `MODEL_FREEZE 후보` = 1단 M3 / 2단 M1. BL-1 비교 후 확정 |
| 결과보고서 | 미착수 | 최종 PDF |
| 발표자료 | 미착수 | 최종 PPT/PDF |
| clean-room 재현 | 미착수 | `submission/outputs/clean_run.log` |
| 포털 제출 | 미착수 | 제출 완료 화면 |

## 의사결정 로그 형식

전체 기록은 `submission/decision_log.md`. 요약:

| 날짜·시각 | 결정 | 근거 파일 |
|---|---|---|
| 10-04 01:20 | PROTOCOL_FREEZE | `config.yaml` |
| 10-04 01:30 | DL-001 섭동 게이트에 정상측 오경보 증가 추가 | `e5_robustness_summary.csv` |
| 10-04 01:31 | DL-002 gate3 허용오차 도입 | `e2_selection_gates.csv` |
| 10-04 01:31 | DL-003 conformal 교정도 게이트 신설 | `e8_conformal_coverage.csv` |
| 10-04 01:36 | DL-004 G1 은 계산예산상 1시드 | `config.yaml` |

## 매 실험 기록 형식

| Run ID | 설정 | 데이터 해시 | 시드 | 결과 파일 | 판정 |
|---|---|---|---|---|---|
| quick-01 | `config.yaml` 동결판 | `f2d61cb3…` / `9fad8c23…` | 20261004 | `submission/outputs/tables/*` | 게이트 통과 = M3 단독 |
| full-01 | + BL-1 G0/G1 | 동일 | 0,1,2 / 0 | `e2_bl1_*.csv` | 진행 중 |

## 오늘의 세 가지

1. ~~`run_all.py --mode quick` 끝까지 실행~~ → **완료 (25초)**
2. ~~정상 블록 CV 결과표 생성~~ → **완료 (`e3_normal_block_cv.csv`)**
3. ~~보고서 핵심 그림 한 장~~ → **완료 (`figures/fig1_burst_timeline.png` 등 5장)**

## 다음 세 가지 (10/5)

1. BL-1 G0/G1 결과로 MODEL_FREEZE 확정
2. 보고서 1~3장 초안 (데이터 진단 / 모델 비교 / 오류분석)
3. 발표자료 12장 골격

## 즉시 에스컬레이션할 위험

- [ ] 같은 설정의 재실행 결과가 달라짐 — *미발생. 시드 고정 확인*
- [x] window 가 시간 공백을 가로지름 — **테스트로 차단** (`test_windows.py`)
- [x] 정상 holdout 정보가 scaler·threshold 에 들어감 — **테스트로 차단** (`test_no_leakage.py`)
- [ ] 보고서 숫자와 생성 CSV 가 다름 — 보고서 착수 시 재점검
- [x] TensorFlow 문제로 전체 파이프라인이 멈춤 — **해소**. TF 2.21.0 설치 완료.
      미설치 환경에서도 torch 대체 / 둘 다 없으면 BL-1만 건너뛰도록 격리됨
- [ ] 제출물에서 소속·학교·로고 발견 — 제출 전 검사
- [ ] 10/7 종료 시점에 PDF/PPT/zip 중 하나라도 없음

## 결과를 본 뒤 생긴 새 위험

| 위험 | 내용 | 대응 |
|---|---|---|
| **완전분리** | BL-0·M1·M3 모두 AP=1.000 / AUROC=1.000 / Recall=1.000 | 탐지 지표로 모델을 고르지 않는다. 보고서에서 "성능 변별 불가 + 날짜 교란"으로 서술 |
| **섭동 취약성** | M1 은 오프셋에서 오경보 +89.7pp, BL-0 는 이득에서 +61.3pp | 노랑(계측 확인) / 빨강(설비 점검) 분리의 근거로 사용 |
| **BL-0 비교정** | 점수 동률로 p-value 해상도 없음(α=0.10에서 실제 0.013) | gate5 로 주력 제외. 기준선으로만 보고 |
