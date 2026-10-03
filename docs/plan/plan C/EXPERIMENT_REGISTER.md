# Plan C 실험 등록부

목적: 이상 test 결과를 본 뒤 실험 정의를 바꾸는 것을 막고, 보고서 숫자의 생성 경로를 남긴다.

## 1. 동결 설정

| 항목 | 기본값 | 대안/민감도 |
|---|---|---|
| gap threshold | 0.5초 | 0.3, 1.0초 |
| window | 10샘플 | 5, 20 |
| stride | 1 | 없음 |
| alarm persistence | 3 window | 1, 5 |
| normal folds | 5 temporal blocks | burst boundary 보존 |
| calibration quantile | q99.0, q99.5, q99.9 모두 보고 | 운영값은 CV로 결정 |
| PCA explained variance | 90%, 95% | component 고정 비교 |
| DPCA lag | 1, 2, 4 | 최대 4 |
| seeds | 42, 1337, 2026 | stochastic 모델만 |

## 2. 실험 상태

| ID | 우선순위 | 상태 | 입력 | 출력 | 의사결정 |
|---|---:|---|---|---|---|
| C-E01 gap audit | P0 | 미착수 | raw CSV | `gap_audit.csv/png` | 전처리 승인 |
| C-E02 metric audit | P0 | 미착수 | labels/scores | `metric_audit.csv` | PA 금지 검증 |
| C-E03 benchmark | P0 | 미착수 | shared folds | `model_comparison.csv` | 후보 축소 |
| C-E04 block CV | P0 | 미착수 | normal blocks | `block_cv.csv/png` | new normal 강건성 |
| C-E05 window sensitivity | P1 | 미착수 | w=5/10/20 | `window_sensitivity.csv` | window 확정 |
| C-E06 ablation | P0 | 미착수 | A/S/R/O/Q | `ablation.csv` | 영향요인 |
| C-E07 sensor stress | P0 | 미착수 | perturbation grid | `robustness.csv/png` | 모델 통과/탈락 |
| C-E08 jitter stress | P1 | 미착수 | synthetic jitter | `jitter.csv/png` | AI2 ACF 해석 |
| C-E09 calibration | P0 | 미착수 | C0/C1/C2 | `coverage.csv/png` | p-value 표현 |
| C-E10 explanation | P0 | 미착수 | alarm windows | `explanation_agreement.csv` | 원인 표현 승인 |
| C-E11 persistence | P1 | 미착수 | k=1/3/5 | `persistence.csv` | 운영 경보 확정 |
| C-E12 clean rerun | P0 | 미착수 | clean env | `clean_run.log` | 제출 승인 |

## 3. 모델 설정 상한

### Isolation Forest

- `n_estimators`: 200, 500
- `max_samples`: 128, 256, `auto`
- `max_features`: 0.7, 1.0
- test contamination 사용 금지

### PCA-MSPC / DPCA

- explained variance: 90%, 95%
- empirical q99.0/q99.5/q99.9
- lag: 1/2/4
- covariance: empirical vs Ledoit-Wolf shrinkage

### LSTM-AE

- 공식 구조 1개, 수정 구조 1개
- learning rate 0.001 고정
- batch 128 고정
- 최대 300 epoch
- early stopping patience 30
- seed 3개

총 조합 수를 늘리기보다 정상 block 일반화를 우선한다.

## 4. 결과 기록 규칙

각 run은 다음을 JSON으로 남긴다.

```json
{
  "run_id": "C-E03_M3_fold2_seed42",
  "code_hash": "...",
  "data_hash_normal": "...",
  "data_hash_outlier": "...",
  "config": {},
  "fit_rows": [],
  "calibration_rows": [],
  "test_rows": [],
  "metrics": {},
  "runtime_sec": 0.0,
  "artifacts": []
}
```

## 5. 모델 동결표

| 후보 | 실행 성공 | B0 개선 | Worst FP | 강건성 | 설명 audit | 최종 판정 |
|---|---|---|---|---|---|---|
| B0 | | 기준 | | | | baseline |
| B1 IF | | | | | | |
| B2 LSTM-AE | | | | | | |
| M1 Mahalanobis | | | | | | |
| M2 PCA-MSPC | | | | | | |
| M3 DPCA | | | | | | |

## 6. 변경 기록

| 시각 | 변경 | 이유 | test 결과를 본 뒤 변경했는가 | 영향받는 실험 |
|---|---|---|---|---|
| | | | | |

## 7. 제출 전 자동 검사

- [ ] 모든 window가 하나의 burst 안에 있음
- [ ] fold별 원본 행 중복 없음
- [ ] scaler/model/calibrator fit에 normal test와 outlier가 없음
- [ ] point adjustment 함수가 없음
- [ ] prediction row가 원본 행과 역추적됨
- [ ] 표의 숫자가 run JSON에서 자동 집계됨
- [ ] stochastic model 3개 seed 완료
- [ ] 실패 run도 삭제하지 않고 로그에 남음
- [ ] 빈 requirements 파일이 아님
- [ ] 절대경로가 없음
- [ ] 블라인드 금지 문자열이 없음
