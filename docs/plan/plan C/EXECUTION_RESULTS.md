# Plan C 실행 결과

실행일: 2026-10-04  
실행 환경: Lenovo 21QNCTO1WW, Ryzen AI 7 PRO 350 (8C/16T), RAM 27.6 GB  
최종 실행: `python plan_c_submission/run_all.py --mode full`

## 결론

Plan C의 최종 운영 후보는 **B1 Isolation Forest**다. 선택은 이상 데이터의 F1만
최대화하지 않고, 정상 시간 블록에서 발생하는 최악 경보 사건 수를 먼저 제한하는
사전 규칙에 따랐다.

| 항목 | 전체 실행 결과 |
|---|---:|
| 평균 window F1 | 0.9492 |
| 평균 Recall | 0.9358 |
| 평균 AP | 0.9936 |
| 최악 정상 블록 경보 사건 | 2 |
| 평균 이상 burst 탐지율 | 0.9255 |
| 전체 실행 시간 | 29.249초 |

M2 PCA-MSPC의 평균 F1은 0.9791, M1 Mahalanobis는 0.9786으로 더 높았지만,
두 모델 모두 최악 정상 블록에서 경보 사건이 5건 발생했다. B1은 기준선 B0와 같은
최악 2건을 유지하면서 B0의 F1 0.6865를 크게 개선했으므로 운영 후보로 선정했다.

## 민감도와 원인 분석

- 20-sample 창은 F1 0.9837로 가장 높았지만, 탐지 지연 증가와 짧은 burst 배제,
  사후 선택 편향을 피하기 위해 사전 고정한 10-sample 창을 유지했다.
- 진폭 특징군 A를 제거하면 F1이 0.9738로 상승했다. 반면 시간형태 특징군 S를
  제거하면 F1이 0.8273으로 하락해, 단순 진폭보다 파형의 시간형태가 핵심임을 보였다.
- 경보 지속성 `k=3`은 평균 정상 경보 1.0건과 이상 burst 탐지율 0.9294의 절충점이다.
  `k=1`은 탐지율 0.9882지만 최악 경보 20건, `k=5`는 최악 경보 1건이지만 탐지율
  0.8941이다.
- 선택 모델의 전체 교란 최저 Recall은 `dropout_ai2`에서 0.8879였다. AI2 채널
  결측 시 Yellow-S로 강등하고 센서/DAQ 점검을 요구해야 한다.
- counterfactual repair 기준 전역 최상위 채널은 AI0_Vibration이었다. 이 결과는
  인과 고장 위치가 아니라 모델 의존 증거로만 해석한다.

## 실험 레지스터 상태

| ID | 상태 | 산출물 |
|---|---|---|
| C-E01 gap audit | 완료 | `data_audit.csv`, 단위 테스트 |
| C-E02 metric audit | 완료 | window/AP/alarm/burst 지표 |
| C-E03 model benchmark | 완료 | `model_comparison.csv` |
| C-E04 normal block CV | 완료 | `block_cv.csv` |
| C-E05 window sensitivity | 완료 | `window_sensitivity.csv` |
| C-E06 feature ablation | 완료 | `feature_ablation.csv` |
| C-E07 sensor stress | 완료 | `robustness.csv` |
| C-E08 jitter stress | 완료 | `robustness.csv` |
| C-E09 calibration audit | 완료 | `calibration_audit.csv` |
| C-E10 explanation audit | 완료 | `explanation.csv` |
| C-E11 alarm persistence | 완료 | `persistence_sensitivity.csv` |
| C-E12 clean rerun | 완료 | `manifest.json`, `RESULTS.md` |

## 하드웨어 결정

Radeon 860M GPU와 AMD NPU는 장치로 확인했지만, 설치된 PyTorch는 CPU 빌드이고
ONNX Runtime에는 DirectML/NPU provider가 없다. 데이터가 작아 전체 통계 파이프라인이
약 29초에 끝나므로, 이번 재현 패키지는 추가 드라이버 의존성이 없는 노트북 네이티브
CPU 실행을 기준으로 고정했다.

## 정직한 제한

- 기존 Plan B의 TensorFlow LSTM-AE 실행 결과는 참고자료로 남겼지만, Python 3.13
  환경에서 TensorFlow를 사용할 수 없어 동일 프로토콜 재학습 표에는 섞지 않았다.
- VUS-PR은 검증되지 않은 구현을 추가하지 않고 AP와 burst/alarm 지표로 대체했다.
- 제공된 이상 CSV는 하나의 날짜/고장 기록이므로, 21개 burst를 21개의 독립 고장으로
  주장하지 않는다.

상세 수치와 예측 결과는 `plan_c_submission/outputs`에 있으며, 장치·버전·데이터 해시는
`manifest.json`에 기록되어 있다.
