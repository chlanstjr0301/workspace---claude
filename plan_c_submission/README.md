# SHIFT-Guard — Plan C 실행 패키지

프레스 유압펌프 진동·전류 데이터의 burst-aware, shift-aware 이상탐지 파이프라인이다.

## 빠른 실행

프로젝트 루트에서:

```bash
python plan_c_submission/run_all.py --mode quick
```

전체 센서 섭동과 3개 시드 반복:

```bash
python plan_c_submission/run_all.py --mode full
```

테스트:

```bash
python -m pytest plan_c_submission/tests -q
```

## 구현 범위

- timestamp 품질 점검과 0.5초 기준 burst 분할
- burst 내부에서만 생성하는 gap-safe window
- 정상 5개 시간 block 교차검증
- B0 범위 규칙, Isolation Forest, shrinkage Mahalanobis, PCA-MSPC, Dynamic PCA
- 정상 calibration 점수 기반 p-value
- 연속 3-window 경보 병합
- 센서 gain/offset/polarity/noise/jitter/dropout 섭동
- 특징군 및 채널 counterfactual repair 설명
- `predictions.csv`, 표, 그림, 실행 manifest 자동 생성

TensorFlow가 설치된 경우에만 LSTM-AE를 사용할 수 있다. 핵심 quick/full 파이프라인은 TensorFlow 없이 동작한다.

## 출력

`outputs/` 아래에 생성된다.

- `manifest.json`: 데이터 해시, 환경, 실행 설정
- `predictions.csv`: 최종 SHIFT-Guard 예측
- `tables/model_comparison.csv`
- `tables/block_cv.csv`
- `tables/robustness.csv`
- `tables/calibration_audit.csv`
- `tables/explanation.csv`
- `figures/*.png`

## 해석 주의

- 이상 CSV는 독립 고장 사건 1건이다.
- acquisition burst 21개를 21건의 고장으로 해석하지 않는다.
- conformal p-value는 정상성 순위이며 고장 사후확률이 아니다.
- 이상 발생 이전 데이터가 없으므로 진짜 예지 lead time을 주장하지 않는다.
