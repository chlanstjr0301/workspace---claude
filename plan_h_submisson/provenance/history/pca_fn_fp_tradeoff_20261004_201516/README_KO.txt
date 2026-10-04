기존 최고 PCA의 FN–FP 절충 실험

먼저 열기: review_report_ko.html
전체 문서: review_report_ko.md

결론: 기존 P1 W20 k2 Q 모델을 유지합니다.
개발에서는 짧은 PCA 보완으로 FN3→2, FP8→0이 가능했지만,
고정 후 과거 평가에서는 FN28→34, FP1→0으로 악화했습니다.
과거 평가를 보고 후보를 재선택하지 않았습니다.

selection과 과거 평가 자료는 이미 노출된 자료이며 새 독립 검증이 아닙니다.
고장 시작 정답이 없어 실제 고장 전 조기예측은 검증하지 못했습니다.

포함:
- protocol.json, hypotheses.md: 사전 규칙과 개발 근거
- all_trials.csv, threshold_sweep.csv: A12+B24 전체 결과
- frozen_selection.json: 평가 전 고정한 세 구성
- baseline_vs_candidates.csv, block_metrics.csv, paired_errors.csv
- predictions/: 원행 연결 점수·판정·오류
- models/: 기준선, 정상 전용 재학습, 짧은 보완 모델
- data/: 원본 CSV의 해시가 같은 사본
- src/, requirements.lock.txt, RUN_EXPERIMENT.sh
- validation_report.json, final_verification.json
- inputs/: 원래 고정 분할, 기준 모델, 직전 검토 근거
- figures/: PR/운영점, FN-FP, F1/F2, 혼동행렬, 오류 시간축

기존 프로젝트에서 한 번에 재실행:
cd /home/lim/hydraulic_ai_alt
bash runs/pca_fn_fp_tradeoff_20261004_201516/RUN_EXPERIMENT.sh

새 실행 폴더에 저장하므로 현재 결과를 덮어쓰지 않습니다.
원본 ZIP과 직전 검토 ZIP의 실제 파일 대조까지 포함하므로 원프로젝트의
두 기존 패키지 및 직전 검토 폴더가 필요합니다. 자세한 경로는 inventory.json에 있습니다.
환경은 프로젝트 .venv를 쓰며 CPU 스레드2, 실행 제한90분입니다.

단일 CSV 추론은 기존 패키지 없이도 다음 코드와 저장 모델로 실행할 수 있습니다.
가상환경을 활성화하고 이 폴더에서:
python src/infer_system.py --model models/baseline.joblib --threshold 1.3078792257682357 --csv INPUT.csv --out NEW_OUTPUT.csv

보완 후보 추론(개선 성공/배포 권고 모델이라는 뜻이 아님):
python src/infer_system.py --model models/warmup_W3.joblib --threshold 1.7186836831529793 --csv INPUT.csv --out NEW_OUTPUT.csv

모델 점수와 Q95 배율은 고장 확률이 아닙니다. 인증 파일·토큰·가상환경은 포함하지 않았습니다.
