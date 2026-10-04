PCA 엄격 재현·개선 검토 (2026-10-04)

먼저 열 파일: review_report_ko.html (브라우저), review_report_ko.md (원문)
결론: 개선 미확인. 원결과 329TP/28FN/1FP가 재현됐고,
개발에서 고정한 median/IQR PCA는 과거 평가313TP/44FN/3FP로 악화했습니다.

기존 논문·CSV·ZIP을 덮어쓰지 않은 추가 검토 패키지입니다.
새 검토에 사용된 원본 CSV 사본은 data/에 있으며 해시가 원본과 같습니다.
models/는 학습 모델, predictions/는 원래 행에 연결된 점수·판정·오류입니다.
all_trials.csv에 54개 운영구성의 모든 내부 검증이 있고,
후처리2개를 더한 총56개 구성입니다. IF seed/fold 반복과 원재현은 별도입니다.

재실행: 기존 /home/lim/hydraulic_ai_alt 프로젝트에서
bash runs/adversarial_pca_20261004_103236/RUN_REVIEW.sh
새 runs/adversarial_pca_<시각>_rerun_<PID>에 결과를 만들며 기존 결과를 덮어쓰지 않습니다.
원ZIP/풀린패키지의 대조 검사를 포함하므로 원프로젝트와 이전 논문 패키지가 필요합니다.
이 추가검토 ZIP만으로 원ZIP 911개 파일 검증을 생략하는 방식으로 실행하지 않습니다.

별도 라벨 없는 CSV 추론 (프로젝트 .venv 활성화 후):
python src/infer_review.py --model models/final_MEDIANIQR_Q.joblib --csv INPUT.csv --target 0.005 --out NEW_OUTPUT.csv
정상 Q95 배율/ECDF는 고장 확률이 아닙니다.

선택/검증: protocol.json, hypotheses.md, frozen_selection.json,
validation.json, saved_inference_validation.json, final_verification.json.
원본으로 확인된 문제와 미검증 한계: critique_matrix.csv.
기존 6항목 평가 근거: rubric_evidence.csv (공식 외부 채점표 원문은 미확보).
필요한 다음 자료: next_data_requirements_ko.md.

이미 노출된 과거 평가를 새 독립 검증으로 부르지 않습니다.
고장 시작 정답이 없어 고장 전 조기탐지를 검증하지 못했습니다.
인증값이나 토큰·가상환경은 포함하지 않았습니다.
