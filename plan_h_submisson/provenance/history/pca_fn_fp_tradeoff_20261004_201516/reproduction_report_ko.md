# 기존 최고 PCA 재현

저장 모델 별도 프로세스: 4347행의 원행 순서·점수·판정이 기존 파일과 일치했다.
정상 전용 재학습: P1 W20 k2 Q와 P0 W1 k2 Q를 원분할/train_role 경계로 재학습하고 normal calibration Q95 정렬 및1%분위수를 다시 계산해도 TP329/FN28/FP1/TN3989를 재현했다. 원임계값 1.3078792257682357.
개발 selection: TP108/FN3/FP8/TN1978.
원본 통계 특징은 mean/std(ddof0)/RMS/min/max, StandardScaler와 fullSVD PCA는 정상 train에만 fit했다. 유효rank>2를 확인했다. Equipment_state·날짜·파일명·행번호는 모델 입력에 없다.
직전 median/IQR는 기존P1 target1%보다 Recall이 낮았으며, 당시FPR제약을 만족하는 후보 중 선택됐다. P1 임계값0.1% 대조보다 내부TP1/FP-2 이득은 있었지만 개발후반 악화는 선택고정 후 확인됐다. 직전탐색 전체를 반복할 근거는 없다.
