# 원결과 재현

저장 모델을 별도 Python 프로세스에서 추론한 4347행의 ID·판정이 기존 파일과 일치했다. 최대 점수 오차 0.
정상 학습에서 원설정을 재학습한 경우도 TP 329, FN 28, FP 1, TN 3989으로 일치했다. 재현과 신규 실험은 분리했다.
과거 FN 28 중 첫19행 FN 26개, fallback FN 13개. 개발 선택에서도 첫19행 FP 8개, FN 3개였다. 과거 오류는 이미 공개된 사실이며 시작부 가설의 착안에 사용했다.
기존 코드의 Q는 표준화 공간 잔차 제곱합이고 T²는 중심화한 PCA transform 좌표/학습 고유값이다. 모든 유지 고유값이 허용오차보다 크며 k<유효 rank를 검사했다.
근거: reproduction.json, predictions/original_*.csv, src/reproduce.py.
