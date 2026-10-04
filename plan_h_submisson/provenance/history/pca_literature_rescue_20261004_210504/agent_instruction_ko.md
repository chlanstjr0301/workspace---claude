# 재현 에이전트 명령문

이 패키지의 `README_ko.md`, `preregistration.json`, `selected_config.json`을 읽고 고정된 실험을 재현하라. 원본과 기존 결과를 수정하지 말고 새 디렉터리에 저장하라. CPU2 threads로 `bash RUN_EXPERIMENT.sh`를 실행하고 오류가 나면 로그를 보존하며 원인을 조사하라. 원선택이나 수치를 맞추기 위해 데이터를 변경하지 마라.

원모델은 동봉 `models/baseline.joblib`의 P1 W20 k2 Q, 원P0 fallback, 고정 threshold1.3078792257682357이다. 이미 제공된 정상train/정상calibration/selection/test를 유지한다. 검출기 fit은 정상train만, 임계값은 정상calibration만 사용한다. S1+S2 선택 뒤 lock을 쓰고 V를 열며, 과거 평가는 baseline+선정1개+미리 고른R0 비교군으로 제한한다. 개선 후보가 없으면 탈락 후보를 골라 test하지 마라.

보조 점수 R0 원점수, R1 원PCA잔차 LW 거리, R2/R3 잔차 RBF-SVDD(정상 고정표본4096, 중앙거리 제곱과2배), R4/R5 원센서 lag1/2 ridge 예측잔차 LW 거리. 추가 calibration FP 예산0/.0005/.001로 총18개를 넘지 않는다. 모든 후보는 A0 OR (available & aux>tau)로 원TP와 원FP를 보존한다. 새로운 창·가중치·seed·후처리 후보를 추가하지 마라.

주 조건은 pooled FN 감소·F2 상승·F1 비하락, 각 S블록 F1/F2 비하락, 모든 해당 블록 추가FP≤floor(.001*Nnormal)와 절대FPR≤1%다. 작은 블록에도 추가FP 최소1개를 허용하지 마라. 동률1e-12 및 등록된 우선순위를 유지한다. V 원FN0이면 개선 통과가 아닌 확인 불가다. 선택 결과를 보고 기준을 바꾸지 마라.

해시, 원행 정렬, 파일/split/gap 상태, 미래값 변경 검사, OR 불변조건, 별도 추론, 전체재실행을 확인하고 한국어로 성공/실패/미확인을 구별해 보고하라. 현재 기대 결론은 선정 후보 없음이지만 실제 재현과 불일치하면 원인을 보고하라. 과거test는 이미 노출됐으므로 독립 최종 검증이라고 부르지 마라. 점수/F2를 고장확률/현장 비용비로 해석하지 마라.
