# PCA 후속 v2 실험 패키지

먼저 **review_report_ko.html**을 브라우저로 열면 된다. 인터넷 없이 표·그림을 볼 수 있다.

선정 구성은 G1_Q999다. 기존 PCA 경보를 보존하고, 고정 R5 lag2 예측잔차가 정상 calibration의99.9백분위수17.54312199063481을 현재와 직전 관측에서 모두 넘을 때 보조 경보를 추가한다. 실제 배포나 기존 모델 교체는 수행하지 않았다.

- S1+S2: B0 FN3/FP8 → 선정 FN2/FP8.
- C0(기존 R5_A0) 대비: FN1→2, FP9→8. 기존 R5보다 한 이상행을 더 놓치는 대가를 공개한다.
- V: B0와 선정 모두 FN0/FP0. 유지 통과이며 FN 감소 재현은 not_assessable.
- H: B0 FN28/FP1 → 선정 FN13/FP1. 이미 공개된 과거 평가에서 관찰된 개선이다.
- 새로운 독립 자료는 없으며 현장 개선이 입증됐다고 주장하지 않는다.

재실행:

```bash
bash RUN_EXPERIMENT.sh
```

프로젝트 `.venv`를 자동 탐색하고 새 `replays/실행시각/`에 저장한다. 기존 결과는 덮어쓰지 않는다. 다른 컴퓨터에서는 Python3.14.4의 전용 가상환경을 만들고 `python -m pip install -r requirements.lock.txt`로 의존성을 준비한 뒤 `PYTHON_BIN=/전용환경/bin/python bash RUN_EXPERIMENT.sh`로 지정한다. 실행환경 잠금은 원래 실제 환경 전체를 기록한 것이며 시스템 Python 변경은 필요하지 않다.

동봉 고정 모델을 재로딩하고 C의 두 분위수를 재계산해 등록값과 일치하는지 확인한다. 전체 재실행에서 데이터/모델/후보를 다시 탐색하지 않는다. 검증에는 R5 파라미터를 정상 T만으로 재구성하는 동일성 검사도 포함한다.

별도 온라인 추론:

```bash
python src/infer_online.py --csv INPUT.csv --out NEW_OUTPUT.csv
```

입력은 한 파일의 한 원분할을 시간순으로 담은 CSV다. 열은 row_id, TimeStamp, AI0_Vibration, AI1_Vibration, AI2_Current다. 라벨이 필요하지 않다. 파일/분할 사이에서는 새 추론을 시작하고, 동일 분할 내0.5초 초과 공백은 R5/확인 이력을 초기화한다. P1의 과거20행은 원정의에 따라 공백을 넘어 유지한다. 연속 스트림 중간부터 입력하면 이전20행 문맥이 없으므로 처음부터 판정한 결과와 같지 않을 수 있다.

파일 안내:

- diagnosis_ko.md, diagnostic_plan.json: 후보 실행 전 진단과 표시 범위
- protocol_v2.json/.sha256: 역할·기준·6개 슬롯·임계값·의존 범위
- selected_config.json/.sha256: V 이전 잠금
- all_trials.csv, block_metrics.csv, followup_metrics.csv: 중복·탈락 포함 전체 수치
- paired_selected_rows.csv, paired_counts.csv: B0 및 C0와 다른 판정
- predictions/, tables/: 원행 예측·입력범위·경보 시각·조건별 분모
- models/: 원 PCA/P0와 고정 R5
- src/: 진단·판정·온라인 추론·검증·분석·보고·재현 코드
- data/, inputs/, manifests/: 원본 CSV·기존 실행 근거·고정 분할
- validation_report.json, end_to_end_replay.json: 회귀/추론/전체 재현 기록
- MANIFEST_SHA256.json: 패키지 파일 무결성 목록

기존 실행과 수치가 다른 경우 실패 로그를 보존하고 원인을 조사해야 한다. 목표 수치에 맞추어 라벨·분할·모델을 바꾸면 안 된다.
