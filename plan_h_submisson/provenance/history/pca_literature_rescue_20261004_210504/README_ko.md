# PCA 문헌 기반 제한 실험 패키지

결론: 사전 조건을 모두 통과한 개선 후보가 없어 기존 최우수 PCA를 유지한다. 기존 모델을 교체하거나 장비에 배포하지 않았다.

먼저 `review_report_ko.html`을 브라우저로 열면 된다. 인터넷 없이 그림과 표를 볼 수 있다. 상세 수치는 `all_trials.csv`, `block_metrics.csv`, `baseline_vs_candidates.csv`, 원본 행별 변화는 `paired_errors.csv`에 있다.

- `data/`: 수정하지 않은 입력 CSV 두 개
- `inputs/`, `manifests/`: 원분할, 원예측, 개발 이력, 원행 대응 및 fit 행
- `preregistration.json`/`.sha256`: 점수 계산 전 후보18개와 선택 규칙
- `selected_config.json`/`.sha256`: V 개봉 전 선정 후보 없음으로 고정
- `models/`: 원PCA/P0와 정상만으로 학습한 보조 모델
- `predictions/`: S1/S2 전체18후보 및 제한된 V/과거평가
- `src/`: 실제 실행·추론·검증·보고서 코드
- `references/`: 확인한 논문의 URL, 해시, 읽은 범위와 자체 작성 요약. 원문 PDF/추출 전문은 재배포하지 않는다.
- `validation_report.json`, `end_to_end_replay.json`: 검사 및 전체 재실행 결과
- `MANIFEST_SHA256.json`: 포함 파일 SHA-256

재실행: 이 폴더에서 `bash RUN_EXPERIMENT.sh`.

기존 프로젝트에서는 프로젝트 `.venv`를 자동 사용한다. 다른 컴퓨터에서는 Python3.14.4의 전용 가상환경에 `requirements.lock.txt` 버전을 설치하고 `PYTHON_BIN=/전용환경/bin/python bash RUN_EXPERIMENT.sh`로 실행한다. 예를 들어 `python3.14 -m venv .venv`, `.venv/bin/python -m pip install -r requirements.lock.txt` 뒤 실행할 수 있다. 새 `replays/<실행시각>/`에 저장하며 기존 결과를 덮어쓰지 않는다. 경로를 인자로 주려면 아직 존재하지 않는 디렉터리를 지정한다.

별도 추론: `python src/infer_aux.py --csv INPUT.csv --out OUTPUT.csv`. 같은 파일/분할의 시간순 한 구간 CSV를 입력하며 열은 TimeStamp, AI0_Vibration, AI1_Vibration, AI2_Current, 추적용 row_id만 허용한다. 기준선 및6개 보조 점수만 출력한다. 연구용 탈락 후보를 배포하라는 뜻이 아니다.

원모델 재학습은 이번 라운드 범위가 아니다. 저장 원모델의 별도 프로세스 추론과 새 보조 모델 재학습을 포함하는 전체 실행 재현을 구분한다. 개발 및 과거 평가 자료는 이미 반복 노출됐으며 새로운 독립 검증/현장 일반화 증거가 아니다.
