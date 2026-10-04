# 프레스 유압펌프 이상 탐지 및 오경보 분석

정상 운전변화의 오경보와 이상 미탐을 함께 평가하는 재현 프로젝트입니다. 원본 CSV는 읽기만 하며 `configs/frozen/rows.csv`가 중복을 제외한 분석 목록입니다. 단일 정상 날짜와 단일 이상 날짜이므로 고장과 날짜별 운전조건을 분리해 검증할 수 없습니다. 실제 고장 시작/정비 시각이 없어 사전 예측 시간을 주장하지 않습니다.

## 재실행

실제 실행은 Python 3.13 CPU 환경입니다. 정확한 패치 버전과 라이브러리는 각 실행의 `manifests/environment.json`에 저장합니다.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python scripts/run_all.py
```

한 명령으로 고정 분할 검증, 모델 학습, 개발 평가, 제한된 보조 실험, 선택 고정, 최종 평가, 그림·HTML 보고서·추론 일치 검사를 실행합니다. 새 실행은 `runs/experiment_<UTC시각>/`를 만들어 기존 결과를 덮어쓰지 않습니다. 첫 실행의 준비 결과가 있으면 분할을 다시 만들지 않습니다. 설치는 네트워크가 허용된 환경에서 수행합니다.

중단한 실행만 이어갈 때:

```bash
.venv/bin/python scripts/run_all.py --resume runs/experiment_20261003_01
```

저장된 모델·점수는 재사용합니다. `selection_lock.json` 이후 설정을 바꿔 최종 평가를 재튜닝하지 않습니다. 최종 완료 실행의 resume은 분석/보고서/검증만 재생성하므로 원래 결과를 보관하려면 새 실행을 사용하십시오.

## 추론

```bash
.venv/bin/python src/infer.py --run runs/experiment_20261003_01 --input data/outlier_data.csv --output new_predictions.csv
```

한 번에 하나의 시간순 센서 스트림을 넣습니다. 여러 파일을 연결하지 않습니다. 입력에 필요한 열은 TimeStamp와 센서3개이며 라벨·파일·행번호는 분류 입력이 아닙니다. 일치 중복은 최초 행만 남기고, 변환실패/누락/비증가 시간은 오류로 알려 임의 보간을 방지합니다. 기존 출력파일은 덮어쓰지 않습니다. 출력의 보정확률은 이 데이터에서의 실험값이며 현장 고장 확률 보장이 아닙니다.

전체 원본 파일 추론은 학습/개발/최종 구분 없이 해당 파일을 하나의 스트림으로 처리하므로 보고서의 최종 평가 표와 같지 않습니다. `src/verify.py`는 최종 분할별 별도 스트림으로 추론해 저장된 평가와 일치하는지 검사합니다.

## 설계와 저장 위치

- `configs/protocol.json`: 결과 확인 전 제한한 후보·예산·선택 규칙.
- `configs/frozen/`: 원본 해시, 고정 행/버스트/분할, 중복 제거 내역.
- `src/`: 준비, 학습, 평가, 분석, 보고서, 추론, 검증 코드.
- `references/`: 가이드북과 확인한 설정·미확인 정보.
- `runs/<실행>/models/`: 모델, 정규화·점수 변환·sigmoid 객체.
- `runs/<실행>/tables/`: 지표, 버스트 비교, 센서 조합/제거, 후처리, 확률, 오류 조건.
- `runs/<실행>/predictions/`: 모델·seed별 행 예측, 선택 모델 FP/FN, 추적 정보.
- `runs/<실행>/manifests/`: 모델별 대상·공통 행·학습 행 및 환경.
- `runs/<실행>/figures/`: 공백을 연결하지 않은 실제 시간축·곡선·신뢰도 그림.
- `runs/<실행>/report.html`: 평가표6항목 대응 보고서.

대상별 native 평가는 예측 가능 행이 달라 모델 순위에 사용하지 않습니다. 공통행 평가는 모든 길이의 교집합, 전체행 운영은 M2 보완입니다. 셋 다 분리 저장합니다. seed42·43·44를 전부 보고하고 선택은 3seed 공통점수 평균입니다. AP와 사다리꼴 PR-AUC는 구분합니다.

LSTM은 가이드북 구조를 PyTorch로 구현했지만 공통 분할, endpoint 라벨, 최대20epoch, 정상 보정 임계값을 사용하여 가이드북 정확 재현은 아닙니다. 검출기는 정상만 학습하나 개발 이상 라벨은 선택·sigmoid에 사용합니다. 이번 실험의 0.5초 버스트/1% FPR/윈도우 후보는 공식 대회 기준이 아닙니다.

## Colab 실행 기록

CLI는 `/home/lim/.local/bin/colab`이며 모든 명령에 `--auth=oauth2`를 명시합니다. 인증 파일/토큰은 산출물에 포함하지 않습니다. 이번 전용 CPU 세션은 `hydraulic-exp-20261003`입니다. 입력 전송 목록과 해시는 `reports/upload_manifest.json`, 후속 코드 전송 내역은 `reports/transfer_log.json`에 기록합니다. 결과 확인 후 이번 생성 세션만 종료합니다. 다른 세션을 변경하지 않습니다.

CPU 학습 최대20epoch×세 seed×두 LSTM 버전×세 길이, 정상 내부 조기종료를 사용합니다. IF300 trees/max_samples256를 고정했습니다. 시간이 더 허용되는 원문800epoch 재현과 신규 날짜 외부 검증은 별도 후속 과제입니다.

데이터 출처: 중소벤처기업부, Korea AI Manufacturing Platform(KAMP), 소성가공 예지보전 AI 데이터셋, 스마트제조혁신추진단(㈜인터엑스), 2022.12.23., www.kamp-ai.kr.
