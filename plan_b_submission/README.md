# Colab LSTM-AE 실험

가이드북 LSTM-AE를 재현하고, 공격1·2가 지적한 항목을 한 번에 하나씩 바꿔 실측한다. 각 실험이 어떤 주장에 답하는지는 `previous research/공격3 - 검증.md` §4에 정리했다.

## 폴더 구성

```
plan_b_submission/
├── lstm_ae_experiments.py   실험 전체 (학습·평가·요약)
├── requirements.txt         Colab 기본 환경에 이미 있음. 로컬 실행용
├── data/                    press_data_normal.csv, press_data_outlier.csv
└── results/                 실행하면 생김
    ├── runs/<설정>_s<시드>.json     실험 1회 결과
    ├── scores/<설정>_s<시드>.npz    window별 점수·채널별 오차 (오류분석용)
    ├── all_runs.csv
    ├── summary.csv
    └── summary.md               보고서용 요약표
```

## 실험 설정

| ID | 바꾼 것 (가이드북 G0 대비) |
|---|---|
| G0 | 없음 — 가이드북 그대로 (abs, naive window, offset 100, 마지막 1시점 MSE, 이상 섞인 P=R 임계값, MSE, MinMax) |
| A1 | abs 제거 |
| A2 | gap-aware windowing |
| A3 | 이상 점수 = window 전체 MSE |
| A4 | 임계값 = 정상 검증 q99.9 |
| A5 | Huber 손실 |
| A6 | RobustScaler |
| P20 | Plan B: abs 제거, gap-aware, offset 0, 전체 MSE, 정상 전용 임계값, Plan B 분할(학습 0~11999 / 임계값 12000~14999 / 테스트 15000~ + 이상 전체), window 20 |
| P10 | P20과 같고 window 10 |

학습 예산은 모든 설정이 같다: 최대 300 epoch, EarlyStopping patience 30, ReduceLROnPlateau patience 15. `--replicate`를 붙이면 G0를 가이드북 원래 예산(800 / 120 / 50)으로 한 번 더 돌려 `G0orig`로 저장한다.

## 실행 순서

Colab 세션 안에서는 업로드 후 `/content/colab`을 작업 디렉터리로 두고 실행한다.

```bash
# 1) 시드 42로 전 설정 한 바퀴 — 먼저 결과 윤곽과 1회 소요 시간을 본다
python lstm_ae_experiments.py --seeds 42

# 2) 나머지 시드
python lstm_ae_experiments.py --seeds 1337 2026

# 3) 가이드북 원래 예산 재현 (G0orig) + LSTM-AE 블록 교차검증 오경보 (P20·P10 각 5회 학습)
python lstm_ae_experiments.py --replicate --blockcv --seeds 42

# 요약표만 다시 만들 때
python lstm_ae_experiments.py --summary
```

- **이어서 돌리기:** 끝난 실험은 `results/runs/`에 json이 남아 건너뛴다. 세션이 끊기면 같은 명령을 다시 실행하면 된다. 단, Colab 세션이 종료되면 세션 안의 파일도 사라지므로 단계마다 `results/`를 내려받아 둔다.
- **소요 시간:** CPU(이 노트북 VM, 2코어)에서는 epoch당 5~10초였다. T4에서는 1단계 첫 실험의 `train_sec`를 보고 전체 시간을 추정한다. 1단계는 9개 실험, 전체는 27개 실험 + G0orig 1개 + 블록 교차검증 10회 학습이다.
- **스모크 테스트:** `python lstm_ae_experiments.py --smoke`는 epoch 2로 전 설정이 도는지만 확인한다(`results_smoke/`에 저장, 수치는 의미 없음).

## Colab CLI로 돌리기 (WSL)

`colab` CLI는 **WSL(또는 리눅스/맥)에서만** 돈다. Windows 쪽 설치본은 `colab_cli/console.py`가 유닉스 전용 모듈 `termios`를 import 해서 `--help`조차 실행되지 않는다. 설치는 `pip install google-colab-cli`(확인한 버전 0.7.4), 실행 파일은 `~/.local/bin/colab`이므로 비로그인 셸에서는 PATH를 직접 넣어야 한다.

알아둘 것:

- `upload`/`download`는 **파일 1개씩만** 받는다(`colab upload <로컬> <원격>`). 폴더는 tar로 묶어 옮긴다.
- `exec -f <로컬.py>`는 로컬 스크립트를 업로드 없이 원격 커널에서 실행한다. 단 스크립트가 읽는 **데이터는 따로 올려야** 한다.
- `exec`의 `--timeout` 기본값이 **30초**다. 학습을 foreground로 돌리면 반드시 올려야 한다.
- 커널 상태는 `exec` 호출 사이에 유지된다. 기본 작업 디렉터리는 `/content`.
- **긴 학습은 foreground로 돌리지 말고** VM 안에서 `setsid nohup`으로 떼어 놓고 로그를 폴링한다. 그래야 로컬에서 세션이 끊겨도 학습이 계속된다.

```bash
cd "/mnt/c/Users/cmsch/Desktop/대회/2026년 제6회 K-인공지능 제조데이터 분석 경진대회/plan_b_submission"
export PATH="$HOME/.local/bin:$PATH"

# 1) 세션 생성 (무료 등급에서도 T4 할당됨. 컴퓨트 유닛을 쓰지 않는다)
colab new -s trainer --gpu T4
colab status -s trainer

# 2) 업로드 — 폴더째 올리려면 tar로 묶는다
tar czf /tmp/colab_exp.tar.gz --exclude=results lstm_ae_experiments.py requirements.txt data
colab upload -s trainer /tmp/colab_exp.tar.gz /content/colab_exp.tar.gz
echo 'import subprocess; subprocess.run("mkdir -p /content/colab && tar xzf /content/colab_exp.tar.gz -C /content/colab", shell=True)' \
  > /tmp/setup.py
colab exec -s trainer -f /tmp/setup.py --timeout 300

# 3) 실행 — 백그라운드로 떼어 놓는다
cat > /tmp/driver.sh <<'SH'
cd /content/colab
python -u lstm_ae_experiments.py --seeds 42
python -u lstm_ae_experiments.py --seeds 1337 2026
python -u lstm_ae_experiments.py --replicate --blockcv --seeds 42
echo "=== ALL DONE ==="
SH
colab upload -s trainer /tmp/driver.sh /content/driver.sh
echo 'import subprocess; subprocess.run("setsid nohup bash /content/driver.sh > /content/driver.log 2>&1 < /dev/null &", shell=True)' \
  > /tmp/launch.py
colab exec -s trainer -f /tmp/launch.py --timeout 120

# 4) 진행 확인 (원하는 만큼 반복)
echo 'import subprocess; print(subprocess.run("tail -40 /content/driver.log", shell=True, capture_output=True, text=True).stdout)' \
  > /tmp/peek.py
colab exec -s trainer -f /tmp/peek.py --timeout 120

# 5) 결과 내려받기 — 단계마다 해 둔다. 묶어서 1개 파일로 받는다
echo 'import subprocess; subprocess.run("cd /content/colab && tar czf /content/results.tar.gz results", shell=True)' \
  > /tmp/pack.py
colab exec -s trainer -f /tmp/pack.py --timeout 300
colab download -s trainer /content/results.tar.gz /tmp/results.tar.gz
tar xzf /tmp/results.tar.gz -C .     # → ./results/

# 6) 끝나면 반드시 반납 (유휴 VM도 자원을 쓴다)
colab stop -s trainer
```

세션이 백엔드에서 정리돼 `exec`가 404/401을 내면 `colab sessions`로 확인하고 `colab new`로 다시 만든 뒤, 2단계부터 반복하면 `results/runs/`에 남은 json 덕분에 끝난 실험은 건너뛴다. 단 세션이 종료되면 VM 안의 파일은 사라지므로, **5단계 다운로드를 단계마다 해 두는 것이 재시작의 전제**다.

## 결과를 받은 뒤

`results/` 폴더를 이 위치에 두면 이어서 분석한다. 보고서에 들어갈 표는 `summary.md`에서 바로 가져오고, 오류분석(FN/FP 위치, 채널별 재구성 오차)은 `scores/*.npz`로 한다.
