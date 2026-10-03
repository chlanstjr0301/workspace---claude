# 환경별 실행 상태

확인일: 2026-10-04 (Asia/Seoul)

## Colab CLI

| 환경 | 상태 | 사용 원칙 |
|---|---|---|
| WSL Ubuntu `~/.local/bin/colab` | **정상**, v0.7.4, OAuth 인증 완료 | Plan B 원격 실행의 유일한 지원 경로 |
| Windows `anaconda3/Scripts/colab` | **실행 불가** | 사용하지 않음 |

Windows 설치본의 실패는 설정 문제가 아니라 플랫폼 제약이다.
`colab_cli/console.py`가 Unix 전용 모듈 `termios`를 import하므로 Windows에서는
`ModuleNotFoundError: No module named 'termios'`가 발생하고 `--help`도 실행되지 않는다.
수정 대상으로 취급하지 않고 WSL 셸 안에서 직접 실행한다.

```bash
wsl -d Ubuntu
~/.local/bin/colab sessions
~/.local/bin/colab usage
```

`\\wsl.localhost\...` 같은 UNC 경로에서 `wsl.exe --cd`를 호출하면
`ERROR_PATH_NOT_FOUND`가 날 수 있으므로, WSL 셸에 먼저 진입한 뒤 Linux 경로에서
명령을 실행한다.

## 가속기 확인

- 무료 등급에서도 CLI로 `colab run --gpu T4`를 실행해 Tesla T4 15,360 MiB 할당을
  실제 확인했다.
- 컴퓨트 유닛 잔액 0.00은 무료 T4 사용 불가를 뜻하지 않는다.
- L4, G4, A100, H100, TPU는 CLI 옵션에 표시되지만 계정 등급과 시점별 가용성에
  따라 달라지며 이 작업공간에서는 검증하지 않았다.
- `colab run`의 기본 timeout은 30초이므로 학습에는 명시적으로 늘리거나,
  유지 세션에서 `setsid nohup`으로 실행한다.

## 현재 변동 상태

2026-10-04 조회 시:

- 활성 Colab assignment: **1개**
- 세션명: `trainer`
- 하드웨어: T4 / GPU / Standard
- 상태: **IDLE**
- 마지막 실행: `/tmp/mark.py`, 2026-10-04 01:58:40
- 표시 잔액: 0.00 compute units
- 표시 사용률: 0.80/hr

이 항목은 시시각각 변한다. 현재 상태가 필요하면 반드시 `colab sessions`와
`colab usage`를 다시 조회한다. 폴더 정리 과정에서는 세션을 중단하지 않았다.

## 로컬 노트북

| 구성요소 | 상태 |
|---|---|
| CPU | AMD Ryzen AI 7 PRO 350, 8코어/16스레드, 정상 |
| RAM | 27.6 GB |
| GPU | AMD Radeon 860M, 장치 인식; 현재 PyTorch는 CPU 빌드 |
| NPU | AMD NPU Compute Accelerator Device, 장치 인식; Python provider 미설치 |
| ONNX Runtime | CPU/Azure provider만 사용 가능 |

Plan C 통계 파이프라인은 전체 실행이 약 29초이므로 로컬 CPU를 기준 환경으로 한다.
Plan B LSTM-AE는 WSL Colab T4를 기준 환경으로 한다.
