# Plan D 진단 실험 모음 (v1–v6 · w1–w5)

`plan_d_submission` 의 제출본 구성을 **하나도 바꾸지 않고** 돌린 진단 실험 27개 표와 해설이다.
두 묶음으로 되어 있다.

| 묶음 | 표 | 질문 | 문서 |
|---|---|---|---|
| **v1–v6** | 18개 | 최종보고서 적대적 검토에서 "문구 수정만으로는 해결되지 않는" 지적에 근거를 만든다 | [01_검토대응_v1-v6.md](01_검토대응_v1-v6.md) |
| **w1–w5** | 9개 | 정상·고장이 다른 날에 수집된 데이터로 조기탐지가 가능한가. 안 되면 무엇을 만들 수 있는가 | [02_조기탐지_w1-w5.md](02_조기탐지_w1-w5.md) |

보고서에 바로 옮겨 쓸 수치만 모은 것 → [03_수치요약.md](03_수치요약.md)
표 목록·질문·검증 기준 → [MANIFEST.csv](MANIFEST.csv)

---

## 불변 원칙

세 실험 묶음 모두 아래를 지켰고, 매 실행마다 코드가 `assert` 로 확인한다.

- **동결 구성 불변**: 1단 M3 PCA-MSPC(특징 23) · 2단 M1 Mahalanobis(진동 15) ·
  conformal p ≤ 0.01 · 빨강 = 같은 버스트 3 window 연속 · 학습 블록 0–2 · 보정 블록 3 · 평가 블록 4
- **바꾸지 않은 파일**: `config.yaml`, `src/` 전체, `outputs/predictions.csv`,
  기존 표 49개 (`e*`, `r*`, `d*`, `c*`)
- **진단 결과를 모델 선정에 쓰지 않는다.** 다른 후보가 더 좋게 나와도 "사후 비교"로만 적는다.
  → `decision_log.md` **DL-018**(v1–v6), **DL-019**(w1–w5)
- **사전 판정 규칙**: 실험마다 *결과를 보기 전에* 정한 해석 규칙이 있고, 코드가 그 규칙을
  표의 `사전판정` 열 또는 stdout 에 직접 찍는다. 결과에 맞춰 규칙을 바꾸지 않았다.

실행 첫 단계에서 재구성한 p₁·p₂ 가 `predictions.csv` 와 일치하는지 확인하고,
일치하지 않으면 중단한다.

---

## 재현

표를 만드는 코드는 이 폴더에 복사해 두지 않았다 (중복 편집 사고 방지).
정본은 `plan_d_submission/` 에 있다.

```bash
cd plan_d_submission

# v1-v6 (약 6초)
python make_review_tables.py
python make_review_tables.py --only E1 E3     # 일부만

# w1-w5 (약 25초). W5 는 v2a 를 읽으므로 make_review_tables.py 가 먼저 돌아야 한다
python make_earlywarning_tables.py
python make_earlywarning_tables.py --only W2 W3

# 전체 파이프라인 (약 61초). 보조 스크립트 루프 끝에 두 모듈이 배선되어 있다
python run_all.py --mode quick
```

산출 위치: `plan_d_submission/outputs/tables/v*.csv`, `w*.csv`
이 폴더의 `tables/` 는 그 **스냅샷**이다.

### 재현성 확인 기록

- `run_all.py --mode quick` 재실행 후 표 **76개 전부 바이트 동일**, `predictions.csv` 동일
- `pytest tests -q` → **13 passed**
- 실행 환경: Python 3.13.9 / Windows 11

> `outputs/` 를 지우고 재실행하면 안 된다. full 모드 산출물 `e2_bl1_g1_planD.csv` 가
> 사라져 `e2_model_comparison.csv` 의 BL-1 행이 빠지고, `e2_selection_gates.csv` 가
> 달라져 **v4a 결과가 변한다.** 제자리 재실행 후 diff 로 검증하는 방식을 쓸 것.

---

## 표 이름 규칙

기존 접두어와 겹치지 않게 골랐다.

| 접두어 | 생성 스크립트 | 내용 |
|---|---|---|
| `e*` | `run_all.py` | 본 파이프라인 (window 민감도·모델 비교·CV·ablation·섭동·coverage) |
| `r*` | `make_report_tables.py`, `make_audit_tables.py` | 보고서 보충표·감사 진단표 |
| `d*` | `make_domain_tables.py`, `make_mofn_table.py`, `make_protocol_tables.py`, `make_supervised_control.py` | 도메인·M-of-N·프로토콜·지도학습 대조군 |
| `c*` | `make_correction_tables.py` | 정정 대응표 |
| **`v*`** | **`make_review_tables.py`** | **검토 대응 진단표** |
| **`w*`** | **`make_earlywarning_tables.py`** | **조기탐지 가능성 진단표** |

---

## 알려진 상태 불일치

v1–v6 작업 지시서가 가정한 저장소 상태와 실제가 다르다. 실험 자체는 영향받지 않았지만,
**보고서 반영은 아직 하지 않았다.** 누가 이어받더라도 같은 벽을 만난다.

| # | 지시서 전제 | 실제 |
|---|---|---|
| 1 | `docs/audit/FINAL_REPORT_ADVERSARIAL_REVIEW.md` | **없음** |
| 2 | 제출본 커밋 `d11ed9d` | 저장소에 **없음** |
| 3 | `report/check_numbers.py` 에 수치·금지표현 추가 | 파일 **없음** |
| 4 | 보고서 표 2-12 · 4-2 에 열 추가 | 두 표 **없음** (현재 2-4/2-5/2-8/2-9/4-1 까지) |
| 5 | `run_all.py` 루프의 `("상호작용 진단표 i1-i4", IT)` 다음에 배선 | 해당 항목·i 표 **없음** → 루프 **맨 끝**에 배선함 |
| 6 | `DL-018` 추가 | 기존은 `DL-015` 까지 → 지시대로 **DL-018** 로 적어 016·017 이 빈다 |

1·3·4 때문에 §7 의 "P0·P1·P2 문구 수정"을 무엇으로 바꿔야 하는지 알 수 없어,
제출본 27쪽 보고서는 손대지 않았다. 반영에 필요한 것은 **검토 문서 하나**다.
