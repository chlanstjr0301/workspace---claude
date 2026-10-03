# K-인공지능 제조데이터 분석 경진대회 작업공간

Plan별 코드와 결과가 섞이지 않도록 다음과 같이 구분한다.

| 구분 | 계획 문서 | 실행 패키지 | 논문 |
|---|---|---|---|
| Plan A | `docs/plan/plan A/` | `plan_a_submission/` | `papers/plan_a/` |
| Plan B | `docs/plan/plan B/` | `plan_b_submission/` | - |
| Plan C (SHIFT-Guard) | `docs/plan/plan C/` | `plan_c_submission/` | `papers/plan_c/` |
| Plan D (CARE-Press) | `docs/plan/plan D/` | `submission/` | `papers/plan_d/` |

> Plan D 실행 패키지만 현재 `submission/`이라는 기존 이름을 유지한다. 2026-10-04
> Windows의 잔여 디렉터리 핸들이 이름 변경을 막고 있어 강제 종료 없이 보존했다.
> 재부팅 또는 핸들 해제 후 `plan_d_submission/`으로 변경하면 된다.

## 공용 폴더

- `data/raw/`: 공용 원본 데이터. Plan별 결과를 두지 않는다.
- `previous research/`: 이전 조사와 공격적 검증 기록.
- `tmp/`: 재생성 가능한 임시 렌더링 파일.

## 결과 위치 원칙

- 모델 실행 결과는 각 실행 패키지 내부에 저장한다. Plan A/C/D는 `outputs/`, Plan B는 재개 가능한 기존 규약 때문에 `results/`를 사용한다.
- 논문 PDF는 각 `papers/plan_*/output/`에 저장한다.
- 프로젝트 루트에는 `outputs/`, `output/`, `paper/`, `scr/` 같은 모호한 이름을 만들지 않는다.

환경별 실행 가능 여부는 `docs/ENVIRONMENT_STATUS.md`를 기준으로 한다.
