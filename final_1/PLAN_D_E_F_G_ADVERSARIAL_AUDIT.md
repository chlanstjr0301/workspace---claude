# Plan D · E · F · G 적대적 감사 보고서 (Red Team)

- 대상 과제: 2026년 제6회 K-인공지능 제조데이터 분석 경진대회, 일반국민/대학(원)생 부문 **문제 ③**
- 감사 기준일: 2026-10-04 (KST) · 제출 마감 2026-10-08(목) 23:59
- 감사 대상 커밋: `origin/main` 878b090를 작업 브랜치에 병합한 상태
- 원칙
  - Plan D–G의 코드와 기존 결과는 수정하지 않았다.
  - 모든 재실행과 검증은 scratchpad의 **임시 복사본**과 **새 출력 폴더**에서 했다.
  - 인증 파일·토큰은 열지 않았다. 세션 기록 파일은 키 이름과 비밀이 아닌 메타데이터만 확인했다.
- 판정 우선순위: ① 공식 공지·과제공개 → ② 공식 결과보고서 양식 → ③ 코드·데이터·재실행 산출물 → ④ 설정 잠금·manifest·decision log → ⑤ 논문·Plan 문서 → ⑥ 자기평가·전략 문서

---

## 1. 감사 범위와 방법

### 1.1 범위

| Plan | 코드·결과 | 문서·논문 | 크기 |
|---|---|---|---|
| D (CARE-Press) | `plan_d_submission/` (`run_all.py`, `src/`, `tests/`, `outputs/`) | `docs/plan/plan D/README.md`, `EXECUTION_BOARD.md`, `plan_d_submission/decision_log.md`, `papers/plan_d/` | 5.8 MB |
| E (PCA 패키지) | `plan_e_submission/code/`, `results/pca_20261003_cpu/` | `paper/manuscript_ko.*`, `documentation/`, `agent/` | 131 MB, 900 파일 |
| F (버스트·단일행 보완) | `plan_f_submission/src/`, `scripts/`, `configs/`, `runs/experiment_20261003_01/` | `paper/논문_한국어.*`, `report.html`, `audit/`, `먼저읽기.txt` | 210 MB, 955 파일 |
| G (Logistic Regression) | `plan_g_submission/src/`, `results/` | `paper_icml_logistic/`, `AGENTS.md`, `BUNDLE_README.md` | 61 MB, 256 파일 |

- `plan_d_submission/outputs_full_backup/`은 저장소에 **존재하지 않는다**(`ls` 결과 없음, 코드·문서 참조도 0건). 이 경로에 대한 검증은 **미검증**이다.
- 내부 지침 준수
  - Plan F `먼저읽기.txt`: 원결과를 덮어쓰지 말 것, `src/verify.py`는 별도 복사본에서 실행할 것 → 준수함.
  - Plan G `AGENTS.md`: `baseline_original.py` 수정 금지, SHA-256 확인 → 수정하지 않았고, 해시는 `3b0855d2…a0a8`로 일치한다.

### 1.2 방법

1. 공식 문서 4종을 처음부터 끝까지 읽었다. 공지사항, 과제공개, 결과보고서 양식, `KAMP 경진대회 - 내 생각.md`가 대상이다.
2. 각 Plan의 최종 예측파일에서 TP/FP/TN/FN, P/R/F1/F2, FPR/FNR, AP, ROC-AUC, Brier, log-loss를 직접 다시 계산했다(§8).
3. 각 Plan을 깨끗한 가상환경에서 재실행했다(§9).
   - D: quick 모드 2회. 기존 outputs 포함본과 outputs 삭제본으로 나눠 돌렸다.
   - E: develop+final 전체 재실행.
   - F: 저장 모델 추론 검증과 `run_all.py` 전체 재실행.
   - G: 저장 결과 분석과 class-weight 재학습.
4. 원본 행 ID와 타임스탬프 수준에서 분할 중복, gap 횡단 window, 이상 라벨 사용 단계를 확인했다.
5. 선택 잠금·완료 파일의 타임스탬프로 E와 F의 개발 순서를 대조했다.
6. 저장소 전체에서 블라인드 위반 문자열(학교명, 사용자 경로, 세션 ID, 장비명)을 검색했다.

### 1.3 데이터 공통 사실 (재계산)

- 정상: 2022-07-12 00:00:00–01:16:55, 20,000행, 599 burst(gap > 0.5 s), 관측 0.556 h
- 이상: 2022-07-17 10:51:07–10:53:53 (**2분 46초**), 600행, 21 burst. 전부 `Equipment_state=1`
- 양성 사건은 **1건**이고, 정상과 이상의 날짜가 완전히 다르다. 고장 시작 시각이 없으므로 **리드타임은 측정할 수 없다.**

---

## 2. 공식 요구사항 기준표

| ID | 공식 요구 (출처) | D | E | F | G |
|---|---|---|---|---|---|
| R1 | 데이터 생산단위·변수·결측·중복·이상치·불균형·검증전략 (문항 1, 15) | 충족(E0·E1·E9) | 충족(audit·coverage) | 충족(audit·가이드북 쪽수) | 부분 |
| R2 | **제조 이상 확률** 예측 모델 (문항 2) | 미충족(`risk_score=1-p`) | 조건부(sigmoid, 동일 사건 보정) | 조건부(sigmoid) | 조건부(LR 확률, 날짜 교란) |
| R3 | 베이스라인 포함 2개 이상 모델 비교(F1 등) | 충족(BL0·BL1·M1–M3) | 충족(P0–P2·I0–I2) | 충족(M0–M4, LSTM 포함) | 부분(LR 변형만) |
| R4 | 최종모델 선정근거, 동일 평가조건 비교 (양식 제2장) | 결함(D-P1-01·02) | 결함(E-P1-02·03) | 결함(F-P1-05) | 결함(G-P1-04) |
| R5 | 주요 영향변수·**상호작용**, FN·FP 집중 공정조건 (문항 3) | 부분(D-P1-04, D-P2-04) | 부분 | 충족에 가까움 | FN 중심 부분 |
| R6 | 사전경보·검사 우선순위·공정점검·작업자 의사결정 (문항 4) | 충족(green/yellow/red + 조치) | **미흡** | 미흡(2-of-3만) | **없음** |
| R7 | 불균형·앙상블·불확실성·확률보정·제조지식 결합 (문항 5) | 부분 | 부분 | 부분 | 부분 |
| R8 | 동일 환경 전처리→학습→추론→결과 자동 실행 (문항 6) | 충족(quick 38 s) | Linux 충족, Windows 미검증 | 충족(전체 재실행) | 저장결과·재학습 부분 |
| S1 | 결과보고서 **PDF**, 양식 6장 + 설문 장 | **없음** | **없음** | **없음**(HTML·PDF 논문은 양식 아님) | **없음** |
| S2 | zip: 소스·requirements·학습데이터·README·**테스트 예측결과** | zip 미구성. 내용물은 대부분 있음 | 미구성 | 미구성 | 미구성, **예측결과 없음** |
| S3 | 발표자료 **PPT와 PDF** | 없음 | 없음 | 없음 | 없음 |
| S4 | 설문 완료화면 캡처 | 없음 | 없음 | 없음 | 없음 |
| S5 | 블라인드: 소속·로고 금지, 성명·팀명만 허용 | 로그에 사용자 경로 | `/home/lim`, 세션 ID | `/home/lim`, 세션명 | `C:\Users\EKR` |
| S6 | 외부데이터 사용 시 출처 명시 | 해당 없음 | 해당 없음(가이드북 참고) | 가이드북 쪽수 인용, 원본 PDF 제외 | 해당 없음 |

> ICML 형식 논문(D·E·G), 한국어 연구논문 PDF/DOCX/HTML(F), `report.html`(F)은 공식 KAMP 결과보고서를 **대체할 수 없다.**

---

## 3. Plan D 감사 (CARE-Press)

**주장 요약**
- 문서 근거: `plan_d_submission/README.md`, `decision_log.md`, `papers/plan_d/main.tex`
- 프로토콜: gap-aware 10샘플 window, 정상 5블록 CV(3 학습 / 1 보정 / 1 holdout), 정상만으로 scaler·threshold 적합, conformal p ≤ 0.01
- 비교 결과: BL0·M1·M3가 AP/AUROC/Recall 1.0으로 완전분리된다. 섭동 시 오경보 증가가 BL0 +61.3, M1 +89.7, M2 +22.3, M3 **+17.0 pp**다.
- 선정: 게이트를 통과한 모델은 M3 하나뿐이다. 최종 구조는 1단 M3, 2단 진동 전용 M1이다.

**재현 확인**: quick 재실행 결과 `outputs/tables/*` 21개 CSV와 `predictions.csv`(15,299행)가 커밋본과 **오차 없이 일치**한다. 테스트는 13개 모두 통과했다.

### D-P1-01 · 선정 게이트를 결과를 본 뒤 3차례 바꿨고, 원 규칙대로라면 BL-0가 선택된다

- **심각도**: P1
- **관련 문항**: 2 (선정근거)
- **예상 감점**: −3 ~ −5
- **공격 대상 주장**: "분석 프로토콜 (결과를 보기 전에 동결)"(`plan_d_submission/README.md:51`), "게이트를 통과한 모델은 M3 PCA-MSPC 단독"(`README.md:150-151`)
- **실제 증거**
  - `decision_log.md:9-13`: PROTOCOL_FREEZE(01:20) 뒤 10–11분 사이에 게이트 3개가 바뀌었다.
    - DL-001(01:30): gate2에 정상측 오경보 증가를 추가
    - DL-002(01:31): gate3에 허용오차 도입
    - DL-003(01:31): gate5 신설
    - 사유는 모두 결과를 본 뒤의 것이다. "엄격 비교 시 … BL-0가 자동 선택됨"(`decision_log.md:11`)이라고 직접 적혀 있다.
  - `e2_selection_gates.csv`:
    - BL-0의 F1 평균은 **0.9734로 최고**이고 최악 블록 FP율도 0.0201로 최저다.
    - M3는 F1 0.9618로 BL0·M1·M3 중 최하위다.
    - M3의 오경보 증가 16.98 pp는 gate2 한계 20 pp를 3 pp 차이로 통과했다. 이 20 pp는 원래 **Recall 하락** 한계였는데 오경보 증가에 그대로 재사용됐다(`selection.py:233-234`).
  - 논문 `main.tex:268-269`는 "We fix the following before examining any result"라고 쓴 뒤 §Amendments(`main.tex:622`)에서 개정을 공개한다. 공개 자체는 양호하지만, 같은 논문 안에서 서술이 충돌한다.
- **재현·반증**: `config.yaml`의 원 게이트(허용오차 0, gate5 없음, gate2 = Recall 하락만)로 `apply_gates`를 재평가하면 BL0만 통과한다. `decision_log.md:11`의 기록과 일치한다.
- **심사위원 질문**: "M3가 이기도록 게이트를 바꾼 것 아닙니까? 원래 규칙의 승자는 무엇이었습니까?"
- **우승 영향**: 문항 2 선정근거의 신뢰도가 떨어진다. 다만 공개 기록이 있어 방어할 여지는 남아 있다.
- **최소 수정안**: README·보고서·논문의 "결과를 보기 전에 동결"을 "프로토콜은 동결, 선정 게이트는 결과 관찰 후 3회 개정(DL-001~003)"으로 바꾼다. 원 게이트와 개정 게이트의 선정 결과를 나란히 표로 싣는다.
- **완료 판정**: 보고서 제2장에 게이트 변경 전·후 비교표가 있고, "결과를 보기 전 동결" 문구가 게이트에 대해서는 0건이다.

### D-P1-02 · "이상 데이터는 선택에 쓰지 않는다"는 주장은 사실이 아니다

- **심각도**: P1
- **관련 문항**: 2
- **예상 감점**: −2 ~ −3
- **공격 대상 주장**: "이상 데이터는 임계값·특징·하이퍼파라미터 선택에 쓰지 않는다"(`run_all.py:12`, `README.md:57`)
- **실제 증거**: 모델 적합과 임계값은 정상만으로 정한다(`run_all.py:171-183`, 테스트 `test_no_leakage.py`). 그러나 **모델 선택**에는 이상 라벨이 들어간다.
  - gate2의 이상 Recall 하락(`selection.py:233`)
  - gate4의 이상 burst 탐지 수(`selection.py:237-238`)
  - 2차 정렬키 `recall_mean`(`selection.py:265-267`)
  - 특징군 ablation 결과(OBS-04)가 해석에 쓰였다.
  - 한계 공개(`decision_log.md:33-34`)도 이상 데이터를 사전실험에서 반복 관찰했음을 인정한다.
- **심사위원 질문**: "정상만 학습했다는 것과 이상 라벨을 전혀 안 썼다는 것은 다른 말 아닙니까?"
- **최소 수정안**: "적합·임계값은 정상만, 모델 선택 게이트 일부(gate2·gate4)와 동률 해소에는 이상 데이터 사용"으로 정정한다.
- **완료 판정**: 제출물에서 "이상 데이터를 선택에 쓰지 않음" 문구 0건

### D-P1-03 · 1단 M3의 완전분리는 전류(CUR) 특징, 즉 날짜·세션 지문에 기대고 있다

- **심각도**: P1
- **관련 문항**: 1·2·3
- **예상 감점**: −3 ~ −5
- **공격 대상 주장**: 완전분리(AP=1.000)와 "탐지력의 출처는 형태·진폭"(`README.md:143-144`)
- **실제 증거**
  - 1단 입력 25차원에 전류 유래 특징 8개가 있다(`features.py:50-68`).
    - A: `A_*_CUR` 3개
    - S: `S_ac1/ac2/zcr_CUR` 3개
    - O: 2개
  - 최종 예측파일의 이상 window 428개에 붙은 첫 번째 기여 특징
    - `S_ac1_CUR` 155, `S_ac2_CUR` 88, `O_absmean_CUR` 39
    - 권고문 분포: **307/428(71.7%)이 "전류센서 이득·오프셋·결선 확인"**
  - M3 설명 안정성 상위 특징에도 `S_ac2_CUR`, `S_ac1_CUR`, `S_zcr_CUR`가 있다(`e4_explanation_stability.csv`).
  - 반대로 2단(진동 전용, `run_all.py:452-454`)은 이상 window의 96.3%에서 p ≤ 0.01이다. 진동만으로도 분리 신호가 있다는 뜻이다.
- **심사위원 질문**: "모델이 고장마다 '전류센서를 점검하라'고 한다면, 고장이 아니라 다른 날 전류 계측 조건을 탐지한 것 아닙니까?"
- **우승 영향**: 출제문의 핵심(정상 운전변화를 이상으로 오인하지 말 것)과 정면으로 충돌한다. 다만 2단 진동 확인 구조가 이미 있어 **방어 가능한 형태**다.
- **최소 수정안**: (새 모델 없이 문서 수정) 간판 성능을 "빨강(1단 ∧ 2단 진동 확인)" 기준으로 바꾸고, 1단 단독 성능은 세션 교란 위험과 함께 보조로 둔다. 노랑의 의미를 "계측·운전조건 변화 가능성"으로 명확히 한다.
- **완료 판정**: 보고서 간판 표가 빨강 기준이고, 1단 기여 특징의 전류 비중이 제3장에 공개되어 있다.

### D-P1-04 · 정상 holdout 블록에서 1단 경보율이 8.2%이고, 대부분 정상 운전변화 구간에 몰려 있다

- **심각도**: P1
- **관련 문항**: 3·4
- **예상 감점**: −2 ~ −4
- **공격 대상 주장**: "노랑 = 계측조건 변화" 분리로 오경보를 관리한다(`README.md:111-120`)
- **실제 증거**: `outputs/predictions.csv`의 `split=='holdout'`(블록 4, 원본 행 16051–19999, 2,929 window)
  - green 2,688 / yellow 238 / red 3
  - 1단 경보율 8.23%, 2단 경보율 1.33%
  - 비녹색 241개 중 217개가 행 16,000–19,000에 있다. Plan A가 찾은 "진폭 저하·상관 반전" 정상 운전변화 구간이다.
  - 106 burst 중 54 burst(51%)에서 노랑이 1회 이상 울린다.
  - 비녹색 권고문: "체결·정렬·**베어링 유격 점검**" 122, "DAQ 동기화" 70
  - 빨강 3 window(행 18984·18985, 19969)는 정상인데 "정밀점검"을 지시한다.
- **CV 추정과 최종 운영 구성이 크게 다르다**
  - 같은 M3로 블록 4를 holdout으로 둔 CV fold 4(학습 1–3, 보정 0)에서는 FP가 **8/2,929 window(0.27%)**였다(`e3_normal_block_cv.csv` 19행).
  - 최종 예측(학습 0–2, 보정 3, `run_all.py:456-457`)에서는 **241/2,929(8.2%)**다. 30배 차이다.
  - 보정 블록 3이 CV에서 FP 3건뿐인 "조용한" 구간이라, 이 블록으로 정한 임계값이 블록 4의 운전변화를 견디지 못한 것이다.
  - 따라서 CV로 보고한 오경보율은 실제 제출 모델의 오경보율을 대표하지 않는다. conformal 교환가능성이 블록 간에 성립하지 않는다는 직접 증거다.
- **재현**: §8.3 스크립트, `e3_normal_block_cv.csv`의 M3 행
- **심사위원 질문**: "정상 운전 15분 동안 burst 절반에서 노랑이 뜨면 작업자가 경보를 믿겠습니까?"
- **최소 수정안**: (모델 변경 없이) 이 결과를 제3장 FP 집중조건의 본문으로 쓴다. 노랑을 "작업자 조치 없음, 기록·누적 모니터링"으로 정의하고, 조치는 빨강에서만 하는 운영안을 제시한다. 이 운영안이 holdout을 보고 정한 것임도 밝힌다.
- **완료 판정**: 제3장에 블록 4의 구간별 경보율 표가 있고, 제4장 노랑 조치가 "기록"으로 정의되어 있다.

### D-P1-05 · 예측파일 기준 성능이 보고되지 않았고, 간판 F1과 단위가 다르다

- **심각도**: P1
- **관련 문항**: 2·6
- **예상 감점**: −2 ~ −3
- **공격 대상 주장**: 간판 F1 0.962(M3), 0.973(BL0)(`e2_model_comparison.csv`, `main.tex:580-585`)
- **실제 증거**
  - 간판 F1은 5-fold 평균이다. fold마다 같은 이상 window 428개를 다시 쓰므로 유병률이 약 12%다.
  - 제출용 `predictions.csv`(holdout 2,929 + fault 428, 유병률 12.75%)로 다시 계산하면:
    - 1단(노랑+빨강): TP 428, FP 241, FN 0 → **F1 0.780**, FPR 0.082
    - 빨강만: TP 375, FP 3, FN 53 → **F1 0.931**, Recall 0.876, FPR 0.0010
  - fit·calibration 행이 들어간 전체 파일 기준으로는 1단 F1 0.647이다. 이는 in-sample 행이 섞인 값이다.
- **최소 수정안**: 보고서에 "CV 평균(모델 비교용)"과 "최종 예측파일 holdout(운영용)" 두 표를 단위·유병률과 함께 싣는다.
- **완료 판정**: 보고서의 운영 성능 수치가 `predictions.csv`에서 한 줄 명령으로 재현된다.

### D-P1-06 · `risk_score=1-p`는 확률이 아니고, 약속한 사후확률 민감도 표가 생성되지 않았다

- **심각도**: P1
- **관련 문항**: 2·5
- **예상 감점**: −2 ~ −4
- **공격 대상 주장**: Plan D 계획 §5.1 "현장 고장률 1%/0.1%/0.01% 사후확률 민감도 표를 별도 제시"(`docs/plan/plan D/README.md:214`)
- **실제 증거**
  - README가 `risk_score`는 사후확률이 아니라고 명시한 점(`README.md:61-62`)은 양호하다.
  - 그러나 holdout 정상 window의 73.1%가 `risk_score > 0.5`이다. holdout+fault 기준 Brier는 **0.487**로, 상수 예측 0.111보다 나쁘다. log-loss는 1.695다.
  - `calibration.py:290-297`의 `posterior_sensitivity` 함수는 `run_all.py`에서 **한 번도 호출되지 않는다**(grep 0건). 관련 출력 표도 없다.
- **심사위원 질문**: "문항 2는 제조 이상 '확률'을 요구합니다. 어디에 확률이 있습니까?"
- **최소 수정안**
  1. 기존 `posterior_sensitivity`를 빨강 TPR·FPR로 호출해 사후확률 표(사전확률 1/0.1/0.01%)를 생성한다.
  2. CV 점수로 교차적합 Platt 보정을 하고 신뢰도 곡선과 Brier를 보고한다. 임계값은 바꾸지 않는다.
- **완료 판정**: 확률 열과 보정 지표가 있고, 그 Brier가 상수 예측보다 낮다. 사후확률 표가 생성된다.

### D-P1-07 · 권고 조치문이 특징 이름에서 물리 원인을 단정하고, 설명 출처가 논문과 다르다

- **심각도**: P1
- **관련 문항**: 3·4
- **예상 감점**: −1 ~ −3
- **공격 대상 주장**: `recommended_action`이 현장 점검 문구라는 주장, 그리고 "vibration-only Mahalanobis monitor … supplies per-feature contributions for root-cause text"(`main.tex:604-606`)
- **실제 증거**
  - `explain.py:167-179`는 특징 접두어만으로 "상·하부 진동 관계 이상 - 체결·정렬·**베어링 유격 점검**" 같은 문구를 만든다. 데이터에 근거한 원인 판정이 아니다.
  - 기여도는 **1단 M3의 SPE 잔차**에서 계산된다(`run_all.py:496-501`, `m1`은 stage1 모델 = M3). 2단 진동 M1이 아니다.
  - 그 결과 이상 window의 72%가 전류 관련 문구를 받는다(D-P1-03).
- **최소 수정안**: 문구를 "점검 후보(원인 아님)"로 바꾸고 논문 서술을 "1단 M3 잔차 기여도"로 정정한다.
- **완료 판정**: 조치 문구에 "원인 아님, 우선 확인 후보" 표기가 있고, 논문과 코드의 설명 출처가 같다.

### D-P2-01 · quick 실행이 기존 출력물(BL-1 CSV)에 숨은 의존성이 있다

- **심각도**: P2
- **관련 문항**: 6
- **예상 감점**: −1
- **실제 증거**: `run_all.py:579-585`는 quick 모드에서도 기존 `outputs/tables/e2_bl1_g1_planD.csv`가 있으면 비교표에 합친다. `outputs/`를 지우고 quick을 돌리면 `e2_model_comparison.csv`가 5행에서 **4행**으로 줄고, BL-1 G0/G1 표 2개가 생성되지 않는다(§9). README "`outputs/` 를 지운 뒤 한 명령으로 전부 다시 생성된다"(`README.md:186-187`)와 다르다. 선정 결과(M3)는 같다.
- **최소 수정안**: README에 "BL-1은 full 모드(약 3.6 h) 또는 동봉 CSV 재사용"을 명시한다.

### D-P2-02 · `outputs/`의 단일 진실 원천이 섞여 있고, 로그가 현재 코드와 맞지 않는다

- **심각도**: P2
- **실제 증거**
  - `run_manifest.json`: mode quick, 2026-10-04 17:26
  - `run_manifest_full.json`: full, 05:14, 13,050.5 s
  - 현재 표는 quick 재실행 결과와 full의 BL-1 CSV가 섞인 상태다.
  - `full_run.log`의 마지막 줄 "final_alarm_summary.csv (1행)"은 현재 파일(4행: fit/calibration/holdout/fault)과 다르다. 로그는 구버전 코드의 기록이다.
  - `outputs_full_backup/`은 저장소에 없다(**미검증**). `papers/plan_d/main.pdf`와 `papers/plan_d/output/main.pdf`는 서로 다른 파일이다(`cmp` 불일치, 첫 쪽 텍스트 차이).
- **최소 수정안**: 최종 SSOT를 `outputs/`(quick 17:26 + BL-1 05:14 CSV)로 선언하고, 구 로그·중복 PDF는 zip에서 제외한다.

### D-P2-03 · 실행 보드가 갱신되지 않았다

- **실제 증거**: `docs/plan/plan D/EXECUTION_BOARD.md:5`는 최종 갱신 01:40이다. BL-1은 "진행 중", 최종모델은 "잠정"(`:17-18`)으로 남아 있다. 반면 `decision_log.md:15`는 05:15에 MODEL_FREEZE를 확정했다. 보드를 zip에 넣으면 불일치가 드러난다.
- **최소 수정안**: zip에서 제외하거나 갱신한다.

### D-P2-04 · FN 분석이 비어 있고, 최종 2단 시스템의 미탐 조건을 분석하지 않았다

- **실제 증거**
  - `e7_error_conditions_summary.csv`의 FN 행은 모두 0이다. M3 단독의 Recall이 1.0이라 분석 대상이 없다.
  - 최종 운영 출력(빨강)의 미탐은 53 window다. 원인은 "연속 3 window" 규칙(`run_all.py:470-478`)으로, burst마다 첫 2 window가 구조적으로 노랑이 된다(burst 17개 × 2 = 34개 + burst 15·16·20 등에서 추가).
- **최소 수정안**: 빨강 기준 FN 표(burst × 위치)를 추가한다.

### D-P2-05 · BL-1(LSTM-AE) 비교는 1 seed이고, 탈락 근거가 근소하다

- **실제 증거**
  - G1은 seed 0 하나(`config.yaml`, DL-004)다. 게이트 2·5는 미평가다(DL-005).
  - G1의 최악 블록 FP율 0.0268은 사후 허용오차 한도 0.0251을 0.0017 차이로 넘는다(`e2_selection_gates.csv`).
  - G0의 가이드북 재현(F1 0.743 vs 문헌 0.748, 3 seed)은 **강점**이다.

### D-P2-06 · 논문의 일부 단정이 수치와 다르다

- **실제 증거**
  - "Every fault window is scored above every normal hold-out window"(`main.tex:324-326`)라고 썼지만, BL-0의 AUROC 평균은 0.999999이고 M3는 0.99986이다. 1.0이 아니다.
  - 섭동은 offset·gain을 **전 채널에 동시에** 준다(`explain.py:125-129`). 현실의 센서 교체는 보통 채널 하나다. 채널별 결과가 없다.

---

## 4. Plan E 감사 (PCA 중심 패키지)

**주장 요약** (`results/pca_20261003_cpu/final_summary.json`)
- 주 운영모델: `P1_W20_K2_Q` (rolling 20행 통계 15차원 → PCA k=2 → Q(SPE)), 목표 FPR 0.01
- 최종: **TP 329, FN 28, FP 1, TN 3,989**, F1 0.9578, AP 0.9871
- F1 대안: 같은 모델, 임계값만 다름(목표 FPR 0.005) → TP 325, FN 32, FP 0
- IF 대표: `I1_W5_ensemble` → F1 0.649

**재계산·재현**
- 예측파일로 계산한 값이 위 수치와 **정확히 일치**한다.
- 두 예측파일의 `score` 열은 동일하고, 임계값만 1.3079 vs 1.4937로 다르다.
- 저장 모델 추론 4,347행이 일치한다.
- 새 폴더에서 develop+final을 처음부터 다시 실행해 선택 잠금과 최종 수치가 모두 동일함을 확인했다(70 s, §9).

### E-P1-01 · 주 운영모델이 수집 공백을 가로지르는 window를 쓴다

- **심각도**: P1
- **관련 문항**: 1·2
- **예상 감점**: −3 ~ −5
- **공격 대상 주장**: 논문 제목 "시간 공백과 예측 가능 범위를 고려한 …", "20행 = 약 1.9초"
- **실제 증거**
  - `code/src/experiment.py:78`: `U` 특징은 `['source_file','split','train_role']` 단위로만 묶는다. burst 경계를 무시한다는 뜻이다.
  - 최종 test에서 모델 경로 행 중 window가 gap을 가로지르는 비율(`rows_U20.csv`의 `crosses_gap`, 예측파일 `elapsed_seconds`로 재확인):
    - 정상 **1,983/3,971(49.9%)**
    - 이상 185/338
    - window 시간 범위: 중앙 3.4 s, 90분위 9.9 s, 최대 **24.1 s**
  - TP 323개 중 172개가 gap 횡단 window에서 나왔다.
  - 같은 길이의 burst-aware 모델 `P2_W20_K2_Q`는 같은 공통행에서 F1 0.974로 unaware 0.987과 비슷하다(`tables/burst_pairs_test.csv`). 그런데 선택되지 않았다.
- **심사위원 질문**: "20행 window가 실제로는 최대 24초, 서로 다른 수집 구간을 이어 붙인 것 아닙니까?"
- **최소 수정안**: (재튜닝 금지) 최종모델은 바꾸지 않는다. "주 모델은 gap 횡단을 허용한다"는 사실과 횡단 비율·시간 범위를 공개한다. burst-aware 결과를 같은 행 기준 민감도로 병기한다.
- **완료 판정**: 보고서에 gap 횡단 비율 표가 있고, "공백을 고려한 모델"이라는 표현이 주 모델에 붙어 있지 않다.

### E-P1-02 · 단일 이상 사건을 보정·선택·최종으로 쪼갰다. 이상 라벨은 선택과 확률보정에 쓰였다

- **심각도**: P1
- **관련 문항**: 2
- **예상 감점**: −2 ~ −4
- **공격 대상 주장**: "선택 잠금 후 최종 1회 평가"를 독립 검증처럼 읽히게 하는 서술
- **실제 증거**: `code/configs/first_experiment/frozen_rows.csv`의 이상 600행 구성

  | 용도 | 원본 행 | burst 수 | 시각 |
  |---|---|---:|---|
  | calibration | 1–132 | 4 | 10:51:07–10:51:34 |
  | selection | 133–243 | 4 | 10:51:39–10:52:08 |
  | test | 244–600 | 13 | 10:52:12–10:53:53 |

  - selection의 이상 111행이 모델 선택에 쓰였다(`protocol.json` `selection`).
  - calibration의 이상 132행이 sigmoid 적합에 쓰였다(`auxiliary.probability`: "fit calibration normal+anomaly").
  - **test는 같은 고장 기록의 뒷부분 1분 41초**다. 독립 고장이 아니다.
  - `protocol.json`에 `"prior_full_data_summaries_seen": true`가 있다.
- **심사위원 질문**: "최종 test의 이상은 선택에 쓴 이상과 같은 고장, 같은 날, 같은 2분짜리 기록 아닙니까?"
- **최소 수정안**: "정상만 적합, 이상 라벨은 선택(111행)과 보정(132행)에 사용, test는 동일 사건의 시간적 후반부"를 제2장 첫 표에 명시한다.
- **완료 판정**: 이상 라벨 사용 단계표가 보고서에 있다.

### E-P1-03 · E는 F의 최종평가가 시작된 뒤에 개발·잠금되었고, 두 Plan은 같은 test를 쓴다

- **심각도**: P1 (Plan 간 선택에 직접 영향)
- **관련 문항**: 2
- **예상 감점**: E를 F보다 낫다는 근거로 쓰면 −3 ~ −5
- **실제 증거**
  - E와 F의 최종 평가 행(4,347행)은 `row_id`와 라벨까지 **완전히 같다**(재확인).
  - E의 분할은 F의 산출물을 그대로 가져왔다("split_origin: first experiment immutable artifacts", `experiment.py:62`. 원 경로 `/home/lim/hydraulic_ai/configs/frozen/rows.csv`).
  - 타임스탬프(UTC):
    - F `selection_lock` 18:24:10 → F `final_complete` 18:50:07
    - E `audit` 18:37:34 → E `selection_lock` 18:38:54 → E `final_summary` 18:43:46
  - E의 개발·잠금은 F의 최종 채점 구간 안에서 일어났다. E 개발자가 F의 test 결과를 보지 않았다는 증거가 없다.
- **심사위원 질문**: "PCA 패키지는 IF 결과를 본 뒤 만든 '대안'입니까? 같은 test에서 두 번 고른 것 아닙니까?"
- **우승 영향**: E의 F1 0.958 > F의 0.918을 "E 선택 근거"로 쓰면 test 재사용 선택 편향이 된다.
- **최소 수정안**: E·F 비교는 "같은 최종 행에서의 사후 비교, 선택 근거로 사용하지 않음"으로 한정한다. Plan 선정은 test 이외 기준으로 한다(평가 보고서 §7).

### E-P1-04 · Windows 심사환경 재현성이 검증되지 않았고, 저장소 사본에서 무결성 검사가 실패한다

- **심각도**: P1
- **관련 문항**: 6
- **예상 감점**: −2 ~ −3
- **실제 증거**
  - 실행 경로가 `RUN_EXPERIMENT.sh`·`VERIFY_PACKAGE.sh`·`BUILD_PAPER.sh`(bash) 전용이다. README가 "Linux/macOS의 bash"라고 명시한다.
  - `code/requirements.lock.txt`의 numpy 2.5.3·pandas 3.0.6은 Python 3.12 이상이 필요하다.
  - 저장소 사본에서 `verify_package.py`가 `AssertionError: Hash mismatch: code/configs/first_experiment/frozen_bursts.csv`로 실패한다.
    - 원인: CSV 10개가 CRLF→LF로 바뀜(LF→CRLF 복원 시 해시 10/10 일치)
    - 로그 12개(`paper/build_console*.log` 등)가 누락됨
  - 해시 검사를 건너뛰면 저장 모델 추론 4,347행은 일치한다.
- **최소 수정안**: Windows용 `run.ps1` 또는 `python` 단일 진입점을 만들고, zip은 git이 아닌 원본 바이트로 만든다. manifest를 재생성한다.
- **완료 판정**: Windows에서 `python` 명령 하나로 검증이 PASS한다.

### E-P1-05 · 현장 활용방안이 사실상 없다

- **심각도**: P1
- **관련 문항**: 4
- **예상 감점**: −4 ~ −6
- **실제 증거**: 후처리는 `none`으로 잠금됐다(`selection_lock.json` `postprocess`). 경보 단계·조치·담당·재교정·센서 결측 대응을 정의한 파일이 없다. 논문도 운영 절차를 다루지 않는다.
- **최소 수정안**: Plan D의 green/yellow/red 운영 정의를 흡수하거나, 최소한 Q 기여도(예측파일의 `*_Q_contribution` 열)를 센서 점검 우선순위로 바꾸는 표를 만든다.

### E-P2-01 · IF의 대표 모델(W5)이 약하게 선정되어 PCA와 IF 비교가 왜곡될 수 있다

- **실제 증거**
  - IF 대표는 selection에서 FPR ≤ 0.01 제약을 통과한 W5다. W10은 0.0136, W20은 0.0156으로 탈락했다.
  - test에서 IF W5는 Recall 0.485, IF W20은 0.896이다(`tables/metrics_test.csv` 1209·1257행).
  - 논문은 같은 입력의 IF W20(320 TP / 6 FP)을 병기한다(`paper/pdf_text.txt:18-19, 295-301`). 이 점은 양호하다.
- **최소 수정안**: 발표에서 "PCA가 IF를 이겼다"는 비교는 W20 대 W20만 쓴다.

### E-P2-02 · 확률보정의 의미는 조건부다

- **실제 증거**: test 기준 sigmoid 확률의 Brier는 **0.0081**, log-loss 0.035다. 상수 예측 0.0754보다 우수하다. 그러나 보정에 쓴 이상은 같은 사건의 132행이고, 유병률은 8.2%로 고정된 평가 조건이다. 논문은 "어느 것도 고장확률은 아니다"(`pdf_text.txt:212`)라고 정확히 쓴다.
- **최소 수정안**: 보고서에 "이 평가 조건에서의 보정 확률, 현장 사전확률은 Bayes로 조정"을 쓴다.

### E-P2-03 · 모든 후보를 test에서 채점해 공개했다

- **실제 증거**: `tables/metrics_test.csv`에 잠금 이후 모든 후보(PCA 78 점수, IF 28)의 test 지표가 있다. 잠금이 먼저였으므로 재튜닝은 아니다. 그러나 독자가 사후에 고르는 것을 막을 수 없다.
- **최소 수정안**: 보고서 본문에는 잠금 모델만 싣는다.

### E-P2-04 · 패키지에 원 작업환경 흔적과 제3자 바이너리가 있다

- **실제 증거**
  - `/home/lim/...` 경로 50회(9개 파일)
  - `manifests/session_record.json`의 `session_id`
  - `tools/`의 Tectonic 바이너리(라이선스 파일 동봉), `paper/fonts`, `icml2026_official.zip`
  - 패키지 크기 131 MB
- **최소 수정안**: KAMP zip에서는 `tools/`, `agent/`, 원격 스크립트, 세션 기록을 제외한다.

### E-P3-01 · "주 운영모델과 F1 대안은 같은 PCA에서 임계값만 다르다"는 사실이다

- **판정**: 두 예측파일의 점수 열이 동일하다(`np.allclose`). 대안은 "4개 목표 FPR 중 selection F1 최대"로 고른 0.005다. 목표 FPR 제약(주)과 F1 최적(대안)이 섞이지 않았다.

---

## 5. Plan F 감사 (버스트·단일행 보완)

**주장 요약** (`runs/experiment_20261003_01/summary.json`)
- 최종모델: M3-L20 Isolation Forest(burst 미고려), window 부족 행은 M2 단일행 IF로 보완, 3 seed 점수 결합, 최근 3회 중 2회 경보
- 최종: **TP 303, FN 54, FP 0, TN 3,990**, Recall 0.8487, F1 0.9182, F2 0.8752, AP 0.9586

**재계산·재현**
- 예측파일 `predictions/main_test.csv`로 계산한 값이 위 수치와 **정확히 일치**한다.
- `src/verify.py`의 모든 검사가 PASS했다(저장 confusion = 예측파일, 단독 추론 = 최종 예측, 잠금 해시 일치).
- 전체 재실행 결과는 §9.

### F-P1-01 · 선택 모델 M3-L20도 burst를 고려하지 않아 window가 공백을 가로지른다

- **심각도**: P1
- **관련 문항**: 1·2
- **예상 감점**: −2 ~ −4
- **공격 대상 주장**: 논문 제목 "관측 공백과 평가 범위를 분리한 …"
- **실제 증거**
  - `src/experiment.py:54`: M1·M4만 `burst_id`로 묶는다. M3는 burst를 무시한다.
  - 최종 test에서 gap 횡단 window: 정상 1,983/3,971, 이상 185/338. E와 같은 행·같은 비율이다.
  - 논문 스스로 "통계 IF 길이20에서 버스트 고려 시 Recall 93.46%→84.97%, F1 0.9565→0.8997"(`paper/논문_한국어.txt:252`)라고 적는다. 정직한 보고다. 다만 최종모델이 "공백을 무시한 쪽"이라는 점은 요약에서 잘 드러나지 않는다.
- **최소 수정안**: 요약과 결론에 "최종모델은 burst 경계를 넘는 window를 허용함"을 명시한다.

### F-P1-02 · FP=0은 2-of-3 후처리의 결과이고, 후처리로 Recall이 떨어졌다

- **심각도**: P1
- **관련 문항**: 2·3
- **예상 감점**: −2 ~ −3
- **실제 증거**
  - 후처리 전: TP 315, FN 42, FP 1 → F1 **0.9361**
  - 후처리 후: TP 303, FN 54, FP 0 → F1 0.9182(`summary.json` `postprocessing_before/after`)
  - TP 12행을 잃고 FP 1행을 지웠다. 후처리는 selection에서 정했으므로 재튜닝은 아니다.
  - 정상 test 관측은 3,990행 × 0.1 s = **0.111 h**(벽시계 15분 51초, 연속된 한 구간 행 16011–20000)다. FP 0건의 Poisson 95% 상한은 약 **27건/h**다.
- **심사위원 질문**: "16분 정상 데이터에서 0건을 '오경보 0'이라고 할 수 있습니까?"
- **최소 수정안**: "FP 0"에는 반드시 관측시간과 상한을 붙이고, 후처리 전·후를 함께 싣는다(논문은 이미 병기함).

### F-P1-03 · native / common / operational 세 평가 범위의 분모가 다르다

- **심각도**: P1 (발표에서 혼동 위험)
- **실제 증거**: 같은 M3_L20, seed 42, 목표 FPR 0.005 기준(`tables/metrics_test.csv` 528·532·536행)

  | 범위 | N | TP | FN | FP | F1 |
  |---|---:|---:|---:|---:|---:|
  | native (window 가능 행만) | 4,309 | 316 | 22 | 4 | 0.960 |
  | common (모든 모델 교집합) | 2,141 | 142 | 11 | 2 | 0.956 |
  | operational (M2 보완, 전체행) | 4,347 | 318 | 39 | 4 | 0.937 |

  - 최종 간판 0.918은 3 seed 결합 + 2-of-3 + operational 기준이다.
  - FN 54 중 18행은 M2 보완 행(window 부족)에서 나왔다.
- **최소 수정안**: 보고서 성능표마다 범위·N·후처리를 열로 표기하고, 모델 순위는 common 기준만 쓴다(README가 이미 이 원칙을 밝힘).

### F-P1-04 · 이상 라벨이 선택과 sigmoid에 쓰였고, test는 같은 사건의 후반부다

- **심각도**: P1 (E-P1-02와 같은 구조)
- **실제 증거**: 분할(`src/prepare.py:55-58`)은 정상 60/20/20, 이상은 개발 40% / 최종 60%다. 개발 이상은 보정 132행과 선택 111행이다(E와 같은 행). README가 "개발 이상 라벨은 선택·sigmoid에 사용"이라고 공개한다(양호). 단일 사건이라는 한계는 E와 같다.

### F-P1-05 · "결과 확인 전 고정"의 선택 규칙과 실제 잠금 내용이 미묘하게 다르다

- **심각도**: P1
- **실제 증거**
  - 프로토콜의 주 규칙은 "selection FPR ≤ 0.01, then Recall desc"(`configs/protocol.json` `selection.primary`)다.
  - 잠금된 main은 `target_fpr 0.005` 임계값이고, selection FPR은 0.0081이다(`selection_lock.json`).
  - main과 `f1_alternative`가 완전히 같은 설정이다(두 블록 동일). 대안 규칙이 별도 후보를 만들지 않았다.
- **최소 수정안**: 잠금이 어떤 규칙으로 0.005를 골랐는지 보고서에 한 줄로 설명하고, main = 대안임을 명시한다.

### F-P2-01 · LSTM 비교는 20 epoch 제한이다

- **실제 증거**: 기본 LSTM 18개 중 16개가 20 epoch 상한에 도달했다(`논문_한국어.txt:152`). 가이드북은 800 epoch·patience 120이다. 논문은 "가이드북 정확 재현이 아님"을 반복해서 밝힌다(`:152`, `:504`). 그러나 "공식 baseline"처럼 읽힐 위험이 있다.
- **최소 수정안**: 보고서 비교표에서 LSTM 행에 "20 epoch 제한, 가이드북 재현 아님"을 붙인다. 가이드북 재현 수치는 Plan D G0(F1 0.743, 800 epoch, 3 seed)에서 인용할 수 있다. 같은 데이터·다른 프로토콜임을 명시한다.

### F-P2-02 · 저장소 사본에서 패키지 검증이 실패한다

- **실제 증거**
  - `python scripts/verify_package.py` → `AssertionError: Changed file: configs/frozen/bursts.csv`
  - CSV 33개가 CRLF→LF로 바뀌었다(복원 시 33/33 일치). `runs/…/driver.log` 1개가 누락됐다.
  - `src/verify.py`는 `verification.json`을 덮어쓴다(README 경고 있음).
- **최소 수정안**: zip을 원본 바이트로 만들고 manifest를 재생성한다.

### F-P2-03 · 고정 lock 파일을 제한망에서 설치할 수 없다

- **실제 증거**: `requirements.lock.txt`의 `torch==2.11.0+cpu`는 PyTorch 전용 인덱스가 필요하다. 감사 환경에서는 해당 인덱스가 프록시에서 403으로 막혔다. PyPI의 `torch==2.11.0`(CUDA 빌드)으로 대체해 검증을 통과했다. 심사환경의 네트워크 정책은 **미검증**이다.
- **최소 수정안**: IF가 최종모델이므로 torch는 선택 의존성으로 분리한다.

### F-P2-04 · 현장 활용이 경보 후처리에 그친다

- **실제 증거**: `report.html`에 "현장활용 10" 절이 있으나 내용은 파일럿 필요성 서술이다. 경보 단계별 작업자 조치·권한·센서 결측 대응이 없다.

### F-P2-05 · 원 작업환경 흔적

- **실제 증거**: `/home/lim` 15회, Colab 세션명 `hydraulic-exp-20261003` 5회, `audit/transfer_log.json`·`upload_manifest.json`이 있다. 가이드북 원본 PDF는 워터마크 때문에 제외했다(양호, `RIGHTS_AND_SOURCES.txt`).

---

## 6. Plan G 감사 (Logistic Regression 분석 묶음)

**주장 요약** (`paper_icml_logistic/main.tex:21`)
- 대상: 575 정상 / 20 이상 burst, 5개 특징, 5-fold × 10회 반복 CV
- 결과: 가중치 None의 F1 0.942 vs balanced 0.776. 놓친 burst 2개(619·620)는 고정적이다.
- 성격: 예측 시스템이 아니라 **class-weight 진단 사례연구**다.

**재현**
- `analyze_results.py`는 저장 결과로 표를 재생성한다(차이 ≤ 5.3e-15). jinja2를 추가 설치해야 했다.
- `logistic_class_weight_ablation.py` 재학습 결과 `metrics.csv`가 오차 0으로 일치한다.
- `audit_paper.py`는 `C:/Users/EKR/.../pdftoppm.exe` 하드코딩 경로 때문에 실패한다.

### G-P0-01 · 공식 제출물(테스트 예측결과·배포 가능한 추론·보고서 PDF)이 없다

- **심각도**: **P0**
- **관련 문항**: 2·4·6 + 제출요건
- **예상 감점**: 제출 불가
- **실제 증거**
  - 예측파일은 50-fold OOF 진단 예측뿐이다(`results/logistic_class_weight_ablation/predictions.csv`, burst × fold). "테스트데이터 예측결과 파일"에 해당하는 독립 평가 예측이 없다.
  - 논문 `main.pdf`는 **미컴파일**이다(`paper_icml_logistic/README.md:5`). `references.bib`은 TODO 주석뿐이다(`references.bib:1-4`). 공식 ICML 스타일도 없다.
  - 새 CSV에 대한 추론 진입점이 없다.
- **최소 수정안**: G를 본체로 쓰지 않는다. FN 분석 그림만 참고 자료로 흡수한다(평가 보고서 §7).

### G-P1-01 · 지도학습 LR은 날짜·세션 분류기일 수 있다

- **심각도**: P1
- **관련 문항**: 2·3
- **예상 감점**: −4 ~ −6
- **실제 증거**
  - 정상 1일·이상 1일이다(`main.tex:41`). 특징의 Cohen's d는 AI0 RMS 7.415, AI0 MaxAbs 7.785다(`main.tex:142`).
  - 같은 고장 기록의 burst 20개가 fold마다 학습 16개 / 평가 4개로 나뉜다. 평가 burst의 "형제"가 학습에 들어가는 **사건 내부 누수**다.
  - 논문은 이를 한계로 밝힌다(`main.tex:201-203`).
- **심사위원 질문**: "날짜가 하나씩뿐인데, LR이 학습한 것이 고장입니까, 7월 17일입니까?"

### G-P1-02 · burst 분할에 라벨을 쓴다

- **심각도**: P1
- **실제 증거**: notebook cell 7은 `gap > 1` 또는 `Equipment_state` 변화 시 새 burst를 만든다(`reference/소성가공.ipynb`). 추론 시에는 라벨이 없으므로 이 분할을 그대로 배포할 수 없다. 이 데이터에서는 파일 간 날짜 gap 때문에 경계가 사실상 같다. gap 기준은 1 s로 다른 Plan(0.5 s)과 다르다. 논문은 이를 한계로 밝힌다(`main.tex:205`).

### G-P1-03 · 반복 CV 50 fold는 독립 표본이 아니다

- **심각도**: P1
- **실제 증거**: burst마다 OOF 10회씩 등장한다. fold당 이상은 4 burst이므로 fold별 Recall은 {0, .25, .5, .75, 1} 다섯 값뿐이다. 이상 20 burst는 2분 46초짜리 고장 1건의 조각이다. 논문이 "do not become 200 independent abnormal examples"(`main.tex:196`)라고 정확히 쓴 점은 양호하다.

### G-P1-04 · 특징 5개와 가중치 선택이 같은 CV 모집단에서 이루어졌다

- **심각도**: P1
- **실제 증거**: notebook cell 38이 같은 50 fold 결과를 보고 `AI0_Kurtosis`를 뺐다. class weight도 같은 fold 평균으로 골랐다(`logistic_class_weight_ablation.py:26-31`). nested CV나 독립 test가 없다. 논문이 공개한다(`main.tex:203`).

### G-P1-05 · 현장 활용·확률보정·조기탐지가 없다

- **심각도**: P1
- **예상 감점**: 문항 4에서 −8 이상
- **실제 증거**: 경보 정책, 작업자 조치, 확률 보정 평가가 없다. LR `predict_proba`는 지도 확률이지만 날짜 교란 상태다.

### G-P2-01 · 고정 80/20 test 개선과 반복 CV 악화가 충돌한다

- **실제 증거**: 전류 ACF 추가 실험 결과
  - 고정 test(119 burst, 이상 4): F1 **0.889 → 1.000**(FP 1 → 0, `results/logistic_current_acf_ablation/metrics.csv`)
  - 반복 CV 평균: F1 **0.776 → 0.763**으로 하락
  - Pearson 버전도 0.776 → 0.767로 하락
  - 보고서는 CV를 주 결과, 고정 test를 부록으로 둔다(`analysis_report.md`). 이 처리는 올바르다. 발표에서 고정 test 수치를 인용하면 선택적 보고가 된다.

### G-P2-02 · baseline 보존 규칙은 지켰다 (위반 없음)

- **실제 증거**
  - `baseline_original.py`의 SHA-256이 `.sha256` 파일과 일치한다.
  - 각 ablation 스크립트가 "유일한 변경 요소" 주석과, 다른 파라미터가 같은지 확인하는 assert를 갖는다(예: `logistic_class_weight_ablation.py:79-81`).
  - LR 실험은 `random_state=42`로 seed를 고정했다. unseeded LSTM baseline과 직접 비교한 수치는 논문에 없다. AGENTS.md 위반은 **발견되지 않았다**.

### G-P2-03 · 환경 파일이 불완전하고 경로가 하드코딩되어 있다

- **실제 증거**: `requirements.txt`에 jinja2가 없다(pandas Styler 사용 시 ImportError). `audit_paper.py`가 `C:/Users/EKR/...` Poppler 경로를 고정했다. `paper_icml_logistic/README.md:12-13`의 실행 예시도 `C:/Users/EKR/miniconda3-pbe/...`다.

### G-P2-04 · FN 분석은 고장 사례 분석이 아니라 같은 기록의 조각 분석이다

- **실제 증거**: burst 619(10샘플)와 620(15샘플)은 짧은 burst다. crop 분석은 "길이 단독 원인 아님"으로 신중하게 결론 낸다(`main.tex:189`). 고장 유형 분석으로 확장하면 과장이 된다.

---

## 7. Plan 간 공통 위험

### COMMON-P0-01 · 공식 KAMP 결과보고서 PDF(6장 양식)가 네 Plan 모두 없다

- **심각도**: **P0**
- **관련 문항**: 전 문항
- **예상 감점**: 미제출이면 평가 불가
- **실제 증거**: 양식은 `docs/announcement/경진대회_결과보고서_양식_일반국민_대학생부문.md:1-110`(표지·서명·6장·설문 장, 휴먼명조). 각 Plan의 문서 상태:
  - D: 영어 ICML 형식 7쪽(`papers/plan_d/output/main.pdf`)
  - E: 한국어 ICML preprint 9쪽(`paper/manuscript_ko.pdf`)
  - F: 한국어 연구논문 17쪽 PDF/DOCX/HTML과 `report.html`
  - G: 미컴파일
  - 어느 것도 양식의 표지·6장 구조·설문 장을 갖추지 않았다. F의 `report.html`은 평가표 6항목에 대응하지만 **HTML**이다(공지: "PDF 파일로 제출").
- **최소 수정안**: 선정 본체(평가 보고서 §7)로 한국어 6장 보고서를 써서 PDF로 낸다.
- **완료 판정**: 양식과 같은 장 제목, 표지, 설문 캡처 장이 있는 PDF

### COMMON-P0-02 · 발표자료(PPT + PDF)가 없다

- **심각도**: P0
- **실제 증거**: D–G 어디에도 `.pptx`가 없다. 저장소의 유일한 PPT는 개인 메모(`docs/KAMP 경진대회 - 내 생각.pptx`)다.

### COMMON-P0-03 · 설문 완료화면 캡처가 없다

- **심각도**: P0 (양식 "필수")

### COMMON-P0-04 · 공식 구성의 소스코드 zip이 없다

- **심각도**: P0
- **실제 증거**: 네 패키지 모두 내부에 원본 CSV를 갖고 있다(D `data/raw/`, E·F `data/`, G `data/raw/`). 그러나 KAMP 구성 zip은 만들어지지 않았다. E·F는 연구 부속물(에이전트 지시문, Colab 기록, 바이너리, 수백 개 중간 산출물)을 포함해 131–210 MB다. G에는 예측결과 파일이 없다.
- **최소 수정안**: 화이트리스트 zip(코드·환경·`data/raw`·README·최종 예측 CSV)을 만들고, 빈 폴더에서 압축을 풀어 실행해 본다.

### COMMON-P1-01 · 날짜·세션 교란 때문에 "고장 탐지" 해석이 식별되지 않는다

- **심각도**: P1 (네 Plan 공통)
- **실제 증거**
  - 정상 07-12와 이상 07-17이 완전히 겹치지 않는다.
  - D 1단과 E·F 특징은 AI2 전류를 쓴다. 전류 0.4–0.9 Hz 성분 비중은 정상 0.994–0.999, 이상 0.043–0.312로 겹치지 않는다(이전 감사에서 재계산).
  - E의 TP 행 중 주 기여 센서가 AI2인 행은 63/329다.
  - G는 진동만 쓰지만 지도학습이다.
- **최소 수정안**: 모든 Plan의 결론을 "관측된 상태 라벨의 구분"으로 한정한다. 정상 내부 운전변화 구간(행 16,000–19,000)의 경보율을 음성 대조군으로 보고한다.

### COMMON-P1-02 · "이상 확률" 요건의 충족 수준이 Plan마다 다르고, 모두 조건부다

- **심각도**: P1

  | Plan | 출력 | Brier (최종 예측) | 상수 예측 Brier | 판정 |
  |---|---|---:|---:|---|
  | D | `risk_score = 1 − p` (conformal) | 0.487 (holdout+fault) | 0.111 | 확률 아님 |
  | E | 정규화 점수 → sigmoid (보정 이상 132행) | 0.0081 (test) | 0.0754 | 평가조건부 보정 확률 |
  | F | 정규화 점수 → sigmoid (보정 이상 132행) | 0.0096 (test) | 0.0754 | 평가조건부 보정 확률 |
  | G | LR `predict_proba` (지도) | 미계산(OOF 진단) | — | 날짜 교란 지도 확률 |

  - 네 Plan 모두 "실제 고장 사후확률"은 **식별할 수 없다**. 독립 고장 사례도, 현장 사전확률도 없기 때문이다.
  - E·F 보정은 같은 사건의 앞부분으로 맞추고 뒷부분으로 평가한 것이다.

### COMMON-P1-03 · 블라인드 위험: 사용자 경로·세션 식별자

- **심각도**: P1 (학교명 아님. 다만 사용자명은 개인 식별에 가깝다)
- **실제 증거**
  - D `outputs/full_run.log`: `C:\Users\cmsch\anaconda3\...` 8회
  - E `/home/lim` 50회와 `session_id`
  - F `/home/lim` 15회와 Colab 세션명
  - G `C:\Users\EKR` 69회
  - 학교명 "경북대학교"는 D–G 밖의 `docs/KAMP 경진대회 - 내 생각.md:5`에만 있다. zip에 넣으면 **P0**이 된다.
- **최소 수정안**: zip 생성 후 `grep -rIiE "경북|대학교|univ|cmsch|/home/lim|EKR|C:\\\\Users|session_id"` 0건을 확인한다.

### COMMON-P1-04 · 단일 이상 사건, 짧은 정상 관측

- **심각도**: P1
- **실제 증거**
  - D: 5-fold의 이상 428 window 재사용, 블록당 정상 관측 약 0.1 h
  - E·F: test 정상 0.111 h에서 FP 1건과 0건 → Poisson 95% 상한 각각 약 43건/h, 27건/h
  - G: 이상 20 burst
  - "FP 0", "Recall 0.85", "303행 탐지"는 모두 **고장 1건의 2분 46초 기록** 내부 수치다.

### COMMON-P2-01 · 조기탐지·리드타임은 측정할 수 없다

- 네 Plan 모두 이를 인정한다: D `README.md:157`, E README, F `README.md:3`, G `main.tex:201`. 문제 제목의 "조기탐지"는 "탐지지연(관측 진입 후 첫 경보)"으로 바꿔 정의해야 한다.

---

## 8. 수치 재계산 결과

모든 값은 각 Plan의 최종 예측파일에서 직접 계산했다. 서로 다른 행은 **같은 평가조건이 아니므로 순위를 매기지 않는다.**

### 8.1 지표표

| Plan · 출력 | 평가 단위 | N | 유병률 | 임계값·후처리 | TP | FP | FN | TN | P | R | F1 | F2 | FPR | AP | ROC-AUC | Brier | 평가 데이터의 선택 사용 |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D 1단(노랑+빨강), holdout+fault | 10샘플 window | 3,357 | 0.128 | p ≤ 0.01 | 428 | 241 | 0 | 2,688 | 0.640 | 1.000 | 0.780 | 0.899 | 0.082 | 0.9998 | 1.000 | 0.487 | 이상: 게이트·동률에 사용 |
| D 빨강, holdout+fault | 〃 | 3,357 | 0.128 | p₁, p₂ ≤ 0.01 + 연속 3 | 375 | 3 | 53 | 2,926 | 0.992 | 0.876 | 0.931 | 0.897 | 0.0010 | — | — | — | 〃 |
| D M3 5-fold 평균(간판) | window, fold 평균 | 5×~3,350 | ~0.12 | p ≤ 0.01 | — | — | — | — | 0.928 | 1.000 | 0.962 | — | 0.0116 | 0.9988 | 0.99986 | — | 〃 |
| E 주(P1_W20_K2_Q) | 행(endpoint, fallback 포함) | 4,347 | 0.082 | 보정 FPR 0.01, 후처리 없음 | 329 | 1 | 28 | 3,989 | 0.997 | 0.922 | 0.958 | 0.936 | 0.00025 | 0.9871 | 0.9962 | 0.0081 | 이상: 선택 111행·보정 132행 |
| E 대안(같은 모델) | 〃 | 4,347 | 0.082 | 보정 FPR 0.005 | 325 | 0 | 32 | 3,990 | 1.000 | 0.910 | 0.953 | 0.927 | 0 | 0.9871 | 0.9962 | — | 〃 |
| F 주(M3_L20 + M2 보완) | 행 | 4,347 | 0.082 | 보정 FPR 0.005 + 2-of-3 | 303 | 0 | 54 | 3,990 | 1.000 | 0.849 | 0.918 | 0.875 | 0 | 0.9586 | 0.9811 | 0.0096 | 〃 |
| F 주, 후처리 전 | 〃 | 4,347 | 0.082 | 보정 FPR 0.005 | 315 | 1 | 42 | 3,989 | 0.997 | 0.882 | 0.936 | 0.903 | 0.00025 | — | — | — | 〃 |
| G None, 50-fold 평균 | burst | 119/fold | 0.034 | p > 0.5 | 3.6/fold | 0 | 0.4/fold | 115 | 1.000 | 0.900 | 0.942 | — | 0 | — | 0.917 | — | 같은 fold로 특징·가중치 선택 |
| G balanced, 50-fold 평균 | burst | 119/fold | 0.034 | p > 0.5 | 3.6 | 1.88 | 0.4 | 113.1 | 0.725 | 0.900 | 0.776 | — | 0.016 | — | 0.948 | — | 〃 |

**비교 가능성 판정**
- **E와 F**: `row_id`와 라벨이 완전히 같은 4,347행이다. 행 단위 지표는 비교할 수 있다. 다만 E가 F의 최종평가 이후에 개발되었으므로(E-P1-03), 이 비교를 **선택 근거로 쓰면 안 된다.** 두 결과가 다른 행은 35행이다.
- **D와 E·F**: 단위(window vs 행), 분할(5-fold vs 고정 60/20/20), 임계값, 후처리가 모두 다르다. D holdout 블록(행 16051–19999)과 E·F 정상 test(행 16011–20000)가 거의 같은 시간대인 점만 공통이다. → **비교 불가**
- **G와 나머지**: 단위(burst), 라벨 기반 분할, 지도학습이다. → **비교 불가**

### 8.2 FAR/h 95% 상한 (정상 관측시간 = 행 × 0.1 s)

| Plan | 정상 관측 | 관측 경보 | FAR/h | 95% 상한 |
|---|---:|---:|---:|---:|
| D M3, CV 블록 0–4 합계 | 0.531 h | 경보 사건 82(23·29·22·3·5) | 154 /h | 블록별 상한 최대 362.7 /h (`e2_model_comparison.csv`) |
| D M3, CV 블록 4 | 0.105 h | 5 | 47.7 | 100.2 /h |
| E test | 0.111 h | FP 1행 | 9.0 | ≈ 42.8 /h |
| F test | 0.111 h | FP 0행 | 0 | ≈ 27.0 /h |
| G | 시간 정보 없음(burst) | — | — | 계산 불가 |

### 8.3 재계산 스크립트 (요지)

```python
import pandas as pd
from sklearn.metrics import f1_score, average_precision_score, brier_score_loss
# D
p = pd.read_csv('plan_d_submission/outputs/predictions.csv')
q = p[p.split.isin(['holdout', 'fault'])]
print(f1_score(q.label, q.alarm_level != 'green'), f1_score(q.label, q.alarm_level == 'red'),
      brier_score_loss(q.label, q.risk_score))                  # 0.780 0.931 0.487
# E
e = pd.read_csv('plan_e_submission/results/pca_20261003_cpu/predictions/operational_P1_W20_K2_Q_target0p01_test.csv')
print(f1_score(e.label, e.prediction), average_precision_score(e.label, e.score))   # 0.9578 0.9871
# F
f = pd.read_csv('plan_f_submission/runs/experiment_20261003_01/predictions/main_test.csv')
print(f1_score(f.label, f.prediction), f1_score(f.label, f.raw_prediction))          # 0.9182 0.9361
print((e.row_id.values == f.row_id.values).all())                                    # True
```

---

## 9. 재현 실행 결과

| Plan | 환경 | 명령 | 결과 | 소요 |
|---|---|---|---|---:|
| D | Py 3.11.15, numpy 2.4.6, pandas 2.3.3, sklearn 1.9.1 | `python run_all.py --mode quick` (기존 outputs 포함 복사본) | 표 21개·`predictions.csv` **완전 일치** | 56 s |
| D | 〃 | 같은 명령, `outputs/` 삭제 복사본 | 선정 M3 동일. BL-1 표 2개 미생성, 비교표 5→4행 | 38 s |
| D | 〃 | `pytest tests -q` | 13 passed | 3.4 s |
| D | — | `--mode full`(LSTM-AE 800 epoch) | **미검증**(원기록 13,050 s) | — |
| E | Py 3.13.14 + `requirements.lock.txt` | `checks.py` | PASS | — |
| E | 〃 | `verify_package.py` | **실패**(CRLF 해시 10개, 누락 12개). 해시 건너뛰면 추론 4,347행 일치 | — |
| E | 〃 | `verify_paper_numbers.py` | PASS(본문 표 7행 등) | — |
| E | 〃 | `experiment.py --stage develop` → `validate_development.py` → `--stage final` (새 폴더) | 선택 잠금 모델·최종 TP/FN/FP/TN **완전 일치** | 70 s |
| E | Windows | `.sh` 스크립트 | **미검증** | — |
| F | 표준 라이브러리 | `scripts/verify_package.py` | **실패**(CRLF 33개, `driver.log` 누락) | — |
| F | Py 3.13 + lock(단, torch는 PyPI cu130 대체) | `src/verify.py --out runs/experiment_20261003_01` (복사본) | 모든 검사 PASS | — |
| F | 〃 | `scripts/run_all.py` (새 run 폴더) | §9.1 참조 | §9.1 |
| G | Py 3.11 + requirements(+jinja2) | `analyze_results.py` | 표 재생성, 최대 차이 5.3e-15 | — |
| G | 〃 | `audit_paper.py` | **실패**(`C:/Users/EKR/...pdftoppm.exe`) | — |
| G | 〃 | `src/logistic_class_weight_ablation.py` (재학습) | `metrics.csv` 차이 0 | — |

### 9.1 Plan F 전체 재실행

- 환경: Python 3.13, `requirements.lock.txt`(torch만 PyPI `2.11.0+cu130`으로 대체), CPU
- 명령: `python scripts/run_all.py` → 새 폴더 `runs/experiment_20261004T084417_371210Z/`
- 소요: **7분 19초**. prepare 검증 → 학습(LSTM 20 epoch × 3 seed × 2 × 3 길이, IF) → 개발 → 잠금 → 최종 → 분석·보고서 → `verify.py` PASS
- 선택 잠금(`M3_L20`, 목표 FPR 0.005, `two_of_three`, selection TP99/FN12/FP16)이 **동일**하다.
- 최종(TP 303 / FN 54 / FP 0 / TN 3,990, F1 0.9182, AP 0.9586)이 **동일**하다. 예측 일치, 점수 최대 차이 4.9e-15.
- IF 계열 표는 모두 일치했다. **LSTM 288행은 TP/FP/FN이 최대 20행까지 달라졌다.** torch 빌드와 하드웨어가 달라 생긴 비결정성으로 보이며, 최종모델(IF)과 선택에는 영향이 없다. LSTM 비교 수치를 보고할 때는 "환경 의존 변동 있음"을 명시해야 한다.

---

## 10. 문서·코드·산출물 불일치

### 10.1 Plan별 핵심 충돌표

| Plan | 충돌 | 쪽 A | 쪽 B | 판정 |
|---|---|---|---|---|
| D | 실행 보드 BL-1 상태 vs 로그·CSV | `EXECUTION_BOARD.md:17-18` "진행 중", "잠정" (01:40) | `full_run.log` 05:13 BL-1 완료, `e2_bl1_g*.csv` 존재, `decision_log.md:15` 05:15 확정 | 보드가 낡음 |
| D | `outputs/` vs `outputs_full_backup/` | `outputs/` = quick 17:26 + full BL-1 CSV | `outputs_full_backup/` 저장소에 없음 | SSOT = `outputs/`, 백업은 미검증 |
| D | 로그 vs 현재 표 | `full_run.log` "final_alarm_summary.csv (1행)" | 현재 파일 4행 | 로그가 구버전 |
| D | 논문 PDF 2종 | `papers/plan_d/main.pdf` | `papers/plan_d/output/main.pdf` | 내용 다름 |
| D | 논문 설명 출처 | `main.tex:604-606` 2단 M1 기여도 | `run_all.py:496-501` 1단 M3 잔차 | 코드 우선 |
| E | 주 임계값 vs F1 대안 | 1.3079 (FPR 0.01) → TP329/FP1 | 1.4937 (FPR 0.005) → TP325/FP0 | 같은 점수, 임계값만 다름(정상) |
| E | IF W5 vs 동일입력 IF W20 | IF 대표 = W5, test Recall 0.485 | IF W20, test Recall 0.896 / FP 6 | 대표 IF가 약함, 논문은 병기 |
| E | 이상 라벨 사용 vs 비지도 표현 | 검출기 정상만 적합 | 선택 111행·보정 132행 이상 사용 | "비지도" 단독 표현 금지 |
| E·F | 무결성 manifest vs 저장소 사본 | `MANIFEST_SHA256.json`, `PACKAGE_SHA256.json` | CRLF→LF, 로그 누락 | 저장소 사본으로는 검증 실패 |
| F | native / common / operational | N 4,309 / 2,141 / 4,347 | F1 0.960 / 0.956 / 0.937 (seed 42, FPR 0.005, 후처리 전) | 분모 표기 필수 |
| F | M3 window 모델 vs M2 보완 | 4,309행 M3 | 38행 M2, FN 54 중 18 | 혼합 결과임을 명시 |
| F | LSTM 20 epoch vs 공식 baseline | `protocol.json` `max_epochs: 20` | 가이드북 800 epoch | "가이드북 재현 아님" 필수 |
| G | 고정 test 개선 vs CV 악화 | ACF 추가 F1 0.889→1.000 | CV 0.776→0.763 | CV 우선 |
| G | seeded ablation vs unseeded baseline | LR `random_state=42` | LSTM `baseline_original.py` 무시드 | 직접 비교 수치 없음(위반 없음) |
| G | 반복 fold 수 vs 독립 사건 수 | 50 fold, burst당 10회 | 이상 사건 1건(20 burst) | 독립 표본 아님 |

### 10.2 D–G 최종모델 표기 대조

| Plan | README | 결과 JSON/CSV | 예측파일 | 논문 | 일치 |
|---|---|---|---|---|---|
| D | 1단 M3 + 2단 M1(진동) | `e2_selection_gates.csv` M3 `passed`, `run_manifest.json` `chosen_stage1_model: M3` | 1단 M3·2단 M1 열 | M3 + vibration-only M1 | 모델은 일치. 설명 출처는 불일치(D-P1-07) |
| E | "주 모델과 F1 대안은 같은 PCA" | `selection_lock.json` `P1_W20_K2_Q` | 동일 키 | P1 W20 | 일치 |
| F | M3-L20 IF + M2 보완 + 3 seed + 2-of-3 | `selection_lock.json` `M3_L20`, `two_of_three` | `actual_model` M3_L20 / M2_L1 | 동일 | 일치 |
| G | 최종모델 개념 없음(진단) | `best_weight.json` None 선택 | OOF만 | "no new … selected" | 제출용 최종모델 없음 |

---

## 11. 제출 금지 주장

| # | 금지 주장 | 해당 | 근거 ID |
|---|---|---|---|
| X1 | "오경보 0건" / "현장 무오경보" (관측시간·상한 없이) | E 대안, F | F-P1-02, COMMON-P1-04 |
| X2 | "고장확률", "이상 발생 확률" (`1−p` 또는 날짜 교란 확률을 그렇게 부름) | D `risk_score`, G | D-P1-06, COMMON-P1-02 |
| X3 | "결과를 보기 전에 모든 선정 규칙을 동결했다" | D | D-P1-01 |
| X4 | "이상 데이터를 어떤 선택에도 쓰지 않았다" / "완전 비지도" | D, E, F | D-P1-02, E-P1-02 |
| X5 | "357개(또는 428개, 20개) 고장 사례를 탐지했다" | E, F, D, G | COMMON-P1-04 |
| X6 | "고장 N초 전에 예측" / "리드타임 확보" | 전체 | COMMON-P2-01 |
| X7 | "베어링 유격이 원인", "전류센서 결함이 원인" | D 권고문 | D-P1-07 |
| X8 | "E(0.958)가 F(0.918)보다 우수해서 선택" | E·F | E-P1-03 |
| X9 | "D의 F1 0.962가 E·F의 0.958/0.918보다 높다"(또는 그 반대) | D–F | §8.1 비교 불가 |
| X10 | "시간 공백을 고려한 최종모델" | E, F 주모델 | E-P1-01, F-P1-01 |
| X11 | "정상·이상이 완전분리되므로 모델이 우수하다" | D, G | D-P1-03, G-P1-01 |
| X12 | "ACF 추가로 F1 1.0 달성" (고정 test) | G | G-P2-01 |
| X13 | "가이드북 LSTM-AE를 공식 베이스라인으로 동일 조건 재현" (20 epoch 결과에 대해) | F | F-P2-01 |
| X14 | ICML·HTML 원고를 결과보고서로 제출 | 전체 | COMMON-P0-01 |
| X15 | 서로 다른 Plan의 최고 수치를 한 모델의 결과처럼 표기 | 전체 | 원칙 |

## 12. 제한적으로 사용 가능한 주장

| # | 주장 | 반드시 붙일 조건 |
|---|---|---|
| L1 | D: 계측조건 섭동 시 오경보 증가 M3 +17.0 pp vs M1 +89.7 pp | 전 채널 동시 섭동, 5 fold, 게이트 한계는 사후 재사용 |
| L2 | D: 가이드북 LSTM-AE 재현 F1 0.743(문헌 0.748), 같은 네트워크를 Plan D 프로토콜에 넣으면 0.959 | G0는 3 seed·800 epoch, G1은 1 seed, window 단위·유병률이 다름 |
| L3 | D 빨강 기준 holdout: Recall 0.876, FP 3 window, F1 0.931 | 블록 4 단일 holdout(0.105 h), 고장 1건, 첫 2 window는 구조적 노랑 |
| L4 | E: 같은 4,347행에서 TP 329 / FP 1 / FN 28 | 이상은 같은 사건의 뒷부분, 선택·보정에 같은 사건 사용, gap 횡단 window 50% |
| L5 | E·F: sigmoid 보정 Brier 0.0081 / 0.0096 | 같은 사건으로 보정, 유병률 8.2% 조건 |
| L6 | F: 버스트 고려가 항상 성능을 높이지는 않는다(IF L20 Recall 93.5→85.0%) | 단일 날짜·단일 사건, 개발에서는 반대 방향 |
| L7 | F: 버스트 첫 9행의 이상 미탐률 36.7% vs 이후 1.2% | 개발 split, 이상 30·81행 |
| L8 | G: balanced 가중치가 Recall은 그대로 두고 정상 FP를 94회(16 burst) 늘림 | 같은 50 fold 내부, 독립 test 없음 |

## 13. 안전하게 사용 가능한 주장

| # | 주장 | 근거 |
|---|---|---|
| S1 | 이상 데이터는 2022-07-17 2분 46초의 단일 고장 기록이고, 정상과 날짜가 다르다 | 원본 CSV |
| S2 | 데이터는 burst 수집이다(정상 599 / 이상 21 burst, 최대 50샘플) | D `e0_burst_summary.csv`, 재계산 |
| S3 | 가이드북의 seq20+offset100(120샘플)이 들어가는 burst는 0개다 | D `e0_burst_summary.csv` `bursts_ge_120_samples=0` |
| S4 | gap-aware windowing으로 정상 window의 25.6%(seq 10) / 50.1%(seq 20)가 제거된다 | D `e1_window_sensitivity.csv` |
| S5 | 리드타임은 이 데이터로 측정할 수 없다 | 이상 파일 전 행 state=1 |
| S6 | `abs()` 전처리는 RMS·MAV·PEAK를 바꾸지 않고 STD·MEAN·P2P·KURT를 바꾼다 | D `e9_abs_invariance.csv` |
| S7 | D·E·F·G의 핵심 수치는 각 패키지의 저장 결과에서 재계산·재현된다 | §8·§9 |
| S8 | 정상 라벨 내부의 행 16,000–19,000 구간에서 여러 모델의 경보가 몰린다 | D holdout 비녹색 217/241, Plan A `fp_breakdown.csv` |

---

## 14. P0/P1/P2/P3 수정 백로그

| 우선 | ID | 대상 | 최소 수정 | 시간 | 완료 판정 |
|---|---|---|---|---:|---|
| P0 | COMMON-P0-01 | 본체 | 한국어 6장 결과보고서 PDF | 12–16 h | 양식 장 구조 일치 |
| P0 | COMMON-P0-04 | 본체 | 화이트리스트 zip + 빈 폴더 실행 | 1.5 h | 단일 명령 성공 |
| P0 | COMMON-P0-02 | 본체 | 발표자료 PPT + PDF | 6–8 h | 두 파일 |
| P0 | COMMON-P0-03 | — | 설문 캡처 | 0.2 h | 이미지 |
| P0 | G-P0-01 | G | 본체로 쓰지 않음(흡수만) | 0 h | — |
| P1 | COMMON-P1-03 | 본체 | 경로·세션·학교명 grep 0건 | 0.5 h | grep 0 |
| P1 | D-P1-06 · COMMON-P1-02 | D | `posterior_sensitivity` 호출 + 교차적합 Platt + 신뢰도 곡선 | 3 h | Brier < 상수 |
| P1 | D-P1-01 · D-P1-02 | D | 게이트 변경 전후 표, 문구 정정 | 1 h | 문구 0건 |
| P1 | D-P1-03 · D-P1-04 · D-P1-05 | D | 빨강 기준 간판, holdout 구간별 경보율 표, 예측파일 지표표 | 2.5 h | 표 3개 |
| P1 | D-P1-07 | D | 권고문 "점검 후보" 정정, 논문 설명 출처 정정 | 0.5 h | 문구 |
| P1 | E-P1-01–05, F-P1-01–05 | E·F | 흡수 시 해당 그림·표에 조건 표기 | 1 h | 표기 |
| P1 | G-P1-01–05 | G | 흡수하지 않음(FN 사례 그림만 참고) | 0 h | — |
| P2 | D-P2-01–06 | D | README BL-1 의존 명시, SSOT 선언, 구 로그·중복 PDF·보드 제외, 빨강 FN 표 | 2 h | zip 검사 |
| P2 | E-P2-01–04, F-P2-01–05, G-P2-01–04 | E·F·G | 흡수 범위에 한해 조건 표기 | 0.5 h | — |
| P3 | 용어 | 전체 | "확률" / "정상성 p" / "위험순위" 용어 통일 | 0.5 h | 용어표 |

**집계**

| 구분 | P0 | P1 | P2 | P3 |
|---|---:|---:|---:|---:|
| Plan D | 0 | 7 | 6 | 0 |
| Plan E | 0 | 5 | 4 | 1 |
| Plan F | 0 | 5 | 5 | 0 |
| Plan G | 1 | 5 | 4 | 0 |
| 공통 | 4 | 4 | 1 | 0 |
