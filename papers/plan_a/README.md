# Paper — ICML 형식

`main.tex` → `output/main.pdf` (5쪽, 2단, letter)

## 빌드

```bash
cd papers/plan_a
pdflatex -output-directory=output main.tex
pdflatex -output-directory=output main.tex    # 2회 (상호참조 해결)
```

MiKTeX / TeX Live 어디서든 컴파일된다. **공식 ICML 스타일 파일 불필요** — `article` 클래스로 ICML 레이아웃(2단, 10pt, Times, 익명 제출)을 재현했다.

## 공식 ICML 스타일로 바꾸려면

[ICML 저자 키트](https://icml.cc/Conferences/2025/AuthorInstructions)에서 `icml2025.sty`, `icml2025.bst`, `fancyhdr.sty`를 받아 같은 폴더에 두고 전처리부를 교체한다.

```latex
\documentclass{article}
\usepackage{microtype,graphicx,subfigure,booktabs,hyperref,amsmath,amssymb}
\usepackage[accepted]{icml2025}   % 제출 시에는 [accepted] 제거(익명)
\icmltitlerunning{Evaluation Pitfalls in Single-Event Industrial Anomaly Detection}
```

그리고 `\twocolumn[...]` 제목 블록을 `\twocolumn[\icmltitle{...}\icmlsetsymbol{...}...]` 로 바꾼다.

> **주의**: 지금 구조에서 `\twocolumn[...]` 의 optional argument 안에 **맨 대괄호 `]` 를 넣으면 인자가 조기 종료된다.** 초록의 신뢰구간을 `[0.852, 0.984]` 가 아니라 `0.852--0.984` 로 쓴 이유다. 수정 시 주의할 것.

## 내용 ↔ 근거 매핑

논문의 모든 수치는 저장소 실행 결과에서 나온 것이다.

| 논문 위치 | 수치 | 출처 |
|---|---|---|
| Abstract, §2.2, Table 1 | 버스트 구조, windowing 잔존율 49.9 %, $F_1$ 0.857→0.818 | `previous research/공격4 - 실험/verify_attack4.py --only V1 V2` |
| §2.3 | 1.8 / 3.6 Hz 선 스펙트럼, Ljung–Box 275/276 | 〃 `--only V5` |
| §3.1 | FAR/h 분해능, 탐지지연 8.78 s 구조적 하한 | `plan_a_submission/src/run_all.py` (측정 한계 블록) |
| §4.1 | 캐리어 파워비 0.959 vs 0.104, 소수 자릿수 지문 | `공격4 - 실험 --only V7`, `_agent_raw/phys*.py` |
| §6, Table 2 | D1 통과 리더보드 | `plan_a_submission/outputs/model_comparison.csv` |
| §6.2 | 페어드 부트스트랩 Δ, P(Δ≤0) | `plan_a_submission/outputs/run_log.txt` |
| §6.3, Table 3 | D1 민감도 | `plan_a_submission/outputs/model_comparison.csv` (d1_ok=False) |
| §6.4, Table 4 | 오경보 집중 91.1 % | `plan_a_submission/outputs/fp_breakdown.csv` |

## 대회 제출물과의 관계

이 논문은 **대회 제출물이 아니다.** 결과보고서는 별도 한글 양식(`docs/announcement/경진대회_결과보고서_양식_*.md`)을 따라야 한다.

다만 다음 용도로 쓸 수 있다.

- 보고서 각 장의 논증 구조·문장 초안 (특히 제1장 데이터 진단, 제3장 오류분석)
- 발표자료의 논리 흐름
- 블라인드 규정상 **저자는 익명**으로 두었으므로 그대로 공유해도 식별 정보가 없다

## 한글 버전이 필요하면

본문을 한국어로 옮기려면 `kotex` 패키지가 필요하다.

```latex
\usepackage{kotex}          % pdflatex
% 또는 xelatex + \setmainhangulfont{맑은 고딕}
```

컴파일러를 `xelatex` 으로 바꾸고 `\usepackage{times}` 를 제거해야 한다.
