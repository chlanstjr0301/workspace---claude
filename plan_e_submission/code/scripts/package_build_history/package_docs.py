from pathlib import Path
import json,shutil
r=Path(Path('reports/current_paper_package.txt').read_text())
shutil.copy2('reports/package_verify.py',r/'code/scripts/verify_package.py')
(r/'README_KO.txt').write_text('''한국어 연구논문 · 코드 · 데이터 · 결과 통합 패키지

먼저 볼 파일
  paper/manuscript_ko.pdf       한국어 논문 PDF (ICML 2026 preprint 형식)
  paper/manuscript_ko.tex       편집 가능한 한국어 LaTeX 원고
  paper/references.bib          참고문헌
  results/pca_20261003_cpu/REPORT.html  상세 실험 보고서와 전체 파일 링크
  agent/EXPERIMENT_AGENT_PROMPT_KO.txt  실험 재현용 에이전트 명령문
  agent/PAPER_AGENT_PROMPT_KO.txt       논문 작성용 에이전트 명령문

실행 (압축을 푼 폴더에서 Linux/macOS의 bash 사용)
  bash VERIFY_PACKAGE.sh       파일 해시와 4,347행 저장 모델 추론 검사
  bash RUN_EXPERIMENT.sh       전용 환경에서 새 폴더에 전체 실험 재현
  bash BUILD_PAPER.sh          LaTeX에서 논문 PDF 다시 빌드

환경
  Python 3.13 또는 3.14 권장, 코드 의존성은 code/requirements.lock.txt에 고정.
  code/.venv만 생성/사용한다. 시스템 Python과 다른 프로젝트는 변경하지 않는다.
  uv가 있으면 사용하고 없으면 표준 venv/pip를 사용한다. 최초 설치는 인터넷 필요.
  논문에는 Linux x86_64용 Tectonic과 한국어 폰트를 포함한다. 다른 OS에서는
  Tectonic을 설치한 뒤 BUILD_PAPER.sh를 사용하거나 LaTeX 환경에서 빌드한다.
  Tectonic의 최초 TeX 번들 다운로드에도 인터넷이 필요할 수 있다.

구조
  paper/          한국어 PDF·LaTeX·BibTeX·벡터 그림·표 원본
  code/           실행 소스·검사·보고서·그림 생성·고정 설정·패키지 잠금
  data/           수정하지 않은 원본 CSV 두 개
  results/        모델·분할·전체 후보·개별 seed·행별 예측·오류·검증 기록
  agent/          복사하여 사용할 수 있는 한국어 에이전트 명령문
  documentation/ 데이터 카드·논문 준비 상태·문헌 확인·경로 수정 내역
  tools/          논문 빌드용 Tectonic (Linux x86_64)

연구 범위
  ICML 수준의 엄밀성을 목표로 작성한 사례연구 원고다. ICML 채택 가능성이나
  학술적 신규성이 이미 충족됐다는 보장이 아니다. 단일 정상/이상 날짜쌍이며
  독립 고장 일반화, 실제 고장 전 조기예측, 압력/품질/비가동 개선은 검증하지 않았다.
  한국어본은 연구용이며 실제 국제학회 투고본 준비·심사·공개 제출은 수행하지 않았다.
  저자/소속, 데이터의 공식 배포처 및 재배포 조건은 연구 책임자가 확인할 항목이다.

보존과 보안
  원본 CSV와 과거 모델 선택은 그대로 보존했다. 실행용 코드의 데이터 경로만
  패키지 상대경로로 바꿨으며 원래 코드도 results 내 source/source_final에 있다.
  인증 토큰, OAuth 파일, Colab 세션 설정, .venv는 압축에 포함하지 않는다.
  원격 스크립트는 과거 실행 기록으로 포함하며 자동 실행하지 않는다.
  기존 결과 파일에서 원래 작업 환경의 경로·세션 ID가 남아 있을 수 있으므로
  이 사용자용 전체 패키지를 학회 익명 부록으로 그대로 제출한 것으로 간주하지 않는다.

기존 결과의 주 모델과 F1 대안은 같은 PCA에서 임계값만 다르다.
오류 분석은 파일명에 target0p01/target0p005가 있는 확정 파일을 사용한다.
기존 진단 파일은 오류 복구 이력 보존을 위해 함께 포함했다.

명령문은 사용자의 실험 요구를 재구성한 실행용 지시문이며 채팅 원문의 전사는 아니다.
''')
(r/'documentation/DATA_CARD_KO.txt').write_text('''데이터 카드
사용자 제공 명칭: 소성가공 예지보전 AI 데이터셋.
파일: data/press_data_normal.csv (20000행), data/outlier_data.csv (600행).
공식 데이터셋 URL/버전/라이선스/설비 ID/동의 및 공개배포 범위: 제공 자료로 확정되지 않음.
이 패키지는 동일 사용자에게 연구 결과를 전달하기 위해 제공된 CSV를 포함한다.
별도의 공개 배포 허가나 새로운 라이선스를 부여하는 문서가 아니다.

열: 빈 이름의 첫 인덱스, TimeStamp, AI0_Vibration, AI1_Vibration, AI2_Current, Equipment_state.
단위와 AI0/AI1의 상부/하부 대응은 미확인. 인덱스/날짜/라벨은 모델 입력에서 제외.
정상 Equipment_state=0, 이상=1. 정상2022-07-12/이상2022-07-17로 날짜-라벨 교란 존재.
정상 추가 중복1행 제거(시각·센서·라벨 동일), 최초행 보존. 분석 정상19999행/이상600행.
결측·변환실패 없음. 대부분0.1초. 0.5초 초과공백 정상598/이상20, 최대16.643/8.572초.
0.5초 버스트 정상599/이상21개. 실제 기계 사이클 또는 독립 고장 사건 아님.
고장시작·정비·압력·품질·비가동·운전부하 이력은 없으며 이를 직접 예측하지 않음.

정상SHA256 f2d61cb3b3108f49a3305d2966b3a31f26942ac8af8ffc9eb531b432b945b0a7
이상SHA256 9fad8c238e63dba3257d8be32a32362548955c772171a95c0d2af61b0435ed2e
분할SHA256 9372ae29ed7d3104a0d88c20b961a2aef034ff7b96b5735aac5f2c0c50de16c2

이미 확인한 최종 자료는 새로운 연구의 미관측 holdout으로 간주하면 안 된다.
후속 방법 선택·추가 검증을 하려면 별도 날짜/설비의 새로운 평가 자료가 필요하다.
''')
(r/'documentation/RESEARCH_READINESS_KO.txt').write_text('''논문 준비 상태와 한계
완료: 한국어 연구논문, 공식 ICML 2026 preprint 스타일, 실제 수치/근거, 수식과 평가 절차,
      동일 입력 IF 비교, 부정적 결과, 코드/데이터/모델/명령문, 저장모델 재현 검사.
완료하지 않은 것: ICML 제출 또는 채택, 공식 형식검사 서버 업로드, 외부 데이터셋 검증,
      여러 설비/날짜/고장사건 검증, 실제 고장전 예측, 실제 현장 개입의 효용 검증.
핵심 연구 기여: 기존 PCA의 새로운 발명이 아니라 관측 범위와 공백 처리를 분리한
      재현 가능한 비교 사례. 자료가 제한된 만큼 일반적인 SOTA 주장은 할 수 없다.
필요한 다음 자료: 날짜 안에서도 정상과 이상이 포함된 여러 날짜/설비, 실제 고장시작과
      정비시각, 부하/운전상태, 독립된 신규 holdout 및 사용/재배포 범위.
새 연구 계획: 새자료 확보 → 날짜/설비 단위 protocol 사전고정 → 동일 예산 baseline →
      최종 1회 평가 → 사건 수준 불확실성/외부검증. 현재 최종자료 재튜닝 금지.
작성자 확인 항목: 실제 저자와 소속, 이해관계, 데이터 provenance와 라이선스,
      AI 지원 내역, 논문 전체 내용에 대한 저자 책임 및 목표 venue의 최신 지침.
패키지 상태: 사용자의 전체 작업물. 역사적 경로·세션 메타데이터가 있어 익명 제출용
      부록과 같지 않다. 토큰/인증 파일은 제외한다. 공개 업로드는 하지 않았다.
''')
(r/'documentation/AI_ASSISTANCE_KO.txt').write_text('''AI 지원 내역
사용자는 모델 후보, 데이터 보존·분할·평가 기준과 작업 범위를 지정했다.
AI 에이전트는 전용 환경 준비, PCA/IF 구현·실행, 오류 복구, 개발 선택, 고정 최종평가,
결과 검사, 도표/보고서/논문 초안 작성 및 패키지화를 수행했다.
수치는 저장된 실행 결과에서 가져왔으며 수행하지 않은 외부검증을 추가하지 않았다.
학술 저자 자격·소속·윤리 및 데이터 이용 책임은 임의로 확정하지 않았다.
에이전트 지시는 agent/ 폴더에 실행용 재구성본으로 포함한다.
''')
refs=[
 {'title':'ICML 2026 Author Instructions','url':'https://icml.cc/Conferences/2026/AuthorInstructions','checked':'official instructions; 8-page submitted main-body rule and LaTeX styles; no submission performed'},
 {'title':'Jackson and Mudholkar 1979','url':'https://www.stat.cmu.edu/technometrics/70-79/VOL-21-03/v2103341.pdf','checked':'original paper accessed in this manuscript task; residual-monitoring scope confirmed'},
 {'title':'Liu, Ting, Zhou 2008','url':'https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf','checked':'author-hosted original paper accessed; isolation principle confirmed'},
 {'title':'Saito and Rehmsmeier 2015','url':'https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432','checked':'original article page; PR perspective'},
 {'title':'Kim et al. version2 2022','url':'https://arxiv.org/abs/2109.05257','checked':'author abstract; point-adjustment critique; no unverified numerical result imported'},
 {'title':'scikit-learn official documentation','url':'https://scikit-learn.org/stable/','checked':'PCA, StandardScaler, average_precision_score, calibration API pages as linked in bibliography'}]
(r/'documentation/REFERENCES_CHECKED.json').write_text(json.dumps(refs,ensure_ascii=False,indent=2))
print('Package documentation and verifier created')
