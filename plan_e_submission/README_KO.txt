한국어 연구논문 · 코드 · 데이터 · 결과 통합 패키지

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
