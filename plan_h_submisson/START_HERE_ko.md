# 먼저 읽기

선정 모델은 **PCA + R5 예측잔차 보조 경보(G1)** 입니다. 기존 PCA B0, R5 모델, 실제 임계값17.54312199063481과 경보 정책은 변경하지 않았습니다.

1. `paper/paper_ko.pdf`: 한국어 연구용 확장 논문. 같은 방법·수치·결론의 `paper/paper_en_icml.pdf`는 ICML2026 공식 익명 형식 연구 초안입니다. 실제 제출·채택 이력이 아닙니다.
2. `reviews/adversarial_review_ko.pdf`: 논문과 함께 읽을 **단점 중심 저자 측 AI 보조 자체 검토**. 실제 ICML 심사·외부 전문가 의견·독립 검증이 아닙니다. CSV에 지적별 페이지와 근거가 연결됩니다.
3. `research_readiness_ko.md`: 연구 기준 충족 범위와 남은 결손.
4. `README.md`: 설치, 저장 모델 추론, 고정 학습 재현, 문서 빌드와 새 자료 입력 명령.
5. `provenance/evidence_map.csv`: 주장→원실행→행→예측→계산 코드 연결.

압축 해제한 `pca_g1_paper_code_data` 폴더에서 실행합니다.

```bash
bash scripts/setup_env.sh
.venv/bin/python scripts/verify_hashes.py
.venv/bin/python scripts/reproduce.py --out work/inference --check
.venv/bin/python code/analysis/build_assets.py
bash scripts/build_papers.sh
# 선택된 구성만 원본부터 재적합; 모델 교체·탐색 아님
.venv/bin/python scripts/train_fixed.py --out work/fixed_training
```

Python3.14와 패키지 최초 설치, Tectonic 최초 TeX 번들 취득에는 인터넷이 필요합니다. 설치된 다른 Tectonic은 `TECTONIC=/실행파일`로 지정합니다. 다른 폴더로 이동해도 원래 프로젝트를 읽지 않습니다. 원본 기록용 역사 코드에는 당시 절대경로가 남아 있지만 기본 진입점에서는 실행하지 않습니다. 실행 출력 경로는 새 폴더여야 합니다.

H의 FN28→13·FP1 유지가 독립 현장 개선의 증거는 아닙니다. 추가 탐지15행은 이미 탐지하던3개 구간의 범위 확대이고, V는 FN0→0 유지입니다. 미사용 고유 관측0행, 신규 독립 자료 검증 미실시입니다. 원본 CSV의 중복1행은 그대로 보존했습니다. 모델 파일 로딩은 신뢰한 패키지에서만 수행하세요.
