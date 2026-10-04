from pathlib import Path
import json,shutil,hashlib,csv,sys,platform,subprocess,importlib.metadata as im
W=Path(__file__).parent;R=W/'pca_g1_paper_code_data';P=Path('/home/lim/hydraulic_ai_alt')
def put(n,s): (R/n).parent.mkdir(parents=True,exist_ok=True);(R/n).write_text(s)
put('README.md','''# PCA + R5 prediction-residual auxiliary alarm (G1)

This user-custody research archive contains Korean and anonymous ICML-2026-format English manuscripts, **not an accepted or submitted paper**. The Korean model name is “PCA + R5 예측잔차 보조 경보(G1)”. Read `paper/paper_ko.pdf` or `paper/paper_en_icml.pdf`, then `reviews/adversarial_review_ko.pdf` (author-side AI-assisted weakness-only self-review, not external peer review), then the provenance and reproduction records.

## Environment and exact commands

Linux x86_64, Python 3.14; exact original packages are pinned. Do not unpickle models from untrusted sources. Work inside the extracted `pca_g1_paper_code_data` directory. An internet connection is needed for initial Python packages and the TeX bundle; no training service, Colab, GPU, purchase, or original project path is required.

```bash
bash scripts/setup_env.sh
.venv/bin/python scripts/verify_hashes.py
.venv/bin/python scripts/reproduce.py --out work/inference --check
.venv/bin/python code/analysis/build_assets.py
bash scripts/build_papers.sh
```

Use a new output path for each run (existing output is refused). `--check` adds the frozen online/batch and input-contract validation; it reuses the historical151 tests rather than silently calling them new experiments. `build_assets.py` reads **stored predictions only**; `reproduce.py` first verifies new predictions against them. Figures and tables regenerated in `paper/` replace only the package's generated assets. Preserve the downloaded ZIP as the immutable original.

To reconstruct only the selected components, without the earlier full search:

```bash
.venv/bin/python scripts/train_fixed.py --out work/fixed_training
```

This fits two PCA pipelines (each includes spectrum and retained-component fits) and one lag-two ridge/covariance R5 on normal T, recalibrates on normal C, and verifies exact alarms against the selected models. Regenerated models are saved under the new output path and never replace `models/`. It reads the frozen row/role manifest and verifies original CSV observations before fitting. Score tolerance and decision equality are recorded separately. No seed or candidate search occurs.

For supplied new data, first validate the input contract without inference:

```bash
.venv/bin/python scripts/validate_input.py --sensors /path/sensors.csv
.venv/bin/python scripts/evaluate_new.py --sensors /path/sensors.csv --provenance /path/provenance.json --labels /path/labels.csv --output /path/new_output
```

Omit `--labels` for unlabeled prediction. Input schema is `data/schema.json`; sensor columns are `row_id,session_id,TimeStamp,AI0_Vibration,AI1_Vibration,AI2_Current`. Labels belong in a separate `row_id,label` file. The checked interface rejects missing/nonfinite sensors, duplicate/nonincreasing times, duplicate IDs and interleaved sessions. Physical units and original equipment metadata remain unknown, so semantic compatibility is not automatically established. Do not designate a report block as an acquisition session. The supplied original H smoke data are reused records, not independent new data.

## Contents and history

- `paper/`: both PDFs, editable LaTeX, shared bibliography, official style, numerical tables and vector figures; fonts and license.
- `code/runtime/`: byte-preserved fixed review source, including the later strict external-input wrapper. `scripts/` provides new portable entrypoints. `code/analysis/` regenerates paper assets.
- `code/historical/`: original precursor scripts. They may contain historical absolute paths and are preserved as evidence, **not executable default entrypoints**. Copied mandatory dependencies are in the portable runtime; no default execution reads these original paths.
- `data/raw/`: byte-identical two originals; `data/analysis/inputs/frozen_rows.csv`: all retained observations with source row, CSV line, timestamp and role. Usage audit retains the excluded duplicate mapping. Original split manifests are included.
- `models/`: actual selected B0 bundle and R5 fitted object; `configs/model_manifest.json`: thresholds and policies.
- `results/frozen/`: predictions and all fixed controls, timing, events, errors; `results/history/`: relevant original PCA/IF comparison, limited-rescue and failed-candidate records. IF records are **historical comparators**, not G1 experiments or results attributed to PCA alone.
- `provenance/`: original preregistrations, selection changes, reports/logs, source inventory, reference audit, evidence map, input-use and experiment-count audits.
- `verification/`: new reproduction/build/relocation/checksum evidence; it is separate from historical experimental counts.
- `reviews/`: weakness-only AI-assisted self-review, criticism evidence and reviewed-manuscript hashes.

**Results:** H B0 TP329/FN28/FP1/TN3989; G1 TP344/FN13/FP1/TN3989. H was already exposed. Added15 rows extend three already-detected groups; no new whole event was found. V confirms maintenance only (FN0/FP0 both; reduction not_assessable). All20599 unique observations were previously used. No independent new-acquisition performance evaluation occurred.

`manifest.json` and `SHA256SUMS` exclude themselves from recursive hash coverage; the external ZIP digest covers the final ZIP. Generated output and caches created after extraction are not listed in the original manifest. Public redistribution permission, authorship, affiliation, support and conflicts are not established; see DATA_LICENSE_NOTES.md and MISSING_ITEMS.md. This archive is not automatically suitable as anonymous public supplemental material.
''')
put('START_HERE_ko.md','''# 먼저 읽기

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
''')
put('DATA_LICENSE_NOTES.md','''# Data provenance and redistribution

Project-provided dataset description: 소성가공 예지보전 AI 데이터셋; two files `press_data_normal.csv` and `outlier_data.csv`. Byte-preserved SHA-256 and row counts are in `results/frozen/data_audit.json` and the final manifest. The original acquisition/provider license text, public-redistribution permission, equipment identity and sensor units were not established from supplied materials. No permission is invented. This package is for the user's custody; it was not uploaded publicly, submitted to a conference, or sent to another person.

The original normal duplicate remains in raw CSV and is excluded only through the documented analysis mapping. No synthetic measurements are mixed with observed performance data. Invalid-input regression fixtures in historical verification are labeled tests, not performance data.

Official ICML style and original notices are retained. Font and Tectonic licenses are included; dependency license notices are copied where installed. Third-party research articles are referenced by verified URL/DOI and scoped reading notes, not redistributed as full texts here. The archive is not automatically anonymized supplemental material: provenance paths and original source records can identify the local project. Confirm data rights and human author disclosures before external release.
''')
put('MISSING_ITEMS.md','''# Actual gaps (not reconstructed as historical facts)

- No separately acquired independent new dataset or unused unique original observation is available.
- Sensor physical units, verified equipment/session identifiers, operating load, physical failure type/start, maintenance, pressure, quality and downtime histories are unavailable.
- A complete immutable historical process ledger, every deleted/overwritten intermediate, and all original inline execution commands could not be recovered. Existing relevant scripts, logs, artifacts and audit evidence are included. New portability/training/document scripts are labeled as newly written, not original historical source.
- The project is not a Git repository; a historical commit chronology cannot be provided.
- Dataset license/public redistribution permission, human authors/affiliations, funding and conflicts are not established.
- Full third-party literature PDFs are intentionally not redistributed; source links, bibliographic verification and inspected-scope notes are included.
- No field runtime/latency or equipment deployment test is claimed. The included compiler binary is Linux x86_64; other platforms require a compatible Tectonic installation. No offline cache/virtual environment is packaged; initial dependencies require network or a user-provided package mirror.

Selected model objects, raw input CSVs, frozen rows, all direct G1 candidates/controls, predictions and mandatory inference dependencies were recovered. Newly reproduced components remain separate from selected models.
''')
put('research_readiness_ko.md','''# 연구 준비 수준 자체 점검

ICML2026은 문서 형식과 연구 검토의 참고 기준이며 채택 가능성을 보장하지 않는다.

| 관점 | 현재 확인한 범위 | 남은 결손 |
|---|---|---|
| 방법 새로움 | PCA 잔차·예측잔차·OR·연속 확인의 정확한 결합 명세 | 각 요소는 기존 방법이며 새로운 범용 학습 이론이나 문헌상 최초라는 근거 없음 |
| 기존 방법과 차이 | 동일 B0를 보존하고 R5 보조 경로에만 연속 조건 부여; threshold/confirmation 대조 | 다른 현장·문제에서의 차별적 이점 미검증 |
| 비교 범위 | 같은 행의 B0/G1/G0 및 이전 threshold 대조; 실패 후보 보존 | 광범위 알고리즘 우월성을 판단하는 비교 아님 |
| 데이터 독립성 | 역할·공백·원행·중복·입력 창 이력 확인 | 모든 고유 관측 기사용, 클래스와 날짜 혼재, 독립 신규 자료 없음 |
| 통계 근거 | 혼동행렬·동일행 paired 변화·사건별 표로 기술 | 의존성과 소수 사건 때문에 유의성·미래 FPR 신뢰구간 주장 없음 |
| 재현성 | 실제 모델·원본·선택 기록·고정 학습 및 추론 경로·별도 해제 검사 | 전 역사 명령·불변 외부 시각·장비 실시간 환경·배포 검증 미확보 |

H에서 FN28→13, FP1 유지의 산술적 사실과 현장 일반화는 다른 주장이다.15개 추가 행은 새 사건0개,3개 기존 관측 구간의 확장이다. V의 FN0/F2=1 유지에서 개선 재현을 요구할 수 없고, 이전 R5의 S2 탈락도 취소되지 않는다. 이러한 한계를 본문에 공개해도 새 독립 자료와 기여 근거의 부족 자체가 해소되지는 않는다. 자세한 단점별 검토는 reviews/adversarial_review_ko.pdf를 참조한다.
''')
put('CHANGELOG.md','''# Package changes relative to original experiment

- Frozen B0/R5 objects, features, thresholds, split, labels and predictions unchanged; runtime sources copied byte-for-byte.
- Added portable orchestration, fixed-setting two-PCA/one-R5 reconstruction, raw-input mapping validation, common paper metric/figure generator and checksum entrypoint. These are new package utilities, not purported historical originals.
- Retained original complete-data helper and later finite-input rejection wrapper separately; no model-level missing-data interpolation or suppression introduced.
- Regenerated manuscript tables from stored predictions. Corrected draft training-target count from an unverified11292 to actual11296 before final publication in package; no experimental file changed.
- Paper build attempt1 failed in koTeX ICU linebreak; used the prior complete manuscript's xeCJK/Noto settings. Attempt2 exposed a bibliography ampersand escaping error; fixed typesetting only. Logs retained.
- English default anonymous ICML2026 style; Korean research preprint layout retained. No accepted option and no invented author/funding/COI facts.
- Adversarial review is a separate author-side AI-assisted weakness review of final manuscripts, not external review.
- New package reconstruction/inference executions occur after the prior usage audit cutoff; their counts are documented separately and are not silently added to historical counts.
''')
put('environment/INSTALL.md','''# Environment

Model/data code was developed and checked on Python3.14.4, Linux x86_64, with numerical-library CPU threads capped at2. Exact installed versions are in requirements.lock.txt. `bash scripts/setup_env.sh` creates a private environment; it never changes system Python. If Python3.14 is unavailable, install it in a user-managed location and set `PYTHON=/path/python3.14` for setup. Compatibility on other versions is not asserted.

The bundled Tectonic Linux x86_64 binary and license come from the original manuscript package. `scripts/build_papers.sh` builds source with it (or TECTONIC override). Tectonic obtains its TeX bundle over the network on first use and uses an ordinary local cache. Required TeX components include ICML2026, xeCJK, fontspec, AMS packages, booktabs, graphicx, algorithm, tabularx, hyperref and natbib. The NotoSansCJKkr font and its license are in paper/fonts; Korean uses the prior xeCJK configuration. Source compiler logs and bundle metadata are retained; no full cache/venv is included.

All detector/model-source assets are local to the extracted archive. Historical source has documentary original paths but is not part of default command execution. No GPU or paid service is required.
''')
# Copy current task request ONLY, never export raw conversations.
found=[]
for session in (Path('/home/lim/.codex/sessions')).rglob('*.jsonl'):
 try:
  for line in session.open():
   j=json.loads(line);p=j.get('payload',{});
   if p.get('role')!='user':continue
   text='\n'.join(c.get('text','') for c in p.get('content',[]) if isinstance(c,dict))
   if '너는 /home/lim/hydraulic_ai_alt 프로젝트의 연구 원고 작성' in text and '[11-6.' in text:found.append((str(session),text))
 except Exception:pass
if found:put('provenance/user_request_ko.txt',found[-1][1]);put('provenance/request_source.json',json.dumps(dict(source=found[-1][0],only_this_request_exported=True),ensure_ascii=False,indent=2))
else:put('provenance/user_request_ko.txt','Exact persisted user request unavailable to extractor. See package scope in README; this is not a verbatim transcript.\n');put('provenance/request_source.json',json.dumps({'exact_request_recovered':False}))
# Copy post-audit reproduction evidence, not output cache.
for n in ['portable_inference_check.json','validation_report.json','data_audit.json','run_status.json']:
 shutil.copy2(W/'inference_check'/n,R/'verification'/('new_'+n))
shutil.copytree(W/'inference_check/logs',R/'verification/inference_logs',dirs_exist_ok=True)
for p in W.glob('paper_build*.log'):shutil.copy2(p,R/'verification'/p.name)
# Fixed model input-role manifest references actual included paths.
m=json.loads((R/'configs/model_manifest.json').read_text());m['package_lineage']=dict(original_run='runs/pca_followup_v2_20261004_214810',normal_fit='data/analysis/inputs/frozen_rows.csv: split=train (normal only)',normal_calibration='split=calibration & label=0',selection='configs/manifests and original protocol blocks; prior labels used',result_source='results/frozen/predictions',runtime='code/runtime',model_identity_preserved=True);put('models/manifest.json',json.dumps(m,ensure_ascii=False,indent=2))
shutil.copy2(P/'runs/pca_g1_frozen_validation_20261004_221914/input_schema.json',R/'data/schema.json')
# Installed third-party licenses, never authentication files.
for name in ['scikit-learn','numpy','scipy','pandas','matplotlib','pymupdf']:
 d=im.distribution(name)
 for f in d.files or []:
  if '.dist-info' in str(f) and any(x in str(f).lower() for x in ['license','copying']) and Path(d.locate_file(f)).is_file():
   dest=R/'environment/licenses'/name/Path(f).name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(d.locate_file(f),dest)
print('documents complete; request exact',bool(found))
