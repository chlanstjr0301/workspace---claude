"""Package existing paper, relevant code/data/results without running experiments."""
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "hydraulic_pump_logistic_paper_bundle.zip"
PREFIX = "hydraulic_pump_logistic_bundle"
EXPERIMENTS = [
    "logistic_current_acf_ablation", "logistic_current_acf_pearson_ablation",
    "logistic_ai0_ai1_corr_ablation", "logistic_class_weight_ablation",
    "logistic_fn_burst_tracking", "logistic_fn_feature_analysis",
    "logistic_burst_length_sensitivity", "logistic_burst_sample_count",
]
SKIP_PARTS = {"__pycache__", ".matplotlib_cache", ".git"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    if ARCHIVE.exists():
        raise FileExistsError(f"Existing archive preserved: {ARCHIVE}")
    selected = set()
    for folder in [ROOT / "paper_icml_logistic"] + [ROOT / "results" / x for x in EXPERIMENTS]:
        assert folder.is_dir(), folder
        selected.update(p for p in folder.rglob("*") if p.is_file()
                        and not any(part in SKIP_PARTS for part in p.relative_to(ROOT).parts))
    qa = ROOT / "tmp/pdfs/paper_icml_logistic"
    if qa.is_dir():
        selected.update(p for p in qa.rglob("*") if p.is_file())
    selected.update(ROOT / "src" / (x + ".py") for x in EXPERIMENTS)
    selected.update(ROOT / p for p in ["AGENTS.md", "reference/소성가공.ipynb",
        "data/raw/press_data_normal.csv", "data/raw/outlier_data.csv",
        "src/LSTM-AutoEncoder/baseline_original.py", "src/LSTM-AutoEncoder/baseline_original.sha256",
        "src/package_logistic_paper.py"])
    baseline = ROOT / "src/LSTM-AutoEncoder/baseline_original.py"
    expected = baseline.with_suffix(".sha256").read_text().split()[0].lower()
    assert digest(baseline.read_bytes()) == expected
    readme = """# Hydraulic pump Logistic Regression 논문·코드·데이터 묶음

## 포함 내용
- paper_icml_logistic/: 영문 main.tex, references.bib(TODO), 표, 300dpi 그림 및 vector PDF/SVG, audit와 논문 재현 코드
- src/: 관련 Logistic Regression 분석·실험 코드, 불변 LSTM reference 및 checksum
- data/raw/: 원본 press_data_normal.csv 및 outlier_data.csv
- reference/: 수정하지 않은 소성가공.ipynb
- results/: 논문과 연결되는 8개 Logistic Regression 실험·진단 결과
- bundle_manifest.json: 모든 포함 파일의 상대 경로, 크기, SHA256

압축 해제 후 이 README가 있는 폴더를 프로젝트 root로 사용한다.
관련 없는 실험, .git, 임시 dependency/vendor, Python·Matplotlib cache는 제외했다.

## 저장 결과로 논문 분석 재현 (모델 재학습 없음)
Python 3.11.16 기준. requirements.txt는 기록된 분석 환경 버전이다.

```shell
python -m pip install -r requirements.txt
python paper_icml_logistic/analyze_results.py
python paper_icml_logistic/audit_paper.py
```

개별 src 실험 스크립트에는 Logistic Regression 학습이 포함될 수 있다.
저장 결과 확인만 할 때는 위 논문 분석·audit 코드만 실행한다.
논문 실행 환경·수치 출처·한계는 paper_icml_logistic/README.md 참조.
PDF figure 렌더 검증의 audit 코드는 원래 환경의 Poppler 경로를 기록하고 있다.
다른 컴퓨터에서는 audit_paper.py의 Poppler 실행 경로를 설치 환경에 맞춰 확인한다.
기존 렌더 QA 자료도 tmp/pdfs/paper_icml_logistic/에 포함했다.

## 논문 PDF
논문 본문은 main.tex이다. 이 묶음을 만들 때 LaTeX 실행기가 없어 main.pdf는 미컴파일이다.
figures/의 PDF들은 개별 그림이며 논문 전체 PDF가 아니다.
official ICML 2026 style file must be added for final submission.
LaTeX 환경이 준비되면 paper_icml_logistic에서 pdflatex를 두 번 실행한다.
references.bib의 문헌 TODO도 제출 전에 해결해야 한다.

## 데이터·모델 조건
기존 5-feature, train-only scaling, 동일 50-fold OOF, class_weight=None과 고정 threshold0.5 기반 진단이다.
Class-weight ablation은 기록된 다른 weight 조건도 포함한다.
불변 baseline_original.py 및 원본 notebook/CSV는 수정하지 않았다.
본 압축 생성은 추가 학습·inference·실험 없이 기존 파일을 묶는 작업이다.
"""
    requirements = "numpy==2.4.6\npandas==3.0.5\nscipy==1.17.1\nscikit-learn==1.9.1\nmatplotlib==3.11.1\n"
    manifest = []
    total_bytes = 0
    with ZipFile(ARCHIVE, "w", compression=ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(selected):
            assert path.is_file(), path
            relative = path.relative_to(ROOT).as_posix()
            data = path.read_bytes()
            z.writestr(f"{PREFIX}/{relative}", data)
            manifest.append(dict(path=relative, bytes=len(data), sha256=digest(data)))
            total_bytes += len(data)
        for name, content in [("BUNDLE_README.md",readme),("requirements.txt",requirements)]:
            data = content.encode("utf-8")
            z.writestr(f"{PREFIX}/{name}",data)
            manifest.append(dict(path=name,bytes=len(data),sha256=digest(data)))
        z.writestr(f"{PREFIX}/bundle_manifest.json",json.dumps(
            dict(format=1,entries=manifest,experiments=EXPERIMENTS,
                 archive_creation_runs_experiments=False),indent=2,ensure_ascii=False).encode("utf-8"))
    with ZipFile(ARCHIVE) as z:
        assert z.testzip() is None, "Archive CRC check failed"
        for entry in manifest:
            assert digest(z.read(f"{PREFIX}/{entry['path']}")) == entry["sha256"]
        must_have=["paper_icml_logistic/main.tex", "paper_icml_logistic/audit.json",
                   "data/raw/press_data_normal.csv", "data/raw/outlier_data.csv", "reference/소성가공.ipynb"]
        assert all(f"{PREFIX}/{p}" in z.namelist() for p in must_have)
    assert digest(baseline.read_bytes()) == expected
    print(json.dumps(dict(archive=str(ARCHIVE),files=len(manifest)+1,
        uncompressed_bytes=total_bytes,zip_bytes=ARCHIVE.stat().st_size,
        sha256=digest(ARCHIVE.read_bytes()),CRC_verified=True,all_file_hashes_verified=True),
        indent=2,ensure_ascii=False))


if __name__ == "__main__":
    main()
