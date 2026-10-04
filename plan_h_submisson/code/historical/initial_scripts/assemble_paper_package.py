from pathlib import Path
import shutil,json
ROOT=Path(__file__).resolve().parents[1];r=Path((ROOT/'reports/current_paper_package.txt').read_text());code=r/'code'
p=code/'src/experiment.py';s=p.read_text().replace("ROOT/'data'","ROOT.parent/'data'");p.write_text(s)
(code/'scripts/reproduce.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export MPLBACKEND=Agg OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python3 scripts/bootstrap.py
run="../results/reproduced_$(date -u +%Y%m%dT%H%M%S)_${RANDOM}"
.venv/bin/python scripts/checks.py
.venv/bin/python src/experiment.py --out "$run" --stage develop
.venv/bin/python scripts/validate_development.py --run "$run"
.venv/bin/python src/experiment.py --out "$run" --stage final
.venv/bin/python scripts/supplement.py --run "$run"
.venv/bin/python scripts/report.py --run "$run"
printf 'Completed: %s\\n' "$run"
''')
(code/'scripts/bootstrap.py').write_text('''"""Install only into this package's code/.venv; do not modify system Python."""
from pathlib import Path
import os,shutil,subprocess,sys
root=Path(__file__).resolve().parents[1];venv=root/'.venv';py=venv/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
uv=shutil.which('uv')
if not uv:
 candidate=Path.home()/'.local/bin/uv'
 if candidate.is_file():uv=str(candidate)
if not py.exists():
 if uv:subprocess.run([uv,'venv',str(venv),'--python',sys.executable],check=True)
 else:subprocess.run([sys.executable,'-m','venv',str(venv)],check=True)
if uv:cmd=[uv,'pip','sync','--python',str(py),str(root/'requirements.lock.txt')]
else:cmd=[str(py),'-m','pip','install','-r',str(root/'requirements.lock.txt')]
subprocess.run(cmd,check=True)
print('Private interpreter:',py)
''')
(r/'RUN_EXPERIMENT.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
bash code/scripts/reproduce.sh
''')
(r/'VERIFY_PACKAGE.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 code/scripts/bootstrap.py
code/.venv/bin/python code/scripts/checks.py
code/.venv/bin/python code/scripts/verify_package.py
''')
(r/'BUILD_PAPER.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Linux x86_64 compiler included; use installed tectonic on other systems.
if command -v tectonic >/dev/null 2>&1; then compiler="$(command -v tectonic)"; else compiler="$PWD/tools/tectonic"; chmod +x "$compiler"; fi
cd paper
"$compiler" --keep-logs manuscript_ko.tex
''')
for p in [r/'RUN_EXPERIMENT.sh',r/'VERIFY_PACKAGE.sh',r/'BUILD_PAPER.sh',code/'scripts/reproduce.sh']:p.chmod(0o755)
# Retain remote scripts as historical evidence but make local entry points portable.
shutil.copy2(ROOT/'scripts/build_paper_assets.py',code/'scripts/build_paper_assets.py')
p=code/'scripts/build_paper_assets.py';s=p.read_text().replace("r=Path((ROOT/'reports/current_paper_package.txt').read_text());run=ROOT/'runs/pca_20261003_cpu'", "r=ROOT.parent;run=r/'results/pca_20261003_cpu'");p.write_text(s)
(code/'requirements-paper.txt').write_text('pymupdf==1.28.2\n# PDF inspection only; model packages are in requirements.lock.txt.\n')
changes={'changes_to_executable_copy':['experiment.py: data root resolves to package/data instead of project-local data','reproduce.sh: portable private environment and unique package/results output','bootstrap.py/verify_package.py: new package utilities','build_paper_assets.py: relative package paths'], 'model_algorithm_changed':False,'historical_source_preserved':'results/pca_20261003_cpu/source and source_final','remote_scripts':'historical scripts retain original dedicated Colab paths; not automatically executed','authentication_files_included':False}
(r/'documentation/PORTABILITY_CHANGES.json').write_text(json.dumps(changes,indent=2))
print('Portable wrappers prepared')
