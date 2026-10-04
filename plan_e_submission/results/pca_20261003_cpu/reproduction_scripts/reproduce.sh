#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export MPLBACKEND=Agg
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
run="runs/pca_$(date -u +%Y%m%dT%H%M%S)_${RANDOM}"
if [[ ! -x .venv/bin/python ]]; then python3 -m venv --without-pip .venv; fi
/home/lim/.local/bin/uv pip sync --python .venv/bin/python requirements.lock.txt
.venv/bin/python scripts/checks.py
.venv/bin/python src/experiment.py --out "$run" --stage develop
.venv/bin/python scripts/validate_development.py --run "$run"
.venv/bin/python src/experiment.py --out "$run" --stage final
.venv/bin/python scripts/supplement.py --run "$run"
.venv/bin/python scripts/report.py --run "$run"
printf 'Completed: %s\n' "$run"
