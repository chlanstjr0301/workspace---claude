#!/usr/bin/env bash
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
printf 'Completed: %s\n' "$run"
