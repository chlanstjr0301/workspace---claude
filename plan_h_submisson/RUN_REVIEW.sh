#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
"$PY" scripts/verify_hashes.py
"$PY" scripts/reproduce.py --out "${1:-work/inference}" --check
"$PY" code/analysis/build_assets.py
bash scripts/build_papers.sh
