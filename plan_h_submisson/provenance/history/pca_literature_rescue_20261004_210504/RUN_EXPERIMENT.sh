#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_PYTHON="${PYTHON_BIN:-}"
if [[ -z "$TASK_PYTHON" ]]; then
  for TASK_CANDIDATE in "$TASK_ROOT/.venv/bin/python" "$TASK_ROOT/../../.venv/bin/python"; do
    if [[ -x "$TASK_CANDIDATE" ]]; then TASK_PYTHON="$TASK_CANDIDATE"; break; fi
  done
fi
if [[ -z "$TASK_PYTHON" ]]; then TASK_PYTHON=python3; fi
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 MPLBACKEND=Agg
"$TASK_PYTHON" "$TASK_ROOT/src/replay.py" "$@"
