#!/usr/bin/env bash
set -euo pipefail
tradeoff_source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
tradeoff_project_dir="$(cd -- "$tradeoff_source_dir/../.." && pwd)"
tradeoff_python="$tradeoff_project_dir/.venv/bin/python"
if [[ ! -x "$tradeoff_python" ]]; then
  echo 'Project .venv missing: install requirements.lock.txt in a dedicated environment and run from PROJECT/runs/THIS_FOLDER.' >&2
  exit 1
fi
tradeoff_run_dir="$tradeoff_project_dir/runs/pca_fn_fp_tradeoff_$(date +%Y%m%d_%H%M%S)_rerun_$$"
mkdir -p "$tradeoff_run_dir/src" "$tradeoff_run_dir/logs"
cp "$tradeoff_source_dir"/src/*.py "$tradeoff_run_dir/src/"
cp "$tradeoff_source_dir/RUN_EXPERIMENT.sh" "$tradeoff_run_dir/RUN_EXPERIMENT.sh"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 MPLBACKEND=Agg
tradeoff_deadline=$((SECONDS + 5400))
for tradeoff_stage in prepare stage_a stage_b freeze final_evaluate validate analyze report verify_outputs; do
  tradeoff_remaining=$((tradeoff_deadline - SECONDS))
  if (( tradeoff_remaining <= 0 )); then
    echo '90 minute budget reached; see completed outputs and logs.' >&2
    exit 124
  fi
  echo "Running $tradeoff_stage -> $tradeoff_run_dir"
  timeout "$tradeoff_remaining" "$tradeoff_python" "$tradeoff_run_dir/src/$tradeoff_stage.py" > "$tradeoff_run_dir/logs/$tradeoff_stage.log" 2>&1
 done
 echo "Completed: $tradeoff_run_dir/review_report_ko.html"
