#!/usr/bin/env bash
set -euo pipefail
review_source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
review_project_dir="$(cd -- "$review_source_dir/../.." && pwd)"
review_python="$review_project_dir/.venv/bin/python"
if [[ ! -x "$review_python" ]]; then
  echo 'Project .venv missing. Install requirements.lock.txt in an isolated Python 3.14 environment, then run from PROJECT/runs/REVIEW_FOLDER.' >&2
  exit 1
fi
review_run_dir="$review_project_dir/runs/adversarial_pca_$(date +%Y%m%d_%H%M%S)_rerun_$$"
mkdir -p "$review_run_dir/src" "$review_run_dir/logs"
cp "$review_source_dir"/src/*.py "$review_run_dir/src/"
cp "$review_source_dir/RUN_REVIEW.sh" "$review_run_dir/RUN_REVIEW.sh"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 MPLBACKEND=Agg
for review_stage in prepare reproduce design develop validate finalize analysis_report additional_audit write_report verify_outputs; do
  echo "Running $review_stage -> $review_run_dir"
  "$review_python" "$review_run_dir/src/$review_stage.py" > "$review_run_dir/logs/$review_stage.log" 2>&1
 done
 echo "Completed: $review_run_dir/review_report_ko.html"
