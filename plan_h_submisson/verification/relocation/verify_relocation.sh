#!/usr/bin/env bash
set -euo pipefail
P=/tmp/pca_g1_publication_verify_env_20261005/bin/python
D=/tmp/pca_g1_publication_work_20261005
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export MPLBACKEND=Agg MPLCONFIGDIR="$D/matplotlib" XDG_CACHE_HOME="$D/cache" TMPDIR="$D"
test ! -e /home/lim/hydraulic_ai_alt/data/press_data_normal.csv
printf 'Original project is masked; no source input access.\n'
"$P" scripts/verify_hashes.py > "$D/hash_verification.json"
"$P" scripts/reproduce.py --out "$D/inference" --check > "$D/inference.log" 2>&1
"$P" scripts/train_fixed.py --out "$D/training" > "$D/training.log" 2>&1
"$P" code/analysis/build_assets.py > "$D/assets.log" 2>&1
bash scripts/build_papers.sh > "$D/documents.log" 2>&1
"$P" scripts/validate_input.py --sensors "$D/inference/external_smoke/sensors.csv" > "$D/input_check.log" 2>&1
"$P" scripts/evaluate_new.py --sensors "$D/inference/external_smoke/sensors.csv" --provenance "$D/inference/external_smoke/provenance.json" --labels "$D/inference/external_smoke/labels.csv" --output "$D/api_replay" > "$D/api.log" 2>&1
printf 'All relocation entrypoints finished.\n'
