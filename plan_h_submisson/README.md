# PCA + R5 prediction-residual auxiliary alarm (G1)

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

A combined entrypoint after setup is `bash RUN_REVIEW.sh work/inference`. Python 3.14 must be installed; setup prefers `uv` and otherwise requires working `venv`/`ensurepip`. The independently installed validation environment is recorded under `verification/`; no virtual environment is shipped. Archived historical builders may retain source-location provenance; the documented runtime and build entrypoints resolve the extracted package root.

`data/provenance.example.json` is metadata format only; confirm rather than assume independence and unit compatibility. Paper build regenerates the self-review manifest from the rebuilt PDFs; use `PYTHON=/path/to/compatible/python bash scripts/build_papers.sh` if your environment is not `.venv`. The editable manuscripts are the authoritative final prose; `provenance/authoring/` preserves assembly-stage scripts only.
