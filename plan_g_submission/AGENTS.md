# Baseline Original and ablation rules

- The immutable reference is `src/LSTM-AutoEncoder/baseline_original.py`.
- Do not edit, refactor, optimize, seed, or silently repair that file for experiments.
- Verify its SHA256 against `src/LSTM-AutoEncoder/baseline_original.sha256` before every ablation.
- Create every ablation by copying the complete reference to a new Python file.
- Each ablation may change exactly one declared experimental factor. Add comments stating the change, its reason, and confirmation that all other conditions are identical.
- Preserve CSVs, abs() on all three inputs, row slices, MinMax fit scope, sequence and future offset, validation/test split, normal-only training validation, model, optimizer/loss/callbacks, last-timestep flatten, original exact P/R equality threshold, strict `>` prediction, and unseeded initialization unless that single factor is the ablation.
- Preserve the original fit() defaults; do not silently add `shuffle=False` or a random seed.
- A threshold-rule failure must be logged. Any compatibility implementation belongs in a separately named file and output directory; it must never replace Baseline Original.
- A deterministic reference, if requested, must be a separate `baseline_seeded.py` copied from the original with seed control as the declared factor. Do not compare an unseeded original run to a seeded experiment as if scaler alone changed.
- Keep library versions and hardware consistent within comparisons, and record environment changes separately from algorithm changes.
- Never select a threshold or tune hyperparameters using final test results.
- Previous `src/scaler_experiment` outputs are historical seeded comparisons based on `guidebook.py`; they are not Baseline Original reproductions. Preserve them and label them accordingly.
- `run_baseline_original_colab.py` is execution infrastructure, not an ablation. It must not inject a seed, deterministic settings, or model changes.
