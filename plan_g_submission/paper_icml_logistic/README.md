# Frozen-model industrial burst case study

`main.tex` is an English research draft with the requested paper sections. It uses a standard two-column article layout because no official ICML style exists in the project. **official ICML 2026 style file must be added for final submission**. It is not represented as an official conference template or submission-ready paper.

No LaTeX engine (`pdflatex`, `xelatex`, `lualatex`, or `tectonic`) was found on PATH or in the inspected local installations. Consequently `main.pdf` has not been compiled. No substitute PDF is presented as LaTeX output. The six figures are supplied as 300-dpi PNG, vector PDF, and SVG. The PDF skill's artifact-start marker ran successfully before authoring them.

`references.bib` is deliberately comment-only, following the requested citation-TODO fallback when no project bibliography exists. Related Work names the citation topics that an author must verify before submission. Author and hardware information remain unknown rather than invented.

## Reproduce the analysis

From the workspace root in PowerShell:

```powershell
& 'C:/Users/EKR/miniconda3-pbe/envs/ai/python.exe' 'paper_icml_logistic/analyze_results.py'
& 'C:/Users/EKR/miniconda3-pbe/envs/ai/python.exe' 'paper_icml_logistic/audit_paper.py'
```

No classifier is fitted. `StandardScaler.fit` recovers statistics from identical saved original training rows only. The script verifies the original model probabilities from existing None coefficients/recovered intercepts and copies all consumed artifacts with exact checksums. No notebook or immutable baseline is edited.

When a genuine LaTeX engine is available:

```powershell
Set-Location paper_icml_logistic
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The current empty bibliography is not run through BibTeX. After adding verified references and the official style, adapt the document preamble and bibliography commands to that legitimate template. Render the compiled PDF and inspect every page before submission; this machine currently cannot verify paper pagination.

## Evidence and units

- Weight metrics: identical 50 saved CV folds, mean and population standard deviation (`ddof=0`) across folds. ROC-AUC and trapezoidal PR-AUC use unchanged saved probabilities. AP is stored separately.
- Paired shifts: None and balanced matched by `(fold, burst_id, label)`. Decision scores use `logit(p)`, with a declared epsilon `1e-15`; no selected probability requires clipping. Crossing counts distinguish dependent observations from unique parents.
- Robustness: every original burst has ten OOF appearances. Burst-mean score strata are predefined at `<=-1`, `abs(score)<1`, and `>=1`; predictions always retain `p>0.5`.
- Geometry: original train-only StandardScaler and train-only normal/abnormal centroids, five dimensions, Euclidean distances. Feature summaries include all 20 abnormal bursts, including 619/620. Signed Cohen's d uses pooled sample variance.
- Crop metrics: deterministic deduplicated start/middle/end crops for lengths 4/6/8/10/12/15/20/25/30; `N>=target`, no padding or new fits. Metrics pool positions within each repeat and report means/std over ten repeats. Parent-mean aggregation is separate and is diagnostic, not an official protocol change.
- Feature drift: one contribution per physical crop, scaled by original normal feature sample SD (`ddof=1`). Raw units are saved but not used to compare unlike feature units.

The original full reference includes 575 normal / 20 abnormal bursts and Recall0.9. Common `N>=30` crop corroboration holds 360 normal / 10 abnormal parents fixed. The earlier strict-shortening `N>15` transition denominator9/14 is not substituted for this experiment's `N>=15` result9/15.

## Files

- `tables/`: audited derived CSVs, table LaTeX, and exact analysis summary JSON.
- `figures/`: six figure sets; each has PNG/PDF/SVG.
- `source_results/`: exact source copies, retaining their workspace-relative directories; `manifest.json` records source path, copy path, SHA256, and byte count. Original ACF-containing data is copied for provenance; the ACF column is never used by the paper classifier analysis.
- `audit.json`, `analysis_validation.md`: machine checks, scope, limitations, and visual review status.

Important limits: only20 abnormal bursts, class and acquisition date completely confounded, no independent test, repeated folds/crops dependent, same-population feature/weight selection, label-aware segmentation, and no causal proof that sample count caused the original two FNs. The unweighted condition ties1:3 on fixed-threshold F1 and does not dominate balanced ROC-AUC.

## Optional verified reading (not manuscript citations)

These URLs were checked as real primary sources during preparation but are not silently added to the requested empty-bibliography draft:

- [Saito and Rehmsmeier, PLOS ONE2015](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432).
- [Pedregosa et al., JMLR2011](https://jmlr.org/papers/v12/pedregosa11a.html).
- [Wu and Keogh, arXiv2009.13807](https://arxiv.org/abs/2009.13807).
- [scikit-learn LogisticRegression documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
