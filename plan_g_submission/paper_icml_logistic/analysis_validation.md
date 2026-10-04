# Analysis validation

Post-authoring source/table/claim/figure checks: 253 passed. Exact numerical claim literals checked: 54.

All source copies match their SHA256 manifests and originals. Class-weight means/std reproduce saved metrics; PR-AUC is recomputed from unchanged probabilities. All selected probabilities are unsaturated. None and balanced pairings preserve labels/folds; crossings are94 normal observations from16 unique parents and zero abnormal/reverse crossings. Both conditions retain18 stableTP and2 persistentFN.

Five paper tables use the audited data, including the handwritten exact feature formulas. Figure traces map to their source CSVs. Training-only centroid coordinates and distances are supplied per fold, not as a single global centroid. Normal and all-abnormal score-distribution summaries include quartiles and both detected/FN subgroups. No classifier is fitted.

The complete draft was checked for explicit acquisition-date confounding, selection optimism, dependency, minority-sample scarcity, cautious crop interpretation, literal None labels, empty bibliography TODOs, and non-official style disclosure. Basic LaTeX environment nesting/file references pass. These are not proof of successful compilation. main.pdf is unavailable because no genuine LaTeX engine exists locally; paper pagination cannot be verified.

All six 300-dpi PNG figures have been visually reviewed. All six vector PDFs were rendered with bundled Poppler and the contact sheet was inspected: no clipping or missing content. The QA contact sheet is at tmp/pdfs/paper_icml_logistic/figure_pdf_contact_sheet.png. Main paper pagination remains unverified because it could not be compiled.
