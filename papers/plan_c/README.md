# SHIFT-Guard ICML 2026 Manuscript

This directory contains the anonymous ICML 2026-format manuscript and its
reproducible vector-figure builder.

## Build

Run from this directory:

```powershell
python build_figures.py
pdflatex -interaction=nonstopmode -halt-on-error shift_guard_icml2026.tex
bibtex shift_guard_icml2026
pdflatex -interaction=nonstopmode -halt-on-error shift_guard_icml2026.tex
pdflatex -interaction=nonstopmode -halt-on-error shift_guard_icml2026.tex
Copy-Item shift_guard_icml2026.pdf output/shift_guard_icml2026.pdf -Force
```

The figure builder reads only frozen result tables under
`plan_c_submission/outputs/tables`. The official `icml2026.sty` and bibliography
style were downloaded from the ICML 2026 author-instruction package without
modification.

The reviewed final PDF is stored at `output/shift_guard_icml2026.pdf`.

The submission manuscript is anonymous. Replace the author, affiliation, and
corresponding-author fields only when preparing a camera-ready version, and then
switch to `\usepackage[accepted]{icml2026}`.
