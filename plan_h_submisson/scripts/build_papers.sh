#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER="${TECTONIC:-$ROOT/environment/tools/tectonic}"
if [[ ! -x "$COMPILER" ]]; then chmod u+x "$COMPILER"; fi
cd "$ROOT/paper"
"$COMPILER" --keep-logs paper_ko.tex
"$COMPILER" --keep-logs paper_en_icml.tex
if [[ -f "$ROOT/reviews/adversarial_review_ko.tex" ]]; then
 "${PYTHON:-$ROOT/.venv/bin/python}" "$ROOT/code/build/write_review.py"
 cd "$ROOT/reviews"
 "$COMPILER" --keep-logs adversarial_review_ko.tex
fi
