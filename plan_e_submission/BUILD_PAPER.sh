#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Linux x86_64 compiler included; use installed tectonic on other systems.
if command -v tectonic >/dev/null 2>&1; then compiler="$(command -v tectonic)"; else compiler="$PWD/tools/tectonic"; chmod +x "$compiler"; fi
cd paper
"$compiler" --keep-logs manuscript_ko.tex
