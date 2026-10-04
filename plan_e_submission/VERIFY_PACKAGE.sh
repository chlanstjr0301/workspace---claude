#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 code/scripts/bootstrap.py
code/.venv/bin/python code/scripts/checks.py
code/.venv/bin/python code/scripts/verify_package.py
code/.venv/bin/python code/scripts/verify_paper_numbers.py
