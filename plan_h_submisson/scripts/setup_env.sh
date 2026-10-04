#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${UV:-$(command -v uv || true)}"
if [[ -z "$UV_BIN" && -x "$HOME/.local/bin/uv" ]]; then UV_BIN="$HOME/.local/bin/uv"; fi
if [[ -n "$UV_BIN" ]]; then
 "$UV_BIN" venv --python "${PYTHON:-python3.14}" "$ROOT/.venv"
 "$UV_BIN" pip sync --python "$ROOT/.venv/bin/python" "$ROOT/environment/requirements.lock.txt"
else
 "${PYTHON:-python3.14}" -m venv "$ROOT/.venv"
 "$ROOT/.venv/bin/python" -m pip install -r "$ROOT/environment/requirements.lock.txt"
fi
