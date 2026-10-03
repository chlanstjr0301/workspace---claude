from __future__ import annotations

import argparse
import sys
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent
sys.path.insert(0, str(PACKAGE))

from src.pipeline import run_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Plan C SHIFT-Guard pipeline")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = run_pipeline(ROOT, args.mode, args.output)
    print(f"Selected fault model: {result['selected_model']}")
    print(f"Outputs: {result['output']}")
    print(result["comparison"].to_string(index=False))


if __name__ == "__main__":
    main()
