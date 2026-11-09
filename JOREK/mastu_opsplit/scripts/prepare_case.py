#!/usr/bin/env python3
"""Create an immutable snapshot of the pinned equilibrium case."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mastu_opsplit.case import prepare_case
from mastu_opsplit.common import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare_case(load_config(args.config), args.output, dry_run=args.dry_run), indent=2))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"prepare_case: {error}\n")


if __name__ == "__main__":
    main()
