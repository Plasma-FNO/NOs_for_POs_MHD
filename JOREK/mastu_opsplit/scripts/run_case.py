#!/usr/bin/env python3
"""Run a prepared case in an allocation, or render a bounded execution plan."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mastu_opsplit.common import load_config
from mastu_opsplit.runner import render_slurm, run_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--render-slurm", action="store_true", help="print sbatch script without submitting")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.render_slurm:
            print(render_slurm(config, args.prepared, args.output, args.config), end="")
            return
        result = run_case(config, args.prepared, args.output, dry_run=args.dry_run)
        print(json.dumps(result, indent=2))
        if not args.dry_run and result["status"] != "completed":
            parser.exit(1, f"run_case: {result['status']}\n")
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"run_case: {error}\n")


if __name__ == "__main__":
    main()
