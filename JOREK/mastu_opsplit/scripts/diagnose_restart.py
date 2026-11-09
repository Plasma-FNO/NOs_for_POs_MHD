#!/usr/bin/env python3
"""Measure physical-volume-weighted drift between two native JOREK restarts."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from mastu_opsplit.common import write_json
from mastu_opsplit.restart import diagnose_restart


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--quadrature-order', type=int, choices=range(2, 17), default=4)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise FileExistsError(args.output)
        if args.dry_run:
            print(json.dumps({'input': str(args.input), 'reference': str(args.reference),
                              'output': str(args.output), 'quadrature_order': args.quadrature_order}))
            return
        report = diagnose_restart(args.input, args.reference, quadrature_order=args.quadrature_order)
        write_json(args.output, report)
        print(json.dumps(report, indent=2))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f'diagnose_restart: {error}\n')


if __name__ == '__main__':
    main()
