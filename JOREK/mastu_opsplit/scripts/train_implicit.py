#!/usr/bin/env python3
"""Implicit native-tangent OpsSplit: one arm (implicit / implicit_rollout / physics) per output directory."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

from mastu_opsplit.common import load_config
from mastu_opsplit.implicit_training import ARMS, run_implicit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--model-config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config['seed'] = args.seed
    results = run_implicit(config, load_config(args.model_config), args.output, arm=args.arm, resume=args.resume)
    if any(not result['evaluation']['audits_passed'] for result in results.values()):
        raise RuntimeError('Rollout numerical/physical audit failed; inspect per-trajectory reports')


if __name__ == '__main__':
    main()
