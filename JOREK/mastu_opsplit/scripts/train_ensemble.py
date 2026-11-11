#!/usr/bin/env python3
"""Train matched ensemble models without the chronological pilot or RHS-label gates."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

from mastu_opsplit.common import load_config
from mastu_opsplit.ensemble_training import run_ensemble


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--model-config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--family', choices=('all', 'ar', 'node', 'opsplit', 'conditional'), default='all')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--linear-branch', choices=('lspline_continuity', 'native_continuity'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'cuda'), help='cuda: device sampling, projection and guards')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config['seed'] = args.seed
    if args.device is not None:
        config['device'] = args.device
    if args.linear_branch is not None:
        config['linear_branch'] = args.linear_branch
    families = ('ar', 'node', 'opsplit') if args.family == 'all' else (args.family,)
    results = run_ensemble(config, load_config(args.model_config), args.output, families=families, resume=args.resume)
    if any(not result['evaluation']['audits_passed'] for result in results.values()):
        raise RuntimeError('Rollout numerical/physical audit failed; inspect per-trajectory reports')


if __name__ == '__main__':
    main()
