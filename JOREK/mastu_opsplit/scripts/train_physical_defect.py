#!/usr/bin/env python3
"""Run the native physical-defect experiment on CUDA with resumable bounded workers.

The regular implicit trainer remains available for historical experiments. This
launcher requires the new physical_defect configuration and never falls back to CPU.
"""
import argparse
import json
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

from mastu_opsplit.common import load_config


def progress(output):
    """Detect a worker that exhausted its budget without advancing a saved result."""
    paths = sorted(output.glob('*/rollout/last.pt')) + sorted(output.glob('*/evaluation_*.json'))
    return [(str(path.relative_to(output)), path.stat().st_mtime_ns, path.stat().st_size) for path in paths]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--model-config', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--arm', choices=('physics', 'implicit_rollout'), required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--setup-cache', type=Path,
                        help='Copy existing training-only fixed_dofs/wall_rows caches; case IDs are checked by the trainer')
    parser.add_argument('--prepared-data-cache', type=Path,
                        help='Reuse a verified immutable dataset setup across bounded workers')
    parser.add_argument('--cpu-budget', type=float, default=300.,
                        help='Checkpoint threshold; leave CPU time for one complete update or trajectory')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = load_config(args.config)
    if not config.get('physical_defect') or config.get('device') != 'cuda':
        parser.error('Use a physical_defect configuration with device: cuda')
    limit = resource.getrlimit(resource.RLIMIT_CPU)[0]
    if not 0 < args.cpu_budget < float('inf') or (limit != resource.RLIM_INFINITY and args.cpu_budget >= limit):
        parser.error('CPU budget must be positive, finite, and below the process CPU limit')
    if args.seed is not None:
        config['seed'] = args.seed
    config['cpu_budget_seconds'] = args.cpu_budget
    if args.prepared_data_cache is not None:
        config['prepared_data_cache'] = str(args.prepared_data_cache.resolve())
    if args.worker:
        from mastu_opsplit.implicit_training import run_implicit
        result = run_implicit(config, load_config(args.model_config), args.output,
                              arm=args.arm, resume=args.resume)
        print(json.dumps(dict(worker_cpu_seconds=time.process_time(), interrupted=bool(result.get('interrupted')))),
              flush=True)
        if result.get('interrupted'):
            raise SystemExit(75)
        if not all(row['evaluation']['audits_passed'] for row in result.values()):
            raise RuntimeError('Native rollout audit failed; completed diagnostic records are preserved')
        return
    if args.output.exists() and not args.resume:
        raise FileExistsError(f'Use a fresh output directory or --resume: {args.output}')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    if args.setup_cache is not None:
        source = json.loads((args.setup_cache/'provenance.json').read_text())
        if source['manifest_sha256'] != config['manifest_sha256'] or any(
                source['config'][key] != config[key] for key in
                ('tangent_sha256', 'F0', 'dirichlet_tolerance', 'source_split_counts', 'split_counts')):
            raise ValueError('Setup cache was identified with different data, tangent, or boundary settings')
        for name in ('fixed_dofs.npz', 'wall_rows.npz'):
            target = args.output/name
            if not target.exists():
                shutil.copy2(args.setup_cache/name, target)
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--worker']
    if '--resume' not in command:
        command.append('--resume')
    while True:
        before = progress(args.output)
        status = subprocess.run(command).returncode
        if status != 75:
            raise SystemExit(status)
        if progress(args.output) == before:
            raise RuntimeError('Worker budget expired without a new checkpoint or evaluation; increase the budget '
                               'within the host limit or use a host without the restrictive CPU limit')
        print('Saved progress; starting the next CUDA worker.', flush=True)


if __name__ == '__main__':
    main()
