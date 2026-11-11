#!/usr/bin/env python3
"""Three real-data updates with checkpoint interruption/resume and bounded evaluation."""
import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from mastu_opsplit.common import load_config, write_json
from mastu_opsplit.ensemble_training import run_ensemble


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--model-config', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    config.update(steps=3, evaluation_cases=1, evaluation_horizons=[4, 16], diagnostic_only=True)
    model_config = load_config(args.model_config)
    run_ensemble(config, model_config, args.output, families=('conditional',), resume=True, stop_after=1)
    first = torch.load(args.output/'conditional/last.pt', map_location='cpu', weights_only=False)
    if first['step'] != 1:
        raise ValueError('Smoke interruption must stop after exactly one update; use a fresh smoke directory')
    results = run_ensemble(config, model_config, args.output, families=('conditional',), resume=True)
    result = results['conditional']
    last = torch.load(args.output/'conditional/last.pt', map_location='cpu', weights_only=False)
    if [row['step'] for row in last['history']] != [1, 2, 3] or last['history'][0] != first['history'][0]:
        raise ValueError('Smoke resume did not preserve the completed update')
    if not result['evaluation']['audits_passed']:
        raise RuntimeError('Smoke physical/numerical audit failed')
    report = dict(checkpoint_resume_passed=True, audits_passed=True, steps=3,
                  parameters=result['parameters'], peak_rss_kib=result['peak_rss_kib'],
                  seconds_per_update=[row['training_seconds'] for row in last['history']],
                  evaluation_seconds=result['evaluation_seconds'], diagnostic_only=True)
    write_json(args.output/'smoke_passed.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
