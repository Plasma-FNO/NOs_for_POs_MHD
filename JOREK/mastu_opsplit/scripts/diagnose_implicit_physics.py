#!/usr/bin/env python3
"""Physics-only (N=0) implicit OpsSplit on TRAINING trajectories: go/no-go before labels and training.

Reports factorization cost, teacher-forced one-step errors and autonomous rollouts with the
ensemble evaluator's metric, beside persistence and a leave-out training-mean increment.
No test trajectory is read.
"""
import argparse
import json
from pathlib import Path
import resource
import sys
import time

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

from mastu_opsplit.common import load_config, sha256_file
from mastu_opsplit.ensemble_training import atomic_json, evaluate_trajectory, prepare_data
from mastu_opsplit.euler_training import FIELDS
from mastu_opsplit.implicit_opsplit import ACTIVE_CHANNELS, ImplicitOpsSplitModel, WeakAssembler
from mastu_opsplit.implicit_training import LabelStore, build_system


def field_mse(space, prediction, truth):
    weights = space.weights/space.weights.sum()
    error = space.basis @ (prediction-truth)
    return np.einsum('pc,p->c', error**2, weights)


def baseline_records(space, states, prediction, horizons):
    weights = space.weights/space.weights.sum()
    squared, mean, moment, out = np.zeros(7), np.zeros((len(weights), 7)), np.zeros((len(weights), 7)), {}
    for step in range(1, max(horizons)+1):
        truth = space.basis @ states[step]
        squared += np.einsum('pc,p->c', (space.basis @ prediction[step]-truth)**2, weights)
        delta = truth-mean
        mean += delta/step
        moment += delta*(truth-mean)
        if step in horizons:
            mse = squared/step
            variation = np.einsum('pc,p->c', moment, weights)/step
            out[step] = dict(normalized_volume_mse=float(mse.mean()), per_field_normalized_mse=dict(zip(FIELDS, mse.tolist())),
                             dynamic_relative_rms=dict(zip(FIELDS, np.sqrt(mse/np.maximum(variation, 1e-30)).tolist())))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--model-config', required=True)
    parser.add_argument('--tangent', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cases', type=int, default=6)
    parser.add_argument('--fractions', type=float, nargs='+', default=[1., .25])
    parser.add_argument('--theta', type=float, default=1.)
    parser.add_argument('--horizons', type=int, nargs='+', default=[16, 32, 96])
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config, model_config = load_config(args.config), load_config(args.model_config)
    torch.set_num_threads(config['threads'])
    report = dict(config=args.config, tangent=args.tangent, tangent_sha256=sha256_file(args.tangent), theta=args.theta,
                  test_trajectories_read=False)
    started = time.monotonic()
    data, space, guard, _, dense, verification = prepare_data(config)
    report['verification_seconds'] = time.monotonic()-started
    train = [case for case in data.cases if case['split'] == 'train']
    t = time.monotonic()
    labels = LabelStore(args.labels, tangent_sha256=report['tangent_sha256'],
                        training_case_ids=[case['case_id'] for case in train])
    system_config = dict(tangent=args.tangent, tangent_sha256=report['tangent_sha256'], dirichlet_tolerance=1e-7,
                         theta=args.theta, F0=0.64)
    system, boundary = build_system(system_config, data, space, verification, output, resume=False, labels=labels)
    report['boundary_rows'] = dict(boundary, seconds=time.monotonic()-t)
    label_paths = {(e['case_id'], e['frame']): e['path'] for e in labels.entries}
    assembler = WeakAssembler(space.geometry)
    for fraction in args.fractions:
        system.factor(fraction)
    report['system'] = system.report()
    report['peak_rss_gib_after_factorization'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20
    atomic_json(output/'report.json', report)
    print(json.dumps(report['system']), flush=True)
    model = ImplicitOpsSplitModel(space, system, {**model_config, 'architecture': {**model_config['architecture'],
                                  **config['architecture']}}, np.ones(len(ACTIVE_CHANNELS)), physical_guard=guard,
                                  physics_only=True)
    model.eval()
    # Evenly spaced training cases; mean increment excludes them (leave-out baseline).
    chosen = [train[i] for i in np.linspace(0, len(train)-1, args.cases).round().astype(int)]
    chosen_ids = {case['case_id'] for case in chosen}
    others = [case for case in train if case['case_id'] not in chosen_ids]
    mean_trajectory = None
    for case in others:
        with h5py.File(case['resolved_path']) as stream:
            x = (stream['native_state'][()]-data.mean)/data.scale
        mean_trajectory = x if mean_trajectory is None else mean_trajectory + x
    mean_increment = (mean_trajectory/len(others))
    mean_increment -= mean_increment[0]
    windows = data.windows('train', max(args.horizons), stride=config['stride'])
    index = {case['case_id']: i for i, case in enumerate(windows.cases)}
    rows = []
    for case in chosen:
        sample = windows[index[case['case_id']]*windows.starts]
        states = sample['state'].numpy()
        row = dict(case_id=case['case_id'])
        # Teacher-forced single steps from truth frames 0, 32, 64.
        one_step = {}
        for fraction in args.fractions:
            substeps = int(round(1/fraction))
            errors, exact, persistence = [], [], []
            for frame in (0, 32, 64):
                start = torch.from_numpy(states[frame:frame+1])
                with torch.no_grad():
                    predicted = model(start, torch.ones(1, 1), substeps=substeps)[0].numpy()
                    forcing = torch.from_numpy(assembler.numpy(np.load(label_paths[case['case_id'], frame]).astype(np.float64)[None]))
                    oracle = start
                    for _ in range(substeps):  # exact label held at q_n, as N_theta(q_n) would be
                        oracle = guard(oracle, system.step(oracle, forcing, fraction))
                errors.append(field_mse(space, predicted, states[frame+1]))
                exact.append(field_mse(space, oracle[0].numpy(), states[frame+1]))
                persistence.append(field_mse(space, states[frame], states[frame+1]))
            one_step[substeps] = dict(physics=np.mean(errors, 0).tolist(), exact_label=np.mean(exact, 0).tolist(),
                                      persistence=np.mean(persistence, 0).tolist())
        row['one_step'] = one_step
        rollouts = {}
        for fraction in args.fractions:
            substeps = int(round(1/fraction))
            t = time.monotonic()
            records = []
            try:
                evaluate_trajectory(model, sample, space, guard, dense, set(args.horizons), substeps, records=records)
                failure = None
            except (FloatingPointError, ValueError, RuntimeError) as error:
                failure = str(error)
            rollouts[substeps] = dict(records=records, failure=failure, seconds=time.monotonic()-t)
        row['physics_only'] = rollouts
        row['persistence'] = baseline_records(space, states, np.broadcast_to(states[:1], states.shape), args.horizons)
        row['mean_increment'] = baseline_records(space, states, states[:1] + mean_increment[:len(states)], args.horizons)
        rows.append(row)
        report['rows'] = rows
        atomic_json(output/'report.json', report)
        summary = {s: [(r['horizon'], round(r['normalized_volume_mse'], 4)) for r in v['records']] for s, v in rollouts.items()}
        print(json.dumps(dict(case=case['case_id'], physics=summary,
                              persistence={h: round(v['normalized_volume_mse'], 4) for h, v in row['persistence'].items()},
                              mean_increment={h: round(v['normalized_volume_mse'], 4) for h, v in row['mean_increment'].items()})),
              flush=True)
    report['elapsed_seconds'] = time.monotonic()-started
    report['peak_rss_gib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20
    atomic_json(output/'report.json', report)


if __name__ == '__main__':
    main()
