#!/usr/bin/env python3
"""Audit a pinned physical-defect campaign before computing paired qualification statistics.

Run directories are relative to --campaign's parent unless absolute. One deterministic
physics run is shared by every learned seed. Missing or failed runs withhold all means.
Bootstrap resamples the same trajectories across seeds; seeds are not independent ICs.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1]/'src')]
from summarize_implicit import FIELDS, aggregate
from mastu_opsplit.common import config_hash, sha256_file, write_json

IDENTITY_KEYS = ('model_config', 'manifest_sha256', 'source_sha256', 'normalization',
                 'source_normalization', 'dirichlet', 'physical_interval', 'labels',
                 'label_scale', 'reference_forcing_retained', 'physical_internal_substeps')
COUNTERS = ('stages', 'boundary_iterations', 'physical_rhs_evaluations', 'learned_network_evaluations')


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def scientific_config(config):
    config = copy.deepcopy(config)
    for key in ('seed', 'prepared_data_cache', 'cpu_budget_seconds'):
        config.pop(key, None)
    # A relocated, hash-pinned profile is the same physical input.
    config['physical_defect']['profile_manifest'].pop('path', None)
    return config


def ratio(left, right):
    return float(left/right) if right > 0 else None


def paired_bootstrap(learned, physics, draws, confidence):
    """learned[seed, case], physics[case]; one common case-index draw for all seeds."""
    denominator = physics[draws].mean(axis=1)
    if np.any(denominator <= 0):
        return dict(interval=None, per_seed_intervals=None,
                    reason='At least one paired draw has zero mean physics error; no draws were discarded.')
    samples = learned[:, draws].mean(axis=2)/denominator[None, :]
    quantiles = [(1-confidence)/2, (1+confidence)/2]
    return dict(interval=np.quantile(samples.mean(axis=0), quantiles).tolist(),
                per_seed_intervals=np.quantile(samples, quantiles, axis=1).T.tolist())


def run_data(run, arm, seed, campaign):
    """Read every intended file, retaining failures even while a job is incomplete."""
    result = dict(path=str(run), arm=arm, seed=seed, status='incomplete', issues=[], failures=[], missing=[])
    provenance = read(run/'provenance.json')
    metrics = read(run/arm/'metrics.json')
    result['metrics_complete'] = metrics is not None and (run/'results.json').exists()
    if provenance is None:
        result['missing'].append('provenance.json')
        return result, None, {}, metrics
    result['provenance_sha256'] = config_hash(provenance)
    result['source_sha256'] = provenance.get('source_sha256')
    config = provenance['config']
    if config.get('diagnostic_only') is not False or config.get('pretrain_steps') != 0:
        result['issues'].append('qualification requires nondiagnostic, unpretrained training')
    checks = dict(steps=campaign['training_updates'], evaluation_horizons=campaign['evaluation_horizons'],
                  evaluation_substeps=campaign['evaluation_requested_substeps'], device='cuda')
    for key, expected in checks.items():
        if config.get(key) != expected:
            result['issues'].append('config mismatch: '+key)
    if config.get('evaluation_cases') not in (None, campaign['expected_test_cases']):
        result['issues'].append('evaluation case cap differs from intended coverage')
    if config['physical_defect']['internal_substeps'] != campaign['internal_substeps']:
        result['issues'].append('internal substeps mismatch')
    if provenance.get('arm') != arm or (seed is not None and config['seed'] != seed):
        result['issues'].append('arm/seed mismatch')
    if campaign.get('source_sha256') is not None and provenance.get('source_sha256') != campaign['source_sha256']:
        result['issues'].append('source differs from campaign pin')
    if campaign.get('manifest_sha256') is not None and provenance.get('manifest_sha256') != campaign['manifest_sha256']:
        result['issues'].append('manifest differs from campaign pin')
    profile = config['physical_defect']['profile_manifest']
    if sha256_file(profile['path']) != profile['sha256']:
        result['issues'].append('profile hash mismatch')
    verification = read(run/'verification.json')
    if verification is None:
        result['missing'].append('verification.json')
    else:
        if not verification.get('all_trajectory_and_initial_hashes_verified') or not verification.get('all_coefficients_finite'):
            result['issues'].append('original dataset verification failed')
        if verification.get('normalization') != provenance['normalization']:
            result['issues'].append('verified normalization mismatch')
    normalization = run/'normalization.npz'
    if not normalization.exists():
        result['missing'].append('normalization.npz')
    else:
        import hashlib
        with np.load(normalization, allow_pickle=False) as saved:
            for name in ('mean', 'scale'):
                if hashlib.sha256(saved[name].tobytes()).hexdigest() != provenance['normalization'][name+'_sha256']:
                    result['issues'].append('saved normalization mismatch: '+name)
            if saved['training_case_ids'].tolist() != provenance['normalization']['training_case_ids']:
                result['issues'].append('saved normalization membership mismatch')
    data = {}
    expected_files = set()
    for index in range(campaign['expected_test_cases']):
        for substeps in campaign['evaluation_requested_substeps']:
            name = f'evaluation_{index:03d}_substeps{substeps}.json'
            expected_files.add(name)
            report = read(run/arm/name)
            if report is None:
                result['missing'].append(name)
                continue
            if report.get('provenance_hash') != result['provenance_sha256']:
                result['issues'].append('stale evaluation provenance: '+name)
            if report.get('failure') is not None:
                result['failures'].append(dict(file=name, failure=report['failure']))
            for record in report['records']:
                key = index, record['horizon'], substeps
                if key in data or record['substeps'] != substeps or record['start_frame'] != 0 or record['horizon'] not in campaign['evaluation_horizons']:
                    result['issues'].append('duplicate/unexpected evaluation record: '+name)
                data[key] = record
    unexpected = {path.name for path in (run/arm).glob('evaluation_*_substeps*.json')}-expected_files
    if unexpected:
        result['issues'].append('unexpected evaluation files: '+','.join(sorted(unexpected)))
    if metrics is None:
        result['missing'].append('metrics.json')
    else:
        result['metrics_sha256'] = sha256_file(run/arm/'metrics.json')
        if metrics.get('arm') != arm or not metrics.get('device', '').startswith('cuda'):
            result['issues'].append('completed metrics do not confirm expected CUDA arm')
        if arm == 'implicit_rollout':
            history = metrics.get('rollout_history', [])
            if metrics.get('selected_step') != campaign['training_updates'] or [row['step'] for row in history] != list(range(1, campaign['training_updates']+1)):
                result['issues'].append('final update/history budget mismatch')
            selected = run/arm/'rollout/selected.pt'
            if not selected.exists():
                result['missing'].append('selected.pt')
            else:
                import torch
                checkpoint = torch.load(selected, map_location='cpu', weights_only=False)
                if checkpoint.get('step') != campaign['training_updates'] or checkpoint.get('selection') != 'final fixed-budget update' or checkpoint.get('provenance_hash') != result['provenance_sha256']:
                    result['issues'].append('selected checkpoint budget/provenance mismatch')
                result['selected_checkpoint_sha256'] = sha256_file(selected)
            result['training_update_seconds'] = sum(row['training_seconds'] for row in history)
        result['device_name'] = metrics.get('device_name')
    return result, provenance, data, metrics


def audit_records(result, provenance, data, metrics, cases, campaign):
    coverage = []
    for horizon in campaign['evaluation_horizons']:
        for substeps in campaign['evaluation_requested_substeps']:
            valid = []
            expected_stages = horizon*substeps*campaign['internal_substeps']*(2 if provenance['config']['physical_defect']['scheme'] == 'ars222' else 1)
            for index, case in enumerate(cases):
                row = data.get((index, horizon, substeps))
                if row is None:
                    continue
                scores = [row.get('normalized_volume_mse'), row.get('raw_normalized_volume_mse')]
                scores += [row.get(key, {}).get(field) for key in ('per_field_normalized_mse', 'raw_per_field_normalized_mse') for field in FIELDS]
                scores += [row.get('dynamic_relative_rms', {}).get(field) for field in FIELDS]
                stage = row.get('stage_statistics', {})
                guard = row.get('guard_statistics', {})
                scores += [stage.get(key) for key in COUNTERS + ('maximum_guard_change', 'maximum_held_dof_change')]
                scores += [guard.get(key) for key in ('limited_field_steps', 'minimum_alpha')]
                scores += [row.get('elapsed_seconds')]
                finite = all(isinstance(value, (int, float)) and np.isfinite(value) and value >= 0 for value in scores)
                expected_network = 0 if result['arm'] == 'physics' else 2*expected_stages
                passed = (row.get('case_id') == case and row.get('audit_passed') is True and finite and
                          stage.get('stages') == expected_stages and stage.get('learned_network_evaluations') == expected_network)
                if not passed:
                    result['failures'].append(dict(case_id=case, horizon=horizon, substeps=substeps,
                        audit_passed=row.get('audit_passed'), finite_scores=finite, actual_case_id=row.get('case_id'),
                        expected_stages=expected_stages, actual_stages=stage.get('stages'),
                        expected_network_calls=expected_network, actual_network_calls=stage.get('learned_network_evaluations')))
                else:
                    valid.append(row)
            complete = len(valid) == len(cases)
            coverage.append(dict(horizon=horizon, requested_substeps=substeps, actual_microsteps=substeps*campaign['internal_substeps'],
                                 valid_records=len(valid), expected_records=len(cases), complete=complete))
            if metrics is not None:
                aggregates = [row for row in metrics['evaluation']['aggregates'] if row['horizon'] == horizon and row['substeps'] == substeps]
                if len(aggregates) != 1:
                    result['issues'].append(f'missing/duplicate aggregate H{horizon}/{substeps}')
                else:
                    saved = aggregates[0]
                    records = [data[index, horizon, substeps] for index in range(len(cases)) if (index, horizon, substeps) in data]
                    if saved['trajectories'] != len(records) or saved['expected_trajectories'] != len(cases) or saved['audit_failures'] != sum(not row['audit_passed'] for row in records) or saved['complete'] != complete:
                        result['issues'].append(f'aggregate coverage mismatch H{horizon}/{substeps}')
                    if complete:
                        expected = np.mean([row['normalized_volume_mse'] for row in valid])
                        if not np.isclose(saved['mean_normalized_volume_mse'], expected, rtol=1e-12, atol=0):
                            result['issues'].append(f'aggregate MSE mismatch H{horizon}/{substeps}')
    result['coverage'] = coverage
    if metrics is not None:
        numerical = sum('file' in failure for failure in result['failures'])
        passed = all(row['complete'] for row in coverage) and not result['failures']
        if metrics['evaluation']['numerical_failures'] != numerical or metrics['evaluation']['audits_passed'] != passed:
            result['issues'].append('aggregate numerical failure/audit summary mismatch')
    result['status'] = ('invalid' if result['issues'] else 'failed' if result['failures'] else
                        'complete' if result['metrics_complete'] and not result['missing'] and all(row['complete'] for row in coverage) else 'incomplete')


def score(rows):
    result = aggregate(rows, len(rows))
    result.update(raw_mean=float(np.mean([row['raw_normalized_volume_mse'] for row in rows])),
        raw_per_field={field:float(np.mean([row['raw_per_field_normalized_mse'][field] for row in rows])) for field in FIELDS},
        limited_field_steps=sum(row['guard_statistics']['limited_field_steps'] for row in rows),
        minimum_alpha=min(row['guard_statistics']['minimum_alpha'] for row in rows),
        maximum_guard_change=max(row['stage_statistics']['maximum_guard_change'] for row in rows),
        maximum_held_dof_change=max(row['stage_statistics']['maximum_held_dof_change'] for row in rows),
        work={key:sum(row['stage_statistics'][key] for row in rows) for key in COUNTERS},
        observed_rollout_and_audit_seconds=sum(row['elapsed_seconds'] for row in rows))
    return result


def summarize(path):
    campaign = read(path)
    if campaign['schema'] != 'mastu_physical_qualification_v1':
        raise ValueError('Unsupported campaign schema')
    seeds = campaign['expected_seeds']
    if len(seeds) != len(set(seeds)) or {run['seed'] for run in campaign['learned']} != set(seeds) or len(campaign['learned']) != len(seeds):
        raise ValueError('Campaign must declare exactly one learned run for every intended seed')
    root = path.resolve().parent
    declared = [('physics', None, campaign['physics']['run_dir'])] + [('implicit_rollout', row['seed'], row['run_dir']) for row in sorted(campaign['learned'], key=lambda row:seeds.index(row['seed']))]
    runs, provenances, records, metrics = {}, {}, {}, {}
    for arm, seed, directory in declared:
        key = 'physics' if seed is None else str(seed)
        try:
            runs[key], provenances[key], records[key], metrics[key] = run_data(root/Path(directory), arm, seed, campaign)
        except (OSError, ValueError, KeyError, TypeError) as error:
            runs[key] = dict(path=str(root/Path(directory)), arm=arm, seed=seed, status='invalid', issues=[str(error)], failures=[], missing=[])
            provenances[key], records[key], metrics[key] = None, {}, None
    issues = []
    reference = next((value for value in provenances.values() if value is not None), None)
    cases = []
    if reference is not None:
        manifest_path = Path(reference['config']['manifest'])
        if sha256_file(manifest_path) != reference['manifest_sha256']:
            issues.append('Manifest file hash mismatch')
        manifest = read(manifest_path)
        cases = [case['case_id'] for case in manifest['cases'] if case['split'] == 'test']
        training = {case['case_id'] for case in manifest['cases'] if case['split'] in ('train', 'validation')}
        if len(cases) != campaign['expected_test_cases'] or len(set(cases)) != len(cases):
            issues.append('Manifest test membership differs from campaign coverage')
        for key, provenance in provenances.items():
            if provenance is None:
                continue
            for name in IDENTITY_KEYS:
                if provenance.get(name) != reference.get(name):
                    runs[key]['issues'].append('Matched identity differs: '+name)
            if scientific_config(provenance['config']) != scientific_config(reference['config']):
                runs[key]['issues'].append('Matched scientific configuration differs')
            if set(provenance['normalization']['training_case_ids']) != training:
                runs[key]['issues'].append('Normalization does not use exactly the declared pooled training cases')
            try:
                audit_records(runs[key], provenance, records[key], metrics[key], cases, campaign)
            except (ValueError, KeyError, TypeError, AttributeError) as error:
                runs[key]['issues'].append('Malformed evaluation: '+str(error))
                runs[key]['status'] = 'invalid'
    all_complete = not issues and all(run['status'] == 'complete' for run in runs.values())
    result = dict(schema='mastu_physical_qualification_summary_v1', campaign=str(path.resolve()), campaign_sha256=sha256_file(path),
        status='complete' if all_complete else 'incomplete_or_failed', issues=issues, runs=runs, expected_seeds=seeds,
        test_case_ids=cases, expected_rollouts=campaign['expected_test_cases']*len(campaign['evaluation_requested_substeps'])*(1+len(seeds)),
        expected_horizon_records=campaign['expected_test_cases']*len(campaign['evaluation_requested_substeps'])*len(campaign['evaluation_horizons'])*(1+len(seeds)),
        expected_aggregates=len(campaign['evaluation_requested_substeps'])*len(campaign['evaluation_horizons'])*(1+len(seeds)),
        qualification_complete=all_complete, qualification_success=False, comparisons=[], primary=None,
        notes=['Test cases were reused from the exploratory study; this is not independent confirmation.',
            'Bootstrap resamples paired trajectories with identical draws across all seeds and the shared physics control; conditional on these fitted seeds.',
            'Seeds are training replicates, not independent initial-condition samples. Seed dispersion is reported separately.',
            'Incomplete or failed intended coverage suppresses all campaign score averages; no survivor averaging.',
            'Physics is a shared deterministic control with zero neural evaluations, not three independent control samples.',
            'Raw error is the final preguard proposal along guarded history, not an independently unguarded rollout.',
            'Observed timing includes CPU audits and solver caches; this is not a controlled throughput comparison.'])
    if not all_complete:
        return result
    boot = campaign['bootstrap']
    draws = np.random.default_rng(boot['seed']).integers(0, len(cases), size=(boot['samples'], len(cases)))
    confidence = boot.get('confidence', .95)
    result['bootstrap'] = dict(boot, confidence=confidence, resampling_unit='paired trajectory', independent_trajectory_count=len(cases), conditional_on_fitted_seeds=True)
    for horizon in campaign['evaluation_horizons']:
        for substeps in campaign['evaluation_requested_substeps']:
            selected = {key:[data[index, horizon, substeps] for index in range(len(cases))] for key,data in records.items()}
            physics = score(selected['physics'])
            learned = {str(seed):score(selected[str(seed)]) for seed in seeds}
            p = np.array([row['normalized_volume_mse'] for row in selected['physics']])
            values = np.array([[row['normalized_volume_mse'] for row in selected[str(seed)]] for seed in seeds])
            bootstrap = paired_bootstrap(values, p, draws, confidence)
            ratios = [ratio(values[i].mean(), p.mean()) for i in range(len(seeds))]
            for index, seed in enumerate(seeds):
                learned[str(seed)].update(mean_ratio=ratios[index],
                    paired_case_ratios=[ratio(left,right) for left,right in zip(values[index],p)],
                    cases_improved=int(np.sum(values[index]<p)),
                    bootstrap_interval=None if bootstrap['per_seed_intervals'] is None else bootstrap['per_seed_intervals'][index],
                    per_field_ratio={field:ratio(learned[str(seed)]['per_field'][field],physics['per_field'][field]) for field in FIELDS})
            per_field = {field:ratio(np.mean([entry['per_field'][field] for entry in learned.values()]),physics['per_field'][field]) for field in FIELDS}
            cell = dict(horizon=horizon, requested_substeps=substeps, actual_microsteps=substeps*campaign['internal_substeps'],
                physics=physics, learned=learned, mean_ratio=ratio(values.mean(),p.mean()),
                seed_ratio_std=float(np.std(ratios,ddof=1)) if len(seeds)>1 and all(value is not None for value in ratios) else None,
                seed_ratio_range=[min(ratios),max(ratios)] if all(value is not None for value in ratios) else None,
                bootstrap=bootstrap, per_field_ratio=per_field,
                per_field_regressions=[field for field,value in per_field.items() if value is not None and value>1],
                raw_mean_ratio=ratio(np.mean([entry['raw_mean'] for entry in learned.values()]),physics['raw_mean']))
            result['comparisons'].append(cell)
    primary = campaign['primary']
    chosen = next(row for row in result['comparisons'] if (row['horizon'],row['requested_substeps']) == (primary['horizon'],primary['requested_substeps']))
    interval = chosen['bootstrap']['interval']
    supported = all(row['mean_ratio'] is not None and row['mean_ratio']<1 for row in chosen['learned'].values()) and interval is not None and interval[1]<1
    material = supported and chosen['mean_ratio']<=primary['maximum_mean_ratio']
    result['primary'] = dict(primary, mean_ratio=chosen['mean_ratio'], bootstrap_interval=interval,
        all_seed_ratios_below_one=all(row['mean_ratio'] is not None and row['mean_ratio']<1 for row in chosen['learned'].values()),
        statistically_supported_gain=supported, material_gain=material)
    result['qualification_success'] = material
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.campaign)
    write_json(args.output, result)
    print(json.dumps(dict(status=result['status'], qualification_success=result['qualification_success'], output=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
