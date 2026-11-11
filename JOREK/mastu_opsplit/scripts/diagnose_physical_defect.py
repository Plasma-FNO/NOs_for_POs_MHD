#!/usr/bin/env python3
"""GPU-only, training-trajectory screen of the full physical BE/ARS222 model.

No optimizer or learned correction is used. Shared release verification may
read all splits; every scored rollout comes from the merged training split.
Completed horizons and failures are saved after each case/substep combination.
"""
import argparse
import copy
import gc
import json
import math
from pathlib import Path
import resource
import shutil
import sys
import time
import traceback

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

from mastu_opsplit.common import config_hash, load_config, sha256_file
from mastu_opsplit.ensemble_training import atomic_json, evaluate_trajectory, prepare_data
from mastu_opsplit.gpu_native import configure_cuda_numerics
from mastu_opsplit.implicit_training import LabelStore, build_model, build_system, check_implicit_config


class CpuBudgetExhausted(RuntimeError):
    pass


def json_safe(value):
    """Keep nonfinite failure diagnostics explicit without producing invalid JSON."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    return value


def copy_setup_cache(source, destination, config):
    provenance = Path(source)/'provenance.json'
    previous = json.loads(provenance.read_text())  # Mandatory: row-cache case IDs alone are insufficient.
    if sha256_file(config['manifest']) != config['manifest_sha256'] or previous.get('manifest_sha256') != config['manifest_sha256']:
        raise ValueError('Setup cache dataset manifest differs from the current pinned release')
    for key in ('tangent_sha256', 'F0', 'dirichlet_tolerance', 'split_counts', 'source_split_counts', 'stride'):
        if previous.get('config', {}).get(key) != config.get(key):
            raise ValueError(f'Setup cache configuration differs: {key}')
    manifest = json.loads(Path(config['manifest']).read_text())
    if previous.get('source_normalization') != manifest['normalization']:
        raise ValueError('Setup cache source normalization differs from the current release')
    training_ids = [case['case_id'] for case in manifest['cases'] if case['split'] in ('train', 'validation')]
    if previous.get('normalization', {}).get('training_case_ids') != training_ids:
        raise ValueError('Setup cache fitted normalization does not use the same merged training cases')
    copied = {'source_provenance': dict(path=str(provenance.resolve()), sha256=sha256_file(provenance)),
              'matched_manifest_sha256': config['manifest_sha256'], 'normalization': previous['normalization']}
    for name in ('fixed_dofs.npz', 'wall_rows.npz'):
        path = Path(source)/name
        digest = sha256_file(path)
        shutil.copyfile(path, destination/name)
        if sha256_file(destination/name) != digest:
            raise ValueError(f'Setup cache copy changed: {name}')
        copied[name] = dict(path=str(path.resolve()), sha256=digest)
    return copied


def release_factors(system):
    """At most one substep size's native LU factors remain live at a time."""
    system.release_adjoints()
    for factor in system._factors.values():
        for solver in getattr(factor, 'lu', {}).values():
            if hasattr(solver, 'close'):
                solver.close()
    system._factors.clear()
    gc.collect()
    torch.cuda.empty_cache()


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--model-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--setup-cache', type=Path, help='Copy verified fixed_dofs.npz and wall_rows.npz from an existing setup')
    parser.add_argument('--case-indices', type=int, nargs='+', default=[0], help='Indices within the merged training split')
    parser.add_argument('--horizons', type=int, nargs='+', default=[4, 16])
    parser.add_argument('--substeps', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--scheme', choices=('be', 'ars222'), required=True)
    return parser.parse_args()


def run(args):
    config, model_config = copy.deepcopy(load_config(args.config)), load_config(args.model_config)
    if config.get('device') != 'cuda' or not isinstance(config.get('physical_defect'), dict):
        raise ValueError('Diagnostic requires device: cuda and an explicit physical_defect configuration')
    config['physical_defect']['scheme'] = args.scheme
    check_implicit_config(config)
    if not args.case_indices or min(args.case_indices) < 0 or len(set(args.case_indices)) != len(args.case_indices):
        raise ValueError('Training case indices must be distinct nonnegative integers')
    if not args.horizons or not args.substeps or min(args.horizons+args.substeps) < 1:
        raise ValueError('Horizons and substeps must be positive integers')
    horizons, substeps_list = sorted(set(args.horizons)), sorted(set(args.substeps))
    configure_cuda_numerics()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; CPU physics rollouts are not permitted by this diagnostic')
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    device = torch.device('cuda')
    args.output.mkdir(parents=True, exist_ok=False)
    started, started_cpu = time.monotonic(), time.process_time()
    report = dict(status='preparing', scheme=args.scheme, config=config, evaluation_split='train',
                  training_updates=0, learned_parameters_used=0, test_rollouts_evaluated=False,
                  data_access_note='Shared release verification may inspect all splits; selected and scored windows are training only.',
                  case_indices=args.case_indices, horizons=horizons, substeps=substeps_list, rows=[],
                  device=str(device), device_name=torch.cuda.get_device_name(device),
                  torch_version=str(torch.__version__), python_version=sys.version)
    def save():
        report.update(elapsed_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-started_cpu,
                      peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
                      peak_torch_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30)
        atomic_json(args.output/'report.json', json_safe(report))
    save()
    try:
        report['setup_cache'] = copy_setup_cache(args.setup_cache, args.output, config) if args.setup_cache else None
        data, space, guard, unused_physical, dense, verification = prepare_data(config)
        del unused_physical
        report['data_verification'] = verification
        if report['setup_cache'] is not None:
            for key in ('training_case_ids', 'mean_sha256', 'scale_sha256'):
                if report['setup_cache']['normalization'].get(key) != verification['normalization'].get(key):
                    raise ValueError(f'Setup cache fitted normalization differs: {key}')
        windows = data.windows('train', max(horizons), stride=config['stride'])
        if max(args.case_indices) >= len(windows.cases):
            raise ValueError('Selected index exceeds the merged training split')
        selected = [windows.cases[index] for index in args.case_indices]
        if any(case['split'] != 'train' for case in selected):
            raise ValueError('Diagnostic attempted to select a non-training trajectory')
        report['selected_cases'] = [dict(case_id=case['case_id'], parent_family_id=case['parent_family_id'],
                                        split=case['split'], start_frame=0, data=case['result']['data']) for case in selected]
        labels = LabelStore(config['labels'], tangent_sha256=config['tangent_sha256'],
                            training_case_ids=[case['case_id'] for case in windows.cases])
        system, boundary = build_system(config, data, space, verification, args.output, resume=False, labels=labels)
        architecture = {**model_config, 'architecture': {**model_config['architecture'], **config['architecture']}}
        model = build_model(config, space, system, architecture, labels.scale, physical_guard=guard,
                            physics_only=True).to(device).eval()
        source = Path(__file__).resolve().parents[1]/'src/mastu_opsplit'
        provenance = dict(config=config, architecture=architecture, labels=labels.identity, boundary_rows=boundary,
                          physical_profiles=model.physical_rhs.provenance,
                          config_file=dict(path=str(args.config.resolve()), sha256=sha256_file(args.config)),
                          model_config_file=dict(path=str(args.model_config.resolve()), sha256=sha256_file(args.model_config)),
                          script_sha256=sha256_file(__file__),
                          source_sha256={p.name: sha256_file(p) for p in sorted(source.glob('*.py'))},
                          manifest_sha256=sha256_file(config['manifest']), verification=verification,
                          normalization=verification['normalization'], source_normalization=data.manifest['normalization'])
        report['provenance_sha256'] = config_hash(provenance)
        atomic_json(args.output/'provenance.json', json_safe(provenance))
        report.update(status='screening', internal_substeps=model.internal_substeps,
                      total_physics_substeps={s: s*model.internal_substeps for s in substeps_list})
        save()
        # Do not start another rollout near the host's fixed worker CPU limit.
        hard_limit = resource.getrlimit(resource.RLIMIT_CPU)[1]
        budget = min(float(config.get('cpu_budget_seconds') or 450.), 550.)
        if hard_limit != resource.RLIM_INFINITY:
            budget = min(budget, max(1., hard_limit-45.))
        report['worker_cpu_budget_seconds'] = budget
        def check_cpu_budget(module, inputs):
            if time.process_time() >= budget:
                raise CpuBudgetExhausted(f'Worker reached {budget:g} CPU seconds before the next macrostep')
        model.register_forward_pre_hook(check_cpu_budget)
        for substeps in substeps_list:
            for index in args.case_indices:
                if time.process_time() >= budget:
                    report['status'] = 'cpu_budget_exhausted'
                    report['remaining_pairs'] = [(s, i) for s in substeps_list for i in args.case_indices
                                                 if not any(r['substeps'] == s and r['case_index'] == i for r in report['rows'])]
                    save()
                    release_factors(system)
                    return report
                sample = windows[index*windows.starts]  # always frame zero, training split only
                model.reset_stage_statistics()
                records, row_started, row_cpu = [], time.monotonic(), time.process_time()
                free, total = torch.cuda.mem_get_info(device)
                row = dict(case_index=index, case_id=sample['case_id'], split='train', start_frame=0,
                           substeps=substeps, records=records, failure=None,
                           device_free_gib_before=free/2**30, device_total_gib=total/2**30)
                try:
                    with torch.no_grad():
                        evaluate_trajectory(model, sample, space, guard, dense, set(horizons), substeps, records=records)
                    torch.cuda.synchronize()
                    if model.stage_statistics['learned_network_evaluations'] != 0:
                        raise AssertionError('Physics-only screen unexpectedly evaluated a learned correction')
                except Exception as error:
                    row['failure'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
                row.update(seconds=time.monotonic()-row_started, cpu_seconds=time.process_time()-row_cpu,
                           stage_statistics=dict(model.stage_statistics), guard_statistics=dict(guard.statistics),
                           system=system.report())
                report['rows'].append(row)
                save()
                print(json.dumps(json_safe(dict(case_id=row['case_id'], substeps=substeps, failure=row['failure'],
                                                horizons=[dict(horizon=r['horizon'], mse=r['normalized_volume_mse'],
                                                               raw_mse=r.get('raw_normalized_volume_mse'), audit=r['audit_passed'])
                                                          for r in records]))), flush=True)
                if row['failure'] is not None and row['failure']['type'] == 'CpuBudgetExhausted':
                    report['status'] = 'cpu_budget_exhausted'
                    report['remaining_pairs'] = [(substeps, index)] + [
                        (s, i) for s in substeps_list for i in args.case_indices
                        if not any(r['substeps'] == s and r['case_index'] == i for r in report['rows'])]
                    save()
                    release_factors(system)
                    return report
            release_factors(system)
        report['status'] = 'completed'
        report['all_rollouts_completed'] = all(row['failure'] is None for row in report['rows'])
        report['all_audits_passed'] = report['all_rollouts_completed'] and all(
            record['audit_passed'] for row in report['rows'] for record in row['records'])
        save()
        return report
    except Exception as error:
        report.update(status='setup_or_driver_failure', failure=dict(type=type(error).__name__, message=str(error),
                                                                   traceback=traceback.format_exc()))
        save()
        raise


if __name__ == '__main__':
    result = run(arguments())
    if result['status'] == 'cpu_budget_exhausted':
        raise SystemExit(75)
