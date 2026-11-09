#!/usr/bin/env python3
"""Generate independent native MAST-U references and a trajectory-split Euler dataset.

Run prepare once, run-case in a four-rank Slurm array, then finalize after the
array. Failed cases remain visible and prevent a complete dataset release.
"""
import argparse
import ctypes
from dataclasses import fields
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import h5py
import numpy as np

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / 'src'))
from mastu_opsplit.common import sha256_file as digest, write_json as write
from mastu_opsplit.numerical import NativeGeometry
from mastu_opsplit.restart import read_restart

RUNS = Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs')
PILOT = RUNS / 'mastu_native_pilot_20261104'
REFERENCE = RUNS / 'mastu_full_opsplit_20261104'
REFINEMENT = RUNS / 'mastu_refinement_20261104'
BASELINE = REFINEMENT / 'runs/baseline_settle1/work/jorek000144.h5'
BINARY_SHA = '98d4348bc6e56d16168e1fc7ed82094036e13f33073ea5ddb18d1366026a4425'
NAMES = ['psi', 'u', 'zj', 'w', 'rho', 'T', 'Vpar']


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sample_cases(seed=20261105):
    """Freeze whole-trajectory assignments before observing any simulation outcomes."""
    rng = np.random.default_rng(seed)
    n = 100
    draws = np.column_stack([(rng.permutation(n) + rng.random(n)) / n for _ in range(6)])
    signs = rng.choice([-1., 1.], size=(n, 2))
    assignments = rng.permutation(['train'] * 60 + ['validation'] * 20 + ['test'] * 20)
    return [dict(case_id=f'mastu_{i:03d}', parent_family_id=f'mastu_ic_{i:03d}',
                 split=str(assignments[i]), center=[.89 + .12*x[0], -.15 + .30*x[1]],
                 width=[.07 + .03*x[2], .12 + .08*x[3]], taper=[.55, .75],
                 amplitude=(signs[i] * (.01 + .04*x[4:6])).tolist())
            for i, x in enumerate(draws)]


def seed_restart(source, target, case, library):
    """Reuse the native product rule for value, s, t and mixed derivative DOFs."""
    pointer = ctypes.POINTER(ctypes.c_double)
    helper = ctypes.CDLL(str(library)).opsplit_perturb_node_
    helper.argtypes = [pointer] * 11
    with h5py.File(source) as f:
        raw = f['values'][()]; original = raw.copy(); xy = f['x'][()]
        boundary = f['boundary'][()]
        axis = np.array([f['psi_axis_t'][-1]]); edge = np.array([f['psi_bnd_t'][-1]])
        zpoints = f['Z_xpoint_t'][:, -1]
    if not edge[0] > axis[0]:
        raise ValueError('Invalid flux normalization')
    center, width, taper, amplitude = [np.asarray(case[key], dtype='f8')
                                      for key in ('center', 'width', 'taper', 'amplitude')]
    if not (np.abs(amplitude) <= .05).all() or not (width > 0).all():
        raise ValueError('Perturbation outside qualified small-amplitude envelope')
    for node in np.flatnonzero((boundary == 0) & (xy[1, 0, 0] > zpoints.min()) &
                              (xy[1, 0, 0] < zpoints.max())):
        arrays = [xy[0, :, 0, node].copy(), xy[1, :, 0, node].copy(), raw[0, :, 0, node].copy(),
                  axis, edge, center, width, taper, amplitude,
                  raw[4, :, 0, node].copy(), raw[5, :, 0, node].copy()]
        helper(*(a.ctypes.data_as(pointer) for a in arrays))
        raw[4, :, 0, node], raw[5, :, 0, node] = arrays[-2:]
    if not np.array_equal(raw[[0, 1, 2, 3, 6]], original[[0, 1, 2, 3, 6]]) or \
       not np.array_equal(raw[:, :, :, boundary != 0], original[:, :, :, boundary != 0]):
        raise ValueError('Perturbation changed unselected fields or boundary nodes')
    changed = np.any(raw != original, axis=(0, 1, 2))
    if not changed.any():
        raise ValueError('Empty perturbation')
    if Path(target).exists():
        raise FileExistsError(target)
    shutil.copyfile(source, target)
    with h5py.File(target, 'r+') as f:
        f['values'][...] = raw
    geometry, state, metadata = read_restart(target, quadrature_order=8)
    samples = np.einsum('eql,elc->eqc', geometry.basis, state[geometry.dof_map])
    if metadata['minimum_signed_jacobian_m2'] <= 0 or not (samples[..., 4:6] > 0).all():
        raise ValueError('Invalid initial geometry or Q8 density/temperature')
    with h5py.File(source) as a, h5py.File(target) as b:
        for key in a:
            if key != 'values':
                np.testing.assert_array_equal(a[key][()], b[key][()])
        vertex = a['vertex'][()].T - 1
        sizes = a['size'][()].transpose(2, 1, 0)
    affected = changed[vertex].any(axis=1)
    jets = original[0, :, 0, :].T[vertex] * sizes
    controls = np.stack([jets[:, :, 0], jets[:, :, 0] + jets[:, :, 1],
                         jets[:, :, 0] + jets[:, :, 2], jets.sum(axis=2)], axis=2)
    upper = float(((controls[affected] - axis[0]) / (edge[0] - axis[0])).max())
    if upper >= 1 or boundary[vertex[affected]].any():
        raise ValueError('Perturbation support reaches separatrix or physical boundary')
    return dict(sha256=digest(target), changed_nodes=int(changed.sum()),
                affected_elements=int(affected.sum()), flux_support_upper_bound=upper,
                positive_q8=True, unselected_fields_boundary_and_metadata_exact=True)


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'plan.json').exists() or (root / 'code').exists():
        raise FileExistsError('Ensemble already prepared or partially prepared')
    code = root / 'code'; code.mkdir()
    shutil.copytree(STUDY / 'src', code / 'src', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(__file__, code / 'build_euler_dataset.py')
    runner_text = (REFERENCE / 'run_reference_full.py').read_text()
    runner_text = runner_text.replace("spec['steps']<=384", "spec['steps']<=768")
    runner_text = runner_text.replace('timeout=7000', 'timeout=14000')
    runner_text = runner_text.replace('384 steps, 7000 seconds', '768 steps, 14000 seconds')
    (code / 'reference_runner.py').write_text(runner_text)
    shutil.copyfile(REFERENCE / 'diagnostics_full.py', code / 'diagnostics_full.py')
    shutil.copyfile(PILOT / 'analyze_pilot.py', code / 'analyze_pilot.py')
    native = (PILOT / 'native_fields.py').read_text()
    native = native.replace(str(STUDY), str(code))
    (code / 'native_fields.py').write_text(native)
    shutil.copyfile(REFINEMENT / 'perturb_helper.f90', code / 'perturb_helper.f90')
    subprocess.run(['gfortran', '-shared', '-fPIC', '-fcheck=all', str(code / 'perturb_helper.f90'),
                    '-o', str(code / 'perturb_helper.so')], check=True)
    shutil.copytree(PILOT / 'prepared/baseline_fine', root / 'prepared/baseline_fine')
    if digest(root / 'prepared/baseline_fine/jorek.exe') != BINARY_SHA:
        raise ValueError('Reference executable changed')
    (root / 'initials').mkdir(); (root / 'cases').mkdir(); (root / 'logs').mkdir()
    shutil.copyfile(BASELINE, root / 'initials/baseline.h5')
    geometry, _, metadata = read_restart(BASELINE)
    with h5py.File(root / 'geometry.h5', 'x') as f:
        for field in fields(NativeGeometry):
            f.create_dataset(field.name, data=getattr(geometry, field.name), compression='gzip', compression_opts=1)
        f.attrs['metadata_json'] = json.dumps(metadata)
    cases = sample_cases()
    for case in cases:
        path = root / 'initials' / (case['case_id'] + '.h5')
        case['initial'] = str(path.relative_to(root))
        case['initial_check'] = seed_restart(root / 'initials/baseline.h5', path, case, code / 'perturb_helper.so')
        print(json.dumps({'prepared': case['case_id'], 'split': case['split']}), flush=True)
    if len({c['initial_check']['sha256'] for c in cases}) != len(cases):
        raise ValueError('Duplicated initial-condition realizations')
    controls = {}; control_sources = {}
    for rate, source in [('coarse', REFINEMENT / 'runs/control_dt64/work'),
                         ('fine', REFERENCE / 'runs/reference_control_dt128_full/work')]:
        destination = root / 'controls' / rate / 'work'; destination.mkdir(parents=True)
        paths = [*sorted(source.glob('jorek??????.h5')), source / 'jorek_restart.h5', source / 'input']
        if len(paths) != 99:
            raise ValueError('Incomplete common control')
        control_sources[rate] = []
        for path in paths:
            shutil.copyfile(path, destination / path.name)
            control_sources[rate].append(dict(source=str(path), sha256=digest(destination / path.name)))
        controls[rate] = str(destination)
    plan = dict(schema='mastu-euler-ensemble-plan-v1', seed=20261105, cases=cases,
                split_counts={'train': 60, 'validation': 20, 'test': 20},
                baseline_sha256=digest(BASELINE), geometry_sha256=digest(root / 'geometry.h5'),
                binary_sha256=BINARY_SHA, controls=controls, control_sources=control_sources,
                coarse=dict(dt=.015625, steps=192, nout=2), fine=dict(dt=.0078125, steps=384, nout=4),
                finer=dict(dt=.00390625, steps=768, nout=8),
                refinement_policy='If coarse/fine fails, compare fine/finer against the same converged fine control; never change the initial condition or split.',
                expected_frames=97, dynamic_error_limit=.01, energy_error_limit=.01,
                physical_window_seconds=[9*4.686377037426928e-7, 12*4.686377037426928e-7],
                code_sha256={str(p.relative_to(root)): digest(p) for p in code.rglob('*') if p.is_file()},
                limits=['One fixed axisymmetric mesh and equilibrium; independent perturbation realizations.',
                        'Spatial convergence and cross-equilibrium generalization are not established.',
                        'Common unperturbed control is diagnostic only and never a training or held-out case.',
                        'Shared scratch requires a separate archival copy for long-term retention.'])
    write(root / 'plan.json', plan)
    return plan


def environment(root, plan):
    for relative, sha in plan['code_sha256'].items():
        if digest(root / relative) != sha:
            raise ValueError('Frozen ensemble code changed: ' + relative)
    sys.path.insert(0, str(root / 'code'))
    load_module('native_fields', root / 'code/native_fields.py')
    runner = load_module('run_native', root / 'code/reference_runner.py')
    runner.ROOT = root; runner.PILOT = root
    return runner


def export_case(root, case, work, report):
    from native_fields import NativeMesh
    paths = sorted(work.glob('jorek??????.h5'))
    if len(paths) != 97:
        raise ValueError('Incomplete reference window')
    mesh = NativeMesh(paths[0]); shape = mesh.read(paths[0])[0].shape
    target = root / 'cases' / (case['case_id'] + '.h5')
    partial = target.with_suffix('.partial.h5')
    if target.exists() or partial.exists():
        raise FileExistsError(target)
    sources = []; times = []
    with h5py.File(partial, 'x') as f:
        state = f.create_dataset('native_state', shape=(97, *shape), dtype='f8',
                                 chunks=(1, min(shape[0], 4096), 7), compression='gzip', compression_opts=1, shuffle=True)
        for i, path in enumerate(paths):
            q, t = mesh.read(path); state[i] = q; times.append(t)
            sources.append(dict(path=str(path.relative_to(root)), sha256=digest(path)))
        times = np.asarray(times)
        np.testing.assert_allclose(np.diff(times), np.diff(times)[0], rtol=1e-10, atol=1e-18)
        f['time_seconds'] = times; f['frame_index'] = np.arange(97)
        metadata = json.dumps(dict(case, schema='mastu-euler-trajectory-v1',
             fields=NAMES, synthetic=False, geometry='../geometry.h5',
             coefficient_semantics='canonical native bicubic value and derivative DOFs, not nodal field samples',
             reference_binary_sha256=BINARY_SHA, convergence_report=report, source_snapshots=sources))
        # Full provenance can exceed HDF5's compact attribute-size limit.
        f.create_dataset('metadata_json', data=metadata, dtype=h5py.string_dtype('utf-8'))
    with h5py.File(partial) as f:
        for i, path in enumerate(paths):
            np.testing.assert_array_equal(f['native_state'][i], mesh.read(path)[0])
    partial.replace(target)
    return dict(path=str(target.relative_to(root)), sha256=digest(target), bytes=target.stat().st_size,
                frames=97, time_seconds=times.tolist())


def run_case(root, index):
    plan = json.loads((root / 'plan.json').read_text())
    if not 0 <= index < len(plan['cases']):
        raise ValueError('Unknown case index')
    case = plan['cases'][index]; name = case['case_id']
    if (root / 'cases' / (name + '.json')).exists():
        raise FileExistsError('Case already assessed')
    runner = environment(root, plan)
    initial = root / case['initial']
    if digest(initial) != case['initial_check']['sha256']:
        raise ValueError('Initial condition changed')
    report = dict(case_id=name, split=case['split'], status='failed')
    try:
        diagnostics = load_module('ensemble_diagnostics', root / 'code/diagnostics_full.py')
        attempts = []
        for rate in ('coarse', 'fine', 'finer'):
            spec = dict(plan[rate], label=name + '_' + rate, restart=str(initial))
            existing = root / 'runs' / spec['label'] / 'result.json'
            if existing.exists():
                provenance = json.loads((existing.parent / 'provenance.json').read_text())
                if json.loads(existing.read_text())['status'] != 'completed' or provenance['spec'] != spec or \
                   provenance['restart_sha256'] != digest(initial) or provenance['binary_sha256'] != BINARY_SHA:
                    raise ValueError('Cannot resume incompatible or failed run')
            else:
                runner.run(spec)
            if rate == 'coarse':
                continue
            previous = 'coarse' if rate == 'fine' else 'fine'
            # The fallback uses one converged common control for both responses.
            # Thus its response error retains, rather than cancels, baseline integration error.
            comparison = diagnostics.compare(root / 'runs' / (name + '_' + previous + '/work'),
                                             root / 'runs' / (name + '_' + rate + '/work'),
                                             plan['controls'][previous], plan['controls']['fine'], order=8)
            comparison['control_convention'] = 'matched-timestep controls' if rate == 'fine' else 'same converged dt128 control for both responses'
            attempts.append(dict(pair=[previous, rate], comparison=comparison))
            report['convergence_attempts'] = attempts
            maximum = max(f['error_over_temporal_std'] for group in comparison['groups'].values()
                          for f in group['fields'].values())
            passed = (comparison['matched_frames'] == 97 and comparison['positive_sampled_rho_T'] and
                      np.isfinite(maximum) and maximum <= plan['dynamic_error_limit'] and
                      max(comparison['energy_relative_errors'].values()) <= plan['energy_error_limit'])
            if passed:
                report.update(status='accepted', max_dynamic_error=maximum, selected_reference_rate=rate,
                              data=export_case(root, case, root / 'runs' / (name + '_' + rate + '/work'), comparison))
                break
        if report['status'] != 'accepted':
            raise ValueError('Timestep convergence, coverage, or Q8 positivity gate failed after refinement')
    except Exception as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        write(root / 'cases' / (name + '.json'), report)
    return report


def finalize(root):
    plan = json.loads((root / 'plan.json').read_text())
    records = []
    for case in plan['cases']:
        path = root / 'cases' / (case['case_id'] + '.json')
        result = json.loads(path.read_text()) if path.exists() else dict(status='missing')
        records.append(dict(case, result=result))
    failed = [c['case_id'] for c in records if c['result']['status'] != 'accepted']
    if failed:
        write(root / 'incomplete.json', dict(ready=False, failed_or_missing=failed, cases=records))
        raise ValueError('Ensemble incomplete: ' + ', '.join(failed))
    if digest(root / 'geometry.h5') != plan['geometry_sha256']:
        raise ValueError('Geometry changed')
    with h5py.File(root / 'geometry.h5') as f:
        geometry = NativeGeometry(**{field.name: f[field.name][()] for field in fields(NativeGeometry)}).validate()
    times = np.asarray(records[0]['result']['data']['time_seconds'])
    mean = np.zeros((len(geometry.dof_ids), 7)); count = 0
    for case in records:
        data = case['result']['data']; path = root / data['path']
        if digest(path) != data['sha256']:
            raise ValueError('Trajectory changed: ' + case['case_id'])
        np.testing.assert_array_equal(data['time_seconds'], times)
        if case['split'] == 'train':
            with h5py.File(path) as f:
                for q in f['native_state']:
                    count += 1; mean += (q - mean) / count
    variance = np.zeros(7)
    for case in records:
        if case['split'] != 'train':
            continue
        with h5py.File(root / case['result']['data']['path']) as f:
            for q in f['native_state']:
                v = np.einsum('eql,elc->eqc', geometry.basis, (q - mean)[geometry.dof_map])
                variance += np.einsum('eqc,eq->c', v*v, geometry.volume_weights) / geometry.volume_weights.sum()
    scale = np.sqrt(variance / count)
    if not np.all(np.isfinite(scale) & (scale > 0)):
        raise ValueError('Invalid training-only normalization')
    normalization = root / 'normalization.npz'
    with normalization.open('xb') as f:
        np.savez(f, mean=mean, scale=scale, training_case_ids=np.array([c['case_id'] for c in records if c['split'] == 'train']),
                 training_frames=count, time_seconds=times)
    manifest = dict(schema='mastu-euler-dataset-v1', ready_for_euler_comparison=True,
        scientific_model_results_available=False, synthetic=False, fields=NAMES,
        plan_sha256=digest(root / 'plan.json'), split_counts=plan['split_counts'], cases=records,
        geometry=dict(path='geometry.h5', sha256=digest(root / 'geometry.h5')),
        normalization=dict(path='normalization.npz', sha256=digest(normalization), fitted_split='train', frames=count),
        observation_interval_seconds=float(times[1] - times[0]), time_basis='one saved-frame interval equals one pseudotime unit',
        reference_solver='unmodified native JOREK model303; Euler is the surrogate integrator',
        families=['opsplit', 'ar', 'node'], limits=plan['limits'], nonlinear_term_labels_required=False)
    write(root / 'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run-case', 'finalize'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--index', type=int)
    args = parser.parse_args(); root = args.root.resolve()
    if args.action == 'prepare':
        prepare(root)
    elif args.action == 'run-case':
        if args.index is None:
            parser.error('run-case requires --index')
        run_case(root, args.index)
    else:
        finalize(root)
