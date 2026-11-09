"""Audit completed short CPU trajectories; no scientific gate is passed here."""
import gc
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mastu_opsplit.common import sha256_file
from mastu_opsplit.restart import read_restart
from mastu_opsplit.numerical import evaluate_native, evaluate_native_grad, model303_advection_blocks
from mastu_opsplit.runner import classify, restart_health


def numeric_rows(text, name):
    rows = {}
    for line in text.splitlines():
        if line.startswith('@' + name + ':'):
            try:
                row = [float(x) for x in line.split(':', 1)[1].split()]
            except ValueError:
                continue
            assert row and np.isfinite(row).all(), name
            rows[row[0]] = row[1:]
    return rows


def relative_residual(left, right):
    left, right = np.asarray(left), np.asarray(right)
    scale = max(float(np.linalg.norm(left)), float(np.linalg.norm(right)))
    assert scale > 0
    return float(np.linalg.norm(left - right) / scale)


def native_balances(text):
    energy = numeric_rows(text, 'energy_conservation')
    # The first printed derivative is a placeholder, then overwritten at the same time.
    energy = np.array([row for row in energy.values() if any(row)])
    rates = numeric_rows(text, 'dparticles_dt')
    sources = numeric_rows(text, 'particlesource')
    fluxes = numeric_rows(text, 'bnd_particle_fluxes')
    times = sorted(rates.keys() & sources.keys() & fluxes.keys())
    assert len(times) >= 3 and len(energy) >= 3
    assert all(len(sources[t]) == 3 and len(fluxes[t]) == 5 and len(rates[t]) == 3 for t in times)
    actual = np.array([rates[t][0] for t in times])
    expected = np.array([sources[t][0] - sum(fluxes[t]) for t in times])
    return {'scope': 'Native printed diagnostics, source communication/live_data.f90 and diagnostics/mod_integrals3D.f90; not independent weak residuals',
            'energy_samples': len(energy),
            'energy_relative_l2_mismatch': relative_residual(energy[:, 0], energy[:, 1]),
            'energy_max_absolute_mismatch_W': float(np.max(np.abs(energy[:, 0] - energy[:, 1]))),
            'particle_samples': len(times),
            'particle_relative_l2_mismatch': relative_residual(actual, expected),
            'particle_max_absolute_mismatch_per_s': float(np.max(np.abs(actual - expected))),
            'particle_formula': 'dN/dt = total source - sum(outward boundary particle fluxes)',
            'independent_balance_acceptance': False}


def audit_run(name):
    run = ROOT / 'runs' / (name + '_20261102')
    result = json.loads((run / 'result.json').read_text())
    work = run / 'attempt_1/work'
    restart = work / 'jorek_restart.h5'
    stdout, stderr = [(run / f'attempt_1/{s}.log').read_text() for s in ('stdout', 'stderr')]
    assert result['status'] == 'completed'
    datasets = []
    with h5py.File(restart) as f:
        def check(key, value):
            if isinstance(value, h5py.Dataset):
                array = value[()]
                assert not np.issubdtype(array.dtype, np.number) or np.isfinite(array).all(), key
                datasets.append(key)
        f.visititems(check)
    return {'job_id': result['scheduler_job_id'], 'historical_status': result['status'],
            'current_log_classification': classify(0, False, stdout + stderr),
            'health': restart_health(restart, expected_steps=96, final_time_s=1.4059131112280784e-6),
            'readable_finite_datasets': len(datasets), 'stderr_sha256': sha256_file(run / 'attempt_1/stderr.log'),
            'balances': native_balances((work / 'macroscopic_vars.dat').read_text())}


def main():
    output = ROOT / 'reports/cpu_short_pilot_validation_20261102.json'
    assert not output.exists(), output
    assert numeric_rows('@a: %"header"\n@a: 1 0 0\n@a: 1 2 3', 'a') == {1.: [2., 3.]}
    assert relative_residual([1., 2.], [1., 2.]) == 0.
    assert relative_residual([1., 2.], [2., 4.]) == .5
    paths = {
        'baseline_initial': 'cpu_fine_mpi4_equilibrium',
        'perturbed_initial': 'cpu_fine_initializer_mpi4',
        'baseline_final': 'cpu_fine_mpi4_baseline96',
        'perturbed_final': 'cpu_fine_perturbed96',
    }
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': sha256_file(__file__),
              'scope': 'Short 1.406 microsecond paired pilot diagnostics; no accepted training trajectories',
              'runs': {name: audit_run(name) for name in ('cpu_fine_mpi4_baseline96', 'cpu_fine_perturbed96')},
              'gate_A_passed': False, 'constraints_checked': False, 'mesh_convergence_checked': False,
              'states': {}, 'perturbation': {}}
    geometry = None
    fields = {}
    for label, name in paths.items():
        path = ROOT / f'runs/{name}_20261102/attempt_1/work/jorek_restart.h5'
        current_geometry, state, metadata = read_restart(path, quadrature_order=8)
        if geometry is None:
            geometry = current_geometry
        else:
            for attribute in ('dof_map', 'basis', 'basis_grad', 'quadrature_rz', 'poloidal_weights', 'boundary_labels'):
                assert np.array_equal(getattr(geometry, attribute), getattr(current_geometry, attribute)), attribute
        del current_geometry
        fields[label] = values = evaluate_native(state, geometry)
        assert (values[..., 4:6] > 0).all(), label
        gradient = evaluate_native_grad(state, geometry)
        selected = model303_advection_blocks(values, gradient, geometry.quadrature_rz[..., 0], 5/3)
        weights = geometry.volume_weights
        report['states'][label] = {'restart_sha256': sha256_file(path), 'metadata': metadata,
            'sampled_min_density_temperature': values[..., 4:6].min(axis=(0, 1)).tolist(),
            'selected_density_pressure_rms_code_time_units': np.sqrt(np.sum(weights[..., None] * selected**2, axis=(0, 1))/weights.sum()).tolist(),
            'selected_block_scope': 'Source-derived local algebra diagnostic only; not validated native labels or advection/diffusion ratio'}
        del gradient, selected, state
        gc.collect()
    weights = geometry.volume_weights
    def rms(value):
        return float(np.sqrt(np.sum(weights * value**2) / weights.sum()))
    for index, name in ((4, 'density'), (5, 'temperature'), (None, 'pressure')):
        def field(label):
            return fields[label][..., index] if index is not None else fields[label][..., 4] * fields[label][..., 5]
        initial = field('perturbed_initial') - field('baseline_initial')
        final = field('perturbed_final') - field('baseline_final')
        initial_rms = rms(initial)
        assert initial_rms > 0
        report['perturbation'][name] = {'initial_rms': initial_rms, 'final_rms': rms(final),
            'final_initial_rms_ratio': rms(final)/initial_rms,
            'relative_l2_change_from_seed': rms(final-initial)/initial_rms,
            'baseline_relative_l2_drift': rms(field('baseline_final')-field('baseline_initial'))/rms(field('baseline_initial')),
            'initial_peak': float(initial.max()), 'final_peak': float(final.max())}
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'perturbation': report['perturbation'], 'runs': report['runs']}, indent=2))


if __name__ == '__main__':
    main()
