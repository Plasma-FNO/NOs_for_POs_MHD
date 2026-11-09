"""One guarded allocation: ordinary/cached parity, then a refined native probe."""
import json
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mastu_opsplit.common import sha256_file
from mastu_opsplit.numerical import evaluate_native, evaluate_native_grad
from mastu_opsplit.restart import read_restart
from mastu_opsplit.runner import classify, restart_health


def rms(values, weights, gradient=False):
    square = np.sum(values * values, axis=-1) if gradient else values * values
    return np.sqrt(np.sum(weights[..., None] * square, axis=(0, 1)) / weights.sum())


def parity(path, reference, protocol, exact=False):
    with h5py.File(path) as a, h5py.File(reference) as b:
        keys = ['x', 'values', 'vertex', 'size', 'boundary', 'constrained', 'n_sons', 'xtime', 'energies']
        arrays = {key: bool(np.array_equal(a[key][()], b[key][()])) for key in keys}
    if exact:
        if not all(arrays.values()):
            raise ValueError(f'ordinary-start reference arrays changed: {arrays}')
        return {'exact_reference_arrays': arrays, 'passed': True}
    geometry, state, metadata = read_restart(path, quadrature_order=protocol['quadrature_order'])
    expected_geometry, expected, expected_metadata = read_restart(reference, quadrature_order=protocol['quadrature_order'])
    for name in ('dof_map', 'dof_ids', 'basis', 'basis_grad', 'quadrature_rz', 'poloidal_weights', 'bounds', 'boundary_labels'):
        if not np.array_equal(getattr(geometry, name), getattr(expected_geometry, name)):
            raise ValueError(f'cached-start reference geometry changed: {name}')
    if metadata['steps'] != expected_metadata['steps'] or abs(metadata['time_seconds'] - expected_metadata['time_seconds']) >= 1e-15:
        raise ValueError('cached-start reference time/steps differ')
    metrics = {}
    passed = True
    for name, evaluator, gradient in [('field', evaluate_native, False), ('gradient', evaluate_native_grad, True)]:
        current, baseline = evaluator(state, geometry), evaluator(expected, geometry)
        error = rms(current - baseline, geometry.volume_weights, gradient)
        norm = rms(baseline, geometry.volume_weights, gradient)
        limit = protocol['absolute_rms_tolerance_code_units'] + protocol[f'relative_{name}_l2_tolerance'] * norm
        checks = error <= limit
        passed &= bool(checks.all())
        metrics[name] = {'absolute_rms_errors': error.tolist(), 'reference_rms': norm.tolist(), 'tolerances': limit.tolist(), 'passed': checks.tolist()}
    if not passed:
        raise ValueError(f'cached-start physical parity failed: {metrics}')
    return {'exact_reference_arrays': arrays, 'passed': passed, 'metrics': metrics}


def main():
    base = Path(sys.argv[1])
    record = json.loads((base / 'authorization.json').read_text())
    for name, value in record['input_hashes'].items():
        if sha256_file(base / name) != value:
            raise ValueError(f'prepared recovery input changed: {name}')
    binary = base / 'source/jorek_model303'
    protocol = json.loads((base / 'parity_protocol.json').read_text())
    all_results = []
    for name, steps, timeout in [('ordinary8', 8, 600), ('cached8', 8, 600), ('refined3', 3, 5400)]:
        work = base / name
        start = time.monotonic()
        with (work / 'input').open() as inp, (work / 'stdout.log').open('w') as out, (work / 'stderr.log').open('w') as err:
            proc = subprocess.run(['timeout', '--signal=TERM', '--kill-after=10s', f'{timeout}s',
                                   'mpirun', '--bind-to', 'none', '-n', '4', str(binary)],
                                  cwd=work, stdin=inp, stdout=out, stderr=err)
        text = (work / 'stdout.log').read_text() + (work / 'stderr.log').read_text()
        result = {'stage': name, 'returncode': proc.returncode, 'status': classify(proc.returncode, proc.returncode == 124, text),
                  'wall_seconds': time.monotonic() - start, 'binary_sha256': sha256_file(binary), 'scientific_gates_passed': []}
        try:
            if result['status'] != 'completed':
                raise ValueError(result['status'])
            if 'Trying to open response file' in text:
                raise ValueError('unexpected STARWALL response load for fixed-boundary case')
            result['health'] = restart_health(work / 'jorek_restart.h5', expected_steps=steps,
                                              final_time_s=2.343188518713464e-7 if steps == 8 else 8.78695694517549e-8)
            if name != 'refined3':
                result['parity'] = parity(work / 'jorek_restart.h5', base / 'reference8.h5', protocol, exact=name == 'ordinary8')
            else:
                geometry, state, metadata = read_restart(work / 'jorek_restart.h5', quadrature_order=8)
                values = evaluate_native(state, geometry)
                if not np.isfinite(values).all() or not (values[..., 4:6] > 0).all():
                    raise ValueError('nonfinite fields or nonpositive sampled density/temperature')
                result['sampled_admissibility'] = {'quadrature_order': 8, 'finite_fields': True,
                    'minimum_density': float(values[..., 4].min()), 'minimum_temperature': float(values[..., 5].min())}
                result['metadata'] = metadata
        except Exception as error:
            result.update(status='failed_validation', error=str(error))
        (work / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        all_results.append(result)
        summary = {'status': 'running' if result['status'] == 'completed' and name != 'refined3' else result['status'],
                   'stages': all_results, 'scientific_gates_passed': []}
        (base / 'result.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        print(json.dumps(result), flush=True)
        if result['status'] != 'completed':
            return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
