"""Run one frozen refined case and verify native output and initialization."""
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np


def audit_restart(path, steps, final_time):
    from mastu_opsplit.common import sha256_file
    from mastu_opsplit.numerical import evaluate_native
    from mastu_opsplit.restart import read_restart
    from mastu_opsplit.runner import restart_health

    health = restart_health(path, expected_steps=steps, final_time_s=final_time)
    if health['steps'] != steps:
        raise ValueError('native step count differs from the frozen run')
    datasets = []
    with h5py.File(path) as stream:
        def check(name, item):
            if isinstance(item, h5py.Dataset):
                values = item[()]
                if np.issubdtype(item.dtype, np.number) and not np.isfinite(values).all():
                    raise ValueError('nonfinite restart dataset: ' + name)
                datasets.append(name)
        stream.visititems(check)
    geometry, state, metadata = read_restart(path, quadrature_order=8)
    fields = evaluate_native(state, geometry)
    if not np.isfinite(fields).all() or (fields[..., 4:6] <= 0).any():
        raise ValueError('nonfinite fields or nonpositive sampled density/temperature')
    if metadata['negative_jacobian_samples']:
        raise ValueError('negative sampled Jacobian')
    result = {'health': health, 'metadata': metadata, 'readable_finite_datasets': len(datasets),
              'minimum_density': float(fields[..., 4].min()),
              'minimum_temperature': float(fields[..., 5].min()), 'restart_sha256': sha256_file(path)}
    del geometry, state, fields
    gc.collect()
    return result


def check_initializer(path, reference, amplitudes, axis, boundary_flux):
    with h5py.File(path) as current, h5py.File(reference) as original:
        for key in ['x', 'vertex', 'size', 'index', 'boundary', 'constrained', 'n_sons']:
            if not np.array_equal(current[key][()], original[key][()]):
                raise ValueError('initialized geometry changed: ' + key)
        actual, baseline = current['values'][()], original['values'][()]
        boundary = original['boundary'][()]
        vertex = original['vertex'][()].T - 1
        sizes = original['size'][()].transpose(2, 1, 0)
    if not np.array_equal(actual[[0, 1, 2, 3, 6]], baseline[[0, 1, 2, 3, 6]]):
        raise ValueError('unselected initialized fields changed')
    if not np.array_equal(actual[:, :, :, boundary != 0], baseline[:, :, :, boundary != 0]):
        raise ValueError('initialized boundary coefficients changed')
    relative = actual[4:6, 0, 0] / baseline[4:6, 0, 0] - 1
    if (relative < -1e-12).any() or (relative > np.asarray(amplitudes)[:, None] + 1e-12).any():
        raise ValueError('initialized nodal amplitudes exceed the frozen positive perturbation')
    changed = np.any(np.abs((actual - baseline)[4:6, :, 0].transpose(2, 1, 0)[vertex]) > 1e-14,
                     axis=(1, 2, 3))
    if np.any(boundary[vertex[changed]]):
        raise ValueError('initializer support touches a boundary element')
    maximum = None
    if changed.any():
        jets = baseline[0, :, 0].T[vertex] * sizes
        control = np.stack([jets[:, :, 0], jets[:, :, 0] + jets[:, :, 1],
                            jets[:, :, 0] + jets[:, :, 2], jets.sum(axis=2)], axis=2)
        maximum = max(float(((control[changed] - a) / (b - a)).max())
                      for a in [axis - 5e-6, axis + 5e-6]
                      for b in [boundary_flux - 5e-6, boundary_flux + 5e-6])
        if maximum >= .85:
            raise ValueError('initializer support approaches the separatrix')
    elif any(amplitudes):
        raise ValueError('requested perturbation has no changed elements')
    if not any(amplitudes) and not np.array_equal(actual, baseline):
        raise ValueError('unperturbed initial fields differ from the validated probe')
    return {'unselected_fields_exact': True, 'boundary_coefficients_exact': True,
            'affected_elements': int(changed.sum()), 'affected_elements_touch_boundary': False,
            'maximum_psi_n_control_hull_with_rounding_interval': maximum,
            'maximum_nodal_relative_perturbations': relative.max(axis=1).tolist()}


def main():
    base, root, label = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    sys.path.insert(0, str(root / 'src'))
    from mastu_opsplit.common import sha256_file
    from mastu_opsplit.runner import classify

    record = json.loads((base / 'manifest.json').read_text())
    for name, expected in record['input_hashes'].items():
        if sha256_file(base / name) != expected:
            raise ValueError('frozen input changed: ' + name)
    work = base / label
    start = time.monotonic()
    with (work / 'input').open() as inp, (work / 'stdout.log').open('x') as out, (work / 'stderr.log').open('x') as err:
        process = subprocess.run(['timeout', '--signal=TERM', '--kill-after=10s', '5400s',
                                  'mpirun', '--bind-to', 'none', '-n', '4', str(base / 'jorek.exe')],
                                 cwd=work, stdin=inp, stdout=out, stderr=err)
    text = (work / 'stdout.log').read_text() + (work / 'stderr.log').read_text()
    result = {'label': label, 'returncode': process.returncode,
              'status': classify(process.returncode, process.returncode == 124, text),
              'process_wall_seconds': time.monotonic() - start, 'scientific_gates_passed': []}
    try:
        if result['status'] != 'completed':
            raise ValueError(result['status'])
        if 'Trying to open response file' in text or 'MUMPS failure during' in text:
            raise ValueError('unexpected fixed-boundary response load or MUMPS error')
        if (work / 'stderr.log').stat().st_size:
            raise ValueError('nonempty native stderr requires review')
        result['initial'] = audit_restart(work / 'jorek000000.h5', 0, 0.)
        result['final'] = audit_restart(work / 'jorek_restart.h5', 48, record['final_time_s'])
        result['initializer'] = check_initializer(work / 'jorek000000.h5', base / 'refined_reference_initial.h5',
                                                  record['runs'][label]['amplitudes'],
                                                  record['reference_psi_axis'], record['reference_psi_boundary'])
        result['workspace_retries'] = text.count('MUMPS workspace retry:')
    except Exception as error:
        result.update(status='failed_validation', error=str(error))
    with (work / 'result.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result), flush=True)
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    sys.exit(main())
