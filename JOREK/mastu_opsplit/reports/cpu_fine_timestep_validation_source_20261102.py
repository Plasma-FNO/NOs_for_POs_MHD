"""Compare 48/96-step baseline and seeded response at the same physical time."""
import gc
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from mastu_opsplit.common import sha256_file
from mastu_opsplit.restart import diagnose_restart, read_restart
from mastu_opsplit.numerical import evaluate_native
from mastu_opsplit.runner import classify


def main():
    output = ROOT/'reports/cpu_fine_timestep_validation_20261102.json'
    assert not output.exists()
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'script_sha256': sha256_file(__file__),
        'scope': 'Short fine-mesh temporal comparison; older 96-step trajectories have a known intermediate-file MPI race and remain diagnostic only',
        'gate_A_passed': False, 'physical_acceptance': False, 'runs': {}, 'comparisons': {}}
    fields = {}
    geometry = None
    for family, older in (('baseline', 'fine_mpi4_baseline96'), ('perturbed', 'fine_perturbed96')):
        paths = []
        for steps, name in ((48, 'io_'+family+'48'), (96, older)):
            run = ROOT/f'runs/cpu_{name}_20261102'
            result = json.loads((run/'result.json').read_text())
            assert result['status'] == 'completed', result
            text = '\n'.join((run/f'attempt_1/{name}.log').read_text() for name in ('stdout', 'stderr'))
            status = classify(0, False, text)
            assert status == ('completed' if steps == 48 else 'hdf5_error'), status
            path = run/'attempt_1/work/jorek_restart.h5'
            current_geometry, state, metadata = read_restart(path, quadrature_order=8)
            assert metadata['steps'] == steps
            if geometry is None:
                geometry = current_geometry
            else:
                for key in ('dof_map', 'basis', 'basis_grad', 'quadrature_rz', 'poloidal_weights', 'boundary_labels'):
                    assert np.array_equal(getattr(geometry, key), getattr(current_geometry, key)), key
            del current_geometry
            key = family+str(steps)
            fields[key] = evaluate_native(state, geometry)
            report['runs'][key] = {'job_id': result['scheduler_job_id'], 'status': status,
                                  'restart_sha256': sha256_file(path), 'metadata': metadata}
            paths.append(path)
            del state
            gc.collect()
        diagnostic = diagnose_restart(paths[0], paths[1], quadrature_order=8)
        errors = {key: field['relative_l2_drift'] for key, field in diagnostic['fields'].items()}
        integrated = {key: abs(value['relative_change']) for key, value in diagnostic['moments_code_units'].items()}
        report['comparisons'][family] = {'full_field_relative_l2': errors,
            'maximum_full_field_relative_l2': max(value for value in errors.values() if value is not None),
            'integrated_density_pressure_relative_difference': integrated,
            'full_field_one_percent_target_met': all(value is not None and value <= .01 for value in errors.values()),
            'integrated_two_percent_target_met': all(value <= .02 for value in integrated.values()),
            'positive_density_temperature_at_sampled_quadrature': diagnostic['positive_at_sampled_quadrature']}
    times = [value['metadata']['time_seconds'] for value in report['runs'].values()]
    assert max(times)-min(times) < 1e-15
    weights = geometry.volume_weights
    perturbations = {}
    for index, name in ((4, 'density'), (5, 'temperature'), (None, 'pressure')):
        differences = {}
        for steps in (48, 96):
            a, b = fields['perturbed'+str(steps)], fields['baseline'+str(steps)]
            differences[steps] = a[..., index]-b[..., index] if index is not None else a[..., 4]*a[..., 5]-b[..., 4]*b[..., 5]
        norm = np.sum(weights*differences[96]**2)
        assert norm > 0
        perturbations[name] = float(np.sqrt(np.sum(weights*(differences[48]-differences[96])**2)/norm))
    report.update(time_s=times[0], perturbation_relative_l2_differences=perturbations,
                  perturbation_one_percent_target_met=all(value <= .01 for value in perturbations.values()))
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'comparisons': report['comparisons'], 'perturbation_relative_l2_differences': perturbations}, indent=2))


if __name__ == '__main__':
    main()
