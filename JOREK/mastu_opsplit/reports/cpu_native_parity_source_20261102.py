"""Apply the frozen physical field/gradient MPI and capacity parity protocol."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from mastu_opsplit.common import sha256_file
from mastu_opsplit.restart import read_restart
from mastu_opsplit.numerical import evaluate_native, evaluate_native_grad


def channel_rms(values, weights, *, gradient=False):
    square = np.sum(values*values, axis=-1) if gradient else values*values
    return np.sqrt(np.sum(weights[..., None]*square, axis=(0, 1))/weights.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    phase_path = ROOT/'reports/cpu_refinement_continuation_20261102.json'
    phase = json.loads(phase_path.read_text())
    protocol_path = Path(phase['parity_protocol'])
    assert sha256_file(protocol_path) == phase['parity_protocol_sha256']
    protocol = json.loads(protocol_path.read_text())
    run = next(run for run in phase['runs'] if run['label'] == args.label)
    output = ROOT/f'reports/cpu_{args.label}_physical_parity_20261102.json'
    assert not output.exists()
    path = Path(run['bulk_output'])/'attempt_1/work/jorek_restart.h5'
    reference = Path(run['prepared'])/'equilibrium_reference.h5'
    result = json.loads((Path(run['bulk_output'])/'result.json').read_text())
    assert result['attempts'][-1]['returncode'] == 0
    geometry, state, metadata = read_restart(path, quadrature_order=protocol['quadrature_order'])
    reference_geometry, reference_state, reference_metadata = read_restart(reference, quadrature_order=protocol['quadrature_order'])
    for name in ('dof_map', 'dof_ids', 'basis', 'basis_grad', 'quadrature_rz', 'poloidal_weights', 'bounds', 'boundary_labels'):
        assert np.array_equal(getattr(geometry, name), getattr(reference_geometry, name)), name
    assert metadata['steps'] == reference_metadata['steps']
    assert abs(metadata['time_seconds']-reference_metadata['time_seconds']) < 1e-15
    np.testing.assert_allclose(channel_rms(np.ones((1, 1, 7)), np.ones((1, 1))), 1.)
    np.testing.assert_allclose(channel_rms(np.ones((1, 1, 7, 2)), np.ones((1, 1)), gradient=True), np.sqrt(2))
    values, expected = evaluate_native(state, geometry), evaluate_native(reference_state, geometry)
    positive = bool((values[..., 4:6] > 0).all())
    fields = {}
    all_passed = positive
    for name, current, baseline, gradient in [('field', values, expected, False),
        ('gradient', evaluate_native_grad(state, geometry), evaluate_native_grad(reference_state, geometry), True)]:
        error = channel_rms(current-baseline, geometry.volume_weights, gradient=gradient)
        norm = channel_rms(baseline, geometry.volume_weights, gradient=gradient)
        tolerance = protocol['absolute_rms_tolerance_code_units']+protocol[f'relative_{name}_l2_tolerance']*norm
        passed = error <= tolerance
        all_passed &= bool(passed.all())
        fields[name] = {key: {'absolute_rms_error': float(error[i]), 'reference_rms': float(norm[i]),
            'relative_l2_error': float(error[i]/norm[i]) if norm[i] else None,
            'absolute_rms_tolerance': float(tolerance[i]), 'passed': bool(passed[i])}
            for i, key in enumerate(('psi', 'u', 'zj', 'w', 'rho', 'T', 'Vpar'))}
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'script_sha256': sha256_file(__file__),
        'protocol_sha256': sha256_file(protocol_path), 'label': args.label, 'job_id': run['job_id'],
        'original_runner_status': result['status'], 'original_result_preserved': True,
        'restart_sha256': sha256_file(path), 'reference_sha256': sha256_file(reference),
        'metadata': metadata, 'raw_maximum_canonical_coefficient_difference': float(np.abs(state-reference_state).max()),
        'positive_density_temperature_at_sampled_quadrature': positive,
        'physical_parity_targets_met': all_passed, 'metrics': fields, 'scientific_gates_passed': []}
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    run.update(physical_parity_targets_met=all_passed, native_parity_validation=str(output),
               native_parity_validation_sha256=sha256_file(output))
    phase_path.write_text(json.dumps(phase, indent=2)+'\n')
    print(json.dumps({'label': args.label, 'physical_parity_targets_met': all_passed,
        'maximum_relative_field_error': max(value['relative_l2_error'] or 0 for value in fields['field'].values()),
        'maximum_relative_gradient_error': max(value['relative_l2_error'] or 0 for value in fields['gradient'].values()),
        'original_runner_status': result['status']}, indent=2))


if __name__ == '__main__':
    main()
