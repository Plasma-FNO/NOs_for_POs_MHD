"""Quantify constitutive temporal-mass differences on dense pilot snapshots."""
import gc
import json
from datetime import datetime, timezone
from pathlib import Path
import re
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from mastu_opsplit.common import sha256_file
from mastu_opsplit.restart import read_restart
from mastu_opsplit.numerical import evaluate_native, evaluate_native_grad
from cpu_short_pilot_validation_source_20261102 import numeric_rows


def main():
    output = ROOT/'reports/cpu_temporal_mass_audit_20261102.json'
    assert not output.exists()
    source = Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/src/jorek/src')
    assembly = (source/'models/model303/mod_elt_matrix_fft.f90').read_text()
    assert re.search(r'amat_77\s*=\s*v\s*\*\s*Vpar\s*\*\s*r0_corr\s*\*\s*F0\*\*2', assembly, re.I)
    run = ROOT/'runs/cpu_refine_stable_parity8_20261102'
    assert json.loads((run/'result.json').read_text())['status'] == 'completed'
    work = run/'attempt_1/work'
    fields = ('x', 'vertex', 'size', 'index', 'boundary')
    coefficients, metadata = {}, {}
    geometry, _, _ = read_restart(work/'jorek000000.h5', quadrature_order=4)
    with h5py.File(work/'jorek000000.h5') as f:
        reference_geometry = {name: f[name][()] for name in fields}
    for step in range(9):
        path = work/f'jorek{step:06d}.h5'
        with h5py.File(path) as f:
            assert all(np.array_equal(f[name][()], reference_geometry[name]) for name in fields)
        current_geometry, coefficients[step], metadata[step] = read_restart(path, quadrature_order=2)
        assert np.array_equal(current_geometry.dof_map, geometry.dof_map)
        del current_geometry
    text = (work/'macroscopic_vars.dat').read_text()
    energy = numeric_rows(text, 'energy_conservation')
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'script_sha256': sha256_file(__file__),
        'native_source_sha256': {str(p.relative_to(source)): sha256_file(p) for p in
            [source/'models/model303/mod_elt_matrix_fft.f90', source/'models/corr_neg_include.f90',
             source/'diagnostics/mod_integrals3D.f90']},
        'scope': 'Instantaneous constitutive mass differences applied to centered physical-time derivatives on pilot states. These are partial contributions, not an exact discrete energy identity.',
        'native_mass_facts': ['Parallel diagonal time coefficient uses rho_corrected*F0^2/R, while generic kinetic energy uses rho*(F0^2+|grad(psi)|^2)/R^2.',
            'Perpendicular gradient temporal mass and pressure-temperature diagonal use corrected density.',
            'Negative rho_min_neg selects the rho_1-based smooth correction; positive raw densities below rho_1 can activate it.'],
        'unaccounted': 'Discrete temporal/spatial linearization, stabilization, source work and modified native boundary rows still need native assembly evidence.',
        'full_energy_acceptance': False, 'scientific_gates_passed': [], 'frames': []}
    weights, radius = geometry.volume_weights, geometry.quadrature_rz[..., 0]
    mu0 = 4e-7*np.pi
    for step in range(2, 8):
        state = coefficients[step]
        dt = metadata[step+1]['time_seconds']-metadata[step-1]['time_seconds']
        assert dt > 0
        derivative = (coefficients[step+1]-coefficients[step-1])/dt
        value, gradient = evaluate_native(state, geometry), evaluate_native_grad(state, geometry)
        rate, gradient_rate = evaluate_native(derivative, geometry), evaluate_native_grad(derivative, geometry)
        rho, temperature, parallel = value[..., 4], value[..., 5], value[..., 6]
        assert (rho > 0).all() and (temperature > 0).all()
        corrected = rho.copy()
        active = rho < .01
        corrected[active] = .005+.005*np.exp((rho[active]-.01)/.005)
        delta = rho-corrected
        bp2 = np.sum(gradient[..., 0, :]**2, axis=-1)/radius**2
        bb2 = bp2+.64**2/radius**2
        def integrate(value):
            return float(np.sum(weights*value)/mu0)
        bpol_power = integrate(rho*bp2*parallel*rate[..., 6])
        parallel_floor = integrate(delta*.64**2/radius**2*parallel*rate[..., 6])
        pressure_floor = integrate(delta*rate[..., 5]/(5/3-1))
        perpendicular_floor = integrate(delta*radius**2*np.sum(gradient[..., 1, :]*gradient_rate[..., 1, :], axis=-1))
        independent_parallel_mass = integrate((rho*bb2-corrected*.64**2/radius**2)*parallel*rate[..., 6])
        np.testing.assert_allclose(bpol_power+parallel_floor, independent_parallel_mass, rtol=1e-12, atol=1e-6)
        path = work/f'jorek{step:06d}.h5'
        with h5py.File(path) as f:
            native_kinetic = float(np.asarray(f['kin_par_tot_t'][()]).ravel()[-1])
            time_code = np.asarray(f['t_now'][()]).item()
        independent_kinetic = .5*integrate(rho*bb2*parallel**2)
        np.testing.assert_allclose(independent_kinetic, native_kinetic, rtol=1e-6, atol=1e-8)
        changed = np.abs(delta) > 64*np.finfo(float).eps
        report['frames'].append({'step': step, 'time_s': metadata[step]['time_seconds'],
            'restart_sha256': sha256_file(path), 'native_energy_minus_dE_dt_rhs_W': energy[time_code],
            'density_correction_branch_volume_fraction': float(weights[active].sum()/weights.sum()),
            'nonroundoff_density_mass_change_volume_fraction': float(weights[changed].sum()/weights.sum()),
            'maximum_density_coefficient_correction': float(np.max(np.abs(delta))),
            'parallel_poloidal_mass_difference_W': bpol_power,
            'parallel_density_correction_mass_difference_W': parallel_floor,
            'pressure_density_correction_mass_difference_W': pressure_floor,
            'perpendicular_density_correction_mass_difference_W': perpendicular_floor,
            'sum_of_partial_mass_difference_powers_W': bpol_power+parallel_floor+pressure_floor+perpendicular_floor,
            'independent_parallel_kinetic_J': independent_kinetic,
            'native_parallel_kinetic_J': native_kinetic})
        del value, gradient, rate, gradient_rate
        gc.collect()
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'density_mass_change_volume_fraction': [min(f['nonroundoff_density_mass_change_volume_fraction'] for f in report['frames']), max(f['nonroundoff_density_mass_change_volume_fraction'] for f in report['frames'])],
        'partial_mass_power_range_W': [min(f['sum_of_partial_mass_difference_powers_W'] for f in report['frames']), max(f['sum_of_partial_mass_difference_powers_W'] for f in report['frames'])]}, indent=2))


if __name__ == '__main__':
    main()
