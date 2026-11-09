"""Audit corrected energy diagnostics and exact unchanged native state."""
import gc
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from mastu_opsplit.common import sha256_file
from mastu_opsplit.restart import read_restart
from mastu_opsplit.numerical import evaluate_native
from cpu_short_pilot_validation_source_20261102 import numeric_rows, native_balances, relative_residual
from cpu_energy_diagnostic_audit_source_20261102 import resistive_integrand


def energy_metrics(text):
    names = ['energy_conservation', 'bnd_fluxes', 'heatingpower', 'dissipative_terms',
             'mag_energy_src', 'mag_energy_balance']
    rows = {name: numeric_rows(text, name) for name in names}
    times = sorted(set.intersection(*(set(value) for value in rows.values())))
    times = [time for time in times if any(rows['energy_conservation'][time])]
    assert len(times) >= 3
    lhs = np.array([rows['energy_conservation'][t][0] for t in times])
    rhs = np.array([rows['energy_conservation'][t][1] for t in times])
    terms = np.array([rows['bnd_fluxes'][t]+[-rows['heatingpower'][t][0]]
                      +[rows['dissipative_terms'][t][0], -rows['mag_energy_src'][t][0],
                        -rows['mag_energy_balance'][t][2]]
                      +rows['dissipative_terms'][t][2:4] for t in times])
    # This case has zero Ohmic/viscous thermal heating, radiation and neutral terms.
    np.testing.assert_allclose(terms.sum(axis=1), rhs, rtol=2e-5, atol=2.)
    scale = sum(np.linalg.norm(terms[:, i]) for i in range(terms.shape[1]))
    mag = np.array([rows['mag_energy_balance'][t] for t in times])
    mag_scale = sum(np.linalg.norm(mag[:, i]) for i in range(1, 5))
    assert scale > 0 and mag_scale > 0
    return {'samples': len(times), 'energy_relative_l2_net_mismatch': relative_residual(lhs, rhs),
            'energy_relative_l2_to_sum_of_component_norms': float(np.linalg.norm(lhs-rhs)/scale),
            'energy_max_absolute_mismatch_W': float(np.max(np.abs(lhs-rhs))),
            'magnetic_relative_l2_net_mismatch': relative_residual(mag[:, 0], mag[:, 5]),
            'magnetic_relative_l2_to_sum_of_component_norms': float(np.linalg.norm(mag[:, 0]-mag[:, 5])/mag_scale),
            'physical_balance_acceptance': False,
            'scope': 'Native printed diagnostics with reconstructed source/sink columns; component scaling does not replace net-balance reporting.'}


def main():
    output = ROOT/'reports/cpu_energy_balance_validation_20261102.json'
    assert not output.exists()
    report = {'created_utc': datetime.now(timezone.utc).isoformat(),
        'script_sha256': sha256_file(__file__), 'scope': 'Model303 magnetic diagnostic correction only; no physical gate promotion',
        'gate_A_passed': False, 'full_energy_acceptance': False, 'runs': {}}
    for family in ['baseline', 'perturbed']:
        run = ROOT/f'runs/cpu_energy_complete_{family}48_20261102'
        reference = ROOT/f'runs/cpu_io_{family}48_20261102'
        result = json.loads((run/'result.json').read_text())
        assert result['status'] == 'completed'
        assert (run/'attempt_1/stderr.log').stat().st_size == 0
        path, oldpath = [p/'attempt_1/work/jorek_restart.h5' for p in [run, reference]]
        with h5py.File(path) as current, h5py.File(oldpath) as old:
            exact = {key: bool(np.array_equal(current[key][()], old[key][()]))
                     for key in ['values', 'x', 'vertex', 'size', 'index', 'boundary', 'xtime', 'energies']}
            assert all(exact.values()), exact
            loss = float(np.asarray(current['ohmic_tot_t'][()]).ravel()[-1])
            unit = np.asarray(current['sqrt_mu0_rho0'][()]).item()
        geometry, state, metadata = read_restart(path, quadrature_order=8)
        values = evaluate_native(state, geometry)
        assert (values[..., 4:6] > 0).all()
        independent_loss = float(np.sum(geometry.volume_weights * resistive_integrand(
            values[..., 5], values[..., 2], geometry.quadrature_rz[..., 0])) / (4e-7*np.pi*unit))
        difference = abs(loss-independent_loss)/independent_loss
        assert difference < 1e-6, (loss, independent_loss)
        text = (run/'attempt_1/work/macroscopic_vars.dat').read_text()
        oldtext = (reference/'attempt_1/work/macroscopic_vars.dat').read_text()
        report['runs'][family] = {'job_id': result['scheduler_job_id'],
            'restart_sha256': sha256_file(path), 'reference_sha256': sha256_file(oldpath),
            'native_arrays_exact': exact, 'metadata': metadata,
            'positive_density_temperature_at_sampled_quadrature': True,
            'native_final_resistive_loss_W': loss, 'independent_q8_resistive_loss_W': independent_loss,
            'resistive_loss_relative_difference': difference,
            'previous_diagnostics': native_balances(oldtext), 'corrected_diagnostics': energy_metrics(text),
            'corrected_particle_diagnostics': native_balances(text)}
        del geometry, state, values
        gc.collect()
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({name: {'exact_state': all(run['native_arrays_exact'].values()),
                           'independent_loss_relative_difference': run['resistive_loss_relative_difference'],
                           'energy': run['corrected_diagnostics']} for name, run in report['runs'].items()}, indent=2))


if __name__ == '__main__':
    main()
