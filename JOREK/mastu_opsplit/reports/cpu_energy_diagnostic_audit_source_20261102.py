"""Endpoint evidence for the model303/native energy-diagnostic mismatch."""
import gc
import json
import re
from datetime import datetime, timezone
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mastu_opsplit.common import sha256_file
from mastu_opsplit.restart import read_restart
from mastu_opsplit.numerical import evaluate_native
from cpu_short_pilot_validation_source_20261102 import numeric_rows


def resistive_integrand(temperature, current, radius):
    # Pinned model303, T_0=.018, eta=1e-5, T_1=1e-4 and correction coefficients=.5,.5.
    assert np.isfinite([temperature, current, radius]).all()
    assert (temperature > 0).all() and (radius > 0).all()
    corrected = temperature.copy()
    mask = temperature < 1e-4
    corrected[mask] = .5e-4 + .5e-4 * np.exp((temperature[mask] - 1e-4)/.5e-4)
    return 1e-5 * (corrected/.018)**(-1.5) * (current/radius)**2


def main():
    output = ROOT / 'reports/cpu_energy_diagnostic_audit_20261102.json'
    assert not output.exists()
    test = np.array([.018])
    np.testing.assert_allclose(resistive_integrand(test, np.ones(1), np.ones(1)), 1e-5)
    source = Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/src/jorek/src')
    report = {'created_utc': datetime.now(timezone.utc).isoformat(),
        'script_sha256': sha256_file(__file__),
        'source_hashes': {str(path.relative_to(source)): sha256_file(path) for path in (
            source / 'models/model303/mod_elt_matrix_fft.f90', source / 'models/mod_plasma_functions.f90',
            source / 'models/corr_neg_include.f90', source / 'diagnostics/mod_integrals3D.f90',
            source / 'communication/live_data.f90')},
        'findings': [
            'Pinned model303 evolves induction with eta(T); eta_ohmic=0 disables thermal Ohmic heating, not resistive magnetic loss.',
            'Native J2_tot uses eta_T_ohm, so native ohmic_tot_t is zero here. The live-data balance consequently omits resistive loss.',
            'Generic diagnostics apply eta_coul_log_dep=true to magnetic source power; legacy model303 element assembly does not use this flag.',
            'Both diagnostic laws need reconciliation before native energy-conservation columns can approve physical acceptance.'
        ],
        'scope': 'Endpoint model303 resistive-loss estimate only, not an independent complete energy balance or a validated source-power correction',
        'physical_acceptance': False, 'gate_A_passed': False, 'runs': {}}
    for name in ('cpu_fine_mpi4_baseline96', 'cpu_fine_perturbed96'):
        work = ROOT / f'runs/{name}_20261102/attempt_1/work'
        path = work / 'jorek_restart.h5'
        text = (work / 'input').read_text()
        for key, expected in {'T_0': .018, 'T_1': 1e-4, 'eta': 1e-5, 'eta_ohmic': 0.,
                              'T_min_neg': -1e12, 'T_min': 1e-20}.items():
            match = re.search(r'^\s*' + key + r'\s*=\s*([^!\s]+)', text, re.I | re.M)
            # eta_ohmic is the verified zero default when absent from the snapshot.
            assert match or key == 'eta_ohmic', key
            actual = float(match[1].lower().replace('d', 'e')) if match else 0.
            assert actual == expected, (key, actual)
        estimates = {}
        for order in (4, 8):
            geometry, state, metadata = read_restart(path, quadrature_order=order)
            values = evaluate_native(state, geometry)
            with h5py.File(path) as stream:
                unit = np.asarray(stream['sqrt_mu0_rho0'][()]).item()
            estimates[str(order)] = float(np.sum(geometry.volume_weights * resistive_integrand(
                values[..., 5], values[..., 2], geometry.quadrature_rz[..., 0])) / (4e-7 * np.pi * unit))
            del geometry, state, values
            gc.collect()
        text = (work / 'macroscopic_vars.dat').read_text()
        rows = numeric_rows(text, 'mag_energy_src')
        report['runs'][name] = {'restart_sha256': sha256_file(path),
            'time_s': metadata['time_seconds'], 'resistive_loss_W_by_quadrature_order': estimates,
            'q4_q8_relative_difference': abs(estimates['4'] - estimates['8']) / estimates['8'],
            'native_generic_magnetic_source_W_at_final_time': rows[max(rows)][0],
            'source_power_limitation': 'Generic source power uses a different resistivity law; it is not a matching sink/source pair.'}
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(report['runs'], indent=2))


if __name__ == '__main__':
    main()
