"""Source-derived viscous work on frozen states; not full boundary acceptance."""
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
from mastu_opsplit.surface import restart_boundary_quadrature
from cpu_short_pilot_validation_source_20261102 import numeric_rows


def main():
    output = ROOT/'reports/cpu_viscous_work_audit_20261102.json'
    assert not output.exists()
    source = Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/src/jorek/src')
    preset = (source/'models/preset_parameters.f90').read_text()
    assert re.search(r'normalized_velocity_profile\s*=\s*\.true\.', preset, re.I)
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'script_sha256': sha256_file(__file__),
        'surface_source_sha256': sha256_file(ROOT/'src/mastu_opsplit/surface.py'),
        'native_source_hashes': {str(path.relative_to(source)): sha256_file(path) for path in
            [source/'models/model303/mod_elt_matrix_fft.f90', source/'models/preset_parameters.f90',
             source/'diagnostics/mod_integrals3D.f90', source/'models/model303/mod_boundary_conditions.f90']},
        'scope': 'Same-time source-derived normalized-profile viscous contributions. Exterior edges are inferred from canonical topology and labels; native modified boundary rows remain unvalidated.',
        'gate_A_passed': False, 'full_energy_acceptance': False,
        'formulas': {'parallel_boundary_work': 'integral(nu_par*Vpar*grad(Vpar).n*dS)/(mu0*time_unit)',
                     'legacy_perpendicular_loss': '-integral(nu(T)*grad(u).grad(w)*dV)/(mu0*time_unit)',
                     'provisional_rhs': 'native corrected magnetic RHS - generic perpendicular loss + legacy perpendicular loss - incoming parallel viscous boundary work'},
        'runs': {}}
    for family in ['baseline', 'perturbed']:
        old = ROOT/f'runs/cpu_io_{family}48_20261102/attempt_1/work'
        current = ROOT/f'runs/cpu_energy_complete_{family}48_20261102/attempt_1/work'
        assert json.loads((current.parents[1]/'result.json').read_text())['status'] == 'completed'
        text = (current/'macroscopic_vars.dat').read_text()
        energy, dissipative, particle_flux = [numeric_rows(text, key) for key in
            ['energy_conservation', 'dissipative_terms', 'bnd_particle_fluxes']]
        namelist = (old/'input').read_text()
        assert 'normalized_velocity_profile' not in namelist.lower()
        for key, value in [('V_0', 0.), ('V_1', 0.), ('visco', 1e-5), ('visco_par', 1e-5)]:
            match = re.search(r'^\s*'+key+r'\s*=\s*([^!\s]+)', namelist, re.I | re.M)
            assert match and float(match[1].lower().replace('d', 'e')) == value
        frames = []
        for step in [8, 16, 24, 32, 40]:
            path = old/f'jorek{step:06d}.h5'
            geometry, state, metadata = read_restart(path, quadrature_order=4)
            values, gradient = evaluate_native(state, geometry), evaluate_native_grad(state, geometry)
            with h5py.File(path) as stream:
                code_time = np.asarray(stream['t_now'][()]).item()
                unit = np.asarray(stream['sqrt_mu0_rho0'][()]).item()
                density_unit = np.asarray(stream['central_density'][()]).item()*1e20/unit
                with h5py.File(current/path.name) as matched:
                    assert np.array_equal(stream['values'][()], matched['values'][()])
            factor = 1/(4e-7*np.pi*unit)
            temperature = values[..., 5].copy()
            mask = temperature < 1e-4
            temperature[mask] = .5e-4+.5e-4*np.exp((temperature[mask]-1e-4)/.5e-4)
            viscosity = 1e-5*(temperature/.018)**(-1.5)
            dot = np.sum(gradient[..., 1, :]*gradient[..., 3, :], axis=-1)
            radius = geometry.quadrature_rz[..., 0]
            legacy = float(np.sum(geometry.volume_weights*(-viscosity*dot))*factor)
            generic = float(np.sum(geometry.volume_weights*(-viscosity*(radius**2*dot
                            +2*radius*values[..., 3]*gradient[..., 1, 0])))*factor)
            np.testing.assert_allclose(generic, dissipative[code_time][2], rtol=2e-6, atol=.1)
            boundary = restart_boundary_quadrature(path)
            assert boundary['unknown_edge_count'] == 0
            bv, bg, normal = boundary['values'], boundary['gradients'], boundary['normals']
            radius_b = boundary['points'][..., 0]
            bp = np.stack((bg[..., 0, 1]/radius_b, -bg[..., 0, 0]/radius_b), axis=-1)
            vp = np.stack((-radius_b*bg[..., 1, 1], radius_b*bg[..., 1, 0]), axis=-1)
            bb2 = .64**2/radius_b**2+np.sum(bp*bp, axis=-1)
            diffusive_flux = -1e-5*np.sum(bg[..., 4, :]*normal, axis=-1)
            diffusive_flux += 1e-5*np.sum(bg[..., 4, :]*bp, axis=-1)*np.sum(bp*normal, axis=-1)/bb2
            reconstructed_particle_flux = np.array([0.,
                np.sum(boundary['surface_weights']*diffusive_flux),
                np.sum(boundary['surface_weights']*bv[..., 4]*bv[..., 6]*np.sum(bp*normal, axis=-1)),
                np.sum(boundary['surface_weights']*bv[..., 4]*np.sum(vp*normal, axis=-1)), 0.])*density_unit
            native_particle_flux = np.array(particle_flux[code_time])
            flux_error = float(np.max(np.abs(reconstructed_particle_flux-native_particle_flux))/np.max(np.abs(native_particle_flux)))
            assert flux_error < 1e-8
            work = float(np.sum(boundary['surface_weights']*1e-5*boundary['values'][..., 6]
                       *np.sum(boundary['gradients'][..., 6, :]*boundary['normals'], axis=-1))*factor)
            lhs, rhs = energy[code_time]
            provisional = rhs-generic+legacy-work
            frames.append({'step': step, 'time_s': metadata['time_seconds'], 'restart_sha256': sha256_file(path),
                'minus_dE_dt_W': lhs, 'native_magnetic_corrected_rhs_W': rhs,
                'generic_perpendicular_loss_W': generic, 'legacy_model303_perpendicular_loss_W': legacy,
                'incoming_parallel_viscous_boundary_work_W': work, 'provisional_rhs_W': provisional,
                'remaining_net_mismatch_W': lhs-provisional,
                'native_snapshot_state_exact': True,
                'reconstructed_particle_fluxes_per_s': reconstructed_particle_flux.tolist(),
                'native_particle_fluxes_per_s': native_particle_flux.tolist(),
                'particle_flux_error_scaled_by_largest_component': flux_error,
                'exterior_edges': len(boundary['element_ids']), 'unknown_edges': boundary['unknown_edge_count'],
                'collapsed_axis_edges': boundary['collapsed_axis_edges'],
                'surface_area_m2': float(boundary['surface_weights'].sum()),
                'native_modified_boundary_rows_checked': False})
            del geometry, state, values, gradient, boundary
            gc.collect()
        report['runs'][family] = frames
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({family: {'max_remaining_net_mismatch_W': max(abs(frame['remaining_net_mismatch_W']) for frame in frames),
                               'parallel_boundary_work_range_W': [min(frame['incoming_parallel_viscous_boundary_work_W'] for frame in frames),
                                                                  max(frame['incoming_parallel_viscous_boundary_work_W'] for frame in frames)]}
                      for family, frames in report['runs'].items()}, indent=2))


if __name__ == '__main__':
    main()
