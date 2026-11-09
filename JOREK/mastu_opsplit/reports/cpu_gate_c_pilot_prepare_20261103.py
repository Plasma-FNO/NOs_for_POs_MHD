"""Prepare a reviewable training-envelope pilot; failed meshes cannot launch it."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mastu_opsplit.case import namelist_overrides
from mastu_opsplit.common import config_hash, load_config, sha256_file, write_json
from mastu_opsplit.sampling import training_envelope_pilot, transport_profile_overrides


def main():
    reports = ROOT / 'reports'
    output = reports / 'cpu_gate_c_pilot_20261103.json'
    destination = ROOT / 'runs/prepared_gate_c_pilot_20261103'
    if output.exists() or destination.exists():
        raise ValueError('preserve existing training-envelope pilot')
    sampling = load_config(ROOT / 'configs/sampling.yaml')
    cases = training_envelope_pilot(sampling)
    reference = Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/spatial_radial_pair_20261103')
    template = reference / 'baseline48/input'
    row = next(line for line in subprocess.check_output(['saldo', '-b', '--dcgp', '-n'], text=True).splitlines() if line.split()[0] == 'FUPA2_FAUNA')
    cols = row.split()
    if min(int(cols[3])-int(cols[4]), int(cols[7])-int(cols[8])) < 32768:
        raise ValueError('insufficient credit for the proposed bounded corner pilot')
    destination.mkdir(parents=True)
    profiles = {name: [1e-5, 0., 0., 99., 99., 0., 0., 0., 0., 0.] for name in ['D_perp', 'ZK_perp']}
    for case in cases:
        path = destination / (case['case_id'] + '.input')
        path.write_text(namelist_overrides(template.read_text(), {
            **transport_profile_overrides(profiles, case['s_perp']),
            'opsplit_amplitude(1)': case['A_rho'], 'opsplit_amplitude(2)': case['A_T']}))
        case.update(input_template=str(path), input_template_sha256=sha256_file(path),
                    status='prepared_development_candidate', center_m=[.95, 0.], width_m=[.08, .16],
                    psi_n_taper=[.55, .75])
    plan = {'status': 'prepared_waiting_for_qualified_mesh_and_native_export', 'submission_allowed': False,
            'scope': 'Training interval only, fixed center/width and short horizon; no OOD accuracy claim.',
            'sampling_config_sha256': sha256_file(ROOT / 'configs/sampling.yaml'), 'template_sha256': sha256_file(template),
            'sampling_source_sha256': sha256_file(ROOT / 'src/mastu_opsplit/sampling.py'),
            'reference_geometry_status': 'Healthy diagnostic radial80 mesh; not accepted at the 1% spatial target.',
            'qualified_coarse_geometry': None, 'qualified_fine_geometry': None,
            'case_count_per_mesh': len(cases), 'cases': cases, 'final_time_s': 1.4059131112280784e-6,
            'steps': 48, 'tstep_code_units': .0625, 'field_relative_target': .01, 'integral_relative_target': .02,
            'required_checks': ['Positive finite q8 geometry/density/temperature, strict 1e-10 equilibrium, clean solver and exact final time.',
                                'Initializer unselected/boundary coefficients unchanged, amplitude bounds and element-level flux-support clearance.',
                                'All eight baseline, perturbed and matched baseline-subtracted fields pass the common-physical-query convergence target.',
                                'Native selected strong/weak labels and term strengths verified; physical balances and native constraints accounted.',
                                'Every refinement, timestep sibling and restart of a parent case remains in its development family.',
                                'Record every failure and actual resource/storage cost; freeze accepted mesh and protocol before production sampling.'],
            'future_sampling_blockers': ['The original strong-amplitude and held-out transport envelopes need separate pilots.',
                                         'Varying centers/widths requires independent support and resolution validation.',
                                         'The doubled test horizon has not been qualified by this short-horizon plan.'],
            'budget': {'authorization': 'User request to resolve Gates A/B/C; this is a prepared development pilot, not a production submission.',
                       'simulation_allocations_max': 20, 'analysis_allocations_max': 8, 'mpi_ranks': 4, 'threads': 1,
                       'simulation_memory_GiB': 256, 'simulation_hours_each_max': 6, 'analysis_memory_GiB': 64,
                       'analysis_hours_each_max': 1, 'rank_CPU_hours_max': 488, 'node_hours_max': 128,
                       'conservative_resource_hours_max': 32768, 'storage_GiB_max': 100, 'attempts_each_max': 1,
                       'saldo_row': row, 'reserved_or_submitted': False},
            'scientific_gates_passed': []}
    plan['candidate_plan_hash'] = config_hash(plan)
    write_json(output, plan)
    print(json.dumps({'report': str(output), 'cases': len(cases), 'submission_allowed': False}))


if __name__ == '__main__':
    main()
