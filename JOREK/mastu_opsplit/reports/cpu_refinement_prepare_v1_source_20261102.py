"""Prepare and submit bounded native mesh-refinement stages with a shared budget."""
import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from mastu_opsplit.common import config_hash, sha256_file
from mastu_opsplit.case import namelist_overrides
from mastu_opsplit.runner import render_slurm, run_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['parity8', 'quad_probe3'], required=True)
    args = parser.parse_args()
    phase_path = ROOT/'reports/cpu_refinement_continuation_20261102.json'
    phase = json.loads(phase_path.read_text())
    budget_path = Path(phase['budget_record'])
    assert sha256_file(budget_path) == phase['budget_record_sha256']
    budget = json.loads(budget_path.read_text())
    build_path = ROOT/'reports/cpu_refinement_capacity_build_20261102.json'
    build = json.loads(build_path.read_text())
    assert build['status'] == 'compiled_and_linked'
    assert sha256_file(build['binary_path']) == build['binary_sha256']
    template = 'energy_complete_baseline48'
    config = yaml.safe_load((ROOT/f'configs/cpu_{template}_20261102.yaml').read_text())
    old = ROOT/f'runs/prepared_cpu_{template}_20261102'
    manifest = json.loads((old/'case.json').read_text())
    assert manifest['config_hash'] == config_hash(config)
    assert all(sha256_file(old/name) == value for name, value in manifest['snapshot_hashes'].items())
    label = 'refine_'+args.stage
    assert not any(run['label'] == label for run in phase['runs'])
    mesh_names = ['n_R', 'n_Z', 'n_flux', 'n_tht', 'n_open', 'n_outer', 'n_inner',
                  'n_leg', 'n_leg_out', 'n_private', 'n_up_leg', 'n_up_leg_out', 'n_up_priv']
    mesh_names += [f'n_ext_block({i})' for i in range(1, 12)]
    mesh = {key: config['input_defaults'][key]*(2 if args.stage == 'quad_probe3' else 1)
            for key in mesh_names}
    steps, minutes, memory = (8, 20, '64G') if args.stage == 'parity8' else (3, 60, '128G')
    dependency = None
    if args.stage == 'quad_probe3':
        parity = next(run for run in phase['runs'] if run['label'] == 'refine_parity8')
        assert parity['status'] == 'completed'
        assert parity['result']['status'] == 'completed'
        dependency = parity['job_id']
    horizon = config['acceptance']['final_time_s']*steps/48
    config['case_id'] = 'mastu-model303-'+label
    config['jorek'].update(executable=build['binary_path'], executable_sha256=build['binary_sha256'])
    config['jorek']['build_config'] += ' n_nodes_max=120001 n_elements_max=120001'
    config['jorek']['derivative_build'].update(refinement_capacity_report=str(build_path),
                                              refinement_capacity_report_sha256=sha256_file(build_path))
    config['input_defaults'].update(mesh)
    config['evolution'].update(nstep_n=steps, tstep_n=.0625, nout=1)
    config['execution'].update(mpi_ranks=16, omp_threads=1, max_attempts=1, timeout_s=minutes*60-60)
    config['slurm'].update(account='fupa2_fauna', partition='dcgp_fua_prod',
                           walltime=f'{minutes//60:02d}:{minutes%60:02d}:00', memory=memory)
    config['acceptance']['final_time_s'] = horizon
    if args.stage == 'quad_probe3':
        config['acceptance'].pop('equilibrium_regression')
    else:
        config['acceptance']['equilibrium_regression']['absolute_tolerance'] = 1e-10
    config['resource_authorization'].update(resource_profile=budget['resource_profile'],
        budget_record=str(budget_path), budget_record_sha256=sha256_file(budget_path),
        approval_source=budget['approval_source'],
        scope='MPI/capacity parity on the existing mesh' if args.stage == 'parity8' else 'Nested 2x mesh three-step cost/health probe',
        requested_cpu_core_hours=16*minutes/60, requested_node_hours=minutes/60)
    path = ROOT/f'configs/cpu_{label}_20261102.yaml'
    prepared = ROOT/f'runs/prepared_cpu_{label}_20261102'
    assert not path.exists() and not prepared.exists()
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    shutil.copytree(old, prepared)
    src = Path(build['source_directory'])
    files = {'jorek.exe': Path(build['binary_path']), 'mesh_settings.f90': src/'models/mod_settings.f90',
        'model_settings.f90': src/'models/model303/mod_model_settings.f90',
        'source_initial_conditions.f90': src/'models/model303/initial_conditions.f90',
        'source_residual.f90': src/'models/model303/mod_elt_matrix_fft.f90',
        'source_defaults.f90': src/'models/preset_parameters.f90', 'source_main.f90': src/'jorek2_main.f90',
        'source_diagnostics.f90': src/'diagnostics/mod_integrals3D.f90',
        'build_config.txt': Path(build['build_settings_summary'])}
    if args.stage == 'parity8':
        files['equilibrium_reference.h5'] = ROOT/'runs/cpu_io_baseline48_20261102/attempt_1/work/jorek000008.h5'
    for name, source in files.items():
        target = prepared/name
        if target.exists():
            target.chmod(0o644)
        shutil.copyfile(source, target)
        target.chmod(0o555 if name == 'jorek.exe' else 0o444)
    target = prepared/'input'
    target.chmod(0o644)
    target.write_text(namelist_overrides((old/'input').read_text(), {**mesh, **config['evolution']}))
    target.chmod(0o444)
    derived = copy.deepcopy(manifest)
    derived.update(case_id=config['case_id'], config_hash=config_hash(config),
        evolution=config['evolution'], output=str(prepared), parent_snapshot=str(old),
        executable_sha256=build['binary_sha256'])
    derived.pop('unchanged_state_reference', None)
    for name, source in files.items():
        derived['files'][name] = {'source': str(source), 'sha256': sha256_file(prepared/name)}
    derived['snapshot_hashes'] = {p.name: sha256_file(p) for p in prepared.iterdir()
                                 if p.is_file() and p.name != 'case.json'}
    target = prepared/'case.json'
    target.chmod(0o644)
    target.write_text(json.dumps(derived, indent=2)+'\n')
    bulk = src.parents[1]
    output = bulk/f'cpu_{label}_20261102'
    (ROOT/f'runs/cpu_{label}_20261102').symlink_to(output, target_is_directory=True)
    script = ROOT/f'reports/cpu_{label}_20261102.sbatch'
    script.write_text(render_slurm(config, prepared, output, path))
    prior = phase['builds']+[run for run in phase['runs'] if 'job_id' in run]
    cpu_hours = sum(run['requested_cpu_core_hours'] for run in prior)+16*minutes/60
    node_hours = sum(run['requested_node_hours'] for run in prior)+minutes/60
    assert cpu_hours <= budget['phase']['requested_cpu_core_hours_max']
    assert node_hours <= budget['phase']['requested_node_hours_max']
    assert len(phase['runs']) < budget['phase']['simulation_submissions_max']
    command = ['sbatch', '--parsable', '--job-name=mastu-cpu-'+label, '--chdir', str(ROOT),
               '--output', str(bulk/f'logs/{label}-%j.out'), '--error', str(bulk/f'logs/{label}-%j.err')]
    if dependency:
        command += ['--dependency=afterok:'+dependency]
    command += [str(script)]
    check = subprocess.run([command[0], '--test-only', *command[2:]], capture_output=True, text=True)
    assert check.returncode == 0, check.stderr
    result = subprocess.run(command, capture_output=True, text=True)
    run = {'label': label, 'config': str(path), 'prepared': str(prepared), 'bulk_output': str(output),
        'script': str(script), 'script_sha256': sha256_file(script), 'preparation_source_sha256': sha256_file(__file__),
        'plan': run_plan(config, prepared, output), 'requested_cpu_core_hours': 16*minutes/60,
        'requested_node_hours': minutes/60, 'status': 'submission_failed',
        'submission': {'command': command, 'returncode': result.returncode, 'stdout': result.stdout,
                       'stderr': result.stderr, 'utc': datetime.now(timezone.utc).isoformat()}}
    if result.returncode == 0:
        run.update(job_id=result.stdout.strip().split(';')[0], status='submitted')
    phase['runs'].append(run)
    phase.update(reserved_cpu_core_hours=cpu_hours, reserved_node_hours=node_hours)
    phase_path.write_text(json.dumps(phase, indent=2)+'\n')
    assert result.returncode == 0, result.stderr
    print(json.dumps({'label': label, 'job_id': run['job_id'], 'reserved_phase_cpu_hours': cpu_hours,
                      'reserved_phase_node_hours': node_hours}, indent=2))


if __name__ == '__main__':
    main()
