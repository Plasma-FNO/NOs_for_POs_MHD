"""Bounded radial, wall-normal and poloidal refinement campaign.

Each native three-step proof gates a matched 48-step pair and its comparison.
The optional second radial level compares against radial160, never radial80.
"""
import argparse
import fcntl
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(REPORTS))
from mastu_opsplit.case import namelist_overrides
from mastu_opsplit.common import sha256_file
from cpu_spatial_next_launch_20261103 import physics_text

PYTHON = '/pitagora_work/FUPB2_UKAEA_ML/vgopakum/envs/cfo/bin/python'
REFERENCE = Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/spatial_radial_pair_20261103')
BASE = REFERENCE.parent / 'spatial_directions_20261103'
RECORD = REPORTS / 'cpu_spatial_directions_20261103.json'
BINARY_SHA = 'dd786f76a5c815976ad54bcd31b744ad2c3d6cafddfcc064dcae8e6761d8ccd6'


def candidates():
    return {
        'radial160': {'direction': 'core radial', 'overrides': {'n_flux': 160}, 'reference': str(REFERENCE)},
        'radial320': {'direction': 'core radial', 'overrides': {'n_flux': 320}, 'reference': str(BASE / 'radial160')},
        'wall60': {'direction': 'wall normal', 'overrides': {f'n_ext_block({i})': 114 if i == 3 else 60 for i in range(1, 12)}, 'reference': str(REFERENCE)},
        'wall50': {'direction': 'wall normal', 'overrides': {f'n_ext_block({i})': 95 if i == 3 else 50 for i in range(1, 12)}, 'reference': str(REFERENCE)},
        'poloidal440': {'direction': 'poloidal including divertor legs', 'overrides': {'n_tht': 440, 'n_leg': 24, 'n_leg_out': 160, 'n_up_leg': 24, 'n_up_leg_out': 160}, 'reference': str(REFERENCE)},
        'poloidal330': {'direction': 'poloidal including divertor legs', 'overrides': {'n_tht': 330, 'n_leg': 18, 'n_leg_out': 120, 'n_up_leg': 18, 'n_up_leg_out': 120}, 'reference': str(REFERENCE)},
    }


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_case(label):
    return json.loads((BASE / label / 'campaign_case.json').read_text())


def save_case(label, case):
    write_json(BASE / label / 'campaign_case.json', case)


def verify_campaign():
    record = json.loads(RECORD.read_text())
    for name, digest in record['frozen_sources'].items():
        if sha256_file(name) != digest:
            raise ValueError('frozen campaign source changed: ' + name)
    return record


def credit(required):
    row = next(line for line in subprocess.check_output(['saldo', '-b', '--dcgp', '-n'], text=True).splitlines() if line.split()[0] == 'FUPA2_FAUNA')
    cols = row.split()
    if min(int(cols[3]) - int(cols[4]), int(cols[7]) - int(cols[8])) < required:
        raise ValueError('insufficient verified shared project credit')
    return row


def submit(label, stage, tasks, memory, hours, body, dependency=()):
    case = load_case(label)
    if stage in case['jobs']:
        raise ValueError('preserve already submitted job: ' + label + '/' + stage)
    work = BASE / label
    script = work / (stage + '.sbatch')
    script.write_text(f'''#!/bin/bash
#SBATCH --job-name=mastu-{label}-{stage}
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_prod
#SBATCH --nodes=1
#SBATCH --ntasks={tasks}
#SBATCH --cpus-per-task=1
#SBATCH --mem={memory}G
#SBATCH --time={hours:02d}:00:00
set -euo pipefail
source {work}/jorek-env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
sha256sum --check {BASE}/startup.sha256 > {work}/{stage}-startup.log
{body}
''')
    subprocess.run(['bash', '-n', str(script)], check=True)
    command = ['sbatch', '--parsable', '--chdir', str(work), '--output', str(work / (stage + '-%j.out')), '--error', str(work / (stage + '-%j.err'))]
    if dependency:
        command += ['--dependency', 'afterok:' + ':'.join(dependency)]
    command.append(str(script))
    preflight = subprocess.run(['sbatch', '--test-only', *command[1:]], capture_output=True, text=True, check=True)
    job = subprocess.check_output(command, text=True).strip()
    case['jobs'][stage] = {'job_id': job, 'script': str(script), 'script_sha256': sha256_file(script), 'scheduler_preflight': preflight.stderr}
    save_case(label, case)
    return job


def start_probe(label):
    verify_campaign()
    credit(512)
    case = load_case(label)
    if case['jobs']:
        raise ValueError('candidate already started: ' + label)
    body = f'''set +e
{PYTHON} {BASE}/{label}/worker.py {BASE}/{label} {ROOT} probe3
probe_status=$?
set -e
{PYTHON} {Path(__file__).resolve()} finish-probe {label} --returncode "$probe_status"
'''
    job = submit(label, 'probe', 4, 256, 2, body)
    print(json.dumps({'candidate': label, 'probe_job': job}), flush=True)


def prepare():
    if RECORD.exists() or BASE.exists():
        raise ValueError('preserve existing directional campaign')
    row = credit(16384)
    if sha256_file(REFERENCE / 'source/jorek_model303') != BINARY_SHA:
        raise ValueError('validated unchanged binary required')
    manifest = json.loads((REFERENCE / 'manifest.json').read_text())
    for label in ['baseline48', 'perturbed48']:
        result = json.loads((REFERENCE / label / 'result.json').read_text())
        if result['status'] != 'completed' or result['binary_sha256'] != BINARY_SHA:
            raise ValueError('healthy matched radial80 reference required')
        for moment, name in [('initial', 'jorek000000.h5'), ('final', 'jorek_restart.h5')]:
            if sha256_file(REFERENCE / label / name) != result[moment]['restart_sha256']:
                raise ValueError('reference output changed')
    BASE.mkdir()
    source_paths = [Path(__file__).resolve(), REPORTS / 'cpu_spatial_directions_analysis_20261103.py',
                    REPORTS / 'cpu_spatial_directions_tests_20261103.xml',
                    REPORTS / 'cpu_spatial_next_launch_20261103.py', REPORTS / 'cpu_spatial_pair_worker_20261103.py',
                    REPORTS / 'cpu_cached_rz_recovery_worker_20261103.py', ROOT / 'src/mastu_opsplit/queries.py',
                    ROOT / 'src/mastu_opsplit/restart.py', ROOT / 'src/mastu_opsplit/numerical.py']
    frozen = {str(p): sha256_file(p) for p in source_paths}
    for label, config in candidates().items():
        work = BASE / label
        (work / 'source').mkdir(parents=True)
        shutil.copyfile(REFERENCE / 'source/jorek_model303', work / 'source/jorek_model303')
        (work / 'source/jorek_model303').chmod(0o555)
        for name in ['worker.py', 'jorek-env.sh']:
            shutil.copyfile(REFERENCE / name, work / name)
        stages = {}
        for stage, steps, amplitude in [('probe3', 3, 0.), ('baseline48', 48, 0.), ('perturbed48', 48, .05)]:
            folder = work / stage
            folder.mkdir()
            overrides = {**config['overrides'], 'nstep_n': steps, 'opsplit_amplitude(1)': amplitude, 'opsplit_amplitude(2)': amplitude}
            (folder / 'input').write_text(namelist_overrides((REFERENCE / stage / 'input').read_text(), overrides))
            for name in ['eqdsk.dat', 'jorek_equil_rz.h5']:
                shutil.copyfile(REFERENCE / 'probe3' / name, folder / name)
            if stage != 'probe3' and physics_text(folder / 'input') != physics_text(REFERENCE / stage / 'input'):
                raise ValueError('physics or numerical controls changed: ' + label)
            stages[stage] = {'steps': steps, 'amplitudes': [amplitude, amplitude], 'timeout_seconds': 6900 if steps == 3 else 20400}
        m = {key: manifest[key] for key in ['final_time_s', 'tstep_code_units', 'field_relative_target', 'integral_relative_target', 'required_field_groups', 'equil_accuracy', 'validated_unchanged_binary_sha256']}
        files = [work / 'worker.py', work / 'jorek-env.sh', work / 'source/jorek_model303'] + [p for stage in stages for p in (work / stage).iterdir()]
        m.update(stages=stages, input_hashes={str(p): sha256_file(p) for p in files},
                 spatial_refinement_scope=f"Independent {config['direction']} count refinement; other directional counts and mesh placement flags fixed. Two-resolution consistency at the fixed timestep.")
        write_json(work / 'manifest.json', m)
        write_json(work / 'campaign_case.json', {**config, 'jobs': {}, 'status': 'prepared', 'manifest_sha256': sha256_file(work / 'manifest.json')})
        frozen[str(work / 'manifest.json')] = sha256_file(work / 'manifest.json')
    budget = {'authorization': 'User requested next radial refinement to reach 1% and wall/poloidal convergence checks.',
              'mpi_ranks': 4, 'threads': 1, 'memory_GiB_simulation': 256, 'probe_allocations_max': 6,
              'probe_hours_each_max': 2, 'full_pair_count_max': 4, 'full_allocations_max': 8,
              'full_hours_each_max': 6, 'analysis_allocations_max': 4, 'analysis_hours_each_max': 1,
              'rank_CPU_hours_max': 244, 'node_hours_max': 64, 'conservative_resource_hours_max': 16384, 'saldo_row': row,
              'policy': 'Start radial160, wall60, poloidal440. Only failed wall/poloidal proofs enable wall50/poloidal330. Only healthy radial160 full pair failing the frozen field/integral/coverage targets enables radial320, compared against radial160. No retry of any candidate; no physics/tolerance/build changes.'}
    write_json(RECORD, {'bulk_output': str(BASE), 'reference': str(REFERENCE), 'budget': budget, 'candidates': candidates(), 'frozen_sources': frozen,
                        'ledger_before': json.loads((REPORTS / 'resource_ledger.json').read_text()), 'status': 'prepared'})
    (BASE / 'startup.sha256').write_text(''.join(value + '  ' + name + '\n' for name, value in frozen.items()))
    for label in ['radial160', 'wall60', 'poloidal440']:
        start_probe(label)


def finish_probe(label, returncode):
    verify_campaign()
    work = BASE / label
    case = load_case(label)
    result = json.loads((work / 'probe3/result.json').read_text()) if (work / 'probe3/result.json').exists() else {'status': 'missing_result'}
    if returncode or result['status'] != 'completed':
        case.update(status='probe_failed', probe_result=result)
        save_case(label, case)
        fallback = {'wall60': 'wall50', 'poloidal440': 'poloidal330'}.get(label)
        if fallback:
            start_probe(fallback)
        return
    if sha256_file(work / 'manifest.json') != case['manifest_sha256'] or result['binary_sha256'] != BINARY_SHA:
        raise ValueError('proof identity changed')
    if result['initial']['metadata']['minimum_signed_jacobian_m2'] <= 0 or result['final']['health']['steps'] != 3:
        raise ValueError('positive geometry and three-step proof required')
    credit(3328)
    manifest = json.loads((work / 'manifest.json').read_text())
    references = {}
    for stage in ['baseline48', 'perturbed48']:
        old = Path(case['reference']) / stage
        old_result = json.loads((old / 'result.json').read_text())
        if old_result['status'] != 'completed' or old_result['binary_sha256'] != BINARY_SHA:
            raise ValueError('reference pair is not healthy')
        if physics_text(old / 'input') != physics_text(work / stage / 'input'):
            raise ValueError('comparison physics or numerical settings differ')
        if old_result['final']['health']['steps'] != 48 or abs(old_result['final']['health']['final_time_s'] - manifest['final_time_s']) > 1e-15:
            raise ValueError('reference horizon differs')
        for name in ['eqdsk.dat', 'jorek_equil_rz.h5']:
            if sha256_file(old / name) != sha256_file(work / stage / name):
                raise ValueError('equilibrium source differs')
        hashes = {}
        for moment, name in [('initial', 'jorek000000.h5'), ('final', 'jorek_restart.h5')]:
            digest = sha256_file(old / name)
            if digest != old_result[moment]['restart_sha256']:
                raise ValueError('reference identity changed')
            hashes[moment + '_sha256'] = digest
        references[stage] = {'work': str(old), **hashes, 'input_sha256': sha256_file(old / 'input')}
    analysis_manifest = work / 'analysis_manifest.json'
    with analysis_manifest.open('x') as stream:
        json.dump({**manifest, 'references': references, 'simulation_manifest_sha256': case['manifest_sha256']}, stream, indent=2)
    # Full jobs depend on this successful proof allocation, including final bookkeeping.
    dependency = [case['jobs']['probe']['job_id']]
    ids = [submit(label, stage, 4, 256, 6, f'{PYTHON} {work}/worker.py {work} {ROOT} {stage}', dependency) for stage in ['baseline48', 'perturbed48']]
    submit(label, 'analysis', 1, 64, 1, f'{PYTHON} {Path(__file__).resolve()} analyze {label}', ids)
    case = load_case(label)
    case.update(status='pair_submitted', probe_result=result, analysis_manifest_sha256=sha256_file(analysis_manifest))
    save_case(label, case)


def analyze(label):
    verify_campaign()
    case = load_case(label)
    work = BASE / label
    if sha256_file(work / 'analysis_manifest.json') != case['analysis_manifest_sha256']:
        raise ValueError('analysis references changed')
    output = REPORTS / f'cpu_spatial_directions_{label}_convergence_20261103.json'
    subprocess.run([PYTHON, str(REPORTS / 'cpu_spatial_directions_analysis_20261103.py'), str(work), str(work / 'analysis_manifest.json'), str(output)], check=True)
    report = json.loads(output.read_text())
    passed = report['candidate_spatial_targets_met_on_covered_volume'] and report['coverage']['full_wall_coverage']
    case.update(status='directional_targets_met' if passed else 'directional_targets_not_met', report=str(output), report_sha256=sha256_file(output))
    save_case(label, case)
    if label == 'radial160' and not passed:
        start_probe('radial320')


def status():
    record = verify_campaign()
    cases = {label: load_case(label) for label in candidates()}
    ids = [job['job_id'] for case in cases.values() for job in case['jobs'].values()]
    if ids:
        accounting = subprocess.check_output(['sacct', '-n', '-P', '-X', '-j', ','.join(ids), '--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS'], text=True)
    else:
        accounting = ''
    result = {'cases': cases, 'accounting': accounting, 'budget': record['budget']}
    write_json(REPORTS / 'cpu_spatial_directions_status_20261103.json', result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'finish-probe', 'analyze', 'status'])
    parser.add_argument('label', nargs='?', choices=list(candidates()))
    parser.add_argument('--returncode', type=int, default=0)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'status':
        status()
    else:
        if not args.label:
            parser.error('candidate label required')
        # One case owns one lock; independent directions can progress concurrently.
        with (BASE / args.label / 'controller.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if args.action == 'finish-probe':
                finish_probe(args.label, args.returncode)
            else:
                analyze(args.label)


if __name__ == '__main__':
    main()
