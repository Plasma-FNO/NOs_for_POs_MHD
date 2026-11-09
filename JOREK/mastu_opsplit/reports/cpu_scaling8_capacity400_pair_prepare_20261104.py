"""Run a matched 48-step pair only after a qualified combined-solver probe."""
import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mastu_opsplit.case import namelist_overrides
from mastu_opsplit.common import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    probes = {'radial320': ('scaling8_capacity400_radial320_20261104','probe3'), 'radial640': ('scaling8_capacity400_radial640_20261104','probe3'), 'poloidal880': ('scaling8_capacity400_20261104','poloidal880'), 'poloidal1760': ('scaling8_capacity400_20261104','probe3')}
    parser.add_argument('label', choices=probes)
    args = parser.parse_args()
    memory = 720 if args.label == 'poloidal1760' else 512
    bulk = Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102')
    parent = bulk / probes[args.label][0]
    probe = parent / probes[args.label][1]
    evidence = json.loads((probe / 'result.json').read_text())
    executable = parent / 'source/jorek_model303'
    if evidence['status'] != 'completed' or sha256_file(executable) != evidence['binary_sha256']:
        raise ValueError('healthy same-binary native probe required')
    for moment, filename in [('initial', 'jorek000000.h5'), ('final', 'jorek_restart.h5')]:
        if sha256_file(probe / filename) != evidence[moment]['restart_sha256']:
            raise ValueError('qualified probe changed')
    text = (probe / 'stdout.log').read_text()
    iterations = re.findall(r'Iteration\s+(\d+): diff=\s*(\S+)', text)
    if 'Fixed boundary equilibrium converged:' not in text or not iterations or float(iterations[-1][1]) >= 1e-10:
        raise ValueError('strict native equilibrium convergence required')
    errors = re.findall(r'MUMPS refinement steps and backward errors:\s*(\d+)\s+(\S+)\s+(\S+)', text)
    if len(errors) < 3 or any(int(n)>5 or not all(math.isfinite(float(x)) and float(x)>=0 for x in [a,b]) or float(a)+float(b)>1e-12 for n,a,b in errors):
        raise ValueError('verified bounded linear solves required')
    qualification = json.loads((ROOT / 'reports/cpu_scaling8_capacity400_qualification_20261104.json').read_text())
    if qualification['status'] != 'original_baseline_perturbed_and_poloidal880_parity_verified' or evidence['binary_sha256'] != qualification['binary_sha256']:
        raise ValueError('qualified combined binary required')
    for name, digest in qualification['input_hashes'].items():
        if sha256_file(name) != digest:
            raise ValueError('qualification proof changed: ' + name)
    if args.label == 'radial320' and not all(evidence.get(k,{}).get('passed',False) for k in ['initial_parity','parity']):
        raise ValueError('radial320 strict physical parity required')
    original = json.loads((parent / 'manifest.json').read_text())
    for name,digest in original['input_hashes'].items():
        if sha256_file(name) != digest:
            raise ValueError('frozen probe source/input changed: '+name)
    base = bulk / f'scaling8_capacity400_pair_{args.label}_20261104'
    if base.exists():
        raise ValueError('preserve existing pair')
    row = next(line for line in subprocess.check_output(['saldo', '-b', '--dcgp', '-n'], text=True).splitlines() if line.split()[0] == 'FUPA2_FAUNA')
    columns = row.split()
    if min(int(columns[3])-int(columns[4]), int(columns[7])-int(columns[8])) < 2 * memory * 6:
        raise ValueError('insufficient current credit for two capped six-hour runs')
    (base / 'source').mkdir(parents=True)
    shutil.copyfile(executable, base / 'source/jorek_model303')
    (base / 'source/jorek_model303').chmod(0o555)
    (base / 'probe3').symlink_to(probe, target_is_directory=True)
    for name in ['worker.py', 'jorek-env.sh']:
        shutil.copyfile(parent / name, base / name)
    original = json.loads((parent / 'manifest.json').read_text())
    manifest = {key: original[key] for key in ['final_time_s', 'tstep_code_units', 'field_relative_target', 'integral_relative_target', 'equil_accuracy']}
    manifest.update({key: original[key] for key in ['qualified_parent','binary_sha256'] if key in original})
    manifest.update(validated_unchanged_binary_sha256=evidence['binary_sha256'],
                    binary_qualification='Same qualified capacity executable across the pair; strict baseline/perturbed/poloidal880 reference parity and fresh refined three-step qualification required.',
                    stages={})
    files = [ROOT / 'reports/cpu_scaling8_capacity400_qualification_20261104.json', base / 'source/jorek_model303', base / 'worker.py', base / 'jorek-env.sh',
             probe / 'result.json', probe / 'jorek000000.h5']
    for label, amplitude in [('baseline48', 0.), ('perturbed48', .05)]:
        work = base / label
        work.mkdir()
        overrides = {'nstep_n': 48, 'nout': 8, 'opsplit_amplitude(1)': amplitude, 'opsplit_amplitude(2)': amplitude}
        (work / 'input').write_text(namelist_overrides((probe / 'input').read_text(), overrides))
        for name in ['eqdsk.dat', 'jorek_equil_rz.h5']:
            shutil.copyfile(probe / name, work / name)
        files.extend(work.iterdir())
        manifest['stages'][label] = {'steps': 48, 'amplitudes': [amplitude, amplitude], 'timeout_seconds': 20400, 'mpi_ranks': 16}
    manifest['input_hashes'] = {str(p): sha256_file(p) for p in files}
    (base / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    python = '/pitagora_work/FUPB2_UKAEA_ML/vgopakum/envs/cfo/bin/python'
    for label in manifest['stages']:
        script = base / (label + '.sbatch')
        script.write_text(f'''#!/bin/bash
#SBATCH --job-name=mastu-capacity-pair-{args.label}-{label}
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_prod
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem={memory}G
#SBATCH --time=06:00:00
set -euo pipefail
source {base}/jorek-env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
{python} {base}/worker.py {base} {ROOT} {label}
''')
        command = ['sbatch', '--parsable', '--chdir', str(base), '--output', str(base / (label + '-%j.out')),
                   '--error', str(base / (label + '-%j.err')), str(script)]
        subprocess.run(['sbatch', '--test-only', *command[1:]], check=True, capture_output=True)
        record = {'status': 'prepared', 'bulk_output': str(base), 'stage': label,
                  'probe': str(probe), 'binary_sha256': evidence['binary_sha256'],
                  'manifest_sha256': sha256_file(base / 'manifest.json'), 'script_sha256': sha256_file(script),
                  'budget': {'authorization': 'User requested continued numerical convergence.', 'mpi_ranks': 16,
                             'memory_GiB': memory, 'wall_hours_max': 6, 'rank_CPU_hours_max': 96, 'saldo_row': row},
                  'scientific_gates_passed': []}
        output = ROOT / f'reports/cpu_scaling8_capacity400_pair_{args.label}_{label}_20261104.json'
        with output.open('x') as stream:
            json.dump(record, stream, indent=2)
        record.update(status='submitted', job_id=subprocess.check_output(command, text=True).strip())
        output.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps({'stage': label, 'job_id': record['job_id'], 'base': str(base)}), flush=True)


if __name__ == '__main__':
    main()
