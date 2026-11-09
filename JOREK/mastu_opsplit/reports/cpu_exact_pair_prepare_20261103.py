"""Run a matched 48-step pair only after a qualified exact-crossing probe."""
import argparse
import json
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
    parser.add_argument('label', choices=['radial80', 'radial160', 'poloidal440', 'core_poloidal440'])
    args = parser.parse_args()
    bulk = Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102')
    parents = {'poloidal440': 'exact_poloidal_crossing_20261103',
               'core_poloidal440': 'exact_core_poloidal_crossing_20261103'}
    parent = bulk / parents.get(args.label, 'exact_core_crossing_20261103')
    probe = parent / ('control80' if args.label == 'radial80' else args.label)
    evidence = json.loads((probe / 'result.json').read_text())
    executable = parent / 'source/jorek_model303'
    if evidence['status'] != 'completed' or sha256_file(executable) != evidence['binary_sha256']:
        raise ValueError('healthy same-binary native probe required')
    for moment, filename in [('initial', 'jorek000000.h5'), ('final', 'jorek_restart.h5')]:
        if sha256_file(probe / filename) != evidence[moment]['restart_sha256']:
            raise ValueError('qualified probe changed')
    if 'Fixed boundary equilibrium converged:' not in (probe / 'stdout.log').read_text():
        raise ValueError('strict native equilibrium convergence required')
    base = bulk / f'exact_pair_{args.label}_20261103'
    if base.exists():
        raise ValueError('preserve existing pair')
    row = next(line for line in subprocess.check_output(['saldo', '-b', '--dcgp', '-n'], text=True).splitlines() if line.split()[0] == 'FUPA2_FAUNA')
    columns = row.split()
    if min(int(columns[3])-int(columns[4]), int(columns[7])-int(columns[8])) < 3072:
        raise ValueError('insufficient current credit for two capped six-hour runs')
    (base / 'source').mkdir(parents=True)
    shutil.copyfile(executable, base / 'source/jorek_model303')
    (base / 'source/jorek_model303').chmod(0o555)
    (base / 'probe3').symlink_to(probe, target_is_directory=True)
    for name in ['worker.py', 'jorek-env.sh']:
        shutil.copyfile(parent / name, base / name)
    original = json.loads((parent / 'manifest.json').read_text())
    manifest = {key: original[key] for key in ['final_time_s', 'tstep_code_units', 'field_relative_target', 'integral_relative_target', 'equil_accuracy']}
    manifest.update(validated_unchanged_binary_sha256=evidence['binary_sha256'],
                    binary_qualification='Unchanged relative to this exact-crossing probe; geometry differs intentionally from the old solver.',
                    stages={})
    files = [base / 'source/jorek_model303', base / 'worker.py', base / 'jorek-env.sh',
             probe / 'result.json', probe / 'jorek000000.h5']
    for label, amplitude in [('baseline48', 0.), ('perturbed48', .05)]:
        work = base / label
        work.mkdir()
        overrides = {'nstep_n': 48, 'nout': 8, 'opsplit_amplitude(1)': amplitude, 'opsplit_amplitude(2)': amplitude}
        (work / 'input').write_text(namelist_overrides((probe / 'input').read_text(), overrides))
        for name in ['eqdsk.dat', 'jorek_equil_rz.h5']:
            shutil.copyfile(probe / name, work / name)
        files.extend(work.iterdir())
        manifest['stages'][label] = {'steps': 48, 'amplitudes': [amplitude, amplitude], 'timeout_seconds': 20400}
    manifest['input_hashes'] = {str(p): sha256_file(p) for p in files}
    (base / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    python = '/pitagora_work/FUPB2_UKAEA_ML/vgopakum/envs/cfo/bin/python'
    for label in manifest['stages']:
        script = base / (label + '.sbatch')
        script.write_text(f'''#!/bin/bash
#SBATCH --job-name=mastu-exact-{args.label}-{label}
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_prod
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=256G
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
                  'budget': {'authorization': 'User requested continued numerical convergence.', 'mpi_ranks': 4,
                             'memory_GiB': 256, 'wall_hours_max': 6, 'rank_CPU_hours_max': 24, 'saldo_row': row},
                  'scientific_gates_passed': []}
        output = ROOT / f'reports/cpu_exact_pair_{args.label}_{label}_20261103.json'
        with output.open('x') as stream:
            json.dump(record, stream, indent=2)
        record.update(status='submitted', job_id=subprocess.check_output(command, text=True).strip())
        output.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps({'stage': label, 'job_id': record['job_id'], 'base': str(base)}), flush=True)


if __name__ == '__main__':
    main()
