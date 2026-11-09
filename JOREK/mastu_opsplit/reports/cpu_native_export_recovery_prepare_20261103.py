"""Recover a pre-launch permissions failure; validate evolved step-8 states."""
import difflib
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
    reports = ROOT / 'reports'
    old_record = json.loads((reports / 'cpu_native_export_20261103.json').read_text())
    old = Path(old_record['bulk_output'])
    base = old.parent / 'native_export_recovery_20261103'
    record_path = reports / 'cpu_native_export_recovery_20261103.json'
    if base.exists() or record_path.exists():
        raise ValueError('preserve existing launch recovery')
    state = subprocess.check_output(['sacct', '-n', '-P', '-X', '-j', old_record['job_id'], '--format=State,ExitCode'], text=True).strip()
    if state != 'FAILED|1:0' or list((old / 'parent8').glob('opsplit-native-*.bin')):
        raise ValueError('expected preserved pre-launch failure')
    if 'could not access' not in (old / 'parent8/stderr.log').read_text():
        raise ValueError('unexpected parent failure')
    row = next(line for line in subprocess.check_output(['saldo', '-b', '--dcgp', '-n'], text=True).splitlines() if line.split()[0] == 'FUPA2_FAUNA')
    cols = row.split()
    if min(int(cols[3])-int(cols[4]), int(cols[7])-int(cols[8])) < 768:
        raise ValueError('insufficient project CPU credit')
    base.mkdir()
    shutil.copytree(old / 'source', base / 'source')
    path = base / 'source/models/model303/opsplit_export_begin.inc'
    original = path.read_text()
    if original.count('index_now == 1') != 1:
        raise ValueError('export frame anchor changed')
    changed = original.replace('index_now == 1', 'index_now == 9')
    path.write_text(changed)
    patch = ROOT / 'patches/0017-native-evolved-export-frame.patch'
    with patch.open('x') as stream:
        stream.writelines(difflib.unified_diff(original.splitlines(True), changed.splitlines(True),
                                             fromfile='a/models/model303/opsplit_export_begin.inc', tofile='b/models/model303/opsplit_export_begin.inc'))
    shutil.copyfile(old / 'parent.exe', base / 'parent.exe')
    (base / 'parent.exe').chmod(0o555)
    for name, source in [('jorek-env.sh', old / 'jorek-env.sh'), ('worker.py', reports / 'cpu_native_export_worker_20261103.py')]:
        shutil.copyfile(source, base / name)
    reference = old.parent / 'spatial_radial_pair_20261103'
    previous = json.loads((old / 'manifest.json').read_text())
    stages = {}
    for label, config in previous['stages'].items():
        work = base / label
        work.mkdir()
        ref = reference / ('perturbed48' if label == 'perturbed8' else 'baseline48')
        initial = ref / 'jorek000008.h5'
        if not initial.exists():
            raise ValueError('evolved native reference frame missing')
        shutil.copyfile(initial, work / 'initial_reference.h5')
        for name in ['input', 'eqdsk.dat', 'jorek_equil_rz.h5']:
            shutil.copyfile(old / label / name, work / name)
        stages[label] = {**config, 'initial_state_sha256': sha256_file(initial), 'expected_final_step': 16,
                         'native_source_frame': str(initial), 'initial_step': 8}
    files = [p for label in stages for p in (base / label).iterdir()]
    files += [base / 'parent.exe', base / 'worker.py', base / 'jorek-env.sh',
              base / 'source/models/model303/mod_elt_matrix_fft.f90', base / 'source/matrix/construct_matrix_mod.f90',
              ROOT / 'src/mastu_opsplit/native_export.py', ROOT / 'src/mastu_opsplit/restart.py', ROOT / 'src/mastu_opsplit/numerical.py']
    files += list((base / 'source/models/model303').glob('opsplit_export_*.inc'))
    budget = {**previous['budget'], 'scope': 'Fresh isolated recovery of copied binary permissions; unchanged parent and instrumented disabled/enabled eight-step regressions from actual step-8 states, plus perturbed labels.', 'saldo_row': row}
    manifest = {**previous, 'stages': stages, 'final_time_s': previous['final_time_s']*2, 'budget': budget,
                'input_hashes': {str(p): sha256_file(p) for p in files}}
    (base / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    inputs = {**manifest['input_hashes'], str(base / 'manifest.json'): sha256_file(base / 'manifest.json')}
    (base / 'startup.sha256').write_text(''.join(digest + '  ' + name + '\n' for name, digest in inputs.items()))
    script = reports / 'cpu_native_export_recovery_20261103.sbatch'
    script.write_text((Path(old_record['script']).read_text()).replace(str(old), str(base)).replace('mastu-native-export', 'mastu-native-export-recovery'))
    subprocess.run(['bash', '-n', str(script)], check=True)
    command = ['sbatch', '--parsable', '--chdir', str(base), '--output', str(base / 'batch-%j.out'), '--error', str(base / 'batch-%j.err'), str(script)]
    preflight = subprocess.run(['sbatch', '--test-only', *command[1:]], text=True, capture_output=True, check=True)
    record = {'status': 'prepared', 'bulk_output': str(base), 'prior_failure_job': old_record['job_id'], 'budget': budget,
              'manifest_sha256': sha256_file(base / 'manifest.json'), 'patch': str(patch), 'patch_sha256': sha256_file(patch),
              'script': str(script), 'script_sha256': sha256_file(script), 'scheduler_preflight': preflight.stderr, 'scientific_gates_passed': []}
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    job = subprocess.check_output(command, text=True).strip()
    record.update(status='submitted', job_id=job)
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'job_id': job, 'bulk_output': str(base)}))


if __name__ == '__main__':
    main()
