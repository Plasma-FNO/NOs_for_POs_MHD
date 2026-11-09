"""Fix include dependencies and measure native repeatability before qualification."""
import difflib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mastu_opsplit.common import sha256_file


def main():
    reports = ROOT / 'reports'
    old_record = json.loads((reports / 'cpu_native_export_recovery_20261103.json').read_text())
    old = Path(old_record['bulk_output'])
    base = old.parent / 'native_bridge_20261103'
    record_path = reports / 'cpu_native_bridge_20261103.json'
    if base.exists() or record_path.exists():
        raise ValueError('preserve existing native bridge experiment')
    state = subprocess.check_output(['sacct', '-n', '-P', '-X', '-j', old_record['job_id'], '--format=State,ExitCode'], text=True).strip()
    if not state.startswith('CANCELLED'):
        raise ValueError('prior development allocation must be stopped')
    parent_proof = json.loads((old / 'parent8/result.json').read_text())
    if parent_proof['status'] != 'completed' or sha256_file(old / 'parent8/jorek_restart.h5') != parent_proof['final']['restart_sha256']:
        raise ValueError('healthy unchanged-parent repeatability reference required')
    row = next(line for line in subprocess.check_output(['saldo', '-b', '--dcgp', '-n'], text=True).splitlines() if line.split()[0] == 'FUPA2_FAUNA')
    cols = row.split()
    if min(int(cols[3])-int(cols[4]), int(cols[7])-int(cols[8])) < 512:
        raise ValueError('insufficient current project CPU credit')
    base.mkdir()
    shutil.copytree(old / 'source', base / 'source')
    path = base / 'source/Makefile'
    original = path.read_text()
    anchor = '$(OBJDIR)/mod_elt_matrix_fft.o $(MODDIR)/mod_elt_matrix_fft.mod: $(CGDEP)'
    if original.count(anchor) != 1:
        raise ValueError('native include dependency anchor changed')
    changed = original.replace(anchor, anchor+' $(wildcard models/$(MODEL)/opsplit_export_*.inc)')
    path.write_text(changed)
    patch = ROOT / 'patches/0018-native-export-include-dependencies.patch'
    with patch.open('x') as stream:
        stream.writelines(difflib.unified_diff(original.splitlines(True), changed.splitlines(True), fromfile='a/Makefile', tofile='b/Makefile'))
    # The previous copied object is newer than its include: force this first rebuild.
    (base / 'source/.obj/mod_elt_matrix_fft.o').unlink()
    for name, source in [('parent.exe', old / 'parent.exe'), ('jorek-env.sh', old / 'jorek-env.sh'),
                         ('worker.py', reports / 'cpu_native_export_worker_20261103.py')]:
        shutil.copyfile(source, base / name)
    (base / 'parent.exe').chmod(0o555)
    previous = json.loads((old / 'manifest.json').read_text())
    for label in previous['stages']:
        work = base / label
        work.mkdir()
        for name in ['input', 'eqdsk.dat', 'jorek_equil_rz.h5', 'initial_reference.h5']:
            shutil.copyfile(old / label / name, work / name)
    files = [p for label in previous['stages'] for p in (base / label).iterdir()]
    files += [base / 'worker.py', base / 'parent.exe', base / 'jorek-env.sh', path,
              base / 'source/models/model303/mod_elt_matrix_fft.f90', base / 'source/matrix/construct_matrix_mod.f90',
              ROOT / 'src/mastu_opsplit/native_export.py', ROOT / 'src/mastu_opsplit/native_balance.py',
              ROOT / 'src/mastu_opsplit/restart.py', ROOT / 'src/mastu_opsplit/numerical.py',
              reports / 'cpu_native_bridge_tests_20261103.xml', old / 'parent8/jorek_restart.h5', old / 'parent8/result.json']
    files += list((base / 'source/models/model303').glob('opsplit_export_*.inc'))
    budget = {**previous['budget'], 'scope': 'One corrected-include build; repeated unchanged-parent, disabled/enabled prototype and perturbed eight-step proofs; all export and parity outcomes retained.',
              'wall_hours_max': 2, 'rank_CPU_hours_max': 8, 'node_hours_max': 2, 'conservative_resource_hours_max': 512, 'saldo_row': row}
    manifest = {**previous, 'budget': budget, 'repeatability_reference': str(old / 'parent8/jorek_restart.h5'),
                'check_linearized_balance': True, 'input_hashes': {str(p): sha256_file(p) for p in files}}
    (base / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    inputs = {**manifest['input_hashes'], str(base / 'manifest.json'): sha256_file(base / 'manifest.json')}
    (base / 'startup.sha256').write_text(''.join(digest+'  '+name+'\n' for name, digest in inputs.items()))
    script = reports / 'cpu_native_bridge_20261103.sbatch'
    text = Path(old_record['script']).read_text().replace(str(old), str(base)).replace('--time=03:00:00', '--time=02:00:00').replace('mastu-native-export-recovery', 'mastu-native-bridge')
    text = text.replace(f'{"/pitagora_work/FUPB2_UKAEA_ML/vgopakum/envs/cfo/bin/python"} {base}/worker.py',
                        f"test $(sha256sum jorek_model303 | cut -d' ' -f1) != a196265da5e7d60d96b092f772aedcb42e3a420ae2db19052f0083b99b27038e\n/pitagora_work/FUPB2_UKAEA_ML/vgopakum/envs/cfo/bin/python {base}/worker.py")
    script.write_text(text)
    subprocess.run(['bash', '-n', str(script)], check=True)
    command = ['sbatch', '--parsable', '--chdir', str(base), '--output', str(base / 'batch-%j.out'), '--error', str(base / 'batch-%j.err'), str(script)]
    preflight = subprocess.run(['sbatch', '--test-only', *command[1:]], text=True, capture_output=True, check=True)
    record = {'status': 'prepared', 'bulk_output': str(base), 'prior_stopped_job': old_record['job_id'], 'budget': budget,
              'manifest_sha256': sha256_file(base / 'manifest.json'), 'patch': str(patch), 'patch_sha256': sha256_file(patch),
              'script': str(script), 'script_sha256': sha256_file(script), 'scheduler_preflight': preflight.stderr, 'scientific_gates_passed': []}
    record_path.write_text(json.dumps(record, indent=2)+'\n')
    job = subprocess.check_output(command, text=True).strip()
    record.update(status='submitted', job_id=job)
    record_path.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'job_id': job, 'bulk_output': str(base)}))


if __name__ == '__main__':
    main()
