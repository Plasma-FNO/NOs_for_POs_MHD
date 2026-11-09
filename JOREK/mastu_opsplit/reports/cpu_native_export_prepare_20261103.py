"""Prepare an isolated native-label build and matched restart regression."""
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

PARENT = Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/restart_capacity_recovery_20261103/source')
REFERENCE = PARENT.parents[1] / 'spatial_radial_pair_20261103'
BASE = PARENT.parents[1] / 'native_export_20261103'


def replace_once(text, anchor, replacement):
    if text.count(anchor) != 1:
        raise ValueError('native source anchor is not unique: ' + anchor[:100])
    return text.replace(anchor, replacement, 1)


def instrument(source):
    """Add diagnostics without rewriting any native RHS or matrix expression."""
    path = source / 'models/model303/mod_elt_matrix_fft.f90'
    original = path.read_text()
    text = replace_once(original, '  implicit none\n\ncontains', '  implicit none\n#include "opsplit_export_state.inc"\n\ncontains')
    text = replace_once(text, 'ELM_p = 0.d0\n', '#include "opsplit_export_declarations.inc"\n\nELM_p = 0.d0\n')
    text = replace_once(text, 'rhs_ij_1   = 0.d0 ;', '#include "opsplit_export_begin.inc"\n\nrhs_ij_1   = 0.d0 ;')
    text = replace_once(text, '            !#  RHS equations end', '            !#  RHS equations end')
    anchor = '            if (use_fft) then\n\n              index_ij ='
    text = replace_once(text, anchor, '#include "opsplit_export_point.inc"\n\n' + anchor)
    text = replace_once(text, 'if (.NOT. use_fft) return', '#include "opsplit_export_finish.inc"\n\nif (.NOT. use_fft) return')
    path.write_text(text)
    patches = list(difflib.unified_diff(original.splitlines(True), text.splitlines(True), fromfile='a/models/model303/mod_elt_matrix_fft.f90', tofile='b/models/model303/mod_elt_matrix_fft.f90'))
    for name in ['state', 'declarations', 'begin', 'point', 'finish']:
        target = source / f'models/model303/opsplit_export_{name}.inc'
        shutil.copyfile(ROOT / f'native/opsplit_export_{name}.f90', target)
        patches += list(difflib.unified_diff([], target.read_text().splitlines(True), fromfile='/dev/null', tofile='b/models/model303/' + target.name))
    path = source / 'matrix/construct_matrix_mod.f90'
    original = path.read_text()
    text = replace_once(original, 'only : element_matrix_fft\n', 'only : element_matrix_fft, opsplit_export_element_id\n')
    text = replace_once(text, '    ! --- Call element_matrix\n', '    opsplit_export_element_id = ielm\n\n    ! --- Call element_matrix\n')
    path.write_text(text)
    patches += list(difflib.unified_diff(original.splitlines(True), text.splitlines(True), fromfile='a/matrix/construct_matrix_mod.f90', tofile='b/matrix/construct_matrix_mod.f90'))
    path = source / 'Makefile'
    original = path.read_text()
    anchor = '$(OBJDIR)/mod_elt_matrix_fft.o $(MODDIR)/mod_elt_matrix_fft.mod: $(CGDEP)'
    text = replace_once(original, anchor, anchor+' $(wildcard models/$(MODEL)/opsplit_export_*.inc)')
    path.write_text(text)
    patches += list(difflib.unified_diff(original.splitlines(True), text.splitlines(True), fromfile='a/Makefile', tofile='b/Makefile'))
    return ''.join(patches)


def main():
    record_path = ROOT / 'reports/cpu_native_export_20261103.json'
    if BASE.exists() or record_path.exists():
        raise ValueError('preserve existing native export experiment')
    binary_sha = 'dd786f76a5c815976ad54bcd31b744ad2c3d6cafddfcc064dcae8e6761d8ccd6'
    if sha256_file(PARENT / 'jorek_model303') != binary_sha:
        raise ValueError('validated parent binary changed')
    row = next(line for line in subprocess.check_output(['saldo', '-b', '--dcgp', '-n'], text=True).splitlines() if line.split()[0] == 'FUPA2_FAUNA')
    cols = row.split()
    if min(int(cols[3])-int(cols[4]), int(cols[7])-int(cols[8])) < 768:
        raise ValueError('insufficient verified CPU project credit')
    BASE.mkdir()
    shutil.copytree(PARENT, BASE / 'source')
    patch = ROOT / 'patches/0016-native-selected-export.patch'
    with patch.open('x') as stream:
        stream.write(instrument(BASE / 'source'))
    for name, original in [('parent.exe', PARENT / 'jorek_model303'), ('jorek-env.sh', REFERENCE / 'jorek-env.sh'),
                           ('worker.py', ROOT / 'reports/cpu_native_export_worker_20261103.py')]:
        shutil.copyfile(original, BASE / name)
    (BASE / 'parent.exe').chmod(0o555)
    stages = {}
    for label, reference_label, enabled in [('parent8', 'baseline48', False), ('disabled8', 'baseline48', False),
                                           ('enabled8', 'baseline48', True), ('perturbed8', 'perturbed48', True)]:
        work = BASE / label
        work.mkdir()
        ref = REFERENCE / reference_label
        result = json.loads((ref / 'result.json').read_text())
        if result['status'] != 'completed' or sha256_file(ref / 'jorek000000.h5') != result['initial']['restart_sha256']:
            raise ValueError('healthy reference initial state required')
        overrides = {'restart': True, 'nstep_n': 8, 'nout': 1, 'opsplit_amplitude(1)': 0., 'opsplit_amplitude(2)': 0.}
        (work / 'input').write_text(namelist_overrides((ref / 'input').read_text(), overrides))
        for source_name, target_name in [('jorek000000.h5', 'initial_reference.h5'), ('eqdsk.dat', 'eqdsk.dat'), ('jorek_equil_rz.h5', 'jorek_equil_rz.h5')]:
            shutil.copyfile(ref / source_name, work / target_name)
        stages[label] = {'export_enabled': enabled, 'initial_state_sha256': result['initial']['restart_sha256'], 'timeout_seconds': 2100}
    source_files = [BASE / 'source/models/model303/mod_elt_matrix_fft.f90', BASE / 'source/matrix/construct_matrix_mod.f90']
    source_files += list((BASE / 'source/models/model303').glob('opsplit_export_*.inc'))
    files = [p for label in stages for p in (BASE / label).iterdir()] + source_files
    files += [BASE / 'worker.py', BASE / 'parent.exe', BASE / 'jorek-env.sh', ROOT / 'src/mastu_opsplit/native_export.py',
              ROOT / 'src/mastu_opsplit/numerical.py', ROOT / 'src/mastu_opsplit/restart.py',
              ROOT / 'reports/cpu_native_export_tests_20261103.xml']
    manifest = {'stages': stages, 'parent_binary_sha256': binary_sha, 'final_time_s': 1.4059131112280784e-6*8/48,
                'input_hashes': {str(p): sha256_file(p) for p in files},
                'parity_protocol': json.loads((ROOT / 'reports/cpu_native_parity_protocol_20261102.json').read_text()),
                'budget': {'authorization': 'Explicit user request to resolve remaining Gates A/B/C blockers for validated data.',
                           'scope': 'One isolated build plus parent/disabled/enabled baseline eight-step restart regressions and enabled perturbed eight-step export.',
                           'allocations_max': 1, 'mpi_ranks': 4, 'threads': 1, 'memory_GiB': 256, 'wall_hours_max': 3,
                           'rank_CPU_hours_max': 12, 'node_hours_max': 3, 'conservative_resource_hours_max': 768,
                           'compiler_processes_max': 2, 'storage_GiB_max': 32, 'attempts_per_stage_max': 1, 'saldo_row': row}}
    (BASE / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    inputs = {**manifest['input_hashes'], str(BASE / 'manifest.json'): sha256_file(BASE / 'manifest.json')}
    (BASE / 'startup.sha256').write_text(''.join(digest + '  ' + name + '\n' for name, digest in inputs.items()))
    script = ROOT / 'reports/cpu_native_export_20261103.sbatch'
    python = '/pitagora_work/FUPB2_UKAEA_ML/vgopakum/envs/cfo/bin/python'
    script.write_text(f'''#!/bin/bash
#SBATCH --job-name=mastu-native-export
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_prod
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=256G
#SBATCH --time=03:00:00
set -euo pipefail
source {BASE}/jorek-env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
sha256sum --check {BASE}/startup.sha256 > {BASE}/startup_verification.log
cd {BASE}/source
timeout --signal=TERM --kill-after=10s 600s make -j2 jorek_model303 > {BASE}/build.out 2> {BASE}/build.err
{python} {BASE}/worker.py {BASE} {ROOT}
''')
    subprocess.run(['bash', '-n', str(script)], check=True)
    command = ['sbatch', '--parsable', '--chdir', str(BASE), '--output', str(BASE / 'batch-%j.out'), '--error', str(BASE / 'batch-%j.err'), str(script)]
    preflight = subprocess.run(['sbatch', '--test-only', *command[1:]], text=True, capture_output=True, check=True)
    record = {'status': 'prepared', 'bulk_output': str(BASE), 'manifest_sha256': sha256_file(BASE / 'manifest.json'),
              'patch': str(patch), 'patch_sha256': sha256_file(patch), 'script': str(script), 'script_sha256': sha256_file(script),
              'budget': manifest['budget'], 'scheduler_preflight': preflight.stderr, 'scientific_gates_passed': []}
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    job = subprocess.check_output(command, text=True).strip()
    record.update(status='submitted', job_id=job)
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'job_id': job, 'bulk_output': str(BASE)}))


if __name__ == '__main__':
    main()
