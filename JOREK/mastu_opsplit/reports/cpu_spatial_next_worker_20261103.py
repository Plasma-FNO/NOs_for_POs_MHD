"""Capacity regression, guarded finer probe, and frozen matched-horizon runs."""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(sys.argv[2])
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'reports'))
from mastu_opsplit.common import sha256_file
from mastu_opsplit.runner import classify
from cpu_spatial_pair_worker_20261103 import audit_restart, check_initializer
from cpu_cached_rz_recovery_worker_20261103 import parity


def main():
    base, stage = Path(sys.argv[1]), sys.argv[3]
    manifest = json.loads((base/'manifest.json').read_text())
    for name, expected in manifest['input_hashes'].items():
        if sha256_file(base/name) != expected:
            raise ValueError('frozen input changed: '+name)
    config = manifest['stages'][stage]
    executable = Path(config.get('binary', str(base/'source/jorek_model303')))
    if stage in ['baseline48', 'perturbed48']:
        probe = json.loads((base/'probe3/result.json').read_text())
        if probe['status'] != 'completed':
            raise ValueError('validated native probe required')
        if 'validated_unchanged_binary_sha256' in manifest:
            if sha256_file(executable) != manifest['validated_unchanged_binary_sha256']:
                raise ValueError('validated unchanged executable identity differs')
        else:
            regression = json.loads((base/'regression8/result.json').read_text())
            if regression['status'] != 'completed':
                raise ValueError('validated capacity parity required for a changed executable')
        if sha256_file(executable) != probe['binary_sha256']:
            raise ValueError('probe binary changed')
    work = base/stage
    start = time.monotonic()
    with (work/'input').open() as inp, (work/'stdout.log').open('x') as out, (work/'stderr.log').open('x') as err:
        proc = subprocess.run(['timeout', '--signal=TERM', '--kill-after=10s', str(config['timeout_seconds'])+'s',
                               'mpirun', '--bind-to', 'none', '-n', '4', str(executable)],
                              cwd=work, stdin=inp, stdout=out, stderr=err)
    text = (work/'stdout.log').read_text()+(work/'stderr.log').read_text()
    result = {'stage': stage, 'status': classify(proc.returncode, proc.returncode==124, text),
              'returncode': proc.returncode, 'process_wall_seconds': time.monotonic()-start,
              'binary_sha256': sha256_file(executable), 'scientific_gates_passed': []}
    try:
        if result['status'] != 'completed':
            raise ValueError(result['status'])
        if (work/'stderr.log').stat().st_size or 'Trying to open response file' in text or 'MUMPS failure during' in text:
            raise ValueError('native stderr, unexpected boundary response, or solver failure requires review')
        if 'equilibrium not fully converged' in text:
            raise ValueError('native equilibrium did not reach the requested stopping tolerance')
        result['initial'] = audit_restart(work/'jorek000000.h5', 0, 0.)
        result['final'] = audit_restart(work/'jorek_restart.h5', config['steps'],
                                         manifest['final_time_s']*config['steps']/48)
        if stage == 'probe3' and 'planned_geometry_reference' in manifest:
            import h5py
            import numpy as np
            with h5py.File(work/'jorek000000.h5') as actual, h5py.File(manifest['planned_geometry_reference']) as expected:
                for key in ['x','vertex','size','index','boundary','constrained','n_sons','neighbours']:
                    if not np.array_equal(actual[key][()], expected[key][()]):
                        raise ValueError('planned geometry changed with validated executable: '+key)
            result['planned_geometry_exact'] = True
        if stage == 'regression8':
            reference = base/config.get('parity_reference', 'reference8.h5')
            if 'parity_reference' in config:
                parent = json.loads((reference.parent/'result.json').read_text())
                if parent['status'] != 'completed' or sha256_file(reference) != parent['final']['restart_sha256']:
                    raise ValueError('same-allocation parent replay not validated')
            result['parity'] = parity(work/'jorek_restart.h5', reference,
                                      manifest['parity_protocol'])
        elif stage in ['baseline48', 'perturbed48']:
            initial_log = (base/'probe3/stdout.log').read_text().split(' initial energies :')[0]
            axis = float(re.findall(r'Psi_axis\s*=\s*([\-+0-9.Ee]+)', initial_log)[-1])
            bnd = float(re.findall(r'Psi_bnd\s*=\s*([\-+0-9.Ee]+)', initial_log)[-1])
            result['initializer'] = check_initializer(work/'jorek000000.h5', base/'probe3/jorek000000.h5',
                                                       config['amplitudes'], axis, bnd)
        result['workspace_retries'] = text.count('MUMPS workspace retry:')
    except Exception as error:
        result.update(status='failed_validation', error=str(error))
    (work/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result), flush=True)
    return 0 if result['status']=='completed' else 1


if __name__ == '__main__':
    sys.exit(main())
