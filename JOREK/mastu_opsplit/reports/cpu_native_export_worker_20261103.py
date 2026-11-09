"""Compare unchanged and instrumented native solves; verify independent labels."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

BASE, ROOT = map(Path, sys.argv[1:3])
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'reports'))
from mastu_opsplit.common import sha256_file
from mastu_opsplit.native_export import validate_export
from mastu_opsplit.native_balance import linearized_volume_balance
from cpu_cached_rz_recovery_worker_20261103 import parity
from cpu_spatial_pair_worker_20261103 import audit_restart


def main():
    manifest = json.loads((BASE / 'manifest.json').read_text())
    for path, digest in manifest['input_hashes'].items():
        if sha256_file(path) != digest:
            raise ValueError('frozen native experiment input changed: ' + path)
    results = {}
    for label, config in manifest['stages'].items():
        work = BASE / label
        executable = BASE / ('parent.exe' if label == 'parent8' else 'source/jorek_model303')
        shutil.copyfile(work / 'initial_reference.h5', work / 'jorek_restart.h5')
        env = dict(os.environ)
        env['MASTU_NATIVE_EXPORT'] = '1' if config['export_enabled'] else '0'
        with (work / 'input').open() as inp, (work / 'stdout.log').open('x') as out, (work / 'stderr.log').open('x') as err:
            process = subprocess.run(['timeout', '--signal=TERM', '--kill-after=10s', str(config['timeout_seconds'])+'s',
                                      'mpirun', '--bind-to', 'none', '-n', '4', str(executable)],
                                     cwd=work, env=env, stdin=inp, stdout=out, stderr=err)
        result = {'status': 'failed_validation', 'returncode': process.returncode, 'binary_sha256': sha256_file(executable), 'scientific_gates_passed': []}
        try:
            text = (work / 'stdout.log').read_text()
            if process.returncode or (work / 'stderr.log').stat().st_size or any(s in text for s in ['MUMPS failure during', 'equilibrium not fully converged', 'Trying to open response file']):
                raise ValueError('native solve failed or emitted unreviewed diagnostics')
            result['final'] = audit_restart(work / 'jorek_restart.h5', config.get('expected_final_step', 8), manifest['final_time_s'])
            if label == 'parent8' and manifest.get('repeatability_reference'):
                try:
                    result['native_repeatability'] = parity(work / 'jorek_restart.h5', manifest['repeatability_reference'], manifest['parity_protocol'])
                except Exception as error:
                    result['native_repeatability'] = {'passed': False, 'error': str(error)}
            if label in ['disabled8', 'enabled8']:
                result['parent_parity'] = parity(work / 'jorek_restart.h5', BASE / 'parent8/jorek_restart.h5', manifest['parity_protocol'])
            if label == 'enabled8':
                result['export_toggle_parity'] = parity(work / 'jorek_restart.h5', BASE / 'disabled8/jorek_restart.h5', manifest['parity_protocol'])
            result['status'] = 'completed'
        except Exception as error:
            result['error'] = str(error)
        # Same-state algebra can be assessed even if a trajectory parity test fails.
        if config['export_enabled'] and list(work.glob('opsplit-native-rank-*.bin')):
            try:
                result['native_export'] = validate_export(work, work / 'initial_reference.h5')
                if manifest.get('check_linearized_balance'):
                    result['native_linearized_volume_balance'] = linearized_volume_balance(
                        work, work / 'initial_reference.h5', work / 'jorek000009.h5', result['native_export'])
                    if result['native_linearized_volume_balance']['status'] != 'interior_linearized_balance_verified':
                        result['status'] = 'failed_validation'
            except Exception as error:
                result['native_export'] = {'status': 'failed_validation', 'error': str(error)}
                result['status'] = 'failed_validation'
        elif config['export_enabled']:
            result.update(status='failed_validation', export_error='native export files missing')
        (work / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        results[label] = result
        (BASE / 'result.json').write_text(json.dumps({'stages': results, 'scientific_gates_passed': []}, indent=2, allow_nan=False) + '\n')
        print(json.dumps({'stage': label, 'status': result['status'], 'export_status': result.get('native_export', {}).get('status')}), flush=True)
        if label == 'parent8' and result['status'] != 'completed':
            return 1
    qualified = all(x['status'] == 'completed' and x.get('native_repeatability', {}).get('passed', True) for x in results.values())
    return 0 if qualified else 1


if __name__ == '__main__':
    sys.exit(main())
