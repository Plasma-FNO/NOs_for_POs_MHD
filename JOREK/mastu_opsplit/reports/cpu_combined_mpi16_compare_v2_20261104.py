"""Require unchanged frozen field/gradient parity before accepting sixteen ranks."""
import json
import math
import re
import sys
from pathlib import Path

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R.parent / 'src'))
from mastu_opsplit.common import sha256_file
from cpu_native_snapshot_parity_20261104 import parity

B = Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102')
base = B / 'combined_mpi16_20261104'
manifest = json.loads((base / 'manifest.json').read_text())
for path, digest in manifest['input_hashes'].items():
    if sha256_file(path) != digest:
        raise ValueError('frozen MPI input changed')
protocol_path = R / 'cpu_native_parity_protocol_20261102.json'
protocol = json.loads(protocol_path.read_text())
report = {'status': 'incomplete', 'scientific_gates_passed': [], 'cases': {},
          'binary_sha256': manifest['binary_sha256'], 'protocol': protocol,
          'snapshot_parity_source_sha256': sha256_file(R / 'cpu_native_snapshot_parity_20261104.py'), 'protocol_sha256': sha256_file(protocol_path), 'source_sha256': sha256_file(__file__),
          'limitation': 'Fresh-start three-step baseline/perturbed MPI parity only. Larger meshes and full trajectories still need validation.'}
for case, current, reference in [
    ('baseline', base / 'baseline16', B / 'combined_refined_solver_20261103/control3'),
    ('perturbed', base / 'perturbed16', base / 'perturbed4'),
]:
    checks = {}
    for work in [current, reference]:
        proof = json.loads((work / 'result.json').read_text())
        if proof['status'] != 'completed' or proof['binary_sha256'] != manifest['binary_sha256']:
            raise ValueError('healthy same-binary MPI cases required')
        log = (work / 'stdout.log').read_text()
        errors = re.findall(r'MUMPS refinement steps and backward errors:\s*(\d+)\s+(\S+)\s+(\S+)', log)
        if len(errors) < 3 or any(int(n) > 5 or not all(math.isfinite(float(x)) and float(x) >= 0 for x in [a,b])
                                 or float(a) + float(b) > 1e-12 for n,a,b in errors):
            raise ValueError('accurate bounded linear solves required')
        for moment, name in [('initial', 'jorek000000.h5'), ('final', 'jorek_restart.h5')]:
            if sha256_file(work / name) != proof[moment]['restart_sha256']:
                raise ValueError('MPI output changed')
    for moment, name in [('initial', 'jorek000000.h5'), ('final', 'jorek_restart.h5')]:
        try:
            checks[moment] = parity(current / name, reference / name, protocol)
        except ValueError as error:
            checks[moment] = {'passed': False, 'error': str(error)}
    report['cases'][case] = checks
report['passed'] = all(x['passed'] for case in report['cases'].values() for x in case.values())
report['status'] = 'mpi16_field_gradient_parity_verified' if report['passed'] else 'failed_mpi16_parity'
with (R / 'cpu_combined_mpi16_parity_20261104.json').open('x') as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps({'status': report['status']}), flush=True)
