"""Check complete fine-mesh physical query coverage before long spatial comparisons."""
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R.parent / 'src'))
from mastu_opsplit.common import sha256_file
from mastu_opsplit.queries import sample_restart
from mastu_opsplit.restart import read_restart

B = Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102')
parent = B / 'combined_refined_solver_20261103'
cases = {
    'wall': (parent / 'control3', parent / 'wall80'),
    'core_poloidal_next': (parent / 'core_poloidal440', B / 'combined_core_poloidal880_20261104/probe3'),
}
label = sys.argv[1]
coarse, fine = cases[label]
output = R / f'cpu_combined_{label}_coverage_20261104.json'
if output.exists():
    raise ValueError('preserve existing coverage evidence')
hashes = {}
qualification = json.loads((R / 'cpu_combined_refined_qualification_20261103.json').read_text())
for work in (coarse, fine):
    result = json.loads((work / 'result.json').read_text())
    if result['status'] != 'completed' or result['binary_sha256'] != qualification['binary_sha256']:
        raise ValueError('healthy same-qualified-binary probes required')
    path = work / 'jorek000000.h5'
    hashes[str(path)] = sha256_file(path)
    if hashes[str(path)] != result['initial']['restart_sha256']:
        raise ValueError('qualified initial output changed')
geometry, _, metadata = read_restart(fine / 'jorek000000.h5', quadrature_order=4)
points = geometry.quadrature_rz.reshape(-1, 2)
weights = geometry.volume_weights.ravel()
query = sample_restart(coarse / 'jorek000000.h5', points)
mask = query['mask']
report = {
    'scope': 'Fine q4 physical query coverage only; no field-convergence approval.',
    'source_sha256': sha256_file(__file__), 'restart_hashes': hashes,
    'queries_source_sha256': sha256_file(R.parent / 'src/mastu_opsplit/queries.py'),
    'fine_metadata': metadata, 'total_queries': len(mask), 'covered_queries': int(mask.sum()),
    'full_domain_coverage': bool(mask.all()),
    'uncovered_volume_m3': float(weights[~mask].sum()),
    'uncovered_rz_m': points[~mask].tolist(), 'scientific_gates_passed': [],
}
for path, digest in hashes.items():
    if sha256_file(path) != digest:
        raise ValueError('source changed during coverage audit')
with output.open('x') as stream:
    json.dump(report, stream, indent=2)
    stream.write('\n')
print(json.dumps({k: report[k] for k in ['total_queries', 'covered_queries', 'full_domain_coverage']}), flush=True)
