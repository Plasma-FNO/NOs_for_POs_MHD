"""Measure radial80/160/320 differences on identical physical queries; no gate approval."""
import gc
import json
import sys
from pathlib import Path

import h5py
import numpy as np

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R.parent / 'src'))
from mastu_opsplit.common import sha256_file
from mastu_opsplit.numerical import evaluate_native
from mastu_opsplit.queries import sample_restart
from mastu_opsplit.restart import read_restart
from cpu_spatial_directions_analysis_20261103 import CHANNELS, GEOMETRY_KEYS, pressure_fields, sample_fixed_geometry
from cpu_spatial_next_launch_20261103 import physics_text

B = Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102')
bases = {80: B / 'combined_pair_control80_20261104',
         160: B / 'combined_pair_radial160_20261104',
         320: B / 'combined_pair_radial320_20261104'}
output = R / 'cpu_combined_radial_order_20261104.json'
if output.exists():
    raise ValueError('preserve existing convergence-order evidence')
qualification = json.loads((R / 'cpu_combined_refined_qualification_20261103.json').read_text())
hashes = {}
for n, base in bases.items():
    for case in ['baseline48', 'perturbed48']:
        result = json.loads((base / case / 'result.json').read_text())
        if result['status'] != 'completed' or result['binary_sha256'] != qualification['binary_sha256']:
            raise ValueError('healthy qualified same-binary trajectory required')
        if result['final']['health']['steps'] != 48 or abs(result['final']['health']['final_time_s'] - 1.4059131112280784e-6) > 1e-15:
            raise ValueError('common frozen horizon required')
        if physics_text(base / case / 'input') != physics_text(bases[320] / case / 'input'):
            raise ValueError('physics or numerical controls differ')
        path = base / case / 'jorek_restart.h5'
        hashes[str(path)] = sha256_file(path)
        if hashes[str(path)] != result['final']['restart_sha256']:
            raise ValueError('trajectory changed')
fine = bases[320] / 'baseline48/jorek_restart.h5'
geometry, state, _ = read_restart(fine, quadrature_order=4)
points = geometry.quadrature_rz.reshape(-1, 2).copy()
weights = geometry.volume_weights.ravel().copy()
fields = {(320, 'baseline48'): pressure_fields(evaluate_native(state, geometry).reshape(-1, 7))}
del geometry, state
gc.collect()
geometry, state, _ = read_restart(bases[320] / 'perturbed48/jorek_restart.h5', quadrature_order=4)
if not np.array_equal(points, geometry.quadrature_rz.reshape(-1, 2)):
    raise ValueError('fine geometry differs between paired runs')
fields[320, 'perturbed48'] = pressure_fields(evaluate_native(state, geometry).reshape(-1, 7))
del geometry, state
gc.collect()
coverage = {}
mask = np.ones(len(points), dtype=bool)
for n in [80, 160]:
    path = bases[n] / 'baseline48/jorek_restart.h5'
    query = sample_restart(path, points)
    coverage[n] = {'total_queries': len(points), 'covered_queries': int(query['mask'].sum())}
    mask &= query['mask']
    fields[n, 'baseline48'] = pressure_fields(query['values'])
    with h5py.File(path) as stream:
        arrays = {key: stream[key][()] for key in GEOMETRY_KEYS}
    fields[n, 'perturbed48'] = pressure_fields(sample_fixed_geometry(
        bases[n] / 'perturbed48/jorek_restart.h5', query, arrays))
    del arrays, query
    gc.collect()
metrics = {}
for case in ['baseline48', 'perturbed48', 'paired_response']:
    values = {n: fields[n, 'perturbed48'] - fields[n, 'baseline48'] if case == 'paired_response'
              else fields[n, case] for n in bases}
    differences = [np.einsum('nc,n->c', (values[a][mask] - values[b][mask])**2, weights[mask])
                   for a, b in [(80, 160), (160, 320)]]
    norm2 = np.einsum('nc,n->c', values[320][mask]**2, weights[mask])
    metrics[case] = {}
    for i, name in enumerate(CHANNELS):
        a, b = [float(np.sqrt(x[i])) for x in differences]
        norm = float(np.sqrt(norm2[i]))
        metrics[case][name] = {
            'difference_80_160_relative_to_320': a / norm if norm else None,
            'difference_160_320_relative_to_320': b / norm if norm else None,
            'observed_order': float(np.log2(a / b)) if a and b else None,
            'difference_decreases': b < a,
        }
for path, digest in hashes.items():
    if sha256_file(path) != digest:
        raise ValueError('output changed during analysis')
report = {'scope': 'Three-level radial sensitivity on common radial320 q4 physical queries at fixed timestep.',
          'limitation': 'Other spatial counts are fixed. Observed order does not certify other directions or continuum error.',
          'source_sha256': sha256_file(__file__), 'restart_hashes': hashes,
          'coverage': coverage, 'full_domain_coverage': bool(mask.all()), 'metrics': metrics,
          'scientific_gates_passed': []}
with output.open('x') as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps({'full_domain_coverage': bool(mask.all()), 'report': str(output)}), flush=True)
