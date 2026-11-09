"""Audit native bicubic geometry at q8 before committing to time evolution."""
import argparse
import json
import sys
from pathlib import Path
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mastu_opsplit.restart import cubic_basis
from mastu_opsplit.common import sha256_file

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('grid', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
points, _ = np.polynomial.legendre.leggauss(8)
s, t = np.meshgrid((points+1)/2, (points+1)/2, indexing='ij')
_, hs, ht = cubic_basis(s.ravel(), t.ravel())
with h5py.File(args.grid) as f:
    vertices = f['vertex'][()].T-1
    sizes = f['size'][()].transpose(2, 1, 0)
    xy = f['x'][()][:, :, 0, :].transpose(2, 1, 0)
assert all(np.isfinite(a).all() for a in (sizes, xy))
bad, minimum = [], np.inf
for start in range(0, len(vertices), 1024):
    v = vertices[start:start+1024]
    jets = xy[v]*sizes[start:start+1024, ..., None]
    rs, rt = [np.einsum('qvd,evdc->eqc', h, jets) for h in (hs, ht)]
    jac = rs[..., 0]*rt[..., 1]-rs[..., 1]*rt[..., 0]
    scale = np.linalg.norm(rs, axis=-1)*np.linalg.norm(rt, axis=-1)
    assert np.isfinite(jac).all()
    minimum = min(minimum, float(jac.min()))
    invalid = (jac <= 64*np.finfo(float).eps*scale).any(axis=1)
    bad.extend({'element': int(start+i+1), 'corners': xy[v[i], 0].tolist(),
                'minimum_jacobian_m2': float(jac[i].min()), 'maximum_jacobian_m2': float(jac[i].max())}
               for i in np.flatnonzero(invalid))
result = {'grid': str(args.grid), 'grid_sha256': sha256_file(args.grid), 'quadrature_order': 8,
          'elements': len(vertices), 'minimum_signed_jacobian_m2': minimum,
          'positive_nonsingular_at_q8': not bad, 'invalid_elements': bad, 'scientific_gates_passed': []}
with args.output.open('x') as f:
    json.dump(result, f, indent=2, allow_nan=False)
    f.write('\n')
print(json.dumps({k: v for k, v in result.items() if k != 'invalid_elements'}))
