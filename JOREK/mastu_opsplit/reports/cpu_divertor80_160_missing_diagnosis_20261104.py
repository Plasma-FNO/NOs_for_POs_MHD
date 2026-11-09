"""Bounded inverse diagnosis of the frozen divertor80/160 uncovered queries."""
from pathlib import Path
from collections import Counter
import json,sys
import h5py
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/'src'))
from mastu_opsplit.restart import cubic_basis
from mastu_opsplit.common import sha256_file
def recover_missing(path, points):
    with h5py.File(path) as f:
        vertices = f["vertex"][()].T - 1
        jets = f["x"][()][:, :, 0, :].transpose(2, 1, 0)[vertices] * f["size"][()].transpose(2, 1, 0)[..., None]
    controls = np.stack([jets[:, :, 0], jets[:, :, 0] + jets[:, :, 1],
                         jets[:, :, 0] + jets[:, :, 2], jets.sum(axis=2)], axis=2)
    lower, upper = controls.min(axis=(1, 2)), controls.max(axis=(1, 2))
    tree = cKDTree((lower + upper) / 2)
    edge_counts = Counter(tuple(sorted((v[i], v[(i+1) % 4]))) for v in vertices for i in range(4))
    results = []
    for point in points:
        candidates = np.unique(np.concatenate([np.flatnonzero(((point >= lower-1e-10) & (point <= upper+1e-10)).all(axis=1)),
                                                tree.query(point, k=8)[1]]))
        best = (float("inf"), None, None)
        for e in candidates:
            def evaluate(st):
                h, hs, ht = cubic_basis(*st)
                return (np.einsum("vd,vdc->c", h, jets[e]) - point,
                        np.stack([np.einsum("vd,vdc->c", hs, jets[e]), np.einsum("vd,vdc->c", ht, jets[e])], axis=1))
            for s in [.05, .5, .95]:
                for t in [.05, .5, .95]:
                    solution = least_squares(lambda st: evaluate(st)[0], [s,t], jac=lambda st: evaluate(st)[1],
                                             bounds=(0.,1.), xtol=1e-14, ftol=None, gtol=None, max_nfev=150)
                    error = float(np.linalg.norm(evaluate(solution.x)[0]))
                    if error < best[0]:
                        best = (error, int(e), solution.x.copy())
                    if error <= 1e-10:
                        break
                if best[0] <= 1e-10:
                    break
            if best[0] <= 1e-10:
                break
        error, e, st = best
        h, hs, ht = cubic_basis(*st)
        position = np.einsum("vd,vdc->c", h, jets[e])
        ds = np.einsum("vd,vdc->c", hs, jets[e]); dt = np.einsum("vd,vdc->c", ht, jets[e])
        boundary = []
        for edge, distance, tangent in [(0, st[1], ds), (1, 1-st[0], dt), (2, 1-st[1], -ds), (3, st[0], -dt)]:
            key = tuple(sorted((vertices[e,edge], vertices[e,(edge+1)%4])))
            if distance < 1e-6 and edge_counts[key] == 1:
                normal = np.array([tangent[1], -tangent[0]]) / np.linalg.norm(tangent)
                boundary.append({"edge": edge, "outward_offset_m": float((point-position) @ normal)})
        results.append({"rz_m": point.tolist(), "recovered": error <= 1e-10, "best_error_m": error,
                        "element": e+1, "st": st.tolist(), "outer_boundary": boundary})
        print(json.dumps(results[-1]), flush=True)
    return results

from mastu_opsplit.restart import read_restart
from mastu_opsplit.queries import sample_restart
B=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102')
coarse=B/'xpoint_handle_pair_divertor80_20261104/baseline48/jorek000000.h5'
fine=B/'xpoint_handle_pair_divertor160_20261104/baseline48/jorek000000.h5'
proof_path=R/'cpu_xpoint_handle_divertor_next_convergence_20261104.json'
proof=json.loads(proof_path.read_text())
for path in [coarse,fine]:
 evidence=json.loads((path.parent/'result.json').read_text())
 assert evidence['status']=='completed' and sha256_file(path)==evidence['initial']['restart_sha256']
hashes={str(p):sha256_file(p) for p in [coarse,fine,proof_path]}
g,_,metadata=read_restart(fine,quadrature_order=4);points=g.quadrature_rz.reshape(-1,2);weights=g.volume_weights.ravel()
mask=sample_restart(coarse,points)['mask'];missing=points[~mask]
assert len(mask)==proof['coverage']['total_queries'] and int(mask.sum())==proof['coverage']['covered_queries']
rows=recover_missing(coarse,missing)
result={'scope':'Bounded inverse diagnosis only; no extrapolation or coverage waiver.', 'input_hashes':hashes, 'analysis_source_sha256':sha256_file(__file__), 'missing_queries':len(rows), 'recovered':sum(x['recovered'] for x in rows), 'outside_boundary':sum(any(b['outward_offset_m']>1e-10 for b in x['outer_boundary']) for x in rows), 'points':rows, 'scientific_gates_passed':[]}
for path,digest in hashes.items():assert sha256_file(path)==digest
with (R/'cpu_divertor80_160_missing_diagnosis_20261104.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps({k:result[k] for k in ['missing_queries','recovered','outside_boundary']}))
