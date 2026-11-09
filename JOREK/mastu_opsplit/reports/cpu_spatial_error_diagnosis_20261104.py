"""Localize preserved spatial errors and distinguish missed inverse maps from domain gaps."""
from collections import Counter
from pathlib import Path
import gc
import json
import sys

import h5py
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R.parent / "src"))
from mastu_opsplit.common import sha256_file
from mastu_opsplit.numerical import evaluate_native
from mastu_opsplit.queries import sample_restart
from mastu_opsplit.restart import cubic_basis, read_restart
from cpu_spatial_directions_analysis_20261103 import GEOMETRY_KEYS, pressure_fields, sample_fixed_geometry

B = Path("/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102")
output = R / "cpu_spatial_error_diagnosis_20261104.json"
if output.exists():
    raise ValueError("preserve existing diagnosis")
report = {"scientific_gates_passed": [], "script_sha256": sha256_file(__file__),
          "scope": "Diagnosis of preserved legacy trajectories; no acceptance limits changed.", "cases": {}}


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


for label in ["core_poloidal440", "radial160"]:
    coarse = B / "exact_pair_radial80_20261103"
    fine = B / f"exact_pair_{label}_20261103"
    initial = fine / "baseline48/jorek000000.h5"
    for base in [coarse, fine]:
        proof = json.loads((base / "baseline48/result.json").read_text())
        if sha256_file(base / "baseline48/jorek000000.h5") != proof["initial"]["restart_sha256"]:
            raise ValueError("frozen initial geometry changed")
    geometry, state, _ = read_restart(initial, quadrature_order=4)
    points = geometry.quadrature_rz.reshape(-1,2).copy()
    weights = geometry.volume_weights.ravel().copy()
    eqfields = evaluate_native(state, geometry).reshape(-1,7)
    psin = (eqfields[:,0] + .10327) / (.10327 - .01228)
    del eqfields, state, geometry
    gc.collect()
    query = sample_restart(coarse / "baseline48/jorek000000.h5", points)
    missing = points[~query["mask"]]
    diagnosis = {"missing_queries": len(missing), "missing_inverse_diagnosis": recover_missing(coarse / "baseline48/jorek000000.h5", missing),
                 "source_hashes": {}, "regions": {}, "region_note": "Approximate initial normalized-flux bins used only for error localization, never acceptance."}
    with h5py.File(coarse / "baseline48/jorek000000.h5") as f:
        old_geometry = {k: f[k][()] for k in GEOMETRY_KEYS}
    fields = {}
    for case in ["baseline48", "perturbed48"]:
        for mesh, base in [("coarse", coarse), ("fine", fine)]:
            path = base / case / "jorek_restart.h5"
            proof = json.loads((base / case / "result.json").read_text())
            if proof["status"] != "completed" or sha256_file(path) != proof["final"]["restart_sha256"]:
                raise ValueError("healthy immutable final trajectory required")
            diagnosis["source_hashes"][str(path)] = sha256_file(path)
            if mesh == "coarse":
                values = sample_fixed_geometry(path, query, old_geometry)
            else:
                geometry, state, _ = read_restart(path, quadrature_order=4)
                if not np.array_equal(points, geometry.quadrature_rz.reshape(-1,2)):
                    raise ValueError("fine geometry changed")
                values = evaluate_native(state, geometry).reshape(-1,7)
                del geometry, state
            fields[mesh,case] = pressure_fields(values)
            gc.collect()
    regions = {"inner_core": psin < .8, "edge_core": (psin >= .8)&(psin < 1.), "outside_core": psin >= 1.,
               "midplane": np.abs(points[:,1]) < 1., "ends": np.abs(points[:,1]) >= 1.}
    mask = query["mask"]
    for case in ["baseline48", "perturbed48", "paired_response"]:
        if case == "paired_response":
            a = fields["coarse","perturbed48"] - fields["coarse","baseline48"]
            b = fields["fine","perturbed48"] - fields["fine","baseline48"]
        else:
            a,b = fields["coarse",case], fields["fine",case]
        error2 = (a-b)**2 * weights[:,None]
        total = error2[mask].sum(axis=0)
        diagnosis["regions"][case] = {name: {channel: float(error2[mask & region,i].sum()/total[i]) if total[i] else 0.
                                             for i,channel in enumerate(["psi","u","zj","w","rho","T","Vpar","pressure"])}
                                        for name,region in regions.items()}
        for channel,i in [("w",3),("zj",2),("Vpar",6)]:
            indices = np.flatnonzero(mask)
            top = indices[np.argsort(error2[indices,i])[-10:][::-1]]
            diagnosis.setdefault("largest_error_points", {}).setdefault(case, {})[channel] = [
                {"rz_m": points[k].tolist(), "psin": float(psin[k]), "fraction_squared_error": float(error2[k,i]/total[i]) if total[i] else 0.}
                for k in top]
    report["cases"][label] = diagnosis
    del fields, query, old_geometry, points, weights, psin, values, a, b, error2
    gc.collect()
    print(json.dumps({"case_done": label, "missing": len(missing)}), flush=True)
with output.open("x") as f:
    json.dump(report, f, indent=2, allow_nan=False)
    f.write("\n")
