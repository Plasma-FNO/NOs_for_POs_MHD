"""Compare the anchored Newton equilibrium against converged native Picard."""
import argparse
import json
import re
import sys
from pathlib import Path
import h5py
import numpy as np
from scipy.optimize import root

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mastu_opsplit.common import sha256_file
from mastu_opsplit.restart import cubic_basis, read_restart
from mastu_opsplit.numerical import evaluate_native, evaluate_native_grad
from cpu_native_parity_source_20261102 import channel_rms


def flux_span(work):
    log = (work / 'stdout.log').read_text().split(' initial energies :')[0]
    def last(key):
        return float(re.findall(r'\b' + key + r'\s*=\s*(\S+)', log)[-1])
    with h5py.File(work / 'jorek000000.h5') as f:
        vertices = f['vertex'][()].T - 1
        sizes = f['size'][()].transpose(2, 1, 0)
        values = f['values'][()][0, :, 0, :].T
        xy = f['x'][()][:, :, 0, :].transpose(2, 1, 0)
    points = {}
    for name in ['axis', 'xpoint1', 'xpoint2']:
        e = int(last('i_elm_' + name)) - 1
        local = values[vertices[e]] * sizes[e]
        def gradient(st):
            _, hs, ht = cubic_basis(*np.clip(st, 0, 1))
            return np.array([(hs * local).sum(), (ht * local).sum()])
        solution = root(gradient, [last('s_' + name), last('t_' + name)], tol=1e-11)
        assert np.linalg.norm(gradient(solution.x)) < 1e-11
        assert (solution.x >= 0).all() and (solution.x <= 1).all()
        h = cubic_basis(*solution.x)[0]
        points[name] = {'element': e + 1, 'st': solution.x.tolist(),
                        'psi': float((h * local).sum()),
                        'rz': np.einsum('vd,vdc->c', h * sizes[e], xy[vertices[e]]).tolist()}
    active = min(['xpoint1', 'xpoint2'], key=lambda n: abs(points[n]['psi'] - points['axis']['psi']))
    return {'points': points, 'active_xpoint': active,
            'delta_psi_GS': points[active]['psi'] - points['axis']['psi']}


def main():
    bulk = Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=bulk / 'wall_equilibrium_relaxation_20261103/control3')
    parser.add_argument('--current', type=Path, default=bulk / 'newton_anchored_equilibrium_20261103/control3')
    parser.add_argument('--normalized-profiles', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/cpu_newton_equilibrium_comparison_20261103.json')
    args = parser.parse_args()
    original, current = args.reference, args.current
    protocol = json.loads((ROOT / 'reports/cpu_native_parity_protocol_20261102.json').read_text())
    paths = [work / 'jorek000000.h5' for work in (original, current)]
    with h5py.File(paths[0]) as a, h5py.File(paths[1]) as b:
        for key in ['x', 'vertex', 'size', 'index', 'boundary', 'constrained', 'n_sons']:
            assert np.array_equal(a[key][()], b[key][()]), key
    geometry, expected, _ = read_restart(paths[0], quadrature_order=protocol['quadrature_order'])
    other_geometry, actual, _ = read_restart(paths[1], quadrature_order=protocol['quadrature_order'])
    assert np.array_equal(geometry.dof_ids, other_geometry.dof_ids)
    del other_geometry
    metrics = {}
    for kind, evaluate in [('field', evaluate_native), ('gradient', evaluate_native_grad)]:
        a, b = evaluate(expected, geometry), evaluate(actual, geometry)
        err = channel_rms(b - a, geometry.volume_weights, gradient=kind == 'gradient')
        norm = channel_rms(a, geometry.volume_weights, gradient=kind == 'gradient')
        tolerance = protocol['absolute_rms_tolerance_code_units'] + protocol[f'relative_{kind}_l2_tolerance'] * norm
        metrics[kind] = {name: {'absolute_rms_error': float(err[i]), 'reference_rms': float(norm[i]),
                              'relative_l2_error': float(err[i] / norm[i]) if norm[i] else None,
                              'passed': bool(err[i] <= tolerance[i])}
                         for i, name in enumerate(['psi', 'u', 'zj', 'w', 'rho', 'T', 'Vpar'])}
        del a, b
    anchor = json.loads((ROOT / 'reports/cpu_newton_flux_anchor_20261103.json').read_text())
    spans = {label: flux_span(work) for label, work in [('picard', original), ('newton', current)]}
    result = {'physical_parity_targets_met': all(x['passed'] for rows in metrics.values() for x in rows.values()),
              'metrics': metrics, 'flux_spans': spans,
              'newton_pressure_derivative_scale': 1. if args.normalized_profiles else spans['newton']['delta_psi_GS'] / anchor['delta_psi_GS'],
              'newton_uses_original_normalized_profiles': args.normalized_profiles,
              'source_hashes': {str(p): sha256_file(p) for p in paths},
              'protocol_sha256': sha256_file(ROOT / 'reports/cpu_native_parity_protocol_20261102.json'),
              'script_sha256': sha256_file(__file__), 'scientific_gates_passed': []}
    output = args.output
    with output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
