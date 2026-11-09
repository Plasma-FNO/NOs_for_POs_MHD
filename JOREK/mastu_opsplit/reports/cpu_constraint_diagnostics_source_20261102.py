"""Provisional native interior weak constraints; exclude boundary and axis rows."""
import gc
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from mastu_opsplit.common import sha256_file
from mastu_opsplit.restart import read_restart
from mastu_opsplit.numerical import model303_constraint_blocks


def main():
    output = ROOT/'reports/cpu_constraint_diagnostics_20261102.json'
    assert not output.exists()
    report = {'created_utc': datetime.now(timezone.utc).isoformat(),
        'script_sha256': sha256_file(__file__),
        'numerical_source_sha256': sha256_file(ROOT/'src/mastu_opsplit/numerical.py'),
        'scope': 'Source-derived interior weak current/vorticity relations only; native modified rows and full constraint elimination remain unvalidated',
        'scientific_threshold_ratified': False, 'gate_A_passed': False, 'gate_B_passed': False,
        'runs': {}}
    for name in ('cpu_fine_mpi4_equilibrium', 'cpu_fine_initializer_mpi4',
                 'cpu_fine_mpi4_baseline96', 'cpu_fine_perturbed96'):
        path = ROOT/f'runs/{name}_20261102/attempt_1/work/jorek_restart.h5'
        with h5py.File(path) as stream:
            indices = stream['index'][()].T.reshape(-1)-1
            excluded_nodes = (stream['boundary'][()] != 0) | (stream['axis_node'][()] != b'F')
            excluded = np.zeros(indices.max()+1, dtype=bool)
            np.logical_or.at(excluded, indices, np.repeat(excluded_nodes, 4))
        orders = {}
        for order in (4, 8):
            geometry, state, metadata = read_restart(path, quadrature_order=order)
            assert len(excluded) == len(state) and (~excluded).any()
            mass, gradient = model303_constraint_blocks(state, geometry)
            fields = {}
            for channel, label in enumerate(('current', 'vorticity')):
                a, b = mass[~excluded, channel], gradient[~excluded, channel]
                scale = float(np.linalg.norm(a) + np.linalg.norm(b))
                error = float(np.linalg.norm(a+b))
                fields[label] = {'relative_residual_to_component_norms': error/scale if scale else None,
                                 'absolute_l2_residual': error, 'component_norm_sum': scale}
            orders[str(order)] = fields
            del geometry, state, mass, gradient
            gc.collect()
        report['runs'][name] = {'restart_sha256': sha256_file(path), 'metadata': metadata,
            'interior_dofs': int((~excluded).sum()), 'excluded_boundary_or_axis_dofs': int(excluded.sum()),
            'quadrature_checks': orders}
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({name: data['quadrature_checks']['8'] for name, data in report['runs'].items()}, indent=2))


if __name__ == '__main__':
    main()
