"""Record release evidence for the native JOREK split; diagnostics do not certify units."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import h5py
import numpy as np
import torch

from data import layout, select
from equations import JorekTerms
from formulation import plan


def fit_constraint(operator, exported):
    x, y = operator.flatten().double(), exported.flatten().double()
    if x.square().sum() == 0 or y.square().sum() == 0:
        return dict(scale=None, relative_residual=None)
    scale = (x*y).sum()/x.square().sum()
    residual = torch.linalg.vector_norm(y-scale*x)/torch.linalg.vector_norm(y)
    return dict(scale=scale.item(), relative_residual=residual.item())


def inspect(entry):
    with h5py.File(entry['path'], 'r') as handle:
        _, r, z = layout(handle, entry['case'])
        op = JorekTerms(torch.as_tensor(r, dtype=torch.float64), torch.as_tensor(z, dtype=torch.float64))
        frame = min(10, entry['frames']-1)
        potential = 'Phi' if entry['case'] == 'es' else 'u'
        x = torch.as_tensor(np.asarray(handle[potential][frame]), dtype=torch.float64)
        psi = torch.as_tensor(np.asarray(handle['Psi'][frame]), dtype=torch.float64) if entry['case'] == 'em' else x*0
        constraints = op.constraints(x, psi)
        omega = torch.as_tensor(np.asarray(handle['omega'][frame]), dtype=torch.float64)
        fits = dict(exported_omega_vs_laplacian_exported_potential=fit_constraint(constraints['omega'], op.interior(omega)))
        if entry['case'] == 'em':
            j = torch.as_tensor(np.asarray(handle['zj'][frame]), dtype=torch.float64)
            fits['exported_zj_vs_delta_star_exported_psi'] = fit_constraint(constraints['j'], op.interior(j))
        attrs = {name: {k: str(v) for k, v in item.attrs.items()} for name, item in handle.items() if item.attrs}
        metadata = dict(keys=sorted(handle.keys()), root_attributes={k: str(v) for k, v in handle.attrs.items()},
                        dataset_attributes=attrs, frames=entry['frames'],
                        potential_key=potential, sampled_R=[float(r[0]), float(r[-1])], sampled_Z=[float(z[0]), float(z[-1])])
    input_path = Path(entry['input'])
    text = input_path.read_text()
    settings = {}
    for line in text.splitlines():
        line = line.split('!')[0].strip()
        match = re.fullmatch(r'(restart|tstep|nstep|nout|central_density|central_mass|visco_old_setup|eta|visco|D_perp|ZK_perp|ZK_par)\s*=\s*(.+)', line, re.I)
        if match:
            settings[match[1].lower()] = match[2].strip()
    return dict(id=entry['id'], path=entry['path'], frame=frame, metadata=metadata,
                input_path=str(input_path), input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
                input_settings=settings, constraint_fits=fits,
                warning='Spatial fit only. This does not identify native units, elapsed time, or the generating revision.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    result = dict(manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  sampling='First three training-family trajectories per regime, seed907; frame10; raw exported values, two-cell cropped interior.',
                  cases={case: dict(plan=plan(case), samples=[inspect(e) for e in select(manifest, case, 'train', 3, 907)])
                         for case in ('es', 'em')})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({case: {'status': row['plan']['status'], 'samples': len(row['samples'])} for case, row in result['cases'].items()}))


if __name__ == '__main__':
    main()
