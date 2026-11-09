#!/usr/bin/env python3
"""Native check of the exact D_perp(1) dependence used by implicit_coefficient.py (training states only).

For each selected TRAINING restart, runs the patched native export (which verifies its closed-form
coefficients against the untouched native RHS at every quadrature point) at D_perp(1) = m*1e-5 for
m = 1 and each requested multiplier, then checks pointwise that

    c(q; m) - c(q; 1) = (m-1) 1e-5 g(q)          in the two density flux channels, zero elsewhere,
    J(q; m) - J(q; 1) = (m-1) 1e-5 dg/djets(q)   in the same two output channels, zero elsewhere,

with g from the closed form. No OOD trajectory is read.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

from mastu_opsplit.common import write_json
from mastu_opsplit.implicit_coefficient import (BASELINE_D_PERP, JET_COLUMNS, PSI_R, PSI_Z, RHO_FLUX_CHANNELS, RHO_R,
                                                RHO_Z, diffusion_direction, diffusion_jacobian)
from mastu_opsplit.implicit_labels import read_export, run_export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True, help='label plan.json (training restarts, executable)')
    parser.add_argument('--states', nargs='+', default=['mastu_000:48', 'mastu_060:92'], help='case:frame')
    parser.add_argument('--multipliers', type=float, nargs='+', default=[2., 10.])
    parser.add_argument('--F0', type=float, default=0.64)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    entries = {(e['case_id'], e['frame']): e for e in plan['entries']}
    train = set(plan['cases'])   # pooled 80 training trajectories only
    report = dict(plan=str(args.plan), executable_dir=plan['executable_dir'], F0=args.F0, baseline=BASELINE_D_PERP,
                  ood_trajectories_read=False, states=[])
    for spec in args.states:
        case_id, frame = spec.split(':')
        entry = entries[case_id, int(frame)]   # KeyError unless it is a planned training state
        if case_id not in train:
            raise ValueError('Only training states may be used')
        with h5py.File(entry['restart']) as stream:
            vertices = stream['vertex'][()].T
        exports, seconds = {}, {}
        for multiplier in [1.] + args.multipliers:
            started = time.monotonic()
            files = run_export(entry['restart'], args.work/f'{case_id}_{frame}_m{multiplier:g}', plan['executable_dir'],
                               namelist={'D_perp(1)': float(multiplier*BASELINE_D_PERP)})
            exports[multiplier] = read_export(files, vertices)
            seconds[multiplier] = time.monotonic()-started
        base = exports[1.]
        jets = base[..., 3:24]
        g = np.stack(diffusion_direction(jets[..., PSI_R], jets[..., PSI_Z], jets[..., RHO_R], jets[..., RHO_Z], args.F0), -1)
        dg = diffusion_jacobian(jets, args.F0)
        rows = []
        for multiplier in args.multipliers:
            other = exports[multiplier]
            delta = (multiplier-1.)*BASELINE_D_PERP
            jets_change = float(np.abs(other[..., :24]-base[..., :24]).max())
            dc = other[..., 34:49]-base[..., 34:49]
            predicted = np.zeros_like(dc)
            predicted[..., list(RHO_FLUX_CHANNELS)] = delta*g
            dJ = (other[..., 49:364]-base[..., 49:364]).reshape(*base.shape[:2], 21, 15)
            predicted_J = np.zeros_like(dJ)
            for i, column in enumerate(JET_COLUMNS):
                predicted_J[..., column, list(RHO_FLUX_CHANNELS)] = delta*dg[..., i, :]
            scale_c = np.abs(base[..., 34:49]).max(axis=(0, 1)) + 1e-300
            scale_J = np.abs(base[..., 49:364]).reshape(*base.shape[:2], 21, 15).max(axis=(0, 1)) + 1e-300
            flux = list(RHO_FLUX_CHANNELS)
            row = dict(multiplier=multiplier, delta=delta, state_change=jets_change,
                       density_flux_change_over_baseline_flux=float(np.abs(dc[..., flux]).max()/scale_c[flux].max()),
                       forcing_error_over_change=float(np.abs(dc-predicted)[..., flux].max()/np.abs(dc[..., flux]).max()),
                       other_channels_max_change_over_scale=float((np.abs(dc).max(axis=(0, 1))/scale_c)[
                           [k for k in range(15) if k not in flux]].max()),
                       tangent_error_over_change=float(np.abs(dJ-predicted_J)[..., flux].max()/np.abs(dJ[..., flux]).max()),
                       tangent_other_max_change_over_scale=float(np.max(np.abs(dJ).max(axis=(0, 1))/scale_J
                                                                       * (np.arange(15) != flux[0]) * (np.arange(15) != flux[1]))),
                       diffusion_coefficient_ratio_a3=float(np.median(other[..., 26]/base[..., 26])))
            rows.append(row)
            print(json.dumps(dict(state=spec, **row)), flush=True)
        report['states'].append(dict(state=spec, restart=entry['restart'], seconds=seconds, rows=rows))
    worst = lambda key: max(r[key] for s in report['states'] for r in s['rows'])
    report['worst'] = {key: worst(key) for key in ('state_change', 'forcing_error_over_change',
                                                   'other_channels_max_change_over_scale', 'tangent_error_over_change',
                                                   'tangent_other_max_change_over_scale')}
    report['passed'] = bool(report['worst']['state_change'] == 0 and report['worst']['forcing_error_over_change'] < 1e-9
                            and report['worst']['other_channels_max_change_over_scale'] < 1e-12
                            and report['worst']['tangent_error_over_change'] < 1e-8
                            and report['worst']['tangent_other_max_change_over_scale'] < 1e-12)
    write_json(args.output, report)
    print(json.dumps(dict(passed=report['passed'], worst=report['worst'])), flush=True)


if __name__ == '__main__':
    main()
