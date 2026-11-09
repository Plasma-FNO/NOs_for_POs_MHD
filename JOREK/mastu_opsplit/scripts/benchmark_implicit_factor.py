#!/usr/bin/env python3
"""Factorization cost and solve accuracy of the implicit system on the real native mesh."""
import argparse
from dataclasses import fields
import json
from pathlib import Path
import resource
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

from mastu_opsplit.implicit_opsplit import ImplicitNativeSystem, load_frozen_tangent
from mastu_opsplit.numerical import NativeGeometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', required=True)
    parser.add_argument('--normalization', required=True)
    parser.add_argument('--fixed', required=True)
    parser.add_argument('--tangent', required=True)
    parser.add_argument('--ordering', default='metis')
    parser.add_argument('--fractions', type=float, nargs='+', default=[1.])
    args = parser.parse_args()
    with h5py.File(args.geometry) as stream:
        geometry = NativeGeometry(**{item.name: stream[item.name][()] for item in fields(NativeGeometry)}).validate()
    with np.load(args.normalization) as saved:
        mean, scale = saved['mean'], saved['scale']
    with np.load(args.fixed) as saved:
        fixed = saved['fixed']
    t = time.monotonic()
    system = ImplicitNativeSystem(geometry, load_frozen_tangent(args.tangent, geometry), mean, scale, fixed,
                                  code_interval=0.031249999999998657, ordering=args.ordering)
    print(json.dumps(dict(setup_seconds=time.monotonic()-t)), flush=True)
    rng = np.random.default_rng(0)
    for fraction in args.fractions:
        factor = system.factor(fraction)
        rhs = rng.normal(size=7*system.ndof)
        timings = {}
        for transpose in (False, True):
            t = time.monotonic()
            x = factor.solve(rhs, transpose=transpose)
            timings['transpose' if transpose else 'forward'] = dict(seconds=time.monotonic()-t,
                relative_residual=factor.relative_residual(rhs, x, transpose=transpose))
        print(json.dumps(dict(fraction=fraction, factor_seconds=factor.seconds, lu_nonzeros=factor.fill,
                              matrix_nonzeros=int(factor.A.nnz), solves=timings,
                              peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)), flush=True)


if __name__ == '__main__':
    main()
