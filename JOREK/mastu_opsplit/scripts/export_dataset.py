#!/usr/bin/env python3
"""Validate an explicit native exchange bundle and export one trajectory.

This command does not infer native residuals from ordinary JOREK restart files.
"""
import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mastu_opsplit.common import load_config
from mastu_opsplit.data import read_geometry, write_geometry, write_trajectory, validate_trajectory
from mastu_opsplit.numerical import NativeGeometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    required = ('geometry_npz', 'trajectory_npz', 'metadata_json', 'geometry_h5')
    for key in required:
        if key not in config:
            raise ValueError(f'export configuration requires {key}; ordinary restart files are insufficient')
    with np.load(config['geometry_npz'], allow_pickle=False) as bundle:
        geometry = NativeGeometry(**{field.name: bundle[field.name] for field in fields(NativeGeometry)})
    with np.load(config['trajectory_npz'], allow_pickle=False) as bundle:
        arrays = dict(bundle)
    metadata = json.loads(Path(config['metadata_json']).read_text())
    checks = validate_trajectory(geometry, arrays, metadata)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.dry_run:
        print(json.dumps({'output': str(args.output), 'geometry_h5': config['geometry_h5'], 'checks': checks,
                          'synthetic': metadata['synthetic'], 'action': 'validate-and-export'}, indent=2))
        return
    geometry_path = Path(config['geometry_h5'])
    if not geometry_path.exists():
        write_geometry(geometry_path, geometry)
    else:
        read_geometry(geometry_path)
    write_trajectory(args.output, geometry, geometry_path, arrays, metadata)
    print(json.dumps({'output': str(args.output), 'checks': checks}, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError) as error:
        print(f'export failed: {error}', file=sys.stderr)
        sys.exit(1)
