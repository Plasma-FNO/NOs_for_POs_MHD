#!/usr/bin/env python3
"""Plan, export and finalize exact native remainder labels for the 80 training trajectories.

  plan:     fix (case, frame) entries, restarts and executable/tangent hashes; refuses an existing plan.
  worker:   run native export steps concurrently inside a Slurm allocation; resumable per state.
  finalize: check completeness and fit the training-only per-channel label scale.
Only training (original train + validation) trajectories are ever exported.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import fields
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

from mastu_opsplit.common import write_json
from mastu_opsplit.implicit_labels import (label_scale, plan_exports, read_export, remainder_from_export,
                                           run_export, sha256)
from mastu_opsplit.implicit_opsplit import ACTIVE_CHANNELS, load_frozen_tangent
from mastu_opsplit.numerical import NativeGeometry


def load_geometry(manifest):
    manifest = Path(manifest)
    record = json.loads(manifest.read_text())['geometry']
    path = manifest.parent/record['path']
    if sha256(path) != record['sha256']:
        raise ValueError('Geometry hash mismatch')
    with h5py.File(path) as stream:
        return NativeGeometry(**{item.name: stream[item.name][()] for item in fields(NativeGeometry)}).validate()


def plan(args):
    root = Path(args.root)
    if (root/'plan.json').exists():
        raise FileExistsError(root/'plan.json')
    root.mkdir(parents=True, exist_ok=True)
    frames = list(range(0, 97, args.frame_stride))
    entries = plan_exports(args.manifest, frames)
    executable = Path(args.executable_dir)
    record = dict(manifest=str(Path(args.manifest).resolve()), manifest_sha256=sha256(args.manifest),
                  tangent=str(Path(args.tangent).resolve()), tangent_sha256=sha256(args.tangent),
                  executable_dir=str(executable.resolve()), executable_sha256=sha256(executable/'jorek.exe'),
                  input_sha256=sha256(executable/'input'), eqdsk_sha256=sha256(executable/'eqdsk.dat'),
                  frames=frames, active_channels=list(ACTIVE_CHANNELS), entries=entries,
                  cases=sorted({entry['case_id'] for entry in entries}),
                  label=('N = F(q) - F(q*) - L*(q - q*) in native weak coefficients [strong,test_R,test_Z] '
                         'for blocks [psi,u,rho,pressure,Vpar]; inactive induction flux channels dropped'),
                  split_policy='original train + validation (pooled 80 training trajectories) only')
    write_json(root/'plan.json', record)
    print(json.dumps(dict(states=len(entries), cases=len(record['cases']), frames=frames)))


def worker(args):
    root = Path(args.root)
    record = json.loads((root/'plan.json').read_text())
    executable = Path(record['executable_dir'])
    for name, key in (('jorek.exe', 'executable_sha256'), ('input', 'input_sha256'), ('eqdsk.dat', 'eqdsk_sha256')):
        if sha256(executable/name) != record[key]:
            raise ValueError(f'Frozen executable input changed: {name}')
    if sha256(record['tangent']) != record['tangent_sha256']:
        raise ValueError('Frozen tangent changed')
    geometry = load_geometry(record['manifest'])
    frozen = load_frozen_tangent(record['tangent'], geometry)
    with h5py.File(record['entries'][0]['restart']) as stream:
        vertices = stream['vertex'][()].T
    states, failures, scratch = root/'states', root/'failures', Path(args.scratch)/f'task{args.task_index}'
    states.mkdir(exist_ok=True)
    failures.mkdir(exist_ok=True)
    todo = [entry for i, entry in enumerate(record['entries'])
            if i % args.task_count == args.task_index
            and not (states/f'{entry["case_id"]}_{entry["frame"]:03d}.json').exists()]
    print(json.dumps(dict(task=args.task_index, remaining=len(todo))), flush=True)

    def one(entry):
        name = f'{entry["case_id"]}_{entry["frame"]:03d}'
        started = time.monotonic()
        work = scratch/name
        try:
            files = run_export(entry['restart'], work, executable)
            exported = time.monotonic()
            export = read_export(files, vertices)
            with h5py.File(entry['data']) as stream:
                state = stream['native_state'][entry['frame']]
            label, metrics = remainder_from_export(export, frozen, geometry, state)
            temporary = states/f'{name}.tmp.npy'
            np.save(temporary, label.astype(np.float32))
            temporary.replace(states/f'{name}.npy')
            write_json(states/f'{name}.json', dict(entry, **metrics, restart_sha256=sha256(entry['restart']),
                       export_seconds=exported-started, total_seconds=time.monotonic()-started,
                       label_sha256=sha256(states/f'{name}.npy')))
            (failures/f'{name}.json').unlink(missing_ok=True)
            return name, None
        except Exception as error:
            write_json(failures/f'{name}.json', dict(entry, error=str(error), traceback=traceback.format_exc()))
            return name, str(error)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    failed = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for future in as_completed([pool.submit(one, entry) for entry in todo]):
            name, error = future.result()
            print(json.dumps(dict(state=name, error=error)), flush=True)
            if error:
                failed.append(name)
    if failed:
        raise RuntimeError(f'{len(failed)} native exports failed; see {failures}')


def finalize(args):
    root = Path(args.root)
    record = json.loads((root/'plan.json').read_text())
    missing = [f'{e["case_id"]}_{e["frame"]:03d}' for e in record['entries']
               if not (root/'states'/f'{e["case_id"]}_{e["frame"]:03d}.json').exists()]
    if missing:
        raise RuntimeError(f'{len(missing)} labels missing, e.g. {missing[:5]}')
    geometry = load_geometry(record['manifest'])
    paths, metrics = [], []
    for entry in record['entries']:
        name = f'{entry["case_id"]}_{entry["frame"]:03d}'
        meta = json.loads((root/'states'/f'{name}.json').read_text())
        if sha256(root/'states'/f'{name}.npy') != meta['label_sha256']:
            raise ValueError(f'Label changed after export: {name}')
        paths.append(root/'states'/f'{name}.npy')
        metrics.append(meta)
    scale = label_scale(paths, geometry.volume_weights)
    summary = dict(plan_sha256=sha256(root/'plan.json'), states=len(paths), cases=len(record['cases']),
                   frames=record['frames'], active_channels=record['active_channels'], label_scale=scale.tolist(),
                   maximum_value_relative_error=max(m['value_relative_error'] for m in metrics),
                   maximum_gradient_relative_error=max(m['gradient_relative_error'] for m in metrics),
                   median_remainder_over_physical_rms={c: float(np.median([m['remainder_over_physical_rms'][c] for m in metrics]))
                                                       for c in metrics[0]['remainder_over_physical_rms']},
                   mean_export_seconds=float(np.mean([m['export_seconds'] for m in metrics])))
    write_json(root/'summary.json', summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('plan')
    p.add_argument('--root', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--tangent', required=True)
    p.add_argument('--executable-dir', required=True)
    p.add_argument('--frame-stride', type=int, default=4)
    w = sub.add_parser('worker')
    w.add_argument('--root', required=True)
    w.add_argument('--scratch', required=True)
    w.add_argument('--task-index', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)))
    w.add_argument('--task-count', type=int, required=True)
    w.add_argument('--concurrency', type=int, default=8)
    f = sub.add_parser('finalize')
    f.add_argument('--root', required=True)
    args = parser.parse_args()
    dict(plan=plan, worker=worker, finalize=finalize)[args.command](args)


if __name__ == '__main__':
    main()
