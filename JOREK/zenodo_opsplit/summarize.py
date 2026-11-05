"""Summarize completed runs without hiding failed/incomplete rollouts."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def summarize(root):
    rows = []
    for path in sorted(root.glob('seed-*/*/provenance.json')):
        provenance = json.loads(path.read_text())
        metrics_path = path.with_name('test_metrics.json')
        row = {'run': str(path.parent.relative_to(root)), 'mode': provenance['mode'],
               'seed': provenance['seed'], 'em_training_trajectories': len(provenance['train_ids']) if provenance['case'] == 'em' else 0,
               'parameters': provenance['parameters'], 'trainable_parameters': provenance['trainable_parameters'],
               'completed_trajectories': None, 'total_trajectories': None}
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            row.update(completed_trajectories=metrics['completed'], total_trajectories=metrics['total'])
            for horizon in (10, 50, 100):
                valid = [t for t in metrics['trajectories'] if len(t['relative_l2_by_frame']) >= horizon]
                row[f'available_at_{horizon}'] = len(valid)
                for key, prefix in [('relative_l2_by_frame', 'shared_relative_l2'), ('persistence_relative_l2_by_frame', 'persistence_shared_relative_l2')]:
                    row[f'{prefix}_{horizon}'] = float(np.mean([t[key][horizon-1][:4] for t in valid])) if valid else None
        rows.append(row)
    if not rows:
        raise ValueError('No run provenance found')
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (root/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    (root/'summary.json').write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n')
    print(json.dumps(rows, indent=2, allow_nan=False))
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    summarize(parser.parse_args().root)
