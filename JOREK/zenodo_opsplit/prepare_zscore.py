"""Create an isolated composite dataset with train-only population z-score statistics."""
import argparse
import json
from pathlib import Path

import torch

from comparison import covered_states, digest, write_json


def fit_zscore(values, windows, horizon):
    """Pool each eligible frame/cell once; no validation or overlapping-window weighting."""
    starts = {}
    for index, start in windows:
        starts.setdefault(index, []).append(start)
    count = 0
    mean = torch.zeros(values.shape[2], dtype=torch.float64)
    m2 = torch.zeros_like(mean)
    for index, offsets in sorted(starts.items()):
        block = values[index, covered_states(offsets, horizon)].double()
        if not torch.isfinite(block).all():
            raise ValueError('Nonfinite field in eligible training frames')
        variance, block_mean = torch.var_mean(block, dim=(0, 2, 3), correction=0)
        n = block.numel() // len(mean)
        delta = block_mean - mean
        total = count + n
        m2 += variance * n + delta.square() * (count * n / total)
        mean += delta * (n / total)
        count = total
    if not count:
        raise ValueError('No eligible training frames')
    std = (m2 / count).sqrt()
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError('Nonfinite z-score statistics')
    # Constant channels encode as zero; retain a finite decoder scale.
    scale = torch.where(std > 0, std, torch.ones_like(std))
    return mean, scale, std, count


def prepare(source, output, stability_epochs=110):
    source = source.resolve()
    meta = json.loads((source / 'comparison.json').read_text())
    if not 1 <= stability_epochs <= meta['protocol']['epochs']:
        raise ValueError('Stability gate must fit within the epoch budget')
    if digest(source / 'data.pt') != meta['data_sha256']:
        raise ValueError('Changed source data cache')
    for name, expected in meta['term_files_sha256'].items():
        if digest(source / 'terms' / name) != expected:
            raise ValueError('Changed source term data: ' + name)
    values = torch.load(source / 'data.pt', map_location='cpu', weights_only=False, mmap=True)
    mean, scale, std, count = fit_zscore(values['train']['exported'], meta['windows'], meta['protocol']['horizon'])
    details = dict(method='zscore', source=str(source), source_comparison_sha256=digest(source / 'comparison.json'),
                   source_data_sha256=meta['data_sha256'], fields=['rho','T','potential','omega','psi','j'][:len(mean)],
                   weighting='Equal eligible training frame/cell weight; union of windows, no overlap duplication',
                   count_per_field=count, mean=mean.tolist(), population_std=std.tolist(), scale=scale.tolist(),
                   constant_channels=(std == 0).tolist(), previous_center=values['mean'].tolist(), previous_scale=values['scale'].tolist(),
                   usage='Network inputs, rollout loss, learned output units and positive-coordinate reference; recurrent exported state stays float64',
                   limitation='Affine standardization does not remove skewness or bound outliers; normalized losses differ from min-max campaign')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'data.pt').symlink_to(source / 'data.pt')
    (output / 'terms').symlink_to(source / 'terms', target_is_directory=True)
    meta['normalization'] = 'Train-only population z-score over the union of valid training windows'
    meta['normalization_details'] = details
    meta['protocol']['composite_stability_epochs'] = stability_epochs
    meta['protocol']['normalization'] = 'zscore'
    meta['protocol']['notes']['normalization_ablation'] = details['usage'] + '; changes the normalization used by the NS-style objective'
    meta['metrics'] = 'NRMSE on z-score fields; R-weighted decoded relative L2 remains directly comparable; persistence; EM delta_psi; failure/completion counts'
    write_json(output / 'comparison.json', meta)
    write_json(output / 'normalization.json', details)
    return details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stability-epochs', type=int, default=110)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(prepare(args.source, args.output, args.stability_epochs), indent=2))


if __name__ == '__main__':
    main()
