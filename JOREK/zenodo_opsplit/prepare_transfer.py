"""Prepare a separate EM composite campaign initialized from a validated ES model."""
import argparse
import json
from pathlib import Path

import torch

from comparison import digest, write_json


def load_fluid_transfer(target, spec):
    """Restore only the ES fluid interface; keep target physics and EM initialization."""
    from composite import CompositeSplit
    checkpoint = Path(spec['checkpoint'])
    if digest(checkpoint) != spec['sha256']:
        raise ValueError('Changed ES transfer checkpoint')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    cfg = saved['model_config']
    if saved['identity']['arm'] != 'composite' or cfg['channels'] != 4 or target.channels != 6:
        raise ValueError('Transfer requires composite ES checkpoint and EM target')
    for key in ('width','modes','layers','frame_dt','substeps','formulation','boundary_strip','positive_update'):
        if cfg[key] != target.config[key]:
            raise ValueError('Incompatible ES transfer interface: ' + key)
    state = saved['state_dict']
    if not torch.equal(state['native_factor'], target.native_factor[:4].cpu()):
        raise ValueError('ES/EM native unit factors differ')
    # The legacy ES checkpoint carried normalized states. Only its neural weights and
    # normalizers transfer; the target retains the corrected physical-state recurrence.
    source = CompositeSplit(state['r'], state['z'], state['mean'], state['scale'],
                            **{k: cfg[k] for k in ('channels','width','modes','layers','em_layers','diffusion','resistivity','frame_dt','substeps')})
    source.load_state_dict(state)
    target.transfer(source, freeze=spec['freeze_fluid'])
    return dict(**spec, source_epoch=saved['next_epoch'], source_best_validation=saved['best'],
                initialization='ES fluid weights and normalization; zero-output EM correction; target linear physics')


def prepare(source_data, checkpoint, target_data, output):
    source_data, checkpoint, target_data = (p.resolve() for p in (source_data, checkpoint, target_data))
    es = json.loads((source_data / 'comparison.json').read_text())
    em = json.loads((target_data / 'comparison.json').read_text())
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    if es['case'] != 'es' or em['case'] != 'em' or saved['model_config']['channels'] != 4:
        raise ValueError('Expected ES source and EM target')
    if saved['identity']['comparison_sha256'] != digest(source_data / 'comparison.json'):
        raise ValueError('ES checkpoint/data identity mismatch')
    source_families = {e['family'] for split in ('train','val') for e in es['entries'][split]}
    overlap = {split: sorted(source_families & {e['family'] for e in em['entries'][split]}) for split in ('val','test')}
    if any(overlap.values()):
        raise ValueError('ES training/selection overlaps held-out EM geometry families')
    for data, meta in ((source_data, es), (target_data, em)):
        if digest(data / 'data.pt') != meta['data_sha256']:
            raise ValueError('Changed data cache: ' + str(data))
    if em['protocol'].get('normalization') == 'zscore':
        raise ValueError('This transfer trial uses the original EM min-max dataset')
    spec = dict(checkpoint=str(checkpoint), sha256=digest(checkpoint), freeze_fluid=True,
                source_data=str(source_data), source_comparison_sha256=digest(source_data / 'comparison.json'),
                heldout_geometry_overlap=overlap)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'data.pt').symlink_to(target_data / 'data.pt')
    (output / 'terms').symlink_to(target_data / 'terms', target_is_directory=True)
    em['protocol']['composite_transfer'] = spec
    em['protocol']['composite_stability_epochs'] = 110
    em['protocol']['lr'] = 1e-4
    em['protocol']['notes']['transfer'] = 'Frozen ES fluid with source normalization; train only EM correction using target min-max input/loss normalization and Adam1e-4'
    write_json(output / 'comparison.json', em)
    write_json(output / 'transfer.json', dict(**spec, source_epoch=saved['next_epoch'], source_best_validation=saved['best'],
               target_data=str(target_data), target_comparison_sha256=digest(target_data / 'comparison.json')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-data','checkpoint','target-data','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    prepare(args.source_data, args.checkpoint, args.target_data, args.output)
    print((args.output / 'transfer.json').read_text())


if __name__ == '__main__':
    main()
