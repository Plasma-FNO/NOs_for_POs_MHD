"""Read-only checkpoint diagnostics: autonomous, teacher-forced and oracle-input rollouts.

Import the frozen campaign code, never change its checkpoints. Oracle interventions
use true previous-frame fields and are diagnostic, not deployable forecasts.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


@torch.no_grad()
def run(args):
    sys.path.insert(0, str(args.campaign / 'code'))
    from es_threefield import ThreeFieldModel, load_release, CROP
    from composite import positive_increment
    from train_comparison import initialization_hash

    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    meta = json.loads((args.data / 'comparison.json').read_text())
    values = torch.load(args.data / 'data.pt', weights_only=False, mmap=True, map_location='cpu')
    mean, scale = (values[k][:3].double().to(args.device) for k in ('mean', 'scale'))
    entries = [e for e in meta['entries']['test'] if meta['test_prefixes'][e['id']] >= 200]
    reference = json.loads((args.campaign / 'opsplit-vorticity/test_metrics.json').read_text())
    ordered = sorted((t['scores']['200']['r_weighted_relative_l2'][0], t['id'])
                     for t in reference['trajectories'] if t['valid_source_steps'] >= 200)
    selected = [ordered[i][1] for i in [0, 7, 14, 21, 29, 36, 43, 50]]
    if args.limit: entries = entries[:args.limit]
    summary = dict(cohort=[e['id'] for e in entries], intervention_ids=selected,
                   selection='Eight quantiles of corrected-OpsSplit 200-step density error; diagnostic selection only',
                   fields=['rho', 'T', 'potential'], steps=200, results={})
    for arm in args.arms:
        saved = torch.load(args.campaign / arm / 'best.pt', map_location=args.device, weights_only=False)
        cfg = saved['identity']['protocol']
        model = ThreeFieldModel(entries[0]['r'], entries[0]['z'], mean, scale,
                                'node' if arm == 'node' else 'opsplit',
                                **({'opsplit_update': cfg['opsplit_update']} if cfg.get('opsplit_update') else {}),
                                **{k: cfg[k] for k in ('width', 'modes', 'layers')}).to(args.device)
        model.load_state_dict(saved['state_dict']); model.eval()
        original = json.loads((args.campaign / arm / 'test_metrics.json').read_text())
        assert initialization_hash(model) == original['identity']['checkpoint']
        captured = {}
        handle = model.network.register_forward_hook(lambda m, i, out: captured.update(rate=out.double()))
        r = model.r
        w = r[None, :].expand(100, 100); w = w / w.sum()
        wi = w[CROP:-CROP, CROP:-CROP]; wi = wi / wi.sum()
        edge = torch.ones_like(w); edge[CROP:-CROP, CROP:-CROP] = 0
        packed = {}
        checks = []
        failures = {}
        for entry in entries:
            source = load_release(entry, 201).double().to(args.device)
            modes = ['free', 'teacher']
            if entry['id'] in selected:
                modes += ['true_phi', 'true_rho_T', 'true_edges', 'true_phi_mean']
                if arm != 'node': modes += ['no_diffusion']
            state = source[0:1, :3].repeat(len(modes), 1, 1, 1)
            failed = {}
            traces = {name: [] for name in ['relative', 'omega_relative', 'error_energy', 'truth_energy',
                       'scaled_rmse', 'mean', 'truth_mean', 'std', 'truth_std', 'outside', 'truth_outside',
                       'edge_error_fraction', 'learned_mean_delta', 'diffusion_mean_delta',
                       'diffusion_rms_delta', 'learned_rms_delta', 'nonpositive', 'phi_mean_error_fraction']}
            for step in range(1, 201):
                previous, truth = source[step-1, :3], source[step, :3]
                state[1] = previous
                for j, mode in enumerate(modes[2:], 2):
                    if mode in failed: continue
                    if mode == 'true_phi': state[j, 2] = previous[2]
                    if mode == 'true_rho_T': state[j, :2] = previous[:2]
                    if mode == 'true_edges': state[j] = state[j]*(1-edge) + previous*edge
                    if mode == 'true_phi_mean': state[j, 2] += ((previous[2]-state[j, 2])*w).sum()
                before = state.clone()
                # Batch one reproduces the reference and isolates failed oracle tests.
                predictions, rates = [], []
                for j, mode in enumerate(modes):
                    result_j = torch.full_like(state[j:j+1], float('nan'))
                    rate_j = result_j.clone()
                    if mode not in failed:
                        try:
                            result_j = model(state[j:j+1])
                            rate_j = captured['rate']
                        except (FloatingPointError, RuntimeError) as error:
                            if mode == 'free': raise
                            failed[mode] = dict(step=step, error=str(error))
                    predictions.append(result_j); rates.append(rate_j)
                rate = torch.cat(rates, 0); result = torch.cat(predictions, 0)
                if arm == 'node' or cfg.get('opsplit_update') == 'additive':
                    candidate = before+rate*scale[None,:,None,None]
                else:
                    candidate = torch.stack([positive_increment(before[:, i], rate[:, i], scale[i])
                                             if i < 2 else before[:, i]+scale[i]*rate[:, i] for i in range(3)], 1)
                if arm == 'node':
                    candidate = result
                if 'no_diffusion' in modes: result[modes.index('no_diffusion')] = candidate[modes.index('no_diffusion')]
                state = result
                err = (state-truth).square()
                energy = (truth.square()*w).sum((-2, -1))
                error = (err*w).sum((-2, -1))
                wm = (state*w).sum((-2, -1)); tm = (truth*w).sum((-2, -1))
                omega = model.omega(state); actual = source[step, 3, CROP:-CROP, CROP:-CROP]
                vals = dict(relative=(error/energy.clamp_min(1e-30)).sqrt(),
                            omega_relative=(((omega-actual).square()*wi).sum((-2, -1)) /
                                            (actual.square()*wi).sum().clamp_min(1e-30)).sqrt(),
                            error_energy=error, truth_energy=energy,
                            scaled_rmse=(err.mean((-2, -1))).sqrt()/scale,
                            mean=wm, truth_mean=tm,
                            std=(((state-wm[..., None, None]).square()*w).sum((-2,-1))).sqrt(),
                            truth_std=(((truth-tm[..., None, None]).square()*w).sum((-2,-1))).sqrt(),
                            outside=(model.normalize(state).abs()>1).double().mean((-2,-1)),
                            truth_outside=(model.normalize(truth).abs()>1).double().mean((-2,-1)),
                            edge_error_fraction=(err*w*edge).sum((-2,-1))/error.clamp_min(1e-30),
                            learned_mean_delta=((candidate-before)*w).sum((-2,-1)),
                            diffusion_mean_delta=((state-candidate)*w).sum((-2,-1)),
                            learned_rms_delta=(((candidate-before).square()*w).sum((-2,-1))).sqrt()/scale,
                            diffusion_rms_delta=(((state-candidate).square()*w).sum((-2,-1))).sqrt()/scale,
                            nonpositive=(state[:,:2]<=0).double().mean((-2,-1)),
                            phi_mean_error_fraction=(wm[:,2]-tm[2]).square()/error[:,2].clamp_min(1e-30))
                for key, value in vals.items(): traces[key].append(value.cpu().numpy())
            for key, rows in traces.items(): packed[entry['id']+'__'+key] = np.stack(rows)
            packed[entry['id']+'__modes'] = np.array(modes)
            failures[entry['id']] = failed
            # Independently reproduce saved cumulative primary metrics.
            cumulative = np.sqrt(packed[entry['id']+'__error_energy'][:,0].sum(0) /
                                 packed[entry['id']+'__truth_energy'].sum(0))
            expected = next(t['scores']['200']['r_weighted_relative_l2'] for t in original['trajectories'] if t['id']==entry['id'])
            checks.append(float(np.max(np.abs(cumulative-expected))))
            assert np.allclose(cumulative, expected, rtol=2e-3, atol=2e-4), (arm, entry['id'], cumulative, expected)
            print(json.dumps(dict(arm=arm, id=entry['id'], modes=modes, cumulative=cumulative.tolist(), failures=failed)), flush=True)
        handle.remove()
        np.savez_compressed(args.output / (arm+'.npz'), **packed)
        result = dict(checkpoint_hash=initialization_hash(model), failures=failures,
                      reproduction_max_abs_difference=max(checks), blocks={})
        for lo, hi in [(1,49),(50,100),(101,150),(151,200)]:
            block = {}
            for mode in ['free','teacher','true_phi','true_rho_T','true_edges','true_phi_mean','no_diffusion']:
                ids = [e['id'] for e in entries if mode in packed[e['id']+'__modes']]
                if not ids: continue
                stats = {}
                for metric in ['relative','scaled_rmse','mean','std','outside','edge_error_fraction',
                               'learned_mean_delta','diffusion_mean_delta','learned_rms_delta','diffusion_rms_delta',
                               'omega_relative','phi_mean_error_fraction']:
                    stack = np.stack([packed[i+'__'+metric][lo-1:hi, list(packed[i+'__modes']).index(mode)] for i in ids])
                    stats[metric] = np.mean(stack,axis=(0,1)).tolist() if np.isfinite(stack).all() else None
                stats['trajectories'] = len(ids)
                stats['failed_by_block_end'] = sum(mode in failures[i] and failures[i][mode]['step'] <= hi for i in ids)
                block[mode] = stats
            result['blocks'][f'{lo}-{hi}'] = block
        summary['results'][arm] = result
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('DIAGNOSTICS COMPLETE',flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--limit',type=int,default=0,help='Limit trajectories for a full-horizon reproduction check')
    p.add_argument('--arms',nargs='+',choices=['node','opsplit','opsplit-vorticity'],default=['node','opsplit','opsplit-vorticity'])
    run(p.parse_args())
