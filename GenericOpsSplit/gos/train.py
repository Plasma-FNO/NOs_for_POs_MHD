"""Data generation, rollout training and evaluation for the generic implicit OpsSplit testbed.

  python -m gos.train data --problem burgers --root ROOT
  python -m gos.train run  --problem burgers --arm implicit_struct --seed 0 --root ROOT
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch

from .models import ARMS, StepModel
from .operators import OseenNS
from .operators import make_linear
from .problems import get_problem, PROBLEMS

CONFIG = {
    'burgers': dict(net=dict(modes=24, width=48, layers=4), iters=5000, window=8, batch=32,
                    horizons=[25, 50, 100, 200], long=1000),
    'kdv': dict(net=dict(modes=24, width=48, layers=4), iters=5000, window=8, batch=32,
                horizons=[25, 50, 100, 200], long=1000),
    'ks': dict(net=dict(modes=32, width=48, layers=4), iters=5000, window=8, batch=32,
               horizons=[5, 10, 20, 50], long=1000),
    'ns2d': dict(net=dict(modes=12, width=32, layers=4), iters=4000, window=6, batch=16,
                 horizons=[10, 20, 40, 80], long=400),
    'burgers_fem': dict(net=dict(modes=24, width=48, layers=4, pad=16), iters=5000, window=8, batch=32,
                        horizons=[12, 25, 50, 100], long=500),
}
SPLIT_SEEDS = dict(train=1, val=2, test=3)
# Largest step-Jacobian amplification of the near-exact flow map over h on test states (set from
# measure_amplification.py); the penalty only acts above what the physics itself does.
JAC_THRESHOLD = dict(burgers=1.035, kdv=1.40, ks=2.23, ns2d=1.15, burgers_fem=1.021)


# ------------------------------------------------------------------------------------------- data
def data_path(root, problem):
    return Path(root)/'data'/f'{problem}.pt'


def generate_data(problem_name, root, device):
    problem = get_problem(problem_name, device)
    out = dict(problem=problem_name, h=problem.h, mu_ref=problem.mu_ref, ood={})
    t0 = time.time()
    for split, seed in SPLIT_SEEDS.items():
        g = torch.Generator().manual_seed(seed)
        out[split] = problem.generate(problem.counts[split], g, batch=256 if problem.dim == 1 else 64)
        print(f'{problem_name} {split}: {tuple(out[split][0].shape)} {time.time()-t0:.0f}s', flush=True)
    for i, value in enumerate(problem.mu_ood):
        g = torch.Generator().manual_seed(100+i)
        out['ood'][float(value)] = problem.generate(problem.counts['ood'], g, ood=value, batch=256 if problem.dim == 1 else 64)
    out['scale'] = float(out['train'][0].std())
    path = data_path(root, problem_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)
    print(f'wrote {path} scale={out["scale"]:.4f} in {time.time()-t0:.0f}s', flush=True)


def load_data(root, problem, device):
    raw = torch.load(data_path(root, problem.name), weights_only=False)
    scale = raw['scale']
    prep = lambda pair: (pair[0].to(device)/scale, (pair[1].double()/problem.mu_ref).float().to(device))
    data = {k: prep(raw[k]) for k in ('train', 'val', 'test')}
    data['ood'] = {k: prep(v) for k, v in raw['ood'].items()}
    return data, scale


# ------------------------------------------------------------------------------------------ metrics
def norms(x):
    return x.flatten(2).norm(dim=-1)


def relative_errors(pred, true):
    e = norms(pred-true)/norms(true)
    return torch.where(torch.isfinite(e), e, torch.full_like(e, float('inf')))


@torch.no_grad()
def rollout(model, frames, mu, dt_mult=1, substeps=1, chunk=64):
    steps = (frames.shape[1]-1)//dt_mult
    preds = [model(frames[i:i+chunk, 0], mu[i:i+chunk], steps, dt_mult, substeps) for i in range(0, len(frames), chunk)]
    return torch.cat(preds), frames[:, dt_mult::dt_mult][:, :steps]


def summarize(pred, true, horizons, dt_mult=1):
    e = relative_errors(pred, true)
    bound = 10*norms(true).amax(1, keepdim=True)
    bounded = torch.isfinite(norms(pred)) & (norms(pred) <= bound)
    out = {}
    for H in horizons:
        s = H//dt_mult - 1
        if H % dt_mult or s >= e.shape[1]:
            continue
        col = e[:, s]
        out[str(H)] = dict(median=float(col.median()), mean_clipped=float(col.clamp(max=2.).mean()),
                           bounded=float(bounded[:, s].float().mean()))
    return out


@torch.no_grad()
def long_rollout(model, q0, mu, steps, reference_norm):
    q, worst = q0, torch.zeros(len(q0), device=q0.device)
    for _ in range(steps):
        q = model.step(q, mu, model.h)
        n = norms(q[:, None])[:, 0]
        n = torch.where(torch.isfinite(n), n, torch.full_like(n, float('inf')))
        worst = torch.maximum(worst, n/reference_norm)
    final = norms(q[:, None])[:, 0]/reference_norm
    final = torch.where(torch.isfinite(final), final, torch.full_like(final, float('inf')))
    return dict(steps=steps, bounded=float((worst <= 10).float().mean()), max_ratio_median=float(worst.median()),
                final_ratio_median=float(final.median()))


def spectrum_error(pred, true, start, decades=6.):
    """KS long-time statistics over the energetically resolved range (true spectrum within `decades` of
    its peak; the float32 dissipation-range floor is excluded): mean |log10| ratio of time-averaged power
    spectra, and the ratio of time-averaged energies."""
    sp = lambda x: torch.fft.rfft(x[:, start:, 0].double(), dim=-1).abs().pow(2).mean((0, 1))
    if not torch.isfinite(pred[:, start:]).all():
        return dict(log10_error=float('inf'), energy_ratio=float('inf'), modes=0)
    sp_p, sp_t = sp(pred), sp(true)
    keep = sp_t >= sp_t[1:].max()*10**(-decades)
    keep[0] = False
    err = (torch.log10(sp_p[keep]+1e-300) - torch.log10(sp_t[keep])).abs().mean()
    energy = pred[:, start:].double().pow(2).mean()/true[:, start:].double().pow(2).mean()
    return dict(log10_error=float(err), energy_ratio=float(energy), modes=int(keep.sum()))


def step_jacobian_norm(model, q, mu, iterations=15):
    """Largest singular value of the one-step map's Jacobian per sample (Euclidean, normalised units)."""
    f = lambda x: model.step(x, mu, model.h)
    v = torch.randn_like(q)
    v = v/norms(v[:, None]).view(-1, *([1]*(q.ndim-1)))
    sigma = None
    for _ in range(iterations):
        _, Jv = torch.func.jvp(f, (q,), (v,))
        _, pullback = torch.func.vjp(f, q)
        w = pullback(Jv)[0]
        nw = norms(w[:, None]).view(-1)
        sigma = nw.sqrt()
        v = w/nw.view(-1, *([1]*(q.ndim-1)))
    return sigma


def operator_diagnostics(model, problem, device):
    """Growth bound of the learned/known L and recovery of the true operator (identifiable range only)."""
    lin = model.linear
    if lin is None:
        return None
    true = make_linear(problem, 'known', device)
    mus = sorted({problem.mu_train[0]/problem.mu_ref, 1.0, problem.mu_train[1]/problem.mu_ref,
                  *[v/problem.mu_ref for v in problem.mu_ood]})
    out = dict(growth=[], recovery={})
    with torch.no_grad():
        if problem.space == 'fourier':
            for m in mus:
                s = lin.symbol(torch.tensor([m], device=device))[0]
                out['growth'].append(dict(mu_t=m, h_max_re=float(problem.h*s.real.max())))
            j = (lin.grid.k2.sqrt()*problem.length/(2*math.pi)).to(device)
            bands = dict(all=(problem.h*true.symbol(torch.ones(1, device=device))[0].abs() <= 10) & lin.grid.mask,
                         low=(j < problem.n/8) & lin.grid.mask, mid=(j >= problem.n/8) & lin.grid.mask)
            for name, (a, b) in dict(L_at_ref=(lin.symbol(torch.ones(1, device=device))[0], true.symbol(torch.ones(1, device=device))[0]),
                                     O1=(lin.components()[1], true.components()[1])).items():
                for band, ident in bands.items():
                    if name == 'O1' or band != 'all':
                        ident = ident & (problem.h*b.abs() <= 10)
                    if not ident.any() or float(b[ident].abs().max()) == 0:
                        continue
                    key = name if band == 'all' else f'{name}_{band}'
                    out['recovery'][key] = float((a-b)[ident].abs().norm()/b[ident].abs().norm())
        else:
            C = torch.linalg.cholesky(lin.mass_dense.double())
            for m in mus:
                L = lin.bands(torch.tensor([m], device=device))
                from .problems import FEM1D
                Ld = FEM1D.dense(L)[0].double()
                S = 0.5*(Ld + Ld.T)
                Ci = torch.linalg.inv(C)
                out['growth'].append(dict(mu_t=m, h_max_re=float(problem.h*torch.linalg.eigvalsh(Ci @ S @ Ci.T).max())))
            from .problems import FEM1D
            for name, (a, b) in dict(L_at_ref=(FEM1D.dense(lin.bands(torch.ones(1, device=device)))[0], FEM1D.dense(true.bands(torch.ones(1, device=device)))[0]),
                                     O1=(FEM1D.dense(lin.components()[1]), FEM1D.dense(true.components()[1]))).items():
                out['recovery'][name] = float((a-b).norm()/b.norm())

            def profiles(L):  # symmetric off-diagonal = element conductance, antisymmetric = skew advection
                return 0.5*(L.diagonal(1) + L.diagonal(-1)), 0.5*(L.diagonal(1) - L.diagonal(-1))
            (c_l, s_l), (c_t, s_t) = (profiles(FEM1D.dense(o.bands(torch.ones(1, device=device)))[0]) for o in (lin, true))
            out['recovery']['conductance'] = float((c_l-c_t).norm()/c_t.norm())
            out['recovery']['skew'] = float((s_l-s_t).norm()/s_t.norm())
    out['propagator'] = propagator_error(lin, true, problem, model.theta, device)
    return out


def propagator_error(lin, true, problem, theta, device):
    """|| R(h M^-1 L_model) - exp(h M^-1 L_true) || / || exp(...) || at mu_t = 1: what a one-step linear
    propagator gets wrong, independently of how L itself is parameterised."""
    one = torch.ones(1, device=device)
    h = problem.h
    with torch.no_grad():
        if problem.space == 'fourier':
            lm, lt = lin.symbol(one)[0].to(torch.complex128), true.symbol(one)[0].to(torch.complex128)
            R = (1 + (1-theta)*h*lm)/(1 - theta*h*lm)
            E = torch.exp(h*lt)
            j = (lin.grid.k2.sqrt()*problem.length/(2*math.pi)).to(device)
            out = {}
            for band, keep in dict(all=lin.grid.mask, low=(j < problem.n/8) & lin.grid.mask).items():
                out[band] = float((R-E)[keep].abs().norm()/E[keep].abs().norm())
            return out
        from .problems import FEM1D
        M = lin.mass_dense.double()
        Lm, Lt = (FEM1D.dense(o.bands(one))[0].double() for o in (lin, true))
        R = torch.linalg.solve(M - theta*h*Lm, M + (1-theta)*h*Lm)
        E = torch.linalg.matrix_exp(h*torch.linalg.solve(M, Lt))
        return dict(all=float((R-E).norm()/E.norm()))


# ------------------------------------------------------------------------------------------- training
def validation_error(model, frames, mu, steps):
    with torch.no_grad():
        pred = model(frames[:, 0], mu, steps)
        e = relative_errors(pred, frames[:, 1:steps+1]).clamp(max=2.)
    return float(e.mean())


def validation_multi(model, frames, mu, steps, dt_mults):
    errs = []
    with torch.no_grad():
        for s in dt_mults:
            n = steps//s
            pred = model(frames[:, 0], mu, n, dt_mult=s)
            errs.append(float(relative_errors(pred, frames[:, s:n*s+1:s]).clamp(max=2.).mean()))
    return sum(errs)/len(errs)


def amplification(model, q, mu, dt):
    """One power-iteration estimate of the largest singular value of the step Jacobian per sample, with
    a graph through the final Jacobian-vector product (used as a training penalty)."""
    f = lambda x: model.step(x, mu, dt)
    view = lambda n: n.view(-1, *([1]*(q.ndim-1)))
    v = torch.randn_like(q)
    v = v/view(norms(v[:, None])[:, 0])
    _, Jv = torch.func.jvp(f, (q,), (v,))
    _, pullback = torch.func.vjp(f, q)
    w = pullback(Jv.detach())[0].detach()
    w = w/view(norms(w[:, None])[:, 0])
    _, Jw = torch.func.jvp(f, (q,), (w,))
    return norms(Jw[:, None])[:, 0]


def train(model, data, cfg, log, checkpoint_path=None, max_cpu_seconds=None):
    frames, mu = data['train']
    val_frames, val_mu = data['val']
    T_train = (frames.shape[1]-1)//2
    dt_mults = cfg.get('dt_mults', [1])
    if dt_mults != [1]:
        validation_error_ = lambda m, f, u, n: validation_multi(m, f, u, n, dt_mults)
    else:
        validation_error_ = validation_error
    params = [dict(params=model.network_parameters(), lr=cfg.get('lr', 1e-3)),
              dict(params=model.operator_parameters(), lr=cfg.get('operator_lr', 1e-2))]
    params = [p for p in params if p['params']]
    history = dict(loss=[], val=[])
    best = dict(val=validation_error_(model, val_frames, val_mu, T_train), iteration=0,
                state={k: v.detach().clone() for k, v in model.state_dict().items()})
    history['val'].append((0, best['val']))
    if not params:
        return best, history, dict(status='no_parameters')
    opt = torch.optim.Adam(params)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg['iters'], eta_min=cfg.get('lr_min', 1e-5))
    window, bad, status, t0 = cfg['window'], 0, 'completed', time.time()
    skipped_gradients = 0
    label_iters = int(cfg.get('label_pretrain', 0.)*cfg['iters'])
    label_scale = None
    first, elapsed, resumes = 1, 0., 0
    cpu_start, elapsed_cpu = time.process_time(), 0.
    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
    if checkpoint_path is not None and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if saved['config'] != cfg:
            raise ValueError(f'Resume configuration differs: {checkpoint_path}')
        model.load_state_dict(saved['state'])
        opt.load_state_dict(saved['optimizer'])
        sched.load_state_dict(saved['scheduler'])
        best, history = saved['best'], saved['history']
        first, elapsed, resumes = saved['iteration']+1, saved['seconds'], saved['resumes']+1
        elapsed_cpu = saved.get('cpu_seconds', 0.)
        bad, label_scale, skipped_gradients = saved['bad'], saved['label_scale'], saved['skipped_gradients']
        torch.set_rng_state(saved['rng_cpu'].cpu())
        if frames.is_cuda:
            torch.cuda.set_rng_state(saved['rng_cuda'].cpu(), frames.device)
        log(f'Resuming after iteration {first-1}; optimizer, scheduler, and RNG restored')
        del saved

    def save_progress(iteration):
        if checkpoint_path is None:
            return
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint_path.with_suffix('.tmp')
        torch.save(dict(config=cfg, state=model.state_dict(), optimizer=opt.state_dict(),
                        scheduler=sched.state_dict(), best=best, history=history, iteration=iteration,
                        seconds=elapsed+time.time()-t0, resumes=resumes, bad=bad, label_scale=label_scale,
                        cpu_seconds=elapsed_cpu+time.process_time()-cpu_start,
                        skipped_gradients=skipped_gradients, rng_cpu=torch.get_rng_state(),
                        rng_cuda=torch.cuda.get_rng_state(frames.device) if frames.is_cuda else None), temporary)
        temporary.replace(checkpoint_path)

    it = first-1
    for it in range(first, cfg['iters']+1):
        stride = dt_mults[it % len(dt_mults)]
        b = torch.randint(len(frames), (cfg['batch'],), device=frames.device)
        s = torch.randint(T_train-window*stride+1, (cfg['batch'],), device=frames.device)
        idx = s[:, None] + stride*torch.arange(window+1, device=frames.device)
        clip = frames[b[:, None], idx]
        loss = torch.zeros((), device=frames.device)
        if cfg.get('label_weight', 0.) > 0:
            states = clip[:, :window].flatten(0, 1)
            mus = mu[b].repeat_interleave(window)
            with torch.no_grad():
                target = model.exact_increment(states, mus)
                if label_scale is None:
                    label_scale = max(float(target.pow(2).mean()), 1e-12)
            label = (model.remainder_increment(states, mus) - target).pow(2).mean()/label_scale
            loss = loss + (label if it <= label_iters else cfg['label_weight']*label)
        if it > label_iters:
            pred = model(clip[:, 0], mu[b], window, dt_mult=stride)
            loss = loss + (pred-clip[:, 1:]).pow(2).mean()
            if cfg.get('jac_weight', 0.) > 0:
                sigma = amplification(model, clip[:, 0], mu[b], model.h*stride)
                loss = loss + cfg['jac_weight']*torch.relu(sigma - cfg['jac_threshold']).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            bad += 1
            if bad >= 100:
                status = 'diverged'
                break
            sched.step()
            continue
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([p for g in params for p in g['params']], 1.0)
        if not torch.isfinite(grad_norm):
            skipped_gradients += 1
            bad += 1
            if bad >= 100:
                status = 'nonfinite_gradients'
                break
            sched.step()
            continue
        bad = 0
        opt.step()
        sched.step()
        if it % 50 == 0:
            history['loss'].append((it, float(loss)))
        if it % cfg.get('val_every', 500) == 0 or it == cfg['iters']:
            v = validation_error_(model, val_frames, val_mu, T_train)
            history['val'].append((it, v))
            if v < best['val'] and it > label_iters:
                best = dict(val=v, iteration=it, state={k: x.detach().clone() for k, x in model.state_dict().items()})
            log(f'it {it} loss {float(loss):.3e} val {v:.4f} ({time.time()-t0:.0f}s)')
            save_progress(it)
        if max_cpu_seconds is not None and time.process_time() >= max_cpu_seconds and it < cfg['iters']:
            if checkpoint_path is None:
                raise ValueError('A CPU budget requires a checkpoint path')
            status = 'checkpointed'
            save_progress(it)
            log(f'Checkpointed iteration {it} within the process CPU budget; rerun to resume')
            break
    if status != 'checkpointed':
        save_progress(it)
    return best, history, dict(status=status, seconds=elapsed+time.time()-t0,
                               cpu_seconds=elapsed_cpu+time.process_time()-cpu_start,
                               skipped_gradients=skipped_gradients, resumes=resumes, iterations=it)


def evaluate(model, problem, data, cfg, device):
    out = {}
    horizons = cfg['horizons']
    pred, true = rollout(model, *data['test'])
    out['test'] = summarize(pred, true, horizons)
    if problem.name == 'ks':
        out['ks_statistics'] = spectrum_error(pred, true, start=50)
    if model.kind != 'ar':
        out['dt_sweep'] = {}
        for dt_mult, sub in ((2, 1), (4, 1), (1, 2)):
            p, t = rollout(model, *data['test'], dt_mult=dt_mult, substeps=sub)
            out['dt_sweep'][f'x{dt_mult}/{sub}'] = summarize(p, t, horizons, dt_mult)
    out['ood'] = {}
    for value, pair in data['ood'].items():
        p, t = rollout(model, *pair)
        out['ood'][str(value)] = summarize(p, t, horizons)
    frames, mu = data['test']
    ref = norms(frames[:16]).amax(1)
    out['long'] = long_rollout(model, frames[:16, 0], mu[:16], cfg['long'], ref)
    q = torch.cat([frames[:8, 0], frames[:8, frames.shape[1]//2]])
    try:
        if cfg.get('skip_jacobian', False):
            raise ValueError('Not measured: inference-only reference solver')
        sigma = step_jacobian_norm(model, q, torch.cat([mu[:8], mu[:8]]))
        out['jacobian_sigma'] = dict(median=float(sigma.median()), max=float(sigma.max()))
    except Exception as exc:  # explicit arms can be non-finite
        out['jacobian_sigma'] = dict(error=repr(exc))
    out['operator'] = (operator_diagnostics(model, problem, device)
                       if cfg.get('operator_diagnostics', True) and not isinstance(model.linear, OseenNS) else None)
    if isinstance(model.linear, OseenNS):
        with torch.no_grad():
            for dt_mult in (1, 4):
                model.step(frames[:16, 40], mu[:16], model.h*dt_mult)
                out[f'gmres_residual_x{dt_mult}'] = float(model.linear.last_residual.max())
    return out


def run_name(arm, dt_mults, label=False, lip=False, init=False):
    return arm + '_lab'*label + '_lip'*lip + '_init'*init + ('' if dt_mults == [1] else '_mdt')


def run(args):
    device = args.device
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; refusing CPU fallback')
    torch.manual_seed(args.seed)
    problem = get_problem(args.problem, device)
    cfg = dict(CONFIG[args.problem])
    if args.iters is not None:
        cfg['iters'] = args.iters
    if args.lr is not None:
        cfg['lr'] = args.lr
    cfg['dt_mults'] = [int(x) for x in args.dt_mults.split(',')]
    if args.labels:
        cfg.update(label_pretrain=0.4, label_weight=1e-2)
    if args.jac_weight:
        cfg.update(jac_weight=args.jac_weight, jac_threshold=args.jac_threshold or JAC_THRESHOLD[args.problem])
    cfg['init_known'] = bool(args.init_known)
    data, scale = load_data(args.root, problem, device)
    model = StepModel(problem, args.arm, cfg['net'], scale, device).to(device)
    if args.init_known:
        model.init_operator_from_known(device)
    name = run_name(args.arm, cfg['dt_mults'], args.labels, bool(args.jac_weight), args.init_known)
    out_dir = Path(args.root)/'runs'/args.problem/f'{name}_s{args.seed}'
    if args.skip_existing and (out_dir/'result.json').exists():
        print('skip', out_dir.name)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(out_dir/'log.txt', 'w')

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg+'\n')
        log_file.flush()

    n_params = sum(p.numel() for p in model.parameters())
    log(f'{args.problem} {args.arm} seed {args.seed}: {n_params} parameters, config {cfg}')
    best, history, info = train(model, data, cfg, log)
    model.load_state_dict(best['state'])
    model.eval()
    t0 = time.time()
    result = dict(problem=args.problem, arm=name, base_arm=args.arm, seed=args.seed, parameters=n_params, config=cfg,
                  train=dict(info, best_val=best['val'], best_iteration=best['iteration']), history=history,
                  eval=evaluate(model, problem, data, cfg, device))
    result['device'] = dict(requested=str(device), torch=torch.__version__,
                           name=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else 'cpu')
    result['eval_seconds'] = time.time()-t0
    torch.save(dict(state=best['state'], arm=args.arm, problem=args.problem, scale=scale), out_dir/'model.pt')
    (out_dir/'result.json').write_text(json.dumps(result, indent=1, default=float))
    log(f'test {json.dumps(result["eval"]["test"])}')
    log(f'long {json.dumps(result["eval"]["long"])} jac {json.dumps(result["eval"]["jacobian_sigma"])}')


def reeval(args):
    """Recompute the evaluation of every saved run of a problem (training untouched)."""
    device = args.device
    problem = get_problem(args.problem, device)
    data, scale = load_data(args.root, problem, device)
    for run_dir in sorted((Path(args.root)/'runs'/args.problem).glob('*_s*')):
        rf, mf = run_dir/'result.json', run_dir/'model.pt'
        if not (rf.exists() and mf.exists()):
            continue
        result = json.loads(rf.read_text())
        ckpt = torch.load(mf, map_location=device, weights_only=False)
        model = StepModel(problem, result.get('base_arm', result['arm']), result['config']['net'], scale, device).to(device)
        model.load_state_dict(ckpt['state'])
        model.eval()
        result['eval'] = evaluate(model, problem, data, result['config'], device)
        rf.write_text(json.dumps(result, indent=1, default=float))
        print('re-evaluated', run_dir.name, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['data', 'run', 'reeval'])
    ap.add_argument('--problem', required=True, choices=sorted(PROBLEMS))
    ap.add_argument('--arm', choices=sorted(ARMS))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--root', required=True)
    ap.add_argument('--iters', type=int)
    ap.add_argument('--lr', type=float, help='network learning rate (default: 0.001)')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--dt-mults', default='1', help='comma-separated training step multiples, e.g. 1,2,4')
    ap.add_argument('--labels', action='store_true', help='exact remainder labels: pretrain then regularise N')
    ap.add_argument('--jac-weight', type=float, default=0., help='weight of the step-amplification penalty')
    ap.add_argument('--jac-threshold', type=float, help='allowed amplification (default: true flow map, JAC_THRESHOLD)')
    ap.add_argument('--init-known', action='store_true', help='initialise learned L from the true operator')
    ap.add_argument('--skip-existing', action='store_true')
    args = ap.parse_args()
    if args.command == 'data':
        generate_data(args.problem, args.root, args.device)
    elif args.command == 'reeval':
        reeval(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
