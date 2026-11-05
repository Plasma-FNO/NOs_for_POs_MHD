"""One rollout objective, sampler and evaluator for AR/NODE/native scratch/pretrained.

Native full training requires an exhaustive audit of this exact initialization.
Term pretraining is a separate stage; no term penalty enters rollout training.
"""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import torch
import parallel

from comparison import (ARMS, digest, write_json, normalize, decoded_fields, load_fields,
                        reference_for, frame_sums, scores)
from model import SplitFNO
from native_terms import NativeOperators
from native_rollout import NativeRollout
from train_es import relative_lp, save_checkpoint


def merge_limiter(target, source):
    """Accumulate counts and worst relaxation fractions across native windows."""
    for key,value in source.items():
        target[key]=min(target.get(key,1.),value) if key.startswith('minimum_') else target.get(key,0)+value


def initialization_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if isinstance(value, torch.Tensor):
            h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class Experiment:
    def __init__(self, data, arm, device, pretrained=None):
        self.data, self.arm, self.device = data, arm, device
        self.meta = json.loads((data/'comparison.json').read_text())
        self.cfg = self.meta['protocol']
        if digest(data/'data.pt') != self.meta['data_sha256']:
            raise ValueError('Changed shared data cache')
        for name, expected in self.meta['term_files_sha256'].items():
            if digest(data/'terms'/name) != expected: raise ValueError('Changed term data: '+name)
        for name, expected in self.meta['source_files'].items():
            stat = Path(name).stat()
            if [stat.st_size, stat.st_mtime_ns] != expected: raise ValueError('Changed source file: '+name)
        self.values = torch.load(data/'data.pt', weights_only=False, map_location='cpu')
        normalization=self.meta.get('normalization_details',{})
        if normalization.get('method')=='zscore':
            self.mean,self.scale=(torch.tensor(normalization[k],dtype=torch.float64,device=device) for k in ('mean','scale'))
        else:
            self.mean, self.scale = (self.values[k].to(device) for k in ('mean','scale'))
        cfg = self.cfg; torch.manual_seed(cfg['seed'])
        self.native = arm.startswith('opsplit')
        self.composite = arm == 'composite'
        self.pretraining = None
        self.transfer = None
        if self.native:
            norms = torch.load(data/'terms/normalization.pt', weights_only=True)
            self.model = NativeOperators(**norms, width=cfg['width'], modes=cfg['modes'])
            if arm == 'opsplit-pretrained':
                if pretrained is None: raise ValueError('Pretrained arm requires a term checkpoint')
                saved = torch.load(pretrained, map_location='cpu', weights_only=False)
                if (saved.get('phase') != 'term_pretraining' or saved.get('case') != self.meta['case']
                    or saved.get('dataset_sha256') != digest(data/'terms/manifest.json')):
                    raise ValueError('Term checkpoint must come from this shared valid-state dataset')
                if saved['model_config'] != self.model.config: raise ValueError('Term architecture differs')
                for key, value in norms.items():
                    if not torch.equal(value, saved['normalization'][key]): raise ValueError('Normalizer mismatch')
                self.model.load_state_dict(saved['state_dict'])
                self.pretraining = dict(checkpoint=str(pretrained.resolve()), sha256=digest(pretrained),
                                        selected_epoch=saved['epoch'],
                                        selected_by='held-out term MSE; additional training stage')
            elif pretrained is not None: raise ValueError('Scratch arm cannot load pretrained weights')
        elif self.composite:
            if pretrained is not None: raise ValueError('Composite scratch cannot load a term checkpoint')
            from composite import CompositeSplit
            from native import read_input, common_coefficients
            entries = self.meta['entries']['train']
            coefficients = [common_coefficients(read_input(e['input']), self.meta['case']) for e in entries]
            constants = {(c['d'], c['eta'], c['nominal_frame_dt']) for c in coefficients}
            if len(constants) != 1: raise ValueError('Composite linear coefficients must be common to the cohort')
            diffusion, resistivity, frame_dt = constants.pop()
            e = entries[0]
            self.model = CompositeSplit(e['r'], e['z'], self.mean, self.scale,
                                        channels=len(self.mean), width=cfg['width'], modes=cfg['modes'],
                                        layers=cfg['layers'], diffusion=diffusion, resistivity=resistivity, frame_dt=frame_dt)
            if 'composite_transfer' in cfg:
                from prepare_transfer import load_fluid_transfer
                self.transfer=load_fluid_transfer(self.model,cfg['composite_transfer'])
        else:
            if pretrained is not None: raise ValueError('Only the pretrained arm accepts a source checkpoint')
            e = self.meta['entries']['train'][0]
            self.model = SplitFNO(e['r'], e['z'], channels=4 if self.meta['case']=='es' else 6,
                                 width=cfg['width'], modes=cfg['modes'], layers=cfg['layers'],
                                 monolithic=True, autoregressive=arm=='ar', paper_architecture=True)
        self.model.to(device)
        self.identity = dict(arm=arm, comparison_sha256=digest(data/'comparison.json'),
                             initialization_sha256=initialization_hash(self.model), device=device,
                             code_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')})

    @lru_cache(maxsize=4)
    def solver(self, split, index):
        entry = self.meta['entries'][split][index]
        q = (self.values[split]['q'][index, :1] if split != 'test' else load_fields(entry, 1)[1])
        return NativeRollout(reference_for(entry, q), q, self.model, device=self.device,compiled_terms=self.device.startswith('cuda'))

    def predictions(self, split, index, exported, q):
        if self.native:
            solver = self.solver(split, index); solver.reset_limiter_stats()
            state = q[:1].to(self.device)
        else:
            values = exported[:1].to(self.device)
            state = values.double() if self.composite else normalize(values,self.mean,self.scale)
        for step in range(1,len(exported)):
            if self.native:
                state = solver.step(state, solver.ref.c['nominal_frame_dt'], self.cfg.get('native_substeps', 8))
                result = decoded_fields(solver.ref, state)[0]
            else:
                try:state = self.model(state)
                except (FloatingPointError,RuntimeError) as error:
                    raise RuntimeError(f'{split} trajectory {self.meta["entries"][split][index]["id"]}, rollout step {step}: {error}') from error
                result = state[0] if self.composite else (state*self.scale[None,:,None,None]+self.mean[None,:,None,None])[0]
            if not torch.isfinite(result).all(): raise FloatingPointError('Nonfinite predicted fields')
            yield result

    def objective(self, split, index, start, horizon):
        exported = self.values[split]['exported'][index, start:start+horizon+1]
        q = self.values[split]['q'][index, start:start+horizon+1]
        loss = torch.zeros((), device=self.device)
        for prediction, truth in zip(self.predictions(split,index,exported,q), exported[1:]):
            loss = loss + relative_lp(normalize(prediction[None],self.mean,self.scale),
                                       normalize(truth[None].to(self.device),self.mean,self.scale))
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite common rollout objective')
        return loss

    def batch_loss(self, windows):
        # Same summed loss and optimizer update; native equations currently process one window at a time.
        if self.native:
            total = 0.;self.last_batch_limiter={}
            for index, start in windows:
                loss = self.objective('train',index,start,self.cfg['horizon'])
                loss.backward(); total += float(loss.detach())
                merge_limiter(self.last_batch_limiter,self.solver('train',index).limiter_stats)
            return total
        frames = torch.stack([self.values['train']['exported'][i,s:s+self.cfg['horizon']+1] for i,s in windows]).to(self.device)
        frames = frames.double() if self.composite else normalize(frames,self.mean,self.scale)
        state, loss = frames[:,0], frames.new_zeros(())
        for target in frames[:,1:].unbind(1):
            state = self.model(state)
            loss = loss+(relative_lp(normalize(state,self.mean,self.scale),normalize(target,self.mean,self.scale))
                         if self.composite else relative_lp(state,target))
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite common rollout objective')
        loss.backward()
        return float(loss.detach())

    @torch.no_grad()
    def validation(self, horizon=None, limit=None):
        self.model.eval(); rows=[];failure=None
        indices=self.meta['val_indices'][:limit]
        try:
            for index in indices[parallel.rank()::parallel.size()]:
                value=self.objective('val',index,0,horizon or self.cfg['train_frames']-1)
                row=dict(id=self.meta['entries']['val'][index]['id'],loss=float(value))
                if self.native:row['limiter']=dict(self.solver('val',index).limiter_stats)
                rows.append(row)
        except (FloatingPointError,RuntimeError) as error:failure=str(error)
        gathered=parallel.gather(dict(rows=rows,error=failure))
        if any(r['error'] for r in gathered):raise RuntimeError(str([r['error'] for r in gathered if r['error']]))
        by_id={r['id']:r for part in gathered for r in part['rows']}
        rows=[by_id[self.meta['entries']['val'][index]['id']] for index in indices]
        return sum(r['loss'] for r in rows)/(len(rows)*(horizon or self.cfg['train_frames']-1)), rows



def audit_trajectory(ex, split, index, starts, smoke=False):
    ex.model.eval();failures=[];worst=None;checked=0;gradient=False;limiter={};gradient_limiter={}
    horizon=ex.cfg['horizon'] if split=='train' or smoke else ex.cfg['train_frames']-1
    for start in starts:
        try:
            with torch.no_grad():value=float(ex.objective(split,index,start,horizon))
            if worst is None or value>worst[0]:worst=(value,start)
        except (FloatingPointError,RuntimeError) as error:
            failures.append(dict(split=split,id=ex.meta['entries'][split][index]['id'],start=start,error=str(error)))
        if ex.native:merge_limiter(limiter,ex.solver(split,index).limiter_stats)
        checked+=1
    if split=='train' and worst is not None:
        try:
            ex.model.zero_grad(set_to_none=True)
            ex.objective(split,index,worst[1],horizon).backward()
            torch.nn.utils.clip_grad_norm_(ex.model.parameters(),1.,error_if_nonfinite=True)
            gradient=True
            if ex.native:gradient_limiter=dict(ex.solver(split,index).limiter_stats)
        except (FloatingPointError,RuntimeError) as error:
            failures.append(dict(split='gradient',id=ex.meta['entries'][split][index]['id'],start=worst[1],error=str(error)))
    return dict(split=split,index=index,windows=checked,gradient_checked=gradient,failures=failures,
                limiter=limiter,gradient_limiter=gradient_limiter)


def initialize_audit_worker(data,arm,device,pretrained,devices=None):
    global _audit_experiment
    torch.set_num_threads(1)
    if devices is not None:torch.cuda.set_device(devices.get())
    _audit_experiment=Experiment(data,arm,device,pretrained)


def audit_worker(task):
    return audit_trajectory(_audit_experiment,*task)


def audit(experiment, path, smoke=False, workers=1, resume=False, max_seconds=0, gpus=1):
    """Exhaustive audit; workers share a GPU, each handling independent trajectories."""
    from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
    import multiprocessing
    if gpus>1 and (workers!=gpus or not experiment.device.startswith('cuda')):raise ValueError('Use exactly one audit worker per GPU')
    ex=experiment;started=time.monotonic();rows=[];elapsed=0.
    deadline=started+max_seconds if max_seconds else None
    if resume:
        previous=json.loads(path.read_text())
        if previous.get('identity')!=ex.identity or previous.get('exhaustive')!=(not smoke):
            raise ValueError('Audit resume requires identical code/data/initialization and scope')
        rows=previous['trajectories'];elapsed=previous['seconds']
    windows=ex.meta['windows'][:2] if smoke else ex.meta['windows']
    indices=sorted({i for i,_ in windows})
    val_indices=ex.meta['val_indices'][:1 if smoke else None]
    tasks=[('train',i,[s for j,s in windows if j==i],smoke) for i in indices]
    tasks += [('val',i,[0],smoke) for i in val_indices]
    expected={(t[0],t[1]):len(t[2]) for t in tasks}
    done={(r['split'],r['index']) for r in rows}
    if len(done)!=len(rows) or any((r['split'],r['index']) not in expected or r['windows']!=expected[(r['split'],r['index'])] for r in rows):
        raise ValueError('Invalid partial audit coverage')
    remaining=[t for t in tasks if (t[0],t[1]) not in done]
    def record(complete):
        failures=[f for row in rows for f in row['failures']]
        gradients=sum(row['gradient_checked'] for row in rows)
        checked=sum(row['windows'] for row in rows if row['split']=='train')
        validated=sum(row['split']=='val' and not row['failures'] for row in rows)
        passed=complete and not failures and checked==len(windows) and gradients==len(indices) and validated==len(val_indices)
        write_json(path,dict(identity=ex.identity,passed=passed,exhaustive=not smoke,completed=complete,
                            workers=workers,gpus=gpus,windows=checked,gradient_trajectories=gradients,
                            validation_trajectories=validated,failures=failures,
                            trajectories=sorted(rows,key=lambda r:(r['split'],r['index'])),seconds=elapsed+time.monotonic()-started))
        return passed
    if workers==1:
        for task in remaining:
            if deadline and time.monotonic()>=deadline:break
            rows.append(audit_trajectory(ex,*task));record(False)
            print(json.dumps(dict(audited_trajectories=len(rows),total=len(tasks))),flush=True)
    else:
        pretrained=Path(ex.pretraining['checkpoint']) if ex.pretraining else None
        context=multiprocessing.get_context('spawn');devices=context.Queue() if gpus>1 else None
        if devices is not None:
            for index in range(gpus):devices.put(index)
        with ProcessPoolExecutor(max_workers=workers,mp_context=context,
                                 initializer=initialize_audit_worker,initargs=(ex.data,ex.arm,ex.device,pretrained,devices)) as pool:
            todo=iter(remaining);pending=set()
            def fill():
                # Do not prequeue uncancellable work beyond the checkpoint budget.
                while len(pending)<workers and (not deadline or time.monotonic()<deadline):
                    task=next(todo,None)
                    if task is None:break
                    pending.add(pool.submit(audit_worker,task))
            fill()
            while pending:
                finished,pending=wait(pending,return_when=FIRST_COMPLETED)
                for future in finished:
                    rows.append(future.result());record(False)
                    print(json.dumps(dict(audited_trajectories=len(rows),total=len(tasks))),flush=True)
                fill()
    return record(len(rows)==len(tasks))


def require_audit(experiment, path):
    report=json.loads(path.read_text())
    if not report.get('passed') or not report.get('exhaustive') or report.get('identity')!=experiment.identity:
        raise ValueError('Full training requires a passed exhaustive audit of this exact data/code/initialization')


@torch.no_grad()
def evaluate(ex, path, smoke=False, deadline=None):
    ex.model.eval(); rows=[]
    evaluation_identity=dict(experiment=ex.identity, checkpoint=initialization_hash(ex.model), smoke=smoke)
    if path.exists():
        previous=json.loads(path.read_text())
        if previous.get('identity')!=evaluation_identity: raise ValueError('Evaluation checkpoint changed')
        if previous.get('complete'): return
        rows=previous['trajectories']
    horizons=sorted({h for h in (10,ex.cfg['train_frames']-1,100,ex.cfg['eval_horizon']) if h<=ex.cfg['eval_horizon']})
    if smoke: horizons=[min(5,ex.cfg['eval_horizon'])]
    entries=ex.meta['entries']['test'][:1 if smoke else None]
    for index,entry in enumerate(entries):
        if index<len(rows): continue
        if deadline and time.monotonic()>=deadline:
            write_json(path,dict(identity=evaluation_identity,complete=False,trajectories=rows))
            raise SystemExit(75)
        available=ex.meta['test_prefixes'][entry['id']]
        requested=min(max(horizons),available)
        row=dict(id=entry['id'],valid_source_steps=available,completed_steps=0,scores={},persistence={},failure=None)
        if requested<1: rows.append(row); continue
        exported,q=load_fields(entry,requested+1)
        sums={}; persists={}; extrapolation={}; delta_error=delta_energy=0.
        r=torch.tensor(entry['r'],device=ex.device,dtype=torch.float64)
        try:
            for step,(pred,truth) in enumerate(zip(ex.predictions('test',index,exported,q),exported[1:]),1):
                truth=truth.to(ex.device)
                for result,prediction in ((sums,pred),(persists,exported[0].to(ex.device))):
                    for key,value in frame_sums(prediction,truth,ex.mean,ex.scale,r).items():
                        result[key]=result.get(key,0)+value
                        if result is sums and step>=ex.cfg['train_frames']:
                            extrapolation[key]=extrapolation.get(key,0)+value
                if ex.meta['case']=='em':
                    delta_error+=float(((pred[4]-truth[4]).double().square()*r[None,:]).mean())
                    delta_energy+=float(((truth[4]-exported[0,4].to(ex.device)).double().square()*r[None,:]).mean())
                row['completed_steps']=step
                if step in horizons:
                    row['scores'][str(step)]=scores({k:v/step for k,v in sums.items()})
                    row['persistence'][str(step)]=scores({k:v/step for k,v in persists.items()})
                    if step>=ex.cfg['train_frames']:
                        row['scores'][str(step)]['temporal_extrapolation']=scores({k:v/(step-ex.cfg['train_frames']+1) for k,v in extrapolation.items()})
                    if ex.meta['case']=='em':
                        row['scores'][str(step)]['delta_psi_relative_l2']=(delta_error/max(delta_energy,1e-30))**.5
                        row['persistence'][str(step)]['delta_psi_relative_l2']=1. if delta_energy>0 else 0.
        except (FloatingPointError,RuntimeError) as error:
            row['failure']=str(error)
        if ex.native:row['limiter']=dict(ex.solver('test',index).limiter_stats)
        rows.append(row)
        write_json(path,dict(identity=evaluation_identity,complete=False,trajectories=rows))
        print(json.dumps(dict(test_id=entry['id'],completed=row['completed_steps'],failure=row['failure'])),flush=True)
    summary={}
    for horizon in horizons:
        eligible=[r for r in rows if r['valid_source_steps']>=horizon]
        completed=[r for r in eligible if str(horizon) in r['scores']]
        item=dict(eligible=len(eligible),completed=len(completed),failed=len(eligible)-len(completed),
                  source_excluded=len(rows)-len(eligible),mean_nrmse=None,mean_r_weighted_relative_l2=None)
        # No survivor-only aggregate masquerading as the common-cohort result.
        if eligible and len(completed)==len(eligible):
            for metric in ('nrmse','r_weighted_relative_l2'):
                item['mean_'+metric]=torch.tensor([r['scores'][str(horizon)][metric] for r in completed]).mean(0).tolist()
        summary[str(horizon)]=item
    write_json(path,dict(identity=evaluation_identity,complete=True,arm=ex.arm,fields=['rho','T','potential','omega']+(['psi','j'] if ex.meta['case']=='em' else []),
                        normalization=ex.meta['normalization']+'; epsilon1e-6',
                        aggregation='per-trajectory space-time ratios, mean across identical eligible trajectories; null if any model fails',
                        horizons=summary,trajectories=rows,smoke=smoke))


def train(ex, output, resume=False, smoke=False, max_seconds=0, train_only=False, stop_after_epoch=None):
    if stop_after_epoch is not None and (smoke or not ex.composite or not 0<stop_after_epoch<=ex.cfg['epochs']):
        raise ValueError('Bounded optimizer check requires a valid composite epoch budget')
    cfg=ex.cfg; optimizer=torch.optim.Adam([p for p in ex.model.parameters() if p.requires_grad],lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,cfg['scheduler_step'],cfg['scheduler_gamma'])
    provenance=dict(identity=ex.identity,protocol=cfg,smoke=smoke,term_weight=0,
                    pretraining=ex.pretraining,transfer=ex.transfer,parameters=sum(p.numel() for p in ex.model.parameters()),
                    effective_batch_size=cfg['batch_size'],native_gradient_accumulation=ex.native,world_size=parallel.size())
    epoch0=offset0=0; best=float('inf'); elapsed=0.; train_sum=0.; seen=0
    if resume:
        if json.loads((output/'provenance.json').read_text())!=provenance: raise ValueError('Resume identity mismatch')
        saved=torch.load(output/'last.pt',map_location='cpu',weights_only=False)
        ex.model.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        epoch0,offset0,best,elapsed=(saved[k] for k in ('next_epoch','next_offset','best','elapsed_seconds'))
        train_sum,seen=saved['train_sum'],saved['seen']
        if 'rank_rng' in saved:parallel.restore_rng(saved['rank_rng'][parallel.rank()],ex.device)
        else:
            torch.set_rng_state(saved['rng'])
            if ex.device.startswith('cuda'):torch.cuda.set_rng_state_all(saved['cuda_rng'])
        history=output/'history.jsonl'
        if parallel.rank()==0 and history.exists():
            history.write_text(''.join(json.dumps(row)+'\n' for row in map(json.loads,history.read_text().splitlines()) if row['epoch']<epoch0))
    elif parallel.rank()==0:
        output.mkdir(parents=True,exist_ok=False);write_json(output/'provenance.json',provenance)
    parallel.gather(None)  # All writers see the created directory before training.
    started=time.monotonic()
    def payload(epoch,offset,rank_rng):
        return dict(state_dict=ex.model.state_dict(),model_config=ex.model.config,identity=ex.identity,
                    optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),next_epoch=epoch,next_offset=offset,
                    best=best,train_sum=train_sum,seen=seen,elapsed_seconds=elapsed+time.monotonic()-started,
                    rank_rng=rank_rng)
    def checkpoint(path,epoch,offset):
        states=parallel.gather(parallel.rng_state(ex.device))
        if parallel.rank()==0:save_checkpoint(path,payload(epoch,offset,states))
    def timed_exit(epoch,offset):
        if parallel.root_flag(bool(max_seconds and time.monotonic()-started>=max_seconds),ex.device):
            checkpoint(output/'last.pt',epoch,offset)
            if parallel.size()>1:
                if parallel.rank()==0:write_json(output/'requeue.json',dict(epoch=epoch,offset=offset))
                parallel.gather(None)  # Commit the checkpoint/marker before any worker exits.
                raise SystemExit(0)
            raise SystemExit(75)
    windows=ex.meta['windows']
    for epoch in range(epoch0,1 if smoke else stop_after_epoch or cfg['epochs']):
        order=torch.randperm(len(windows),generator=torch.Generator().manual_seed(cfg['seed']+epoch)).tolist()
        if smoke: order=order[:min(2*cfg['batch_size'] if ex.composite else 2,len(order))]
        ex.model.train()
        for offset in range(offset0 if epoch==epoch0 else 0,len(order),cfg['batch_size']):
            timed_exit(epoch,offset)
            batch=[windows[i] for i in order[offset:offset+cfg['batch_size']]]
            optimizer.zero_grad(set_to_none=True)
            try: value=parallel.batch_loss(ex,batch)
            except (FloatingPointError,RuntimeError) as error:
                if parallel.rank()==0:write_json(output/'failure.json',dict(epoch=epoch,offset=offset,windows=batch,error=str(error)))
                raise
            torch.nn.utils.clip_grad_norm_(ex.model.parameters(),cfg['grad_clip'],error_if_nonfinite=True);optimizer.step()
            train_sum+=value;seen+=len(batch)
            if ex.native or (offset//cfg['batch_size'])%20==0:
                checkpoint(output/'last.pt',epoch,min(offset+cfg['batch_size'],len(order)))
            progress=dict(epoch=epoch,offset=offset,loss=value/len(batch)/cfg['horizon'],seconds=elapsed+time.monotonic()-started)
            if ex.native:progress['limiter']=ex.last_batch_limiter
            if parallel.rank()==0:print(json.dumps(progress),flush=True)
        timed_exit(epoch,len(order))
        try:
            score,validation=ex.validation(horizon=cfg['horizon'] if smoke and not ex.composite else None,limit=1 if smoke else None)
        except (FloatingPointError,RuntimeError) as error:
            checkpoint(output/'failed-validation.pt',epoch,len(order))
            if parallel.rank()==0:write_json(output/'failure.json',dict(stage='validation',epoch=epoch,error=str(error)))
            raise
        row=dict(epoch=epoch,train_relative_l2=train_sum/(seen*cfg['horizon']),val_relative_l2=score,
                 lr=optimizer.param_groups[0]['lr'],elapsed_seconds=elapsed+time.monotonic()-started)
        scheduler.step();improved=score<best;best=min(best,score);train_sum=0.;seen=0
        if improved:checkpoint(output/'best.pt',epoch+1,0)
        checkpoint(output/'last.pt',epoch+1,0)
        if parallel.rank()==0:
            with (output/'history.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            write_json(output/'validation.json',dict(epoch=epoch,trajectories=validation))
        offset0=0
    if stop_after_epoch is not None:
        if parallel.rank()==0:
            history=[json.loads(line) for line in (output/'history.jsonl').read_text().splitlines()]
            if [row['epoch'] for row in history]!=list(range(stop_after_epoch)):
                raise ValueError('Incomplete bounded optimizer check')
            write_json(output/'stability.json',dict(identity=ex.identity,passed=True,completed_epochs=stop_after_epoch,
                       validation_horizon=cfg['train_frames']-1,validation_trajectories=len(ex.meta['val_indices']),
                       checkpoint_sha256=digest(output/'last.pt'),history=history))
        return
    if not train_only:
        if parallel.size()!=1:raise ValueError('Distributed training requires a separate single-GPU evaluation stage')
        saved=torch.load(output/'best.pt',map_location=ex.device,weights_only=False);ex.model.load_state_dict(saved['state_dict'])
        evaluate(ex,output/'test_metrics.json',smoke,started+max_seconds if max_seconds else None)


def require_composite_stability(ex, output):
    report=json.loads((output/'stability.json').read_text())
    required_epochs=ex.cfg.get('composite_stability_epochs',26 if ex.meta['case']=='em' else 12)
    if (report.get('identity')!=ex.identity or not report.get('passed') or report.get('completed_epochs',0)<required_epochs
        or report.get('validation_horizon')!=ex.cfg['train_frames']-1
        or report.get('validation_trajectories')!=len(ex.meta['val_indices'])):
        raise ValueError(f'Composite full training requires {required_epochs} checked optimizer epochs with identical code/data')
    saved=torch.load(output/'last.pt',map_location='cpu',weights_only=False)
    if saved['identity']!=ex.identity or saved['next_epoch']<report['completed_epochs']:
        raise ValueError('Composite continuation must resume the checked optimizer campaign')
    if saved['next_epoch']==report['completed_epochs'] and saved['next_offset']==0:
        if digest(output/'last.pt')!=report['checkpoint_sha256']:raise ValueError('Changed checked checkpoint')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arm',choices=(*ARMS,'composite'),required=True);p.add_argument('--pretrained',type=Path)
    p.add_argument('--device',default='cuda');p.add_argument('--audit',action='store_true')
    p.add_argument('--audit-gpus',type=int,default=1)
    p.add_argument('--audit-workers',type=int,default=1,help='One worker per GPU by default; concurrent backward graphs can exhaust device memory')
    p.add_argument('--train-only',action='store_true');p.add_argument('--evaluate-only',action='store_true')
    p.add_argument('--audit-report',type=Path);p.add_argument('--resume',action='store_true')
    p.add_argument('--smoke',action='store_true');p.add_argument('--max-seconds',type=float,default=0)
    p.add_argument('--stop-after-epoch',type=int,help='Bounded composite optimizer/validation check; saves a resumable checkpoint')
    a=p.parse_args();torch.set_num_threads(1);parallel.initialize(a.device)
    if parallel.size()>1 and (a.audit or a.evaluate_only or not a.train_only):p.error('Distributed launch supports --train-only; audit/evaluation have separate stages')
    if a.audit_workers<1:p.error('Positive audit worker count required')
    ex=Experiment(a.data,a.arm,a.device,a.pretrained)
    if a.evaluate_only:
        saved=torch.load(a.output/'best.pt',map_location=a.device,weights_only=False)
        if saved['identity']!=ex.identity:raise ValueError('Evaluation checkpoint identity mismatch')
        ex.model.load_state_dict(saved['state_dict'])
        evaluate(ex,a.output/'test_metrics.json',a.smoke,time.monotonic()+a.max_seconds if a.max_seconds else None)
    elif a.audit:
        if a.output.exists() and not a.resume:raise FileExistsError(a.output)
        if not audit(ex,a.output,a.smoke,1 if a.smoke else a.audit_workers,a.resume,a.max_seconds,a.audit_gpus):
            report=json.loads(a.output.read_text())
            raise SystemExit(2 if report['completed'] else 75)
    else:
        if (ex.native or ex.composite) and not a.smoke:
            if not a.audit_report:p.error('Native/composite full training requires --audit-report')
            require_audit(ex,a.audit_report)
            if ex.composite and a.stop_after_epoch is None:
                if not a.resume:p.error('Composite full training must resume the successful optimizer check')
                require_composite_stability(ex,a.output)
        train(ex,a.output,a.resume,a.smoke,a.max_seconds,a.train_only,a.stop_after_epoch)


if __name__=='__main__':main()
