"""ES three-field AR, Euler NODE and composite OpsSplit, in release-scaled coordinates.

Vorticity is an interior diagnostic/optional loss target, never a model state.
The shared trainer supplies optimizer, scheduling, exact resume and validation
selection. No common-native state or physical-time conversion is performed.
"""
import argparse
import json
from pathlib import Path
import time

import h5py
import torch
from torch import nn

from composite import InteriorDiffusion, positive_increment
from model import fno
from data import layout, read_frames
from comparison import digest, write_json, frame_sums, scores
from train_es import relative_lp
from train_comparison import train, audit, require_audit, require_composite_stability, initialization_hash

ARMS = ('ar', 'node', 'opsplit', 'opsplit-vorticity')
# Calibrated release-potential/vorticity relation, not a state normalization.
OMEGA_FACTOR = -6.460061919207894e-5
# Equivalent density diffusion per stored frame; time is saved-frame index.
DIFFUSION_PER_FRAME = 5e-6
CROP = 4


def laplacian(phi, r, z):
    """Fourth-order cylindrical Laplacian, returning only the valid inner grid."""
    dr, dz = r[1]-r[0], z[1]-z[0]
    c = phi[...,2:-2,2:-2]
    rr = (-phi[...,2:-2,4:]+16*phi[...,2:-2,3:-1]-30*c+16*phi[...,2:-2,1:-3]-phi[...,2:-2,:-4])/(12*dr**2)
    zz = (-phi[...,4:,2:-2]+16*phi[...,3:-1,2:-2]-30*c+16*phi[...,1:-3,2:-2]-phi[...,:-4,2:-2])/(12*dz**2)
    first = (-phi[...,2:-2,4:]+8*phi[...,2:-2,3:-1]-8*phi[...,2:-2,1:-3]+phi[...,2:-2,:-4])/(12*dr)
    return (rr+zz+first/r[2:-2])[...,2:-2,2:-2]


def load_release(entry, frames):
    with h5py.File(entry['path'], 'r') as handle:
        keys, _, _ = layout(handle, 'es')
        return torch.from_numpy(read_frames(handle, keys, 0, frames))


class ThreeFieldModel(nn.Module):
    def __init__(self, r, z, mean, scale, method, width=64, modes=32, layers=6, opsplit_update='additive'):
        super().__init__()
        if method not in ('ar', 'node', 'opsplit'): raise ValueError('Unknown three-field method')
        if opsplit_update not in ('additive', 'softplus'): raise ValueError('Unknown OpsSplit update map')
        if len(mean)!=3 or len(scale)!=3: raise ValueError('Exactly three normalizers required')
        for name,value in dict(r=r,z=z,mean=mean,scale=scale).items():
            self.register_buffer(name,torch.as_tensor(value,dtype=torch.float64).detach().clone())
        if torch.any(self.scale<=0) or not torch.isfinite(self.scale).all(): raise ValueError('Invalid scale')
        self.method=method
        self.opsplit_update=opsplit_update
        self.config=dict(channels=3,fields=['rho','T','potential'],method=method,width=width,modes=modes,layers=layers,
                         normalization='fixed scalar preconditioning then train-only minmax',state='uncentered preconditioned float64',
                         time_basis='saved-frame index; dt=1',native_state_conversion=False,
                         density_diffusion_per_frame=DIFFUSION_PER_FRAME if method=='opsplit' else 0.,
                         opsplit_update=opsplit_update if method=='opsplit' else None,
                         omega_factor=OMEGA_FACTOR,omega_crop=CROP)
        self.network=fno(3,3,width,modes,layers,True)
        if method=='opsplit':
            nn.init.zeros_(self.network.projection.fcs[-1].weight)
            nn.init.zeros_(self.network.projection.fcs[-1].bias)
            self.particle=InteriorDiffusion(r,z,DIFFUSION_PER_FRAME,1.)

    def normalize(self,x):
        return (x-self.mean[:,None,None])/self.scale[:,None,None]

    def omega(self,x):
        return OMEGA_FACTOR*laplacian(x[:,2].double(),self.r,self.z)

    def forward(self,x):
        if x.ndim!=4 or x.shape[1]!=3: raise ValueError('State must be [batch,3,Z,R]')
        x=x.double()
        if not torch.isfinite(x).all(): raise FloatingPointError('Nonfinite input')
        positive=self.method=='opsplit' and self.opsplit_update=='softplus'
        if positive and torch.any(x[:,:2]<=0): raise FloatingPointError('Nonpositive OpsSplit input')
        rate=self.network(self.normalize(x).float()).double()
        if self.method=='ar':
            result=rate*self.scale[None,:,None,None]+self.mean[None,:,None,None]
        elif self.method=='node':
            result=x+rate*self.scale[None,:,None,None]
        else:
            candidate=x+rate*self.scale[None,:,None,None]
            if positive:
                candidate=torch.stack((positive_increment(x[:,0],rate[:,0],self.scale[0]),
                                       positive_increment(x[:,1],rate[:,1],self.scale[1]),candidate[:,2]),1)
            result=torch.stack((self.particle(candidate[:,0]),candidate[:,1],candidate[:,2]),1)
        if not torch.isfinite(result).all(): raise FloatingPointError('Nonfinite prediction')
        if positive and torch.any(result[:,:2]<=0): raise FloatingPointError('Nonpositive OpsSplit prediction')
        return result

    def load_state_dict(self,state_dict,strict=True,assign=False):
        state_dict=state_dict.copy();state_dict.pop('_metadata',None)
        return super().load_state_dict(state_dict,strict=strict,assign=assign)


class ThreeFieldExperiment:
    def __init__(self,data,arm,device='cuda',omega_weight=100.,opsplit_update='additive'):
        if arm not in ARMS or omega_weight<0: raise ValueError('Invalid arm/weight')
        self.data,self.arm,self.device=Path(data),arm,device
        self.meta=json.loads((self.data/'comparison.json').read_text())
        if self.meta['case']!='es': raise ValueError('Three-field workflow is ES-only')
        if digest(self.data/'data.pt')!=self.meta['data_sha256']: raise ValueError('Changed shared data')
        for name,expected in self.meta['source_files'].items():
            stat=Path(name).stat()
            if [stat.st_size,stat.st_mtime_ns]!=expected: raise ValueError('Changed source: '+name)
        self.values=torch.load(self.data/'data.pt',map_location='cpu',weights_only=False,mmap=True)
        self.mean,self.scale=(self.values[k][:3].double().to(device) for k in ('mean','scale'))
        self.omega_mean,self.omega_scale=(self.values[k][3:4].double().to(device) for k in ('mean','scale'))
        source=self.meta['protocol']
        keys=('epochs','train_trajectories','train_frames','batch_size','horizon','width','modes','layers','lr','weight_decay',
              'scheduler_step','scheduler_gamma','grad_clip','eval_horizon','seed')
        self.cfg={k:source[k] for k in keys}
        self.weight=float(omega_weight) if arm=='opsplit-vorticity' else 0.
        self.cfg.update(fields=['rho','T','potential'],normalization='scalar [1e20,1,1e3] then existing train-only minmax',
                        native_state_conversion=False,vorticity_weight=self.weight,omega_factor=OMEGA_FACTOR,omega_crop=CROP,
                        opsplit_update=opsplit_update if arm.startswith('opsplit') else None,
                        auxiliary_loss='batch-summed mean interior ((derived_omega-target)/training_half_range)^2, summed over five steps',
                        checkpoint_selection='primary three-field 49-step relative L2 only, shared across arms',
                        train_relative_l2='primary loss only; optimizer additionally uses weighted vorticity MSE when enabled',
                        composite_stability_epochs=min(12,source['epochs']))
        self.meta['protocol']=self.cfg
        self.native=False;self.composite=arm.startswith('opsplit');self.pretraining=None;self.transfer=None
        torch.manual_seed(self.cfg['seed'])
        entry=self.meta['entries']['train'][0]
        self.model=ThreeFieldModel(entry['r'],entry['z'],self.mean,self.scale,'opsplit' if self.composite else arm,
                                   opsplit_update=opsplit_update,
                                   **{k:self.cfg[k] for k in ('width','modes','layers')}).to(device)
        self.identity=dict(arm=arm,comparison_sha256=digest(self.data/'comparison.json'),protocol=self.cfg,
                           initialization_sha256=initialization_hash(self.model),device=device,
                           code_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')})
        self.batch_counter=0

    def losses(self,frames):
        frames=frames.to(self.device).double();state=frames[:,0,:3]
        primary=state.new_zeros(());auxiliary=state.new_zeros(())
        for truth in frames[:,1:].unbind(1):
            state=self.model(state)
            primary=primary+relative_lp(self.model.normalize(state),self.model.normalize(truth[:,:3]))
            if self.weight:
                error=(self.model.omega(state)-truth[:,3,CROP:-CROP,CROP:-CROP])/self.omega_scale[0]
                auxiliary=auxiliary+error.square().mean((1,2)).sum()
        if not torch.isfinite(primary+self.weight*auxiliary): raise FloatingPointError('Nonfinite objective')
        return primary,auxiliary

    def objective(self,split,index,start,horizon):
        try:
            p,a=self.losses(self.values[split]['exported'][index:index+1,start:start+horizon+1])
        except (FloatingPointError,RuntimeError) as error:
            raise RuntimeError(f'{split} {self.meta["entries"][split][index]["id"]} start{start}: {error}') from error
        return p+self.weight*a if split=='train' else p

    def batch_loss(self,windows):
        frames=torch.stack([self.values['train']['exported'][i,s:s+self.cfg['horizon']+1] for i,s in windows])
        primary,auxiliary=self.losses(frames)
        (primary+self.weight*auxiliary).backward()
        if self.batch_counter%20==0:
            n=len(windows)*self.cfg['horizon']
            print(json.dumps(dict(primary_loss=float(primary.detach())/n,omega_scaled_mse=float(auxiliary.detach())/n,
                                  omega_weight=self.weight,total_loss=float((primary+self.weight*auxiliary).detach())/n)),flush=True)
        self.batch_counter+=1
        # Shared history remains a comparable primary-field metric, not an augmented loss mislabeled L2.
        return float(primary.detach())

    @torch.no_grad()
    def validation(self,horizon=None,limit=None):
        self.model.eval();horizon=horizon or self.cfg['train_frames']-1;rows=[]
        for i in self.meta['val_indices'][:limit]:
            p,a=self.losses(self.values['val']['exported'][i:i+1,:horizon+1])
            rows.append(dict(id=self.meta['entries']['val'][i]['id'],loss=float(p),omega_scaled_mse=float(a)/horizon if self.weight else None))
        return sum(row['loss'] for row in rows)/(len(rows)*horizon),rows


def aggregate(rows,horizons):
    result={}
    for h in horizons:
        eligible=[r for r in rows if r['valid_source_steps']>=h]
        complete=[r for r in eligible if str(h) in r['scores']]
        item=dict(eligible=len(eligible),completed=len(complete),failed=len(eligible)-len(complete),source_excluded=len(rows)-len(eligible),
                  mean_nrmse=None,mean_r_weighted_relative_l2=None,mean_derived_omega_r_weighted_relative_l2=None)
        if eligible and len(complete)==len(eligible):
            for metric in ('nrmse','r_weighted_relative_l2'):
                item['mean_'+metric]=torch.tensor([r['scores'][str(h)][metric] for r in complete]).mean(0).tolist()
            item['mean_derived_omega_r_weighted_relative_l2']=sum(r['scores'][str(h)]['derived_omega']['r_weighted_relative_l2'][0] for r in complete)/len(complete)
        result[str(h)]=item
    return result


@torch.no_grad()
def evaluate(ex,path,smoke=False,deadline=None):
    ex.model.eval();rows=[]
    identity=dict(experiment=ex.identity,checkpoint=initialization_hash(ex.model),smoke=smoke)
    if path.exists():
        old=json.loads(path.read_text())
        if old['identity']!=identity: raise ValueError('Changed evaluation identity')
        if old.get('complete'): return
        rows=old['trajectories']
    horizons=sorted({h for h in (10,ex.cfg['train_frames']-1,100,ex.cfg['eval_horizon']) if h<=ex.cfg['eval_horizon']})
    if smoke: horizons=[min(5,ex.cfg['eval_horizon'])]
    entries=ex.meta['entries']['test'][:1 if smoke else None]
    for index,entry in enumerate(entries):
        if index<len(rows): continue
        if deadline and time.monotonic()>=deadline:
            write_json(path,dict(identity=identity,complete=False,trajectories=rows));raise SystemExit(75)
        available=ex.meta['test_prefixes'][entry['id']];requested=min(max(horizons),available)
        row=dict(id=entry['id'],valid_source_steps=available,completed_steps=0,scores={},persistence={},failure=None,
                 nonpositive_predicted_cells=dict(rho=0,T=0))
        if requested<1: rows.append(row);continue
        source=load_release(entry,requested+1).double().to(ex.device);state=source[:1,:3]
        accum={name:{} for name in ('model','persistence','omega','omega_persistence','extrapolation','omega_extrapolation')}
        r=ex.model.r
        try:
            for step,truth in enumerate(source[1:],1):
                state=ex.model(state);prediction=state[0];omega=ex.model.omega(state)
                row['nonpositive_predicted_cells']['rho']+=int((prediction[0]<=0).sum())
                row['nonpositive_predicted_cells']['T']+=int((prediction[1]<=0).sum())
                groups=dict(model=frame_sums(prediction,truth[:3],ex.mean,ex.scale,r),
                            persistence=frame_sums(source[0,:3],truth[:3],ex.mean,ex.scale,r),
                            omega=frame_sums(omega,truth[3:4,CROP:-CROP,CROP:-CROP],ex.omega_mean,ex.omega_scale,r[CROP:-CROP]),
                            omega_persistence=frame_sums(source[0,3:4,CROP:-CROP,CROP:-CROP],truth[3:4,CROP:-CROP,CROP:-CROP],ex.omega_mean,ex.omega_scale,r[CROP:-CROP]))
                if not all(torch.isfinite(v).all() for g in groups.values() for v in g.values()): raise FloatingPointError('Nonfinite evaluation metric')
                for group,values in groups.items():
                    for key,value in values.items():accum[group][key]=accum[group].get(key,0)+value
                if step>=ex.cfg['train_frames']:
                    for group,target in (('model','extrapolation'),('omega','omega_extrapolation')):
                        for key,value in groups[group].items():accum[target][key]=accum[target].get(key,0)+value
                row['completed_steps']=step
                if step in horizons:
                    row['scores'][str(step)]=scores({k:v/step for k,v in accum['model'].items()})
                    row['scores'][str(step)]['derived_omega']=scores({k:v/step for k,v in accum['omega'].items()})
                    row['persistence'][str(step)]=scores({k:v/step for k,v in accum['persistence'].items()})
                    row['persistence'][str(step)]['omega']=scores({k:v/step for k,v in accum['omega_persistence'].items()})
                    if step>=ex.cfg['train_frames']:
                        count=step-ex.cfg['train_frames']+1
                        row['scores'][str(step)]['temporal_extrapolation']=scores({k:v/count for k,v in accum['extrapolation'].items()})
                        row['scores'][str(step)]['temporal_extrapolation']['derived_omega']=scores({k:v/count for k,v in accum['omega_extrapolation'].items()})
        except (FloatingPointError,RuntimeError) as error:row['failure']=str(error)
        rows.append(row);write_json(path,dict(identity=identity,complete=False,trajectories=rows))
        print(json.dumps(dict(test_id=entry['id'],completed=row['completed_steps'],failure=row['failure'])),flush=True)
    write_json(path,dict(identity=identity,complete=True,arm=ex.arm,fields=['rho','T','potential'],
                        derived_field='omega: interior only, never a predicted state',omega_crop=CROP,
                        normalization='shared training-only minmax; epsilon1e-6',
                        aggregation='per-trajectory space-time ratios; common-cohort mean null if any eligible rollout fails',
                        horizons=aggregate(rows,horizons),trajectories=rows,smoke=smoke))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--omega-weight',type=float,default=100.)
    p.add_argument('--opsplit-update',choices=('additive','softplus'),default='additive')
    p.add_argument('--device',default='cuda');p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--audit',action='store_true');p.add_argument('--audit-report',type=Path)
    p.add_argument('--stop-after-epoch',type=int);p.add_argument('--evaluate-only',action='store_true')
    p.add_argument('--max-seconds',type=float,default=0)
    a=p.parse_args();torch.set_num_threads(1)
    if a.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA allocation required')
    ex=ThreeFieldExperiment(a.data,a.arm,a.device,a.omega_weight,a.opsplit_update)
    if a.evaluate_only:
        saved=torch.load(a.output/'best.pt',map_location=a.device,weights_only=False)
        if saved['identity']!=ex.identity:raise ValueError('Evaluation checkpoint identity mismatch')
        ex.model.load_state_dict(saved['state_dict'])
        evaluate(ex,a.output/'test_metrics.json',deadline=time.monotonic()+a.max_seconds if a.max_seconds else None)
    elif a.audit:
        if a.output.exists() and not a.resume:raise FileExistsError(a.output)
        if not audit(ex,a.output,workers=1,resume=a.resume,max_seconds=a.max_seconds):
            result=json.loads(a.output.read_text());raise SystemExit(2 if result['completed'] else 75)
    else:
        if ex.composite and not a.smoke:
            if not a.audit_report:p.error('OpsSplit training requires its initialization audit')
            require_audit(ex,a.audit_report)
            if a.stop_after_epoch is None:
                if not a.resume:p.error('OpsSplit continuation must resume the checked optimizer state')
                require_composite_stability(ex,a.output)
        if a.smoke:ex.composite=True  # Shared smoke performs two full batches and a full validation prefix.
        train(ex,a.output,resume=a.resume,smoke=a.smoke,max_seconds=a.max_seconds,train_only=True,stop_after_epoch=a.stop_after_epoch)
        if a.smoke:
            evaluate(ex,a.output/'test_metrics.json',smoke=True)


if __name__=='__main__':main()
