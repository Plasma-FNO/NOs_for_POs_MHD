"""Validation-only, guard-aware ablation of a completed Euler checkpoint.

Removing L from an already trained N tests dependence, not retrained causality.
Derivative comparison measures the coarse learned scatter/spline approximation
against exact native bicubic derivatives of the same input field.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mastu_opsplit.euler import NativeSplineSpace,EulerSplineModel,field_loss
from mastu_opsplit.euler_training import load_pilot,physical_linear_from_reference,roll,write,FIELDS
from mastu_opsplit.physical_guard import PhysicalGuard
from mastu_opsplit.restart import read_restart


def diagnose(pilot, output):
    pilot=Path(pilot);p=json.loads((pilot/'provenance.json').read_text())
    torch.set_num_threads(2)
    g,data,times,verified=load_pilot(p['config'])
    if verified['files']!=p['files']:raise ValueError('Reference provenance changed')
    space=NativeSplineSpace(g);norm=np.load(pilot/'normalization.npz')
    mean,scale=norm['mean'],norm['scale'];q=torch.from_numpy((data-mean)/scale)
    q8,_,_=read_restart(p['files'][0][0]['path'],quadrature_order=8)
    guard=PhysicalGuard(g,q8,mean,scale)
    m=EulerSplineModel('opsplit',space,p['model_config'],linear_branch=p['linear_branch'],
                      physical_guard=guard,linear_initialization=p['linear_initialization'],
                      physical_linear=physical_linear_from_reference(p['config'],g,mean,scale,times,verified))
    m.load_state_dict(torch.load(pilot/'opsplit/best.pt',weights_only=False,map_location='cpu')['model']);m.eval()
    linear=m.linear_rhs;result={'pilot':str(pilot),'window':[24,32],'test_used':False,'branches':[]}
    def rms(x):
        v=space.sample(x).numpy()
        return np.sqrt(np.einsum('bpc,p->c',v*v,space.weights/space.weights.sum())/len(v))
    with torch.no_grad():
        for sequence in q[:,24:32]:
            scores={}
            for factor in (1.,0.,-1.):
                guard.reset_statistics()
                m.linear_rhs=lambda state,values=None:factor*linear(state,values)
                pred=roll(m,sequence,torch.ones(1,1))
                scores[str(factor)]={'mse':float(field_loss(pred[1:],sequence[1:],space)),
                    'per_field_mse':dict(zip(FIELDS,(rms(pred[1:]-sequence[1:])**2).tolist())),
                    'guard':dict(guard.statistics)}
                guard.reset_statistics()
            m.linear_rhs=linear
            state=sequence[:1];values=space.sample(state)
            neural=space.project(m.operator(values.float(),m.coordinates,torch.ones(1,1)).double())
            l=linear(state,values)
            guarded_delta=guard(state,state+neural+l)-guard(state,state+neural)
            raw_grad=np.einsum('eqlr,elc->eqcr',g.basis_grad,state[0].numpy()[g.dof_map]).reshape(-1,7,2)
            source=m.operator.normalized_coordinates(m.coordinates,1).to(values.float())
            coeff=m.operator.backbone._encode(values.float().transpose(1,2),source)
            approx=torch.stack((m.derivatives(coeff,(1,0)),m.derivatives(coeff,(0,1))),-1)[0].numpy()
            w=space.weights/space.weights.sum()
            gradient_error=np.sqrt(np.einsum('pcr,p->c',(approx-raw_grad)**2,w)/np.maximum(np.einsum('pcr,p->c',raw_grad**2,w),1e-30))
            result['branches'].append({'ablation':scores,
                'linear_rms':dict(zip(FIELDS,rms(l).tolist())),
                'guarded_linear_effect_rms':dict(zip(FIELDS,rms(guarded_delta).tolist())),
                'lspline_vs_native_gradient_relative_rms':dict(zip(FIELDS,gradient_error.tolist()))})
    write(output,result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();diagnose(args.pilot,args.output)
