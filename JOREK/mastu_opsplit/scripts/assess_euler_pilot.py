"""Plot completed pilot metrics and audit its native linear/constraint contributions."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mastu_opsplit.euler import NativeSplineSpace, EulerSplineModel
from mastu_opsplit.restart import read_restart
from mastu_opsplit.euler_training import FIELDS, write, load_pilot, physical_linear_from_reference


def constraints(coefficients, geometry):
    local=coefficients[geometry.dof_map]
    q=np.einsum('eql,elc->eqc',geometry.basis,local)
    g=np.einsum('eqlr,elc->eqcr',geometry.basis_grad,local)
    active=geometry.boundary_labels==0
    result={}
    for name,field,auxiliary in (('current',0,2),('vorticity',1,3)):
        w=geometry.volume_weights
        if field==0:w=w/geometry.quadrature_rz[...,0]**2
        parts=[np.einsum('eql,eq,eq->el',geometry.basis,q[...,auxiliary],w),
               np.einsum('eqlr,eqr,eq->el',geometry.basis_grad,g[...,field,:],w)]
        assembled=[]
        for part in parts:
            a=np.zeros(len(geometry.dof_ids));np.add.at(a,geometry.dof_map.ravel(),part.ravel())
            assembled.append(a[active])
        result[name]=float(np.linalg.norm(assembled[0]+assembled[1])/
                           max(np.linalg.norm(assembled[0])+np.linalg.norm(assembled[1]),1e-30))
    return result


def assess(output):
    output=Path(output)
    results=json.loads((output/'results.json').read_text())['results']
    provenance=json.loads((output/'provenance.json').read_text())
    # Revalidate the pinned native binary/completion provenance as well as data hashes.
    geometry,data,times,verified=load_pilot(provenance['config'])
    if verified['files']!=provenance['files']:raise ValueError('Reference data changed after training')
    audit={'reference_provenance_rechecked':True,'independent_trajectory_test':False,'constraints':{}}
    for family in results:
        audit['constraints'][family]=[]
        for branch in range(len(data)):
            pred=np.load(output/family/f'last_prediction_{branch}.npy')
            audit['constraints'][family].append({'prediction':constraints(pred,geometry),
                                                 'reference':constraints(data[branch,-1],geometry)})
    space=NativeSplineSpace(geometry)
    normalizer=np.load(output/'normalization.npz');mean,scale=normalizer['mean'],normalizer['scale']
    torch.set_num_threads(2)
    guard=None
    if provenance['config'].get('physical_guard',False):
        from mastu_opsplit.physical_guard import PhysicalGuard
        q8,_,_=read_restart(provenance['files'][0][0]['path'],quadrature_order=8)
        guard=PhysicalGuard(geometry,q8,mean,scale,positivity_basis=provenance['config'].get('positivity_basis','q8'))
    model=EulerSplineModel('opsplit',space,provenance['model_config'],
                           linear_branch=provenance.get('linear_branch','native_bicubic'),
                           physical_guard=guard,
                           linear_initialization=provenance.get('linear_initialization','positive'),
                           physical_linear=physical_linear_from_reference(provenance['config'],geometry,mean,scale,times,verified))
    model.load_state_dict(torch.load(output/'opsplit/best.pt',weights_only=False,map_location='cpu')['model'])
    model.eval();ratios=[]
    with torch.no_grad():
        for branch in range(len(data)):
            state=torch.from_numpy(((data[branch,24]-mean)/scale)[None])
            neural=space.project(model.operator(space.sample(state).float(),model.coordinates,torch.ones(1,1)).double())
            linear=model.linear_rhs(state)
            def rms(x):
                values=space.sample(x)[0].numpy()
                return np.sqrt((values**2*space.weights[:,None]).sum(0)/space.weights.sum())
            ratios.append(dict(zip(FIELDS,(rms(linear)/np.maximum(rms(neural),1e-30)).tolist())))
    audit['linear_over_neural_rms_at_validation_start']=ratios
    if guard is not None:
        # Independent denser sampling, including vertices: not the limiter's Q8 grid.
        q12,_,_=read_restart(provenance['files'][0][0]['path'],quadrature_order=12)
        vertices=np.unique(geometry.dof_map[:,::4])
        audit['independent_positivity_Q12_and_vertices']={}
        for family in results:
            audit['independent_positivity_Q12_and_vertices'][family]=[]
            for branch in range(len(data)):
                minimum=np.full(2,np.inf)
                for state in np.load(output/family/f'prediction_{branch}.npy',mmap_mode='r')[1:]:
                    values=np.einsum('eql,elc->eqc',q12.basis,state[q12.dof_map][...,4:6])
                    minimum=np.minimum(minimum,values.min(axis=(0,1)))
                    minimum=np.minimum(minimum,state[vertices,4:6].min(axis=0))
                audit['independent_positivity_Q12_and_vertices'][family].append(
                    {'minimum_density_temperature':minimum.tolist(),'positive':bool((minimum>0).all())})
    write(output/'audit.json',audit)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    x=np.arange(7)
    for branch,ax in enumerate(axes):
        for i,family in enumerate(('ar','node','opsplit')):
            values=results[family]['test_by_branch'][branch]['dynamic_relative_rms']
            ax.bar(x+(i-1)*.23,[values[f] for f in FIELDS],width=.23,label=family.upper())
        ax.set_xticks(x,FIELDS);ax.set_yscale('log');ax.set_title(('Perturbation','Paired control')[branch])
        ax.set_ylabel('Error / held-out temporal standard deviation');ax.grid(axis='y',alpha=.2)
    axes[0].legend();fig.suptitle(f"MAST-U Euler pilot: same-parent temporal holdout, {provenance['config']['steps']} updates/model")
    fig.savefig(output/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    assess(p.parse_args().output)
