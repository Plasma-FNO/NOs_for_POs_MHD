"""Native bicubic full-volume OpsSplit interface for the supported MAST-U pilot.

The five blocks are [psi,u,rho,pressure,Vpar]. Each has [strong,test_R,test_Z]
coefficients, integrated against 2*pi*R*dR*dZ. The latter two are negative
physical fluxes. Current/vorticity and native boundary equations are solver rows,
not outputs of the neural operator. The native executable advances its own state.
"""
from pathlib import Path
import hashlib
import json
import struct

import numpy as np
import torch
from torch import nn

from .learning import FieldSplineOperator

BLOCKS = ('psi', 'u', 'rho', 'pressure', 'Vpar')
ROWS = (0, 1, 4, 5, 6)


def native_jets(coefficients, geometry):
    """Evaluate coefficients and exact physical bicubic derivatives, with autograd."""
    if coefficients.ndim != 3 or coefficients.shape[1:] != (len(geometry.dof_ids), 7):
        raise ValueError('Expected [batch,canonical_native_dof,7] coefficients')
    if not torch.isfinite(coefficients).all():
        raise ValueError('Nonfinite native coefficients')
    dofs = torch.as_tensor(geometry.dof_map, device=coefficients.device)
    basis = torch.as_tensor(geometry.basis, dtype=coefficients.dtype, device=coefficients.device)
    gradient = torch.as_tensor(geometry.basis_grad, dtype=coefficients.dtype, device=coefficients.device)
    local = coefficients[:, dofs]
    return (torch.einsum('eql,belc->beqc', basis, local),
            torch.einsum('eqlr,belc->beqcr', gradient, local))



def normalize_current_fields(values, mean, scale):
    """Preserve small native perturbations by centering before the float32 cast."""
    mean = torch.as_tensor(mean, dtype=torch.float64, device=values.device)
    scale = torch.as_tensor(scale, dtype=torch.float64, device=values.device)
    return ((values.double() - mean) / scale).float()


def linear_terms(values, gradients, radius, frozen):
    """Fixed linear transport action; NumPy or Torch arrays, arbitrary leading axes."""
    if values.shape[-1] != 7 or gradients.shape != (*values.shape, 2):
        raise ValueError('Seven values and their physical gradients required')
    if frozen.shape != (*values.shape[:-1], 10) or radius.shape != values.shape[:-1]:
        raise ValueError('Frozen coefficient/coordinate shape mismatch')
    is_torch = isinstance(values, torch.Tensor)
    c = values.new_zeros((*values.shape[:-1], 5, 3)) if is_torch else np.zeros((*values.shape[:-1], 5, 3), dtype=values.dtype)
    c[..., 0, 0] = frozen[..., 0]*values[..., 2]/radius**2
    c[..., 1, 1:] = -frozen[..., 1, None]*gradients[..., 3, :]
    for row, field, offset in ((2, 4, 2), (3, 5, 5)):
        c[..., row, 1] = -frozen[..., offset]*gradients[..., field, 0]-frozen[..., offset+1]*gradients[..., field, 1]
        c[..., row, 2] = -frozen[..., offset+1]*gradients[..., field, 0]-frozen[..., offset+2]*gradients[..., field, 1]
    c[..., 4, 1] = -frozen[..., 8]*gradients[..., 6, 0]+frozen[..., 9]*values[..., 6]
    c[..., 4, 2] = -frozen[..., 8]*gradients[..., 6, 1]
    return c


def centered_linear_forcing(values, gradients, radius, frozen):
    """F(q*) + L(q-q*); the frozen baseline/tangent record is immutable metadata."""
    if frozen.shape[:-1] != values.shape[:-1] or frozen.shape[-1] not in (46,361):
        raise ValueError('Expected frozen transport or full baseline tangent channels')
    if isinstance(values, torch.Tensor):
        reference_gradients = torch.stack((frozen[...,17:24],frozen[...,24:31]),dim=-1)
    else:
        reference_gradients = np.stack((frozen[...,17:24],frozen[...,24:31]),axis=-1)
    dv=values-frozen[...,10:17];dg=gradients-reference_gradients
    if frozen.shape[-1]==361:
        if isinstance(values,torch.Tensor):
            jets=torch.cat((dv,dg[...,0],dg[...,1]),dim=-1)
            action=torch.einsum('...jo,...j->...o',frozen[...,46:].reshape(*values.shape[:-1],21,15),jets)
        else:
            jets=np.concatenate((dv,dg[...,0],dg[...,1]),axis=-1)
            action=np.einsum('...jo,...j->...o',frozen[...,46:].reshape(*values.shape[:-1],21,15),jets)
        action=action.reshape(*values.shape[:-1],5,3)
    else:
        action=linear_terms(dv,dg,radius,frozen[...,:10])
    return action+frozen[...,31:46].reshape(*values.shape[:-1],5,3)


def assemble_terms(terms, geometry):
    """Differentiable native weak assembly, including shared derivative DOFs."""
    if terms.ndim != 5 or terms.shape[1:] != (*geometry.basis.shape[:2], 5, 3):
        raise ValueError('Expected [batch,element,quadrature,5,3] weak channels')
    basis = torch.as_tensor(geometry.basis, dtype=terms.dtype, device=terms.device)
    gradient = torch.as_tensor(geometry.basis_grad, dtype=terms.dtype, device=terms.device)
    weights = torch.as_tensor(geometry.volume_weights, dtype=terms.dtype, device=terms.device)
    local = (torch.einsum('beqc,eql,eq->belc', terms[..., 0], basis, weights)
             + torch.einsum('beqcr,eqlr,eq->belc', terms[..., 1:], gradient, weights))
    assembled = terms.new_zeros((terms.shape[0], len(geometry.dof_ids), 5))
    assembled.index_add_(1, torch.as_tensor(geometry.dof_map.ravel(),device=terms.device),local.flatten(1,2))
    return assembled


class FullOpsSplitOperators(nn.Module):
    """Five independent NOs predict the complete centered residual F(q)-F(q*)-L*(q-q*) in weak form."""
    def __init__(self, bounds, model_config):
        super().__init__()
        mask=torch.ones(5,3);mask[0,1:]=0
        self.register_buffer('channel_mask',mask,persistent=False)
        self.blocks = nn.ModuleDict({name: FieldSplineOperator(7, 3, bounds, model_config,
                                   condition_transport=True) for name in BLOCKS})

    def forward(self, fields, source_coords, parameters, query_coords=None):
        return torch.stack([self.blocks[name](fields,source_coords,parameters,query_coords)
                            for name in BLOCKS], dim=-2)*self.channel_mask


def read_linear(path):
    path=Path(path)
    with path.open('rb') as f:
        header=f.read(12)
        if len(header)!=12: raise ValueError('Truncated linear header')
        magic, ne, nq=struct.unpack('<3i',header)
        if magic not in (303046,303361) or ne<1 or nq!=16: raise ValueError('Invalid frozen linear header')
        channels=361 if magic==303361 else 46
        expected=12+ne*4*4+ne*nq*channels*8
        if path.stat().st_size!=expected: raise ValueError('Invalid frozen operator size')
        vertices=np.fromfile(f,dtype='<i4',count=ne*4).reshape(ne,4)
        frozen=np.fromfile(f,dtype='<f8').reshape(ne,nq,channels)
    if (vertices<1).any() or len(np.unique(vertices,axis=0))!=ne or not np.isfinite(frozen).all():
        raise ValueError('Invalid native element identities or frozen coefficients')
    return vertices,frozen


def write_prediction(path, step, prediction, vertices, frozen_path):
    """Atomic, finite, dimension- and geometry-checked native inference handoff."""
    reference,_=read_linear(frozen_path)
    if not np.array_equal(vertices,reference): raise ValueError('Neural forcing geometry mismatch')
    prediction=np.asarray(prediction,dtype='<f8')
    if prediction.shape!=(len(vertices),16,5,3) or not np.isfinite(prediction).all():
        raise ValueError('Invalid full nonlinear prediction')
    if type(step) is not int or not 1<=step<2**31: raise ValueError('Invalid native step stamp')
    path=Path(path); temporary=path.with_suffix('.tmp')
    with temporary.open('xb') as f:
        f.write(struct.pack('<4i',303015,step,len(vertices),16)); prediction.tofile(f)
    temporary.replace(path)


def freeze_export(export_files, restart, output, *, tangent=False):
    """Freeze common-baseline transport and preserve native pointwise oracle labels."""
    import h5py
    from .restart import read_restart
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    geometry,state,metadata=read_restart(restart)
    with h5py.File(restart) as f: vertices=f['vertex'][()].T.astype('<i4')
    lookup={tuple(v):i for i,v in enumerate(vertices)}
    width=364 if tangent else 49
    record=np.dtype([('vertices','<i4',(4,)),('query','<i4'),('data','<f8',(width,))])
    data=np.full((len(vertices),16,width),np.nan)
    for path in export_files:
        if Path(path).stat().st_size%record.itemsize: raise ValueError('Truncated native export')
        for row in np.fromfile(path,dtype=record):
            e=lookup[tuple(row['vertices'])]; q=int(row['query'])-1
            if not 0<=q<16: raise ValueError('Invalid quadrature index')
            if np.isfinite(data[e,q]).all() and not np.array_equal(data[e,q],row['data']):
                raise ValueError('Conflicting duplicate native export records')
            data[e,q]=row['data']
    if not np.isfinite(data).all(): raise ValueError('Missing or nonfinite native quadrature data')
    if not np.allclose(data[...,:2],geometry.quadrature_rz,rtol=0,atol=1e-12):
        raise ValueError('Native export and spline query ordering differ')
    from .numerical import evaluate_native,evaluate_native_grad
    if not np.allclose(data[...,3:10],evaluate_native(state,geometry),rtol=1e-10,atol=1e-13):
        raise ValueError('Native export is not from the specified baseline state')
    gradients=np.stack((data[...,10:17],data[...,17:24]),axis=-1)
    if not np.allclose(gradients,evaluate_native_grad(state,geometry),rtol=1e-8,atol=1e-10):
        raise ValueError('Native gradient mismatch')
    frozen=data[...,24:34].copy()
    with (output/'opsplit_linear.bin').open('xb') as f:
        f.write(struct.pack('<3i',303361 if tangent else 303046,len(vertices),16));vertices.tofile(f)
        np.concatenate((frozen,data[...,3:24],data[...,34:49],data[...,49:]),axis=-1).astype('<f8').tofile(f)
    physical=data[...,34:49].reshape(len(vertices),16,5,3)
    linear=linear_terms(data[...,3:10],gradients,data[...,0],frozen)
    if tangent:
        linear=np.einsum('eqjo,eqj->eqo',data[...,49:].reshape(len(vertices),16,21,15),data[...,3:24]).reshape(len(vertices),16,5,3)
    with h5py.File(output/'baseline_terms.h5','x') as f:
        f['vertices']=vertices;f['coordinates']=data[...,:2];f['values']=data[...,3:10]
        f['gradients']=gradients;f['frozen_transport']=frozen
        f['physical']=physical;f['uncentered_linear']=linear
        f['baseline_forcing']=physical;f['nonlinear']=np.zeros_like(physical)
        if tangent:f['fixed_coefficient_tangent']=data[...,49:].reshape(len(vertices),16,21,15)
        f.attrs['channels']='[psi,u,rho,pressure,Vpar] x [strong,test_R,test_Z]'
        f.attrs['measure']='2*pi*R*dR*dZ; gradient channels are negative physical flux'
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest=dict(baseline=str(Path(restart).resolve()),baseline_sha256=sha(restart),
                  linear_sha256=sha(output/'opsplit_linear.bin'),metadata=metadata,linearization='fixed transport/source full spatial tangent' if tangent else 'transport only',
                  exports={str(p):sha(p) for p in export_files},training_ready=False)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest
