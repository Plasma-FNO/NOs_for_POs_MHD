"""cuDSS 0.7.1 FP64 sparse LU with GPU implicit adjoints.

The pinned runtime is installed separately from the shared Python environment.
NVIDIA API: https://docs.nvidia.com/cuda/cudss/0.7.1/functions.html
Only symbolic analysis/reordering may use the host. Matrix values, RHS, solutions,
numeric factorization and adjoint solves remain on the GPU. No CPU fallback.
cuDSS 0.7.1 has no transpose solve: the adjoint explicitly factorizes A.T.
"""
import ctypes as C
from functools import lru_cache
import os
from pathlib import Path

import torch

DEFAULT_LIBRARY='/pitagora_work/FUPB2_UKAEA_ML/vgopakum/software/jorek-cudss-0.7.1.6/nvidia/cu12/lib/libcudss.so.0'


@lru_cache(maxsize=1)
def library():
    path=os.environ.get('JOREK_CUDSS_LIBRARY',DEFAULT_LIBRARY)
    if not Path(path).is_file():raise RuntimeError('Install nvidia-cudss-cu12==0.7.1.6 or set JOREK_CUDSS_LIBRARY')
    lib=C.CDLL(path)
    ptr=C.c_void_p;out=C.POINTER(ptr);integer=C.c_int;size=C.c_size_t;i64=C.c_int64
    signatures={
        'cudssCreate':[out],'cudssDestroy':[ptr],'cudssSetStream':[ptr,ptr],
        'cudssConfigCreate':[out],'cudssConfigDestroy':[ptr],
        'cudssConfigSet':[ptr,integer,ptr,size],
        'cudssDataCreate':[ptr,out],'cudssDataDestroy':[ptr,ptr],
        'cudssDataGet':[ptr,ptr,integer,ptr,size,C.POINTER(size)],
        'cudssMatrixCreateCsr':[out,i64,i64,i64,ptr,ptr,ptr,ptr,integer,integer,integer,integer,integer],
        'cudssMatrixCreateDn':[out,i64,i64,i64,ptr,integer,integer],
        'cudssMatrixSetValues':[ptr,ptr],'cudssMatrixDestroy':[ptr],
        'cudssExecute':[ptr,integer,ptr,ptr,ptr,ptr,ptr],
        'cudssGetProperty':[integer,C.POINTER(integer)],
    }
    for name,args in signatures.items():
        getattr(lib,name).argtypes=args;getattr(lib,name).restype=integer
    version=[]
    for key in range(3):
        value=integer();check(lib.cudssGetProperty(key,C.byref(value)),'version');version.append(value.value)
    if version!=[0,7,1]:raise RuntimeError(f'cuDSS ABI expects 0.7.1, found {version}')
    return lib


def check(status,operation):
    if status:raise RuntimeError(f'cuDSS {operation} failed with status {status}')


class _System:
    def __init__(self,matrix):
        self.lib=library();self.device=matrix.device;self.stream=torch.cuda.current_stream(self.device)
        self.handle=C.c_void_p();self.config=C.c_void_p();self.data=C.c_void_p()
        self.a=C.c_void_p();self.x=C.c_void_p();self.b=C.c_void_p()
        self.rows=matrix.crow_indices().to(torch.int32);self.cols=matrix.col_indices().to(torch.int32)
        self.n=matrix.shape[0];self.ready=False
        try:
            self.call('cudssCreate',C.byref(self.handle))
            self.call('cudssSetStream',self.handle,C.c_void_p(self.stream.cuda_stream))
            self.call('cudssConfigCreate',C.byref(self.config))
            deterministic=C.c_int(1)
            self.call('cudssConfigSet',self.config,25,C.byref(deterministic),C.sizeof(deterministic))
            self.call('cudssDataCreate',self.handle,C.byref(self.data))
            # CUDA_R_32I=10, CUDA_R_64F=1; general/full/zero-based CSR.
            self.call('cudssMatrixCreateCsr',C.byref(self.a),self.n,self.n,len(self.cols),
                      self.rows.data_ptr(),None,self.cols.data_ptr(),matrix.values().data_ptr(),10,1,0,0,0)
        except Exception:
            self.close();raise

    def call(self,name,*args):check(getattr(self.lib,name)(*args),name)

    def matches(self,matrix):
        return torch.equal(self.rows,matrix.crow_indices()) and torch.equal(self.cols,matrix.col_indices())

    def solve(self,matrix,rhs,factorize=True):
        # cuDSS 0.7.1 deterministic kernels support one RHS only. Reuse the
        # numeric factorization across columns, retaining per-column residuals.
        if rhs.ndim==2:
            return torch.stack([self.solve(matrix,column,factorize=i==0)
                                for i,column in enumerate(rhs.unbind(1))],1)
        self.values=matrix.values().detach();self.rhs=rhs.contiguous();self.output=torch.empty_like(self.rhs)
        self.call('cudssMatrixSetValues',self.a,self.values.data_ptr())
        if not self.x.value:
            self.call('cudssMatrixCreateDn',C.byref(self.x),self.n,1,self.n,self.output.data_ptr(),1,0)
            self.call('cudssMatrixCreateDn',C.byref(self.b),self.n,1,self.n,self.rhs.data_ptr(),1,0)
        else:
            self.call('cudssMatrixSetValues',self.x,self.output.data_ptr())
            self.call('cudssMatrixSetValues',self.b,self.rhs.data_ptr())
        if not self.ready:
            self.call('cudssExecute',self.handle,3,self.config,self.data,self.a,self.x,self.b)
            self.ready=True
        if factorize:self.call('cudssExecute',self.handle,4,self.config,self.data,self.a,self.x,self.b)
        self.call('cudssExecute',self.handle,1008,self.config,self.data,self.a,self.x,self.b)
        info=C.c_int();written=C.c_size_t()
        self.call('cudssDataGet',self.handle,self.data,0,C.byref(info),C.sizeof(info),C.byref(written))
        self.stream.synchronize()
        if info.value:raise RuntimeError(f'cuDSS device error {info.value}')
        residual=torch.sparse.mm(matrix,self.output[:,None])[:,0]-self.rhs
        if not torch.isfinite(self.output).all() or residual.norm()>1e-8*self.rhs.norm().clamp_min(1e-30):
            raise RuntimeError(f'GPU sparse solve failed true residual check: relative={float(residual.norm()/self.rhs.norm().clamp_min(1e-30)):.3e}, rhs_norm={float(self.rhs.norm()):.3e}')
        return self.output

    def close(self):
        # Cached systems are only reused on their creation device/stream.
        if getattr(self,'handle',None) is None:return
        with torch.cuda.device(self.device):
            self.stream.synchronize()
            for name in ('a','x','b'):
                ptr=getattr(self,name)
                if ptr.value:self.lib.cudssMatrixDestroy(ptr);ptr.value=None
            if self.data.value:self.lib.cudssDataDestroy(self.handle,self.data);self.data.value=None
            if self.config.value:self.lib.cudssConfigDestroy(self.config);self.config.value=None
            if self.handle.value:self.lib.cudssDestroy(self.handle);self.handle.value=None


_SYSTEMS=[]


def clear_cache():
    for system in _SYSTEMS:system.close()
    _SYSTEMS.clear()


def direct_solve(matrix,rhs):
    if not rhs.is_cuda or rhs.dtype!=torch.float64:raise ValueError('cuDSS requires CUDA float64 inputs')
    matrix=matrix.detach().to_sparse_csr()
    stream=torch.cuda.current_stream(rhs.device)
    for system in _SYSTEMS:
        if (system.device==rhs.device and system.stream.cuda_stream==stream.cuda_stream and
                system.n==len(rhs) and len(system.cols)==matrix._nnz() and system.matches(matrix)):
            break
    else:
        if len(_SYSTEMS)>=16:_SYSTEMS.pop(0).close()
        system=_System(matrix);_SYSTEMS.append(system)
    try:return system.solve(matrix,rhs)
    except Exception:
        _SYSTEMS.remove(system);system.close();raise


class GPUSparseSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx,values,rhs,indices):
        matrix=torch.sparse_coo_tensor(indices,values,(len(rhs),len(rhs)),is_coalesced=True)
        result=direct_solve(matrix,rhs)
        ctx.save_for_backward(values,indices,result)
        return result

    @staticmethod
    def backward(ctx,gradient):
        values,indices,result=ctx.saved_tensors
        transpose=torch.sparse_coo_tensor(indices.flip(0),values,(len(result),len(result))).coalesce()
        adjoint=direct_solve(transpose,gradient.contiguous())
        values_grad=-adjoint[indices[0]]*result[indices[1]]
        return values_grad.sum(-1) if result.ndim==2 else values_grad,adjoint,None
