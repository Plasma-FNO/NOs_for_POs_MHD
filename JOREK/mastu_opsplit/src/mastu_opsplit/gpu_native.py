"""Device (CUDA) kernels for the fixed native linear maps and sparse direct solves.

Every product is deterministic by construction: a fixed sparse matrix is stored as row-length
buckets of padded rows (sliced ELL), and each bucket is one dense gather plus an in-order row
sum written to distinct rows; there are no atomics. cuSPARSE CSR products are avoided because
they are not bitwise repeatable on H100 (probe 2026-11-09). Sparse LU uses NVIDIA cuDSS 0.7.1
in deterministic mode through ctypes. 0.7.1 has no transpose solve (CUDSS_CONFIG_SOLVE_MODE is
"only off"), so transposed systems are factorised explicitly, lazily, when an adjoint is needed.
configure_cuda_numerics() sets the cuBLAS workspace for determinism and disables TF32.
"""
import ctypes as C
from functools import lru_cache
import os
from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
import torch

CUDSS_LIBRARY = '/pitagora_work/FUPB2_UKAEA_ML/vgopakum/software/jorek-cudss-0.7.1.6/nvidia/cu12/lib/libcudss.so.0'
ANALYSIS, FACTORIZATION, SOLVE = 3, 4, 1008               # cudssPhase_t
CONFIG_DETERMINISTIC_MODE, DATA_INFO, DATA_LU_NNZ = 25, 0, 1
R_64F, R_32I = 1, 10                                     # cudaDataType
MAX_TEMPORARY = 1 << 25                                  # elements per gather chunk (256 MB in float64)


def canonical_device(device):
    """torch.device with an explicit CUDA index ('cuda' -> 'cuda:<current>'), so devices compare equal."""
    device = torch.device(device)
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    return device


def configure_cuda_numerics():
    """Reproducible full-float32 CUDA numerics; call before CUDA starts.

    cuBLAS needs a fixed workspace under torch's deterministic mode. TF32 is disabled: cuDNN enables it by
    default, which put the remainder network 2e-4 (outputs) / 6e-4 (gradients) away from the CPU float32
    reference; with it off the devices agree to 1e-7 / 2e-6.
    """
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


class DeviceCSR:
    """Fixed sparse matrix on a device; deterministic y = M x for x of shape [ncols] or [ncols, k]."""
    def __init__(self, matrix, device, dtype=torch.float64):
        csr = sp.csr_matrix(matrix, dtype=np.float64)
        csr.sum_duplicates()
        self.shape, self.nnz, self.device, self.dtype = csr.shape, int(csr.nnz), canonical_device(device), dtype
        lengths = np.diff(csr.indptr)
        widths = np.where(lengths > 0, 2**np.ceil(np.log2(np.maximum(lengths, 1))).astype(np.int64), 0)
        index = torch.int32 if csr.shape[1] < 2**31 else torch.int64
        self.buckets = []
        for width in np.unique(widths[widths > 0]):
            rows = np.flatnonzero(widths == width)
            valid = np.arange(width)[None] < lengths[rows][:, None]
            offsets = np.where(valid, csr.indptr[rows][:, None] + np.arange(width)[None], 0)
            cols = np.where(valid, csr.indices[offsets], 0)
            vals = np.where(valid, csr.data[offsets], 0.)
            self.buckets.append((torch.as_tensor(rows, device=self.device),
                                 torch.as_tensor(cols, dtype=index, device=self.device),
                                 torch.as_tensor(vals, dtype=dtype, device=self.device)))
        self.stored = int(sum(cols.numel() for _, cols, _ in self.buckets))

    def __matmul__(self, x):
        if x.shape[0] != self.shape[1] or x.device != self.device:
            raise ValueError(f'Operand {tuple(x.shape)} on {x.device} does not match {self.shape} on {self.device}')
        x = x.to(self.dtype)
        trailing = tuple(x.shape[1:])
        columns = int(np.prod(trailing)) if trailing else 1
        out = x.new_zeros((self.shape[0],) + trailing)
        for rows, cols, vals in self.buckets:
            step = max(1, MAX_TEMPORARY//(cols.shape[1]*columns))
            for start in range(0, len(rows), step):
                c, v = cols[start:start+step], vals[start:start+step]
                gathered = x.index_select(0, c.reshape(-1)).reshape(c.shape + trailing)
                out[rows[start:start+step]] = (v.reshape(v.shape + (1,)*len(trailing))*gathered).sum(1)
        return out


def device_matrix(owner, name, build, device, transpose=False):
    """DeviceCSR of build() (or its transpose), cached on `owner` per device."""
    cache = owner.__dict__.setdefault('_device_matrices', {})
    key = (name, str(canonical_device(device)), transpose)
    if key not in cache:
        matrix = build()
        cache[key] = DeviceCSR(matrix.T if transpose else matrix, device)
    return cache[key]


def device_tensor(owner, name, build, device, dtype=None):
    """Constant tensor build() on `device`, cached on `owner`."""
    cache = owner.__dict__.setdefault('_device_tensors', {})
    key = (name, str(canonical_device(device)))
    if key not in cache:
        cache[key] = torch.as_tensor(np.asarray(build()), dtype=dtype, device=device)
    return cache[key]


def apply_batched(matrix, x):
    """[B, ncols, C] -> [B, nrows, C] with a DeviceCSR, in float64."""
    batch, n, channels = x.shape
    flat = x.to(torch.float64).transpose(0, 1).reshape(n, batch*channels)
    return (matrix @ flat).reshape(-1, batch, channels).transpose(0, 1)


@lru_cache(maxsize=1)
def cudss():
    path = os.environ.get('JOREK_CUDSS_LIBRARY', CUDSS_LIBRARY)
    if not Path(path).is_file():
        raise RuntimeError('cuDSS 0.7.1 not found: install nvidia-cudss-cu12==0.7.1.6 or set JOREK_CUDSS_LIBRARY')
    lib = C.CDLL(path)
    ptr, out, integer, size, i64 = C.c_void_p, C.POINTER(C.c_void_p), C.c_int, C.c_size_t, C.c_int64
    signatures = {
        'cudssCreate': [out], 'cudssDestroy': [ptr], 'cudssSetStream': [ptr, ptr],
        'cudssConfigCreate': [out], 'cudssConfigDestroy': [ptr], 'cudssConfigSet': [ptr, integer, ptr, size],
        'cudssDataCreate': [ptr, out], 'cudssDataDestroy': [ptr, ptr],
        'cudssDataGet': [ptr, ptr, integer, ptr, size, C.POINTER(size)],
        'cudssMatrixCreateCsr': [out, i64, i64, i64, ptr, ptr, ptr, ptr, integer, integer, integer, integer, integer],
        'cudssMatrixCreateDn': [out, i64, i64, i64, ptr, integer, integer], 'cudssMatrixDestroy': [ptr],
        'cudssExecute': [ptr, integer, ptr, ptr, ptr, ptr, ptr], 'cudssGetProperty': [integer, C.POINTER(integer)],
    }
    for name, arguments in signatures.items():
        getattr(lib, name).argtypes = arguments
        getattr(lib, name).restype = integer
    version = []
    for key in range(3):
        value = integer()
        _check(lib.cudssGetProperty(key, C.byref(value)), 'cudssGetProperty')
        version.append(value.value)
    if version != [0, 7, 1]:
        raise RuntimeError(f'cuDSS ABI expects 0.7.1, found {version}')
    return lib


def _check(status, name):
    if status:
        raise RuntimeError(f'cuDSS {name} failed with status {status}')


class CudssSolver:
    """Deterministic cuDSS LU of one fixed square sparse matrix on a CUDA device (column-by-column solves)."""
    supports_transpose = False

    def __init__(self, matrix, device):
        started = time.monotonic()
        csr = sp.csr_matrix(matrix, dtype=np.float64)
        csr.sum_duplicates()
        csr.sort_indices()
        if csr.shape[0] != csr.shape[1]:
            raise ValueError('cuDSS LU needs a square matrix')
        self.lib, self.n, self.device = cudss(), csr.shape[0], canonical_device(device)
        self.handle, self.config, self.data = C.c_void_p(), C.c_void_p(), C.c_void_p()
        self.a, self.x, self.b = C.c_void_p(), C.c_void_p(), C.c_void_p()
        with torch.cuda.device(self.device):
            self.stream = torch.cuda.current_stream(self.device)
            self.rows = torch.as_tensor(csr.indptr.astype(np.int32), device=self.device)
            self.cols = torch.as_tensor(csr.indices.astype(np.int32), device=self.device)
            self.values = torch.as_tensor(csr.data, device=self.device)
            self.rhs = torch.zeros(self.n, dtype=torch.float64, device=self.device)
            self.out = torch.zeros(self.n, dtype=torch.float64, device=self.device)
            try:
                self._call('cudssCreate', C.byref(self.handle))
                self._call('cudssSetStream', self.handle, C.c_void_p(self.stream.cuda_stream))
                self._call('cudssConfigCreate', C.byref(self.config))
                flag = C.c_int(1)
                self._call('cudssConfigSet', self.config, CONFIG_DETERMINISTIC_MODE, C.byref(flag), C.sizeof(flag))
                self._call('cudssDataCreate', self.handle, C.byref(self.data))
                self._call('cudssMatrixCreateCsr', C.byref(self.a), self.n, self.n, len(self.cols), self.rows.data_ptr(),
                           None, self.cols.data_ptr(), self.values.data_ptr(), R_32I, R_64F, 0, 0, 0)
                self._call('cudssMatrixCreateDn', C.byref(self.x), self.n, 1, self.n, self.out.data_ptr(), R_64F, 0)
                self._call('cudssMatrixCreateDn', C.byref(self.b), self.n, 1, self.n, self.rhs.data_ptr(), R_64F, 0)
                self._execute(ANALYSIS)
                self._execute(FACTORIZATION)
                info = self._get(DATA_INFO, C.c_int)
                if info:
                    raise RuntimeError(f'cuDSS factorization info {info}')
                self.fill = int(self._get(DATA_LU_NNZ, C.c_int64))
            except Exception:
                self.close()
                raise
        self.seconds = time.monotonic()-started

    def _call(self, name, *arguments):
        _check(getattr(self.lib, name)(*arguments), name)

    def _execute(self, phase):
        self._call('cudssExecute', self.handle, phase, self.config, self.data, self.a, self.x, self.b)

    def _get(self, key, kind):
        value, written = kind(), C.c_size_t()
        self._call('cudssDataGet', self.handle, self.data, key, C.byref(value), C.sizeof(value), C.byref(written))
        self.stream.synchronize()
        return value.value

    def solve(self, rhs, transpose=False):
        if transpose:
            raise ValueError('cuDSS 0.7.1 has no transpose solve; factorise the transpose')
        if rhs.shape[0] != self.n or rhs.device != self.device:
            raise ValueError('Right-hand side does not match the factorised system')
        if rhs.ndim == 2:
            return torch.stack([self.solve(column) for column in rhs.unbind(1)], 1)
        with torch.cuda.device(self.device):
            if torch.cuda.current_stream(self.device).cuda_stream != self.stream.cuda_stream:
                raise RuntimeError('cuDSS solves must run on the stream used at factorisation')
            self.rhs.copy_(rhs)
            self._execute(SOLVE)
            return self.out.clone()

    def close(self):
        lib = getattr(self, 'lib', None)
        if lib is None or not getattr(self, 'handle', None) or not self.handle.value:
            return
        with torch.cuda.device(self.device):
            torch.cuda.synchronize(self.device)
            for handle in (self.a, self.x, self.b):
                if handle.value:
                    lib.cudssMatrixDestroy(handle)
                    handle.value = None
            if self.data.value:
                lib.cudssDataDestroy(self.handle, self.data)
                self.data.value = None
            if self.config.value:
                lib.cudssConfigDestroy(self.config)
                self.config.value = None
            lib.cudssDestroy(self.handle)
            self.handle.value = None

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001 - interpreter shutdown
            pass


class SuperLUSolver:
    """CPU SuperLU with the torch interface of CudssSolver (tests and CPU runs of the torch backend)."""
    supports_transpose = True

    def __init__(self, matrix, permutation=None, pivot_threshold=.01):
        started = time.monotonic()
        A = sp.csc_matrix(matrix)
        self.n, self.permutation = A.shape[0], permutation
        if permutation is None:
            self.lu = splu(A, permc_spec='COLAMD')
        else:
            self.lu = splu(A[permutation][:, permutation].tocsc(), permc_spec='NATURAL', diag_pivot_thresh=pivot_threshold,
                           options=dict(SymmetricMode=True))
        self.fill = int(self.lu.L.nnz + self.lu.U.nnz)
        self.seconds = time.monotonic()-started

    def solve(self, rhs, transpose=False):
        b = rhs.detach().cpu().numpy()
        trans = 'T' if transpose else 'N'
        if self.permutation is None:
            x = self.lu.solve(b, trans=trans)
        else:
            x = np.empty_like(b)
            x[self.permutation] = self.lu.solve(np.ascontiguousarray(b[self.permutation]), trans=trans)
        return torch.as_tensor(x, dtype=rhs.dtype, device=rhs.device)


def direct_solver(matrix, device, *, permutation=None, pivot_threshold=.01):
    """cuDSS on CUDA devices, SuperLU on the CPU."""
    if canonical_device(device).type == 'cuda':
        return CudssSolver(matrix, device)
    return SuperLUSolver(matrix, permutation, pivot_threshold)
