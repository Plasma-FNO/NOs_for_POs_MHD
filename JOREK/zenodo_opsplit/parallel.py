"""Data parallelism preserving the global window batch and summed gradients."""
import datetime
import os
import torch
import torch.distributed as dist


def size():
    return dist.get_world_size() if dist.is_initialized() else 1


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def initialize(device):
    if int(os.environ.get('WORLD_SIZE','1'))==1:return
    if device.startswith('cuda'):torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl' if device.startswith('cuda') else 'gloo',timeout=datetime.timedelta(minutes=60))


def gather(value):
    if size()==1:return [value]
    values=[None]*size();dist.all_gather_object(values,value);return values


def root_flag(value, device):
    if size()==1:return value
    flag=torch.tensor(int(value) if rank()==0 else 0,device=device)
    dist.broadcast(flag,0);return bool(flag)


def sum_gradients(model):
    if size()==1:return
    parameters=list(model.parameters())
    used=torch.tensor([p.grad is not None for p in parameters],device=parameters[0].device,dtype=torch.int32)
    dist.all_reduce(used)
    pending=[]
    for p,active in zip(parameters,used.tolist()):
        if not active:continue  # Preserve None: Adam must not decay unused parameters.
        if p.grad is None:p.grad=torch.zeros_like(p)
        if not p.grad.is_contiguous():p.grad=p.grad.contiguous()
        value=torch.view_as_real(p.grad) if p.grad.is_complex() else p.grad
        pending.append(dist.all_reduce(value,op=dist.ReduceOp.SUM,async_op=True))
    for work in pending:work.wait()


def batch_loss(ex, windows):
    if size()==1:return ex.batch_loss(windows)
    value=0.;error=None;local=windows[rank()::size()]
    try:
        if local:value=ex.batch_loss(local)
    except (FloatingPointError,RuntimeError) as exc:error=f'rank {rank()}: {exc}'
    rows=gather(dict(value=value,error=error,limiter=ex.last_batch_limiter if ex.native and local and error is None else {}))
    failures=[r['error'] for r in rows if r['error']]
    if failures:raise RuntimeError('; '.join(failures))
    sum_gradients(ex.model)
    if ex.native:
        from train_comparison import merge_limiter
        ex.last_batch_limiter={}
        for row in rows:merge_limiter(ex.last_batch_limiter,row['limiter'])
    return sum(row['value'] for row in rows)


def rng_state(device):
    return dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state() if device.startswith('cuda') else None)


def restore_rng(state,device):
    torch.set_rng_state(state['cpu'])
    if device.startswith('cuda'):torch.cuda.set_rng_state(state['cuda'])
