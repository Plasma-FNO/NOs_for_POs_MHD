"""Global-batch, optimizer and resumable checkpoint equivalence on CPU ranks."""
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from train_comparison import Experiment,train
from test_comparison import fixture
import parallel


def _training_worker(rank,data,output,init,interrupt,resume):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+str(init),rank=rank,world_size=2)
    try:
        train(Experiment(Path(data),'node','cpu'),Path(output),resume=resume,
              max_seconds=1e-9 if interrupt else 0,train_only=True)
    except SystemExit as error:
        assert interrupt and error.code==0
    finally:dist.destroy_process_group()


def test_distributed_resume_preserves_global_batch_and_adam(tmp_path):
    data=fixture(tmp_path,epochs=2)
    train(Experiment(data,'node','cpu'),tmp_path/'serial',train_only=True)
    output=tmp_path/'distributed'
    mp.spawn(_training_worker,args=(data,output,tmp_path/'initial-group',True,False),nprocs=2,join=True)
    assert (output/'requeue.json').exists()
    mp.spawn(_training_worker,args=(data,output,tmp_path/'resume-group',False,True),nprocs=2,join=True)
    serial=torch.load(tmp_path/'serial/last.pt',weights_only=False)
    actual=torch.load(output/'last.pt',weights_only=False)
    assert actual['next_epoch']==2 and actual['next_offset']==0 and len(actual['rank_rng'])==2
    for key,value in serial['state_dict'].items():
        if isinstance(value,torch.Tensor):torch.testing.assert_close(actual['state_dict'][key],value,atol=2e-6,rtol=5e-5)
    assert actual['scheduler']==serial['scheduler']
    assert len((output/'history.jsonl').read_text().splitlines())==2


class _ComplexExperiment:
    native=False
    def __init__(self):
        self.model=torch.nn.Module()
        self.model.weight=torch.nn.Parameter(torch.tensor([.3+.4j,.6-.1j]))
        self.model.unused=torch.nn.Parameter(torch.tensor(2.))
    def batch_loss(self,windows):
        loss=sum((self.model.weight*x).abs().square().sum() for x in windows)
        loss.backward();return float(loss.detach())


def _gradient_worker(rank,init):
    torch.set_num_threads(1);dist.init_process_group('gloo',init_method='file://'+str(init),rank=rank,world_size=2)
    try:
        for windows in ([1.,2.,3.],[2.]):
            ex=_ComplexExperiment();reference=_ComplexExperiment()
            expected=reference.batch_loss(windows);actual=parallel.batch_loss(ex,windows)
            assert abs(actual-expected)<1e-6
            torch.testing.assert_close(ex.model.weight.grad,reference.model.weight.grad)
            assert ex.model.unused.grad is None
    finally:dist.destroy_process_group()


def test_complex_gradients_empty_shard_and_unused_parameters(tmp_path):
    mp.spawn(_gradient_worker,args=(tmp_path/'gradient-group',),nprocs=2,join=True)
