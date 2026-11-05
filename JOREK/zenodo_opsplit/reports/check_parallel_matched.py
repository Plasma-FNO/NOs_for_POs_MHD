"""Compare one global matched batch: serial versus distributed gradients/Adam."""
from pathlib import Path
import sys,argparse,json,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
if '--code' in sys.argv:sys.path.insert(0,sys.argv[sys.argv.index('--code')+1])
import torch
import parallel
from train_comparison import Experiment

def main():
 p=argparse.ArgumentParser();p.add_argument('--code',type=Path);p.add_argument('--fast-sparse',action='store_true');p.add_argument('--case',choices=['es','em'],required=True);p.add_argument('--arm',default='opsplit-pretrained');p.add_argument('--reference',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 torch.set_num_threads(1);parallel.initialize('cuda')
 if a.fast_sparse:
  import gpu_sparse,ctypes
  original=gpu_sparse._System.call
  def fast_call(self,name,*args):
   if name=='cudssConfigSet' and args[1]==25:ctypes.cast(args[2],ctypes.POINTER(ctypes.c_int)).contents.value=0
   return original(self,name,*args)
  gpu_sparse._System.call=fast_call
 root=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/jorek-matched-20261104')/a.case
 ex=Experiment(root/'data',a.arm,'cuda',root/'terms/best.pt' if a.arm.endswith('pretrained') else None)
 order=torch.randperm(len(ex.meta['windows']),generator=torch.Generator().manual_seed(ex.cfg['seed'])).tolist()
 windows=[ex.meta['windows'][i] for i in order[:ex.cfg['batch_size']]]
 i,s=windows[0];warm=ex.objective('train',i,s,ex.cfg['horizon']);warm.backward();del warm
 ex.model.zero_grad(set_to_none=True);torch.cuda.synchronize();parallel.gather(None);torch.cuda.reset_peak_memory_stats();start=time.monotonic()
 loss=parallel.batch_loss(ex,windows);torch.cuda.synchronize();seconds=time.monotonic()-start
 measurements=parallel.gather(dict(seconds=seconds,peak_memory_bytes=torch.cuda.max_memory_allocated()))
 grads={n:p.grad.detach().cpu().clone() for n,p in ex.model.named_parameters() if p.grad is not None}
 optimizer=torch.optim.Adam(ex.model.parameters(),lr=ex.cfg['lr'],weight_decay=ex.cfg['weight_decay'])
 torch.nn.utils.clip_grad_norm_(ex.model.parameters(),ex.cfg['grad_clip'],error_if_nonfinite=True);optimizer.step()
 parameters={n:p.detach().cpu().clone() for n,p in ex.model.named_parameters()}
 if parallel.rank()==0:
  row=dict(case=a.case,arm=a.arm,global_batch=len(windows),world_size=parallel.size(),windows=windows,loss=loss,measurements=measurements,identity=ex.identity)
  if a.reference:
   old=torch.load(a.reference,weights_only=False)
   assert old['row']['windows']==windows
   previous_identity=old['row']['identity']
   for key in ('arm','comparison_sha256','initialization_sha256','device'):assert previous_identity[key]==ex.identity[key]
   for name in ('native_rollout.py','native.py','gpu_sparse.py','native_terms.py','model.py','equations.py','comparison.py'):
    assert previous_identity['code_sha256'][name]==ex.identity['code_sha256'][name]
   errors={}
   for name,new,previous in [('gradient',grads,old['gradients']),('adam_parameters',parameters,old['parameters'])]:
    errors[name]=(sum(float((new[k]-previous[k]).abs().square().sum()) for k in new)/max(sum(float(previous[k].abs().square().sum()) for k in new),1e-30))**.5
   row['relative_errors']=errors;row['loss_relative_error']=abs(loss-old['row']['loss'])/max(abs(loss),1e-30)
   row['speedup']=old['row']['measurements'][0]['seconds']/max(x['seconds'] for x in measurements)
   row['passed']=errors['gradient']<5e-5 and errors['adam_parameters']<1e-6 and row['loss_relative_error']<1e-6
  else:
   torch.save(dict(row=row,gradients=grads,parameters=parameters),a.output.with_suffix('.pt'))
  a.output.write_text(json.dumps(row,indent=2)+'\n');print(json.dumps({k:v for k,v in row.items() if k not in ['identity','windows']}),flush=True)
 parallel.gather(None)
 if parallel.size()>1:torch.distributed.destroy_process_group()
if __name__=='__main__':main()
