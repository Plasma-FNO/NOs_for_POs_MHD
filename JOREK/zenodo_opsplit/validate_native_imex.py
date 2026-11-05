"""Independent validation-family IMEX free rollouts, including magnetic perturbation error."""
import argparse,json,time
from pathlib import Path
import h5py,numpy as np,torch
from data import select,layout
from native import read_input,common_coefficients,decode_release,wall_trace,NativeReference
from native_imex import IMEXReference
from validate_native import relative


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=2);p.add_argument('--horizon',type=int,default=5)
    args=p.parse_args();torch.set_num_threads(1)
    manifest=json.loads(args.manifest.read_text());report=dict(split='val',methods='old viscosity selected using training residuals; static initial wall trace',cases={})
    for case in ('es','em'):
        rows=[]
        for entry in select(manifest,case,'val',args.samples,907):
            c=common_coefficients(read_input(entry['input']),case);c['viscosity']='old'
            with h5py.File(entry['path']) as f:
                keys,r,z=layout(f,case)
                raw=torch.tensor(np.stack([f[k][:args.horizon+1] for k in keys],axis=1))
            q=decode_release(raw);r,z=torch.tensor(r),torch.tensor(z)
            ref=NativeReference(r,z,wall_trace(q[:1],r,z,c),c);solver=IMEXReference(ref,q[:1])
            result=dict(id=entry['id'],substeps={})
            for substeps in (8,16):
                x=q[:1].clone();history=[];failure=None;started=time.monotonic()
                try:
                    for step in range(1,args.horizon+1):
                        x=solver.step(x,c['nominal_frame_dt'],substeps)
                        item=dict(frame=step,relative_l2=relative(x,q[step:step+1])[0])
                        if case=='em':item['delta_psi_relative_l2']=relative(x[:,3]-q[:1,3],q[step:step+1,3]-q[:1,3])[0]
                        history.append(item)
                except (ValueError,RuntimeError,FloatingPointError) as error:failure=str(error)
                result['substeps'][str(substeps)]=dict(completed=failure is None,failure=failure,frames=history,seconds=time.monotonic()-started)
                print(case,entry['id'],substeps,'completed',failure is None,'frames',len(history),flush=True)
            rows.append(result)
        report['cases'][case]=rows
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
