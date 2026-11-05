"""Training-only release checks for native units, momentum convention and FD rollouts."""
import argparse,json,time
from pathlib import Path
import h5py,numpy as np,torch
from data import select,layout
from native import (read_input,decode_release,common_coefficients,wall_trace,NativeReference)


def relative(a,b,crop=4):
    a,b=a[...,crop:-crop,crop:-crop],b[...,crop:-crop,crop:-crop]
    return ((a-b).square().sum((-2,-1))/b.square().sum((-2,-1)).clamp_min(1e-30)).sqrt().tolist()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=5);args=p.parse_args()
    torch.set_num_threads(1)
    m=json.loads(args.manifest.read_text());report=dict(cases={},native_rollout_ready=False)
    for case in ('es','em'):
        rows=[]
        for entry in select(m,case,'train',args.samples,907):
            inputs=read_input(entry['input']);c=common_coefficients(inputs,case)
            with h5py.File(entry['path']) as f:
                keys,r,z=layout(f,case)
                raw=torch.tensor(np.stack([f[k][:22] for k in keys],axis=1))
            q=decode_release(raw);r,z=torch.tensor(r),torch.tensor(z)
            rr,zz=torch.meshgrid(r,z,indexing='xy')
            expected_rho=torch.full_like(rr,inputs['rho_0']);expected_T=torch.full_like(rr,inputs['t_0'])
            for i in range(1,int(inputs['n_jropes'])+1):
                radius2=(rr-inputs[f'r_jropes({i})'])**2+(zz-inputs[f'z_jropes({i})'])**2
                shape=(1-radius2/inputs[f'w_jropes({i})']**2).clamp_min(0)**2
                expected_rho+=inputs[f'rho_jropes({i})']*shape;expected_T+=inputs[f't_jropes({i})']*shape
            if case=='es':expected_T/=1e4
            ic=[
                float((q[0,0]-expected_rho).norm()/expected_rho.norm()),
                float((q[0,1]/q[0,0]-expected_T).norm()/expected_T.norm())]
            ref=NativeReference(r,z,wall_trace(q[:1],r,z,c),c)
            row=dict(id=entry['id'],initial_density_temperature_relative_error=ic,frames=[])
            for t in (5,10,20):
                dq=(q[t+1:t+2]-q[t-1:t])/(2*c['nominal_frame_dt'])
                record=dict(frame=t,conventions={})
                for visc in ('old','new'):
                    ref.c['viscosity']=visc
                    rhs=ref.rhs(q[t:t+1])
                    record['conventions'][visc]=relative(rhs,dq)[0]
                row['frames'].append(record)
            # Initial free step, no later-frame boundary values or auxiliary fields.
            ref.c['viscosity']='old';row['initial_step']={}
            for n in (8,16):
                x=q[:1].clone();h=c['nominal_frame_dt']/n;started=time.monotonic()
                try:
                    with torch.no_grad():
                        for _ in range(n):
                            k1=ref.rhs(x);k2=ref.rhs(x+.5*h*k1);x=x+h*k2
                            if not torch.isfinite(x).all():raise FloatingPointError('Nonfinite state')
                    row['initial_step'][str(n)]=dict(completed=True,error=relative(x,q[1:2])[0],seconds=time.monotonic()-started)
                except (ValueError,RuntimeError,FloatingPointError) as error:
                    row['initial_step'][str(n)]=dict(completed=False,error=str(error),seconds=time.monotonic()-started)
            rows.append(row);print(case,entry['id'],row['initial_step'],flush=True)
        report['cases'][case]=rows
    report['note']='These checks support a reference reconstruction, not identification of the exact generating revision. FD term labels are not FEM residual labels.'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
