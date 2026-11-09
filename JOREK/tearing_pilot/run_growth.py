"""Fresh-start growth-window convergence study using an isolated HDF5 fix."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import h5py
import numpy as np
from run_pilot import sha

ROOT=Path(__file__).resolve().parent


def input_for(text,steps,dt,restart,nout=1):
    for key,value in {'restart':'.t.' if restart else '.f.','nstep_n':steps,'tstep_n':dt,'nout':nout}.items():
        text,count=re.subn(rf'(?mi)^(\s*{key}\s*=)[^!\n]*',lambda m:m[1]+' '+str(value),text)
        if count!=1:raise ValueError((key,count))
    return text


def run_case(out,name,steps,dt,restart,binary,cap,nout=1):
    work=out/name;work.mkdir()
    (work/'input').write_text(input_for((ROOT/'snapshot/input').read_text(),steps,dt,restart,nout))
    if restart:shutil.copyfile(ROOT/'snapshot/begin.h5',work/'jorek_restart.h5')
    start=time.monotonic()
    with (work/'input').open() as inp,(work/'logfile').open('w') as log:
        p=subprocess.run(['timeout','--kill-after=10',str(cap),'mpirun','--bind-to','none','-n','2',str(binary)],
                         cwd=work,stdin=inp,stdout=log,stderr=subprocess.STDOUT)
    log=(work/'logfile').read_text(errors='replace')
    result=dict(returncode=p.returncode,wall_seconds=time.monotonic()-start,dt_native=dt,requested_steps=steps,restart=restart,nout=nout,timeout_seconds=cap,
                error_lines=[l for l in log.splitlines() if re.search(r'HDF5-DIAG|assertion.*fail|aborting|not converged|failed to converge|NaN|SIGSEGV',l,re.I)])
    path=work/'jorek_restart.h5'
    if path.exists():
        with h5py.File(path) as f:
            result.update(final_step=int(f['index_now'][0]),final_time_native=float(f['t_now'][0]),finite=bool(np.isfinite(f['values'][()]).all()))
            result['horizon_complete']=result['final_step']==steps+(30 if restart else 0) and abs(result['final_time_native']-(steps*dt+(90000 if restart else 0)))<1e-9
            if name=='regression':
                with h5py.File(ROOT/'snapshot/end.h5') as ref:
                    result['reference_max_abs_error']=float(np.max(abs(f['values'][()]-ref['values'][()])))
                    result['reference_pass']=result['reference_max_abs_error']<=1e-12
    result['healthy_completion']=p.returncode==0 and not result['error_lines'] and result.get('finite',False) and result.get('horizon_complete',False)
    if not restart and (work/'jorek_equil_rz.h5').exists():
        try:
            with h5py.File(work/'jorek_equil_rz.h5') as f:
                for key in f:
                    if isinstance(f[key],h5py.Dataset): f[key][()]
                result['equilibrium_export_readable']=True
        except Exception as e:result.update(equilibrium_export_readable=False,equilibrium_export_error=str(e))
        result['equilibrium_export_count']=log.count('filename = jorek_equil_rz.h5')
        result['healthy_completion'] &= result['equilibrium_export_readable'] and result['equilibrium_export_count']==1
    elif not restart:result['healthy_completion']=False
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dt',type=float,nargs='*');parser.add_argument('--nout',type=int,default=1);parser.add_argument('--max-case-seconds',type=int,default=900);args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise SystemExit('Slurm allocation required')
    out=ROOT/('growth_'+os.environ['SLURM_JOB_ID']);out.mkdir(exist_ok=False)
    binary=ROOT/'single_writer_build/jorek_model199'
    results=dict(job_id=os.environ['SLURM_JOB_ID'],binary=str(binary),binary_sha256=sha(binary),runner_sha256=sha(Path(__file__)),
                 patch_sha256=sha(ROOT/'single_writer.patch'),mpi_ranks=2,omp_threads=1,cases={})
    specs=[('regression',1,3000.,True)]
    if args.dt:
        specs += [(f'dt{dt:g}',round(81000/dt),dt,False) for dt in args.dt]
    else:
        specs += [('dt3000',31,3000.,False),('dt1500',54,1500.,False),('dt750',108,750.,False),('dt375',216,375.,False)]
    for name,steps,dt,restart in specs:
        result=run_case(out,name,steps,dt,restart,binary,args.max_case_seconds,1 if restart else args.nout)
        results['cases'][name]=result
        (out/'run_report.json').write_text(json.dumps(results,indent=2)+'\n')
        print(name,json.dumps(result),flush=True)
        if not result['healthy_completion'] or (name=='regression' and not result['reference_pass']):
            raise SystemExit('Case failed validation; remaining cases skipped')


if __name__=='__main__':main()
