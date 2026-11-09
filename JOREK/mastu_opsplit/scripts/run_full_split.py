"""Bounded native full OpsSplit rollout; run within the prepared Slurm environment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys

import h5py
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mastu_opsplit.case import namelist_overrides
from mastu_opsplit.common import load_config,write_json
from mastu_opsplit.full_rollout import read_linear
from mastu_opsplit.restart import read_restart
from mastu_opsplit.runner import classify,restart_health


def run(config,output,*,dry_run=False):
    required={'prepared_case','initial_restart','frozen_operator','mode','dt','steps','nout'}
    if not required<=config.keys():raise ValueError('Missing rollout keys: '+str(required-config.keys()))
    mode=config['mode'];steps=config['steps'];dt=config['dt'];nout=config['nout']
    scheme=config.get('scheme','euler')
    if scheme not in ('euler','cnab2','sbdf2'):raise ValueError('Unknown OpsSplit time scheme')
    if mode not in ('numerical','recombine','neural'):raise ValueError('Unknown full rollout mode')
    if type(steps) is not int or not 1<=steps<=1536 or not np.isfinite(dt) or not 0<dt<=.015625:
        raise ValueError('Pilot bound: at most 1536 steps and dt<=0.015625')
    if not 0<steps*dt<=3 or type(nout) is not int or not 1<=nout<=steps:
        raise ValueError('Pilot bound: at most the approved 1.406-microsecond window')
    source=Path(config['prepared_case']).resolve();initial=Path(config['initial_restart']).resolve()
    frozen=Path(config['frozen_operator']).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest=json.loads((source/'case.json').read_text())
    if manifest.get('full_split_protocol')!='full_split_protocol.json':raise ValueError('Not a frozen full-split executable')
    for relative,sha in manifest['snapshot_hashes'].items():
        if digest(source/relative)!=sha:raise ValueError('Frozen case changed: '+relative)
    vertices,_=read_linear(frozen)
    linear_manifest=json.loads((frozen.parent/'manifest.json').read_text())
    if digest(frozen)!=linear_manifest['linear_sha256']:raise ValueError('Frozen operator hash mismatch')
    baseline=Path(linear_manifest['baseline'])
    if digest(baseline)!=linear_manifest['baseline_sha256']:raise ValueError('Baseline changed')
    with h5py.File(initial) as f,h5py.File(baseline) as b:
        if not np.array_equal(vertices,f['vertex'][()].T):raise ValueError('Frozen operator element mismatch')
        for key in ('x','size','index','vertex','boundary','n_sons','constrained'):
            if not np.array_equal(f[key][()],b[key][()]):raise ValueError('Frozen operator geometry mismatch: '+key)
        start=int(f['index_now'][0]);t0=float(f['t_now'][0]);unit=float(f['sqrt_mu0_rho0'][0])
    if mode=='neural' and not Path(config.get('checkpoint','')).is_file():
        raise ValueError('Neural mode requires a full_opsplit_v1 checkpoint')
    spec=dict(config=config,initial_sha256=digest(initial),linear_sha256=digest(frozen),
              executable_sha256=digest(source/'jorek.exe'),start_step=start,
              start_seconds=t0*unit,end_seconds=(t0+steps*dt)*unit,
              scheme='native M/constraints; F(q*) + implicit frozen L(q-q*) + explicit complete nonlinear volume remainder',
              boundaries='native deterministic coupled boundary operators',training=False)
    if dry_run:return spec
    if not os.environ.get('SLURM_JOB_ID') or int(os.environ.get('SLURM_NTASKS','0'))!=4:
        raise RuntimeError('Full pilot requires a Slurm allocation with four MPI ranks')
    output.mkdir(parents=True);work=output/'work';work.mkdir()
    write_json(output/'provenance.json',spec)
    for source_file,dest in ((initial,'jorek_restart.h5'),(initial,f'jorek{start:06d}.h5'),
                             (source/'eqdsk.dat','eqdsk.dat'),(frozen,'opsplit_linear.bin'),
                             (frozen.parent/'manifest.json','opsplit_manifest.json')):
        if dest=='opsplit_linear.bin' and source_file.stat().st_dev==work.stat().st_dev:
            os.link(source_file,work/dest)
        else:
            shutil.copyfile(source_file,work/dest)
    overrides=dict(restart=True,regrid=False,tstep_n=dt,nstep_n=steps,nout=nout,write_ps=False)
    (work/'input').write_text(namelist_overrides((source/'input').read_text(),overrides))
    if mode=='neural':
        command=[sys.executable,str(Path(__file__).with_name('infer_full_split.py')),
                 '--checkpoint',str(Path(config['checkpoint']).resolve())]
        script=work/'opsplit_infer.sh';script.write_text('#!/bin/sh\nset -eu\nexec '+shlex.join(command)+'\n');script.chmod(0o755)
    env=dict(os.environ,OPSSPLIT_MODE=mode,OPSSPLIT_SCHEME=scheme,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    timeout=int(config.get('timeout_seconds',7000))
    if not 1<=timeout<=22000:raise ValueError('Timeout outside pilot bound')
    timed_out=False
    with (work/'input').open('rb') as stdin,(output/'stdout.log').open('wb') as stdout,(output/'stderr.log').open('wb') as stderr:
        process=subprocess.Popen(['mpirun','--bind-to','none','-n','4',str(source/'jorek.exe')],
                                 stdin=stdin,stdout=stdout,stderr=stderr,cwd=work,env=env,start_new_session=True)
        try:code=process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(process.pid,signal.SIGKILL);code=process.wait()
    status=classify(code,timed_out,(output/'stdout.log').read_text(errors='replace')+(output/'stderr.log').read_text(errors='replace'))
    result=dict(status=status,returncode=code)
    try:
        if status!='completed':raise RuntimeError(status)
        result['health']=restart_health(work/'jorek_restart.h5',expected_steps=start+steps,final_time_s=(t0+steps*dt)*unit)
        from mastu_opsplit.numerical import evaluate_native
        geometry,state,_=read_restart(work/'jorek_restart.h5',quadrature_order=8)
        samples=evaluate_native(state,geometry)
        result['min_density_temperature']=samples[...,4:6].min(axis=(0,1)).tolist()
        if not np.isfinite(samples).all() or (samples[...,4:6]<=0).any():raise ValueError('Invalid final native state')
    except Exception as exc:
        result.update(status='failed',error=str(exc))
    write_json(output/'result.json',result)
    if result['status']!='completed':raise RuntimeError(result['error'])
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    print(json.dumps(run(load_config(a.config),a.output,dry_run=a.dry_run),indent=2))
