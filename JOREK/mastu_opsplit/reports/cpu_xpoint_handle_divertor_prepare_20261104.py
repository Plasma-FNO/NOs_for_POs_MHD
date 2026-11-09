"""Qualified divertor320 probe after measured divertor80/160 Vpar failure."""
import argparse,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports';sys.path.insert(0,str(ROOT/'src'))
from mastu_opsplit.common import sha256_file
from mastu_opsplit.case import namelist_overrides
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('resolution',type=int,choices=[320]);args=parser.parse_args()
B=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102');parent=B/'xpoint_handle_20261104';base=B/f'xpoint_handle_divertor{args.resolution}_20261104';record_path=R/f'cpu_xpoint_handle_divertor{args.resolution}_20261104.json'
if base.exists() or record_path.exists():raise ValueError('preserve earlier probe')
control=json.loads((parent/'control3/result.json').read_text());binary=parent/'source/jorek_model303';digest=sha256_file(binary)
assert control['status']=='completed' and control['binary_sha256']==digest and control['initial_parity']['passed'] and control['parity']['passed']
original=json.loads((parent/'manifest.json').read_text())
for p,h in original['input_hashes'].items():
 if sha256_file(p)!=h:raise ValueError('parent changed: '+p)
row=next(line for line in subprocess.check_output(['saldo','-b','--dcgp','-n'],text=True).splitlines() if line.split()[0]=='FUPA2_FAUNA');c=row.split()
if min(int(c[3])-int(c[4]),int(c[7])-int(c[8]))<2048:raise ValueError('insufficient credit')
(base/'source').mkdir(parents=True);shutil.copyfile(binary,base/'source/jorek_model303');(base/'source/jorek_model303').chmod(0o555)
shutil.copyfile(parent/'jorek-env.sh',base/'jorek-env.sh')
worker=(parent/'worker.py').read_text()
anchor="    config = manifest['stages'][stage]"
guard='''    qualification = {}
    for label in ['control3','perturbed3','wall80']:
        proof_path = Path(manifest['qualified_parent'])/label/'result.json'
        proof = json.loads(proof_path.read_text())
        if proof['status'] != 'completed' or proof['binary_sha256'] != manifest['binary_sha256']:
            raise ValueError('all same-binary parent controls must complete')
        if not all(proof[k]['passed'] for k in ['initial_parity','parity']):
            raise ValueError('strict parent physical parity required')
        for moment,name in [('initial','jorek000000.h5'),('final','jorek_restart.h5')]:
            if sha256_file(proof_path.parent/name) != proof[moment]['restart_sha256']:
                raise ValueError('parent snapshot changed')
        qualification[str(proof_path)] = sha256_file(proof_path)
    (base/'parent_qualification.json').write_text(json.dumps(qualification,indent=2)+'\\n')
'''
assert worker.count(anchor)==1;worker=worker.replace(anchor,guard+anchor)
anchor="        if stage == 'probe3' and 'planned_geometry_reference' in manifest:"
audit='''        import h5py
        import numpy as np
        from mastu_opsplit.restart import cubic_basis
        with h5py.File(work/'jorek000000.h5') as f:
            v=f['vertex'][()].T-1;sizes=f['size'][()].transpose(2,1,0);xy=f['x'][:,:,0,:].transpose(2,1,0)
        q,_=np.polynomial.legendre.leggauss(16)
        a,b=np.meshgrid((q+1)/2,(q+1)/2,indexing='ij')
        _,hs,ht=cubic_basis(a.ravel(),b.ravel());minimum=float('inf')
        for start in range(0,len(v),512):
            jets=xy[v[start:start+512]]*sizes[start:start+512,...,None]
            rs,rt=[np.einsum('qvd,evdc->eqc',h,jets) for h in [hs,ht]]
            jac=rs[...,0]*rt[...,1]-rs[...,1]*rt[...,0]
            if not np.isfinite(jac).all() or (jac<=0).any():raise ValueError('q16 native mesh fold')
            minimum=min(minimum,float(jac.min()))
        result['minimum_q16_jacobian_m2']=minimum
'''
assert worker.count(anchor)==1;worker=worker.replace(anchor,audit+anchor);(base/'worker.py').write_text(worker)
ref=parent/'probe3';work=base/'probe3';work.mkdir()
(work/'input').write_text(namelist_overrides((ref/'input').read_text(),{'nstep_n':3,'nout':1,'n_leg':48,'n_up_leg':48,'n_leg_out':args.resolution,'n_up_leg_out':args.resolution}))
for name in ['eqdsk.dat','jorek_equil_rz.h5']:shutil.copyfile(ref/name,work/name)
files=[base/'source/jorek_model303',base/'worker.py',base/'jorek-env.sh',parent/'manifest.json',*work.iterdir()]
manifest={k:original[k] for k in ['final_time_s','tstep_code_units','field_relative_target','integral_relative_target','equil_accuracy','parity_protocol']}
manifest.update(stages={'probe3':dict(steps=3,mpi_ranks=16,amplitudes=[0.,0.],timeout_seconds=13800)},input_hashes={str(p):sha256_file(p) for p in files},qualified_parent=str(parent),binary_sha256=digest,scope='Same frozen physics, full q8 field health and independent q16 geometry; parent controls must pass strict physical parity before execution.')
(base/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');script=base/'run.sbatch'
script.write_text(f'''#!/bin/bash
#SBATCH --job-name=mastu-xpoint-divertor{args.resolution}
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_prod
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=512G
#SBATCH --time=04:00:00
set -euo pipefail
source {base}/jorek-env.sh
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/pitagora_work/FUPB2_UKAEA_ML/vgopakum/envs/cfo/bin/python {base}/worker.py {base} {ROOT} probe3
''')
cmd=['sbatch','--parsable','--chdir',str(base),'--output',str(base/'batch-%j.out'),'--error',str(base/'batch-%j.err'),str(script)]
subprocess.run(['sbatch','--test-only',*cmd[1:]],check=True,capture_output=True)
record=dict(status='prepared',bulk_output=str(base),dependencies=[],binary_sha256=digest,manifest_sha256=sha256_file(base/'manifest.json'),script_sha256=sha256_file(script),budget=dict(mpi_ranks=16,memory_GiB=512,wall_hours_max=4,rank_CPU_hours_max=64,saldo_row=row),scientific_gates_passed=[])
record_path.write_text(json.dumps(record,indent=2)+'\n');record.update(status='submitted',job_id=subprocess.check_output(cmd,text=True).strip());record_path.write_text(json.dumps(record,indent=2)+'\n');print(args.resolution,record['job_id'])
