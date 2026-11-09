"""Bounded integer-neighbor rebuild, native parity, and balanced mesh planning."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mastu_opsplit.case import namelist_overrides
from mastu_opsplit.common import sha256_file


def main():
    reports=ROOT/'reports';record_path=reports/'cpu_fast_mesh_20261103.json'
    if record_path.exists():
        raise ValueError('fast mesh pipeline already prepared')
    bulk=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102')
    parent=bulk/'spatial_pair_20261103';base=bulk/'fast_mesh_20261103';base.mkdir()
    row=next(x for x in subprocess.check_output(['saldo','-b','--dcgp','-n'],text=True).splitlines() if x.split()[0]=='FUPA2_FAUNA')
    cols=row.split()
    if min(int(cols[3])-int(cols[4]),int(cols[7])-int(cols[8]))<512:
        raise ValueError('insufficient verified project credit')
    budget={'authorization':'Current explicit user diagnosis and next refinement request',
            'scope':'One integer-only neighbor-search build/parity and balanced directional geometry allocation. No changed-capacity solver is accepted for evolution.',
            'allocations_max':1,'attempts_max':1,'nodes':1,'mpi_ranks':4,'threads':1,'memory_GiB':128,
            'wall_seconds':7200,'compiler_processes_max':2,'rank_CPU_hours_max':8,'node_hours_max':2,
            'conservative_resource_hours_max':512,'saldo_row':row,'spatial_field_target':.01,
            'geometry_candidates':['balanced200','balanced160'],'capacity_selection_limit':155000}
    budget_path=reports/'cpu_fast_mesh_authorization_20261103.json';budget_path.write_text(json.dumps(budget,indent=2)+'\n')
    patch=ROOT/'patches/0015-preserve-exact-neighbours-with-incidence-search.patch'
    parents={'source160':bulk/'restart_capacity_recovery_20261103/source',
             'source240':bulk/'spatial_next_20261103/source'}
    for label,source in parents.items():
        shutil.copytree(source,base/label)
        subprocess.run(['patch','--batch','-p1','-i',str(patch)],cwd=base/label,check=True,capture_output=True)
    old=bulk/'spatial_next_parity_retry_20261103/parent8'
    shutil.copyfile(old/'jorek_restart.h5',base/'reference8.h5')
    shutil.copyfile(old/'jorek000000.h5',base/'reference0.h5')
    shutil.copyfile(parent/'jorek-env.sh',base/'jorek-env.sh')
    shutil.copyfile(reports/'cpu_spatial_next_worker_20261103.py',base/'worker.py')
    shutil.copyfile(reports/'cpu_fast_mesh_controller_20261103.py',base/'controller.py')
    mesh={'n_tht':330,'n_open':6,'n_outer':48,'n_inner':12,'n_leg':18,'n_leg_out':120,
          'n_private':12,'n_up_leg':18,'n_up_leg_out':120,'n_up_priv':12,
          **{f'n_ext_block({i})':114 if i==3 else 60 for i in range(1,12)}}
    stages={}
    for label,steps,flux in [('regression8',8,None),('balanced200',0,200),('balanced160',0,160)]:
        work=base/label;work.mkdir()
        text=(old/'input').read_text() if flux is None else (parent/'baseline48/input').read_text()
        overrides={'nstep_n':steps,'nout':8}
        if flux is not None:
            overrides.update(mesh,n_flux=flux)
        (work/'input').write_text(namelist_overrides(text,overrides))
        for name in ['eqdsk.dat','jorek_equil_rz.h5']:
            shutil.copyfile(parent/'baseline48'/name,work/name)
        stages[label]={'steps':steps,'amplitudes':[0.,0.],'timeout_seconds':1800,'bulk_output':str(work)}
    objects=['mod_elt_matrix_fft.o','mod_mumps.o','initial_conditions.o','mod_settings.o','jorek2_main.o']
    files=[base/name for name in ['reference8.h5','reference0.h5','worker.py','controller.py','jorek-env.sh']]
    files += [p for label in stages for p in (base/label).iterdir()]
    files += [base/label/'grids/grid_utils/update_neighbours_basic.f90' for label in parents]
    authorization={'budget':budget,'stages':stages,'mesh_overrides':mesh,
                   'patch_sha256':sha256_file(patch),'native_integer_tests_sha256':sha256_file(reports/'cpu_exact_neighbors_tests_20261103.xml'),
                   'parity_protocol':json.loads((reports/'cpu_native_parity_protocol_20261102.json').read_text()),
                   'scientific_objects_before':{name:sha256_file(base/'source160/.obj'/name) for name in objects},
                   'input_hashes':{str(p):sha256_file(p) for p in files}}
    (base/'authorization.json').write_text(json.dumps(authorization,indent=2)+'\n')
    script=reports/'cpu_fast_mesh_20261103.sbatch';python='/pitagora_work/FUPB2_UKAEA_ML/vgopakum/envs/cfo/bin/python'
    script.write_text(f'''#!/bin/bash
#SBATCH --job-name=mastu-exact-fast-mesh
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_prod
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
set -euo pipefail
source {base}/jorek-env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd {base}/source160
timeout --signal=TERM --kill-after=10s 600s make -j2 jorek_model303 > {base}/build160.out 2> {base}/build160.err
cp jorek_model303 {base}/solver160.exe
chmod 555 {base}/solver160.exe
cd {base}/source240
timeout --signal=TERM --kill-after=10s 600s make -j2 jorek_model303 > {base}/build240.out 2> {base}/build240.err
cp jorek_model303 {base}/planner240.exe
chmod 555 {base}/planner240.exe
{python} {base}/controller.py {base} {ROOT}
''')
    record={'budget':budget,'bulk_output':str(base),'authorization_sha256':sha256_file(base/'authorization.json'),
            'script_sha256':sha256_file(script),'status':'prepared'}
    record_path.write_text(json.dumps(record,indent=2)+'\n')
    command=['sbatch','--parsable','--chdir',str(base),'--output',str(base/'batch-%j.out'),'--error',str(base/'batch-%j.err'),str(script)]
    subprocess.run(['bash','-n',str(script)],check=True);subprocess.run(['sbatch','--test-only',*command[1:]],check=True)
    record.update(job_id=subprocess.check_output(command,text=True).strip(),status='submitted')
    record_path.write_text(json.dumps(record,indent=2)+'\n')
    path=reports/'cpu_spatial_next_20261103.json';main_record=json.loads(path.read_text())
    main_record['jobs']['directional_geometry_folded']={**main_record['jobs'].pop('directional_plan'),'active':False}
    main_record['jobs']['directional_plan']={'job_id':record['job_id'],'kind':'simulation','budget_record':str(budget_path)}
    main_record['status']='balanced_fast_geometry_submitted';path.write_text(json.dumps(main_record,indent=2)+'\n')
    print(json.dumps({'job_id':record['job_id'],'output':str(base)}))


if __name__=='__main__':
    main()
