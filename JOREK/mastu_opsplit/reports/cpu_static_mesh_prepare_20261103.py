"""Bounded geometry-only repair using supported native grid-placement controls."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mastu_opsplit.case import namelist_overrides
from mastu_opsplit.common import sha256_file


def main():
    reports=ROOT/'reports';path=reports/'cpu_static_mesh_20261103.json'
    if path.exists():
        raise ValueError('geometry repair already prepared')
    bulk=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102')
    parent=bulk/'fast_mesh_20261103';base=bulk/'static_mesh_20261103';base.mkdir()
    row=next(x for x in subprocess.check_output(['saldo','-b','--dcgp','-n'],text=True).splitlines() if x.split()[0]=='FUPA2_FAUNA')
    cols=row.split()
    if min(int(cols[3])-int(cols[4]),int(cols[7])-int(cols[8]))<512:
        raise ValueError('insufficient verified credit')
    budget={'authorization':'Current explicit diagnosis and next spatial refinement request',
            'scope':'One zero-step native mesh-quality repair allocation, at most3 pre-frozen candidates; no invalid geometry or unqualified fields accepted.',
            'allocations_max':1,'attempts_max':1,'nodes':1,'mpi_ranks':4,'threads':1,'memory_GiB':128,
            'wall_seconds':7200,'rank_CPU_hours_max':8,'node_hours_max':2,'conservative_resource_hours_max':512,
            'saldo_row':row,'selection':'First positive native/q8 geometry <=155000: core240 with unchanged exterior attachment counts; wall2x equidistant, wall2x native, then original wall spacing. No evolution used for selection.',
            'spatial_target':.01,'physical_coefficients_changed':False}
    original=bulk/'spatial_pair_20261103/baseline48'
    mesh={'n_flux':240}
    stages={}
    for label,wall_factor,equidistant in [('radial240wall2eq',2,True),('radial240wall2',2,False),('radial240',1,False)]:
        work=base/label;work.mkdir()
        overrides={**mesh,'nstep_n':0,'nout':8,'opsplit_amplitude(1)':0.,'opsplit_amplitude(2)':0.,'n_tht_equidistant':False,
                   **{f'n_ext_block({i})':(76 if i==3 else 40)*wall_factor for i in range(1,12)},
                   **{f'n_ext_equidistant({i})':equidistant for i in range(1,12)}}
        (work/'input').write_text(namelist_overrides((original/'input').read_text(),overrides))
        for name in ['eqdsk.dat','jorek_equil_rz.h5']:
            shutil.copyfile(original/name,work/name)
        stages[label]={'steps':0,'amplitudes':[0.,0.],'timeout_seconds':1800,'binary':str(parent/'planner240.exe'),'bulk_output':str(work)}
    for name in ['worker.py','jorek-env.sh','qualification.json']:
        shutil.copyfile(parent/name,base/name)
    files=[base/name for name in ['worker.py','jorek-env.sh','qualification.json']]+[parent/'planner240.exe']
    files += [p for label in stages for p in (base/label).iterdir()]
    manifest={'budget':budget,'stages':stages,'final_time_s':1.4059131112280784e-6,'mesh_overrides':mesh,
              'geometry_only':True,'input_hashes':{str(p):sha256_file(p) for p in files},
              'qualified_scientific_solver':'Original validated160001-capacity executable; planner240 geometry only.'}
    (base/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    controller=base/'mesh_quality_driver.py';controller.write_text('''import json,subprocess,sys,h5py\nfrom pathlib import Path\nb=Path(sys.argv[1]);root=sys.argv[2];attempts=[]\nfor label in ['radial240wall2eq','radial240wall2','radial240']:\n run=subprocess.run([sys.executable,str(b/'worker.py'),str(b),root,label],check=False)\n result=json.loads((b/label/'result.json').read_text());attempts.append({'label':label,'returncode':run.returncode,'status':result['status']})\n (b/'attempts.json').write_text(json.dumps(attempts,indent=2)+'\\n')\n if run.returncode:continue\n with h5py.File(b/label/'jorek_restart.h5') as f:nodes=int(f['n_nodes'][()].item());elements=int(f['n_elements'][()].item())\n if max(nodes,elements)<=155000:\n  (b/'selection.json').write_text(json.dumps({'selected':label,'nodes':nodes,'elements':elements,'policy':'First positive native and q8 geometry<=155000; no evolution fields used in selection.'},indent=2)+'\\n');break\nelse:raise ValueError('All frozen quality candidates failed native/q8 geometry or capacity checks')\n''')
    script=reports/'cpu_static_mesh_20261103.sbatch'
    script.write_text(f'''#!/bin/bash
#SBATCH --job-name=mastu-geometry-quality
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
/pitagora_work/FUPB2_UKAEA_ML/vgopakum/envs/cfo/bin/python {controller} {base} {ROOT}
''')
    record={'budget':budget,'bulk_output':str(base),'manifest_sha256':sha256_file(base/'manifest.json'),
            'controller_sha256':sha256_file(controller),'script_sha256':sha256_file(script)}
    path.write_text(json.dumps(record,indent=2)+'\n')
    command=['sbatch','--parsable','--chdir',str(base),'--output',str(base/'batch-%j.out'),'--error',str(base/'batch-%j.err'),str(script)]
    subprocess.run(['bash','-n',str(script)],check=True);subprocess.run(['sbatch','--test-only',*command[1:]],check=True)
    record['job_id']=subprocess.check_output(command,text=True).strip();path.write_text(json.dumps(record,indent=2)+'\n')
    master=reports/'cpu_spatial_next_20261103.json';main_record=json.loads(master.read_text())
    main_record['jobs']['equidistant_geometry_folded']={**main_record['jobs'].pop('directional_plan'),'active':False}
    main_record['jobs']['directional_plan']={'job_id':record['job_id'],'kind':'simulation','budget_record':str(path)}
    main_record['status']='fixed_attachment_geometry_submitted';master.write_text(json.dumps(main_record,indent=2)+'\n')
    print(record['job_id'])


if __name__=='__main__':
    main()
