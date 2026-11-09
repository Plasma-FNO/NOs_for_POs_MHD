"""Isolated correction of reversed same-curve intersections; strict healthy controls first."""
import difflib,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports';sys.path.insert(0,str(ROOT/'src'))
from mastu_opsplit.common import sha256_file
B=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102');parent=B/'scaling8_capacity400_20261104';base=B/'reversed_crossing_v2_20261104';record_path=R/'cpu_reversed_crossing_v2_20261104.json'
if base.exists() or record_path.exists():raise ValueError('preserve previous attempt')
qualification=json.loads((R/'cpu_scaling8_capacity400_qualification_20261104.json').read_text())
assert sha256_file(parent/'source/jorek_model303')==qualification['binary_sha256']
for p,h in qualification['input_hashes'].items():
 if sha256_file(p)!=h:raise ValueError('qualification changed: '+p)
row=next(line for line in subprocess.check_output(['saldo','-b','--dcgp','-n'],text=True).splitlines() if line.split()[0]=='FUPA2_FAUNA');c=row.split()
if min(int(c[3])-int(c[4]),int(c[7])-int(c[8]))<4320:raise ValueError('insufficient credit')
# Construct and check all edits before starting the source copy.
edits={};helper=ROOT/'native/opsplit_refine_reversed_pair.f90'
name='grids/grid_utils/find_crossing.f90';old=(parent/'source'/name).read_text();edits[name]=(old,old+'\n'+helper.read_text())
for kind in ['central','leg']:
 name=f'grids/grid_utils/define_{kind}_grid.f90';old=(parent/'source'/name).read_text();s=old
 anchor='real*8              :: R_cub1d(4), Z_cub1d(4)'
 assert s.count(anchor)==1
 s=s.replace(anchor,anchor+'\nreal*8, allocatable :: crossing_R(:,:,:),crossing_Z(:,:,:)\ninteger :: crossing_status,crossing_repairs')
 if kind=='central':
  anchor='!----------------------------------- The main part (without legs, outer and inner)\ndo i=1,n_flux+n_open+n_outer+n_inner'
  allocate='allocate(crossing_R(4,n_flux+n_open+n_outer+n_inner+1,n_tht), &\n         crossing_Z(4,n_flux+n_open+n_outer+n_inner+1,n_tht))'
  current='i+1';pair='i:i+1';targets='flux_list%psi_values(i-1:i)';success='        nwpts%k_cross(i+1,j) = k'
 else:
  anchor='!----------------------------------- The main part (without legs, outer and inner)\ndo i=1,n_surf_tot'
  allocate='allocate(crossing_R(4,n_surf_tot,n_seg),crossing_Z(4,n_surf_tot,n_seg))'
  current='i';pair='i-1:i';targets='flux_list%psi_values(i_flux(i-1:i))';success='        nwpts%k_cross(i,j) = k'
 assert s.count(anchor)==1
 s=s.replace(anchor,allocate+'\ncrossing_R=0.d0; crossing_Z=0.d0; crossing_repairs=0\n'+anchor)
 # Both smooth and fallback successes in the leg routine record the actual curve.
 assert s.count(success)==(1 if kind=='central' else 2)
 previous='i' if kind=='central' else 'i-1'
 block=f'''{success}
        crossing_R(:,{current},j)=R_cub1d; crossing_Z(:,{current},j)=Z_cub1d
        if (i > 1) then
          if (nwpts%k_cross({previous},j) == k .and. &
              all(crossing_R(:,{previous},j) == R_cub1d) .and. &
              all(crossing_Z(:,{previous},j) == Z_cub1d)) then
            if (nwpts%t_tht({current},j) < nwpts%t_tht({previous},j)-1.d-12) then
              call opsplit_refine_reversed_pair(node_list,element_list,nwpts%ielm_flux({pair},j), &
                   {targets},R_cub1d,Z_cub1d, &
                   nwpts%s_flux({pair},j),nwpts%t_flux({pair},j),nwpts%t_tht({pair},j), &
                   nwpts%RR_new({pair},j),nwpts%ZZ_new({pair},j),crossing_status)
              if (crossing_status /= 0) then
                write(*,*) 'Exact reversed {kind} pair failed: ',i,j
                error stop 'unresolved reversed flux intersections'
              endif
              crossing_repairs=crossing_repairs+1
              write(*,*) 'Exact reversed {kind} pair repaired: ',i,j
            endif
          endif
        endif'''
 if kind=='central':block=block.replace('if (i > 1) then','if (i > n_flux+1) then')
 s=s.replace(success,block)
 anchor="write(*,*) '                 Defining new nodes'";assert s.count(anchor)==1
 s=s.replace(anchor,"deallocate(crossing_R,crossing_Z)\n"+anchor)
 edits[name]=(old,s)
sys.path.insert(0,str(ROOT/'native'))
from boundary_incidence import replace_boundary_incidence
name='grids/grid_utils/update_boundary_types.f90';old=(parent/'source'/name).read_text();fast_boundary=replace_boundary_incidence(old)
base.mkdir();shutil.copytree(parent/'source',base/'source',ignore=shutil.ignore_patterns('.git','docs','reg_tests'))
patch=ROOT/'patches/0035-limit-reversed-correction-to-open-surfaces.patch'
with patch.open('x') as f:
 for name,(old,new) in edits.items():
  (base/'source'/name).write_text(new);f.writelines(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+name,tofile='b/'+name))
fast_source=base/'source/grids/grid_utils/update_boundary_types.f90';fast_source.write_text(fast_boundary)
fast_patch=ROOT/'patches/0036-cache-distinct-boundary-node-incidence.patch'
with fast_patch.open('x') as f:f.writelines(difflib.unified_diff((parent/'source/grids/grid_utils/update_boundary_types.f90').read_text().splitlines(True),fast_boundary.splitlines(True),fromfile='a/grids/grid_utils/update_boundary_types.f90',tofile='b/grids/grid_utils/update_boundary_types.f90'))
for name in ['worker.py','jorek-env.sh']:shutil.copyfile(parent/name,base/name)
original=json.loads((parent/'manifest.json').read_text());manifest={k:original[k] for k in ['final_time_s','tstep_code_units','field_relative_target','integral_relative_target','equil_accuracy','parity_protocol']}
files=[helper,patch,fast_patch,fast_source,ROOT/'native/boundary_incidence.py',*[base/'source'/name for name in edits],base/'worker.py',base/'jorek-env.sh',R/'cpu_native_snapshot_parity_20261104.py'];stages={}
for label,ref in [('control3',parent/'control3'),('perturbed3',parent/'perturbed3'),('wall80',B/'scaling8_wall80_20261104/probe3'),('probe3',B/'combined_balanced2_attachment_20261104/probe3')]:
 work=base/label;work.mkdir()
 for name in ['input','eqdsk.dat','jorek_equil_rz.h5']:shutil.copyfile(ref/name,work/name)
 stages[label]=dict(steps=3,mpi_ranks=16,amplitudes=[0.,0.],timeout_seconds=14400 if label=='probe3' else 3600)
 if label!='probe3':
  proof=json.loads((ref/'result.json').read_text());assert proof['status']=='completed'
  stages[label]['parity_reference']=str(ref/'jorek_restart.h5');files.extend([ref/'result.json',ref/'jorek000000.h5',ref/'jorek_restart.h5'])
 files.extend(work.iterdir())
manifest.update(stages=stages,input_hashes={str(p):sha256_file(p) for p in files},scope='Closed-core intersections are excluded. Cached distinct-element incidence preserves boundary classification. Only reversed adjacent intersections on an identical polar curve trigger exact native flux correction; both roots must restore order within the original parent cells and polar segment. Healthy baseline/perturbed/wall80 parity is required. Balanced mesh remains diagnostic until all geometry and health checks pass.')
(base/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');script=base/'run.sbatch'
script.write_text(f'''#!/bin/bash
#SBATCH --job-name=mastu-reversed-open-v2
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_prod
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=720G
#SBATCH --time=06:00:00
set -euo pipefail
source {base}/jorek-env.sh
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd {base}/source
timeout --signal=TERM --kill-after=10s 1500s make -j2 jorek_model303 > {base}/build.out 2> {base}/build.err
for stage in control3 perturbed3 wall80 probe3; do
 /pitagora_work/FUPB2_UKAEA_ML/vgopakum/envs/cfo/bin/python {base}/worker.py {base} {ROOT} "$stage"
done
''')
cmd=['sbatch','--parsable','--chdir',str(base),'--output',str(base/'batch-%j.out'),'--error',str(base/'batch-%j.err'),str(script)]
pre=subprocess.run(['sbatch','--test-only',*cmd[1:]],check=True,text=True,capture_output=True)
record=dict(status='prepared',bulk_output=str(base),manifest_sha256=sha256_file(base/'manifest.json'),script_sha256=sha256_file(script),scheduler_preflight=pre.stderr,budget=dict(authorization='Continue full spatial convergence as requested.',mpi_ranks=16,memory_GiB=720,wall_hours_max=6,rank_CPU_hours_max=96,saldo_row=row),scientific_gates_passed=[])
record_path.write_text(json.dumps(record,indent=2)+'\n');record.update(status='submitted',job_id=subprocess.check_output(cmd,text=True).strip());record_path.write_text(json.dumps(record,indent=2)+'\n');print(record['job_id'])
