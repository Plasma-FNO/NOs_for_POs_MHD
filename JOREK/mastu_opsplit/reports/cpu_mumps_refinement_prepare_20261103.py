from pathlib import Path
import sys,json,shutil,subprocess,difflib
ROOT=Path('/pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/JOREK/mastu_opsplit');R=ROOT/'reports';sys.path.insert(0,str(ROOT/'src'))
from mastu_opsplit.case import namelist_overrides
from mastu_opsplit.common import sha256_file
B=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102');old=B/'native_bridge_20261103';base=B/'mumps_refinement_20261103'
assert not base.exists()
row=next(x for x in subprocess.check_output(['saldo','-b','--dcgp','-n'],text=True).splitlines() if x.split()[0]=='FUPA2_FAUNA');cols=row.split();assert min(int(cols[3])-int(cols[4]),int(cols[7])-int(cols[8]))>=512
base.mkdir();shutil.copytree(old/'source',base/'source',ignore=shutil.ignore_patterns('.git','docs','reg_tests'))
p=base/'source/solvers/mod_mumps.f90';before=p.read_text();after=before.replace('    mmss%initialized = .true.', '''    ! Bound refinement and retain MUMPS' rollback on a deteriorating residual.
    mmss%mumps_par%icntl(10) = 5
    mmss%mumps_par%cntl(2) = 1.d-16
    mmss%mumps_par%icntl(11) = 2
    mmss%initialized = .true.''',1)
# Print the documented host diagnostics immediately after the single-RHS solve.
after=after.replace('    integer                 :: ierr\n    \n    mmss%mumps_par%rhs', '    integer                 :: ierr, my_id\n    \n    mmss%mumps_par%rhs',1)
target='    call mumps_check(mmss, "solve")';assert after.count(target)==1
after=after.replace(target,target+'''
    call MPI_COMM_RANK(mmss%comm,my_id,ierr)
    if (my_id == 0) write(*,*) 'MUMPS refinement steps and backward errors:', &
      mmss%mumps_par%infog(15),mmss%mumps_par%rinfog(7:8)
''')
p.write_text(after);patch=ROOT/'patches/0029-refine-mumps-solution-and-report-backward-error.patch';patch.write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/solvers/mod_mumps.f90',tofile='b/solvers/mod_mumps.f90')))
shutil.copyfile(old/'jorek-env.sh',base/'jorek-env.sh');shutil.copyfile(old/'worker.py',base/'worker.py')
manifest=json.loads((old/'manifest.json').read_text());manifest['stages']={k:{**manifest['stages'][k],'expected_final_step':9,'timeout_seconds':1800} for k in ['enabled8','perturbed8']};manifest['final_time_s']=9*.0625*4.686377037426928e-7
# Existing worker's parent/toggle parity is irrelevant to a one-step accuracy diagnosis;
# compare against each preserved same-initial-state one-step reference instead.
p=base/'worker.py';s=p.read_text();a=s.index("            if label in ['disabled8', 'enabled8']:");z=s.index("            result['status'] = 'completed'",a)
s=s[:a]+'''            result['old_solver_one_step_parity'] = parity(work / 'jorek_restart.h5',
                Path(manifest['old_bridge']) / label / 'jorek000009.h5', manifest['parity_protocol'])
'''+s[z:];p.write_text(s)
files=[p,base/'jorek-env.sh',base/'source/solvers/mod_mumps.f90',patch,ROOT/'src/mastu_opsplit/native_balance.py',ROOT/'src/mastu_opsplit/native_export.py']
for label in manifest['stages']:
 work=base/label;work.mkdir()
 for name in ['initial_reference.h5','eqdsk.dat','jorek_equil_rz.h5']:shutil.copyfile(old/label/name,work/name)
 (work/'input').write_text(namelist_overrides((old/label/'input').read_text(),{'nstep_n':1}))
 files.extend(work.iterdir())
budget={'authorization':'User requested solver convergence and science-plan completion.','scope':'One isolated MUMPS accuracy build and two single-step restart proofs, baseline and perturbed. Original fields, timestep and thresholds preserved.','mpi_ranks':4,'memory_GiB':128,'wall_hours_max':2,'rank_CPU_hours_max':8,'saldo_row':row}
manifest.update(old_bridge=str(old),budget=budget,input_hashes={str(p):sha256_file(p) for p in files});(base/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
batch=R/'cpu_mumps_refinement_20261103.sbatch';batch.write_text(f'''#!/bin/bash
#SBATCH --job-name=mastu-mumps-refinement
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_prod
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
set -euo pipefail
source {base}/jorek-env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd {base}/source
timeout --signal=TERM --kill-after=10s 600s make -j2 jorek_model303 > {base}/build.out 2> {base}/build.err
/pitagora_work/FUPB2_UKAEA_ML/vgopakum/envs/cfo/bin/python {base}/worker.py {base} {ROOT}
''')
command=['sbatch','--parsable','--chdir',str(base),'--output',str(base/'batch-%j.out'),'--error',str(base/'batch-%j.err'),str(batch)];subprocess.run(['sbatch','--test-only',*command[1:]],capture_output=True,check=True)
record={'kind':'simulation','status':'prepared','bulk_output':str(base),'budget':budget,'script_sha256':sha256_file(batch),'manifest_sha256':sha256_file(base/'manifest.json'),'scientific_gates_passed':[]}
p=R/'cpu_mumps_refinement_20261103.json';p.write_text(json.dumps(record,indent=2)+'\n');record.update(status='submitted',job_id=subprocess.check_output(command,text=True).strip());p.write_text(json.dumps(record,indent=2)+'\n');print(record['job_id'])
