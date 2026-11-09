import sys,json,shutil,subprocess,hashlib
from pathlib import Path
from datetime import datetime,timezone
root=Path('/pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/JOREK/mastu_opsplit')
sys.path.insert(0,str(root/'src'))
from mastu_opsplit.common import sha256_file
from mastu_opsplit.case import namelist_overrides
reports=root/'reports'; phasepath=reports/'cpu_refinement_continuation_20261102.json'; phase=json.loads(phasepath.read_text())
budget=json.loads(Path(phase['budget_record']).read_text()); assert sha256_file(phase['budget_record'])==phase['budget_record_sha256']
assert len(phase['runs'])<budget['phase']['simulation_submissions_max']
cpu=sum(x['requested_cpu_core_hours'] for x in phase['builds']+phase['runs'])+4/3
node=sum(x['requested_node_hours'] for x in phase['builds']+phase['runs'])+1/3
assert cpu<=128 and node<=8
parent=json.loads((reports/'cpu_refinement_capacity160k_build_20261102.json').read_text())
assert sha256_file(parent['binary_path'])==parent['binary_sha256']
bulk=Path(parent['source_directory']).parents[1]; dest=bulk/'contour_bounds_debug_20261102'; dest.mkdir()
src=dest/'source'; shutil.copytree(parent['source_directory'],src)
(src/'.obj/reorder_and_clean_flux_surfaces.o').unlink()
work=dest/'work'; work.mkdir()
old=bulk/'cpu_refine_quad_probe3_20261102/attempt_1/work'
for name in ['eqdsk.dat','jorek_equil_rz.h5']: shutil.copyfile(old/name,work/name)
(work/'input').write_text(namelist_overrides((old/'input').read_text(),{'regrid_from_rz':True,'nstep_n':0}))
script=reports/'cpu_contour_bounds_debug_20261102.sbatch'
env=root/'runs/prepared_equilibrium_dcgp_authorized_20261102/jorek-env.sh'
script.write_text(f'''#!/bin/bash
#SBATCH --job-name=mastu-contour-bounds
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_prod
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=00:20:00
set -euo pipefail
source {env}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd {src}
make -j2 .obj/reorder_and_clean_flux_surfaces.o F90FLAGS='-g -fcheck=bounds -fbacktrace' > {dest}/build.out 2> {dest}/build.err
make -j2 jorek_model303 >> {dest}/build.out 2>> {dest}/build.err
sha256sum jorek_model303 > {dest}/binary.sha256
cd {work}
set +e
timeout --signal=TERM --kill-after=10s 1050s mpirun --bind-to none -n 4 {src}/jorek_model303 < input > {dest}/stdout.log 2> {dest}/stderr.log
result=$?
printf '%s\\n' "$result" > {dest}/returncode.txt
exit "$result"
''')
record={'label':'contour_bounds_debug','status':'prepared','diagnostic_only':True,'equations_modified':False,'source_directory':str(src),'bulk_output':str(dest),'script':str(script),'script_sha256':sha256_file(script),'requested_cpu_core_hours':4/3,'requested_node_hours':1/3,'startup_hashes':{p.name:sha256_file(p) for p in work.iterdir()},'parent_build_report_sha256':sha256_file(reports/'cpu_refinement_capacity160k_build_20261102.json'),'bounds_checked_module':'reorder_and_clean_flux_surfaces','reuse_RZ_equilibrium':True}
cmd=['sbatch','--parsable','--chdir',str(dest),'--output',str(dest/'batch-%j.out'),'--error',str(dest/'batch-%j.err'),str(script)]
check=subprocess.run(['sbatch','--test-only',*cmd[1:]],capture_output=True,text=True); assert check.returncode==0,check.stderr
out=subprocess.run(cmd,capture_output=True,text=True); assert out.returncode==0,out.stderr
record.update(job_id=out.stdout.strip(),status='submitted',submitted_utc=datetime.now(timezone.utc).isoformat())
phase['runs'].append(record); phase.update(reserved_cpu_core_hours=cpu,reserved_node_hours=node)
phasepath.write_text(json.dumps(phase,indent=2)+'\n')
(reports/'cpu_contour_bounds_debug_20261102.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
