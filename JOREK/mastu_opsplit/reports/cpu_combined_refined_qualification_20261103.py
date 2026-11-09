"""Validate the complete isolated solver stack without promoting scientific gates."""
from pathlib import Path
import json,re,subprocess,sys
import numpy as np
R=Path(__file__).resolve().parent;ROOT=R.parent;sys.path.insert(0,str(ROOT/'src'))
from mastu_opsplit.common import sha256_file
B=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102')
base=B/'combined_refined_solver_20261103';reference=B/'tight_exact_picard_reference_20261103/picard1'
manifest=json.loads((base/'manifest.json').read_text())
for name,digest in manifest['input_hashes'].items():
    if sha256_file(name)!=digest:raise ValueError('frozen combined input changed: '+name)
binary=sha256_file(base/'source/jorek_model303')
report={'status':'incomplete','scientific_gates_passed':[],'binary_sha256':binary,
        'scope':'Combined Newton/core/wall geometry and MUMPS probe qualification; full spatial and physical acceptance remains separate.',
        'max_solver_backward_error':1e-12,'stages':{},'manifest_sha256':sha256_file(base/'manifest.json'),
        'script_sha256':sha256_file(__file__)}
for label,config in manifest['stages'].items():
    work=base/label;proof=json.loads((work/'result.json').read_text());text=(work/'stdout.log').read_text()
    if proof['status']!='completed' or proof['binary_sha256']!=binary:raise ValueError('healthy same-binary probe required: '+label)
    for moment,name in [('initial','jorek000000.h5'),('final','jorek_restart.h5')]:
        if sha256_file(work/name)!=proof[moment]['restart_sha256']:raise ValueError('probe output changed')
    iterations=re.findall(r'Iteration\s+(\d+): diff=\s*(\S+)',text)
    iteration,update=iterations[-1];update=float(update)
    tolerance=1e-12 if label=='control1' else 1e-10
    if 'Fixed boundary equilibrium converged:' not in text or update>=tolerance:raise ValueError('strict GS convergence required')
    if proof['final']['health']['steps']!=config['steps']:raise ValueError('probe horizon differs')
    samples=np.array([[float(x.replace('D','E')) for x in row] for row in re.findall(
        r'MUMPS refinement steps and backward errors:\s*(\d+)\s+(\S+)\s+(\S+)',text)])
    if len(samples)<config['steps'] or not np.isfinite(samples).all() or (samples<0).any():raise ValueError('complete finite MUMPS diagnostics required')
    if samples[:,0].max()>5 or np.sum(samples[:,1:],axis=1).max()>report['max_solver_backward_error']:raise ValueError('linear solve backward error too large')
    report['stages'][label]={'equilibrium_iterations':int(iteration),'last_raw_update':update,
        'steps':config['steps'],'cycle_fallback_activated':'Stagnant X-point two-cycle:' in text,
        'maximum_mumps_backward_error':float(np.sum(samples[:,1:],axis=1).max()),
        'maximum_refinement_steps':int(samples[:,0].max()),
        'minimum_q8_jacobian_m2':proof['final']['metadata']['minimum_signed_jacobian_m2'],
        'minimum_density':proof['final']['minimum_density'],'minimum_temperature':proof['final']['minimum_temperature'],
        'result_sha256':sha256_file(work/'result.json')}
if report['stages']['control1']['cycle_fallback_activated']:raise ValueError('stable control unexpectedly changed root-search branch')
def physical_input(work):
    lines=[]
    for line in (work/'input').read_text().splitlines():
        left=line.split('=',1)[0].strip().lower()
        if left in ['amix','newton_gs_fixbnd','delta_psi_gs']:continue
        if line.strip():lines.append(line.strip())
    return lines
if physical_input(base/'control1')!=physical_input(reference):raise ValueError('control/reference physical inputs differ')
refproof=json.loads((reference/'result.json').read_text())
if refproof['status']!='completed':raise ValueError('healthy tight Picard reference required')
output=R/'cpu_combined_refined_parity_20261103.json'
subprocess.run([sys.executable,str(R/'cpu_newton_equilibrium_compare_20261103.py'),'--reference',str(reference),
    '--current',str(base/'control1'),'--normalized-profiles','--output',str(output)],check=True,stdout=subprocess.DEVNULL)
parity=json.loads(output.read_text());report['control_parity']=parity['physical_parity_targets_met'];report['parity_report_sha256']=sha256_file(output)
report['status']='combined_numerical_probes_verified' if report['control_parity'] else 'failed_control_parity'
report['limitations']=['Full-horizon spatial/timestep convergence and physical energy/source/flux balance remain pending.',
    'Root tracking is qualified on these fresh-start probes; restart persistence is not established.',
    'MUMPS backward errors are native diagnostics. Independent interior balance/repeatability proofs remain separate from boundary/axis and autonomous-backend acceptance.']
with (R/'cpu_combined_refined_qualification_20261103.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'status':report['status'],'scientific_gates_passed':[]}),flush=True)
