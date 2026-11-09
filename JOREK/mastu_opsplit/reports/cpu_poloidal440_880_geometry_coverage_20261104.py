"""Geometry-only physical coverage of the preserved 440/880 initial meshes."""
import json
import sys
from pathlib import Path
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/'src'))
from mastu_opsplit.common import sha256_file
from mastu_opsplit.restart import read_restart
from mastu_opsplit.queries import sample_restart
B=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102')
coarse=B/'combined_refined_solver_20261103/core_poloidal440/jorek000000.h5'
fine=B/'combined_core_poloidal880_20261104/probe3/jorek000000.h5'
proof=json.loads((R/'cpu_core_poloidal880_geometry_diagnosis_20261104.json').read_text())
assert sha256_file(fine)==proof['sha256'] and all(not row['folds'] for row in proof['quadrature'].values())
parent=json.loads((coarse.parent/'result.json').read_text())
assert parent['status']=='completed' and sha256_file(coarse)==parent['initial']['restart_sha256']
hashes={str(p):sha256_file(p) for p in [coarse,fine]}
g,_,metadata=read_restart(fine,quadrature_order=4);points=g.quadrature_rz.reshape(-1,2);weights=g.volume_weights.ravel()
q=sample_restart(coarse,points);mask=q['mask']
result={'scope':'Geometry-only comparison. The old 880 MHD failure remains rejected; positive preserved initial geometry permits a coverage diagnosis independent of evolution. Fresh stronger-scaling geometry identity must be checked separately.',
 'input_hashes':hashes,'analysis_source_sha256':sha256_file(__file__),'queries_source_sha256':sha256_file(R.parent/'src/mastu_opsplit/queries.py'),
 'fine_metadata':metadata,'total_queries':len(mask),'covered_queries':int(mask.sum()),'full_domain_coverage':bool(mask.all()),'uncovered_volume_m3':float(weights[~mask].sum()),'uncovered_rz_m':points[~mask].tolist(),'scientific_gates_passed':[]}
for p,h in hashes.items():assert sha256_file(p)==h
with (R/'cpu_poloidal440_880_geometry_coverage_20261104.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps({k:result[k] for k in ['total_queries','covered_queries','full_domain_coverage']}))
