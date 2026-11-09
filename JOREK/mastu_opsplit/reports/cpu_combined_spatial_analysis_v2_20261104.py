"""Apply the frozen spatial protocol to qualified combined-solver trajectory pairs."""
import json
import math
import re
import subprocess
import sys
from pathlib import Path

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R.parent / 'src'))
from mastu_opsplit.common import sha256_file
from cpu_spatial_next_launch_20261103 import physics_text

B = Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102')
comparisons = {'wall': ('control80', 'wall80'), 'core_poloidal': ('control80', 'core_poloidal440'), 'core_poloidal_next': ('core_poloidal440', 'core_poloidal880')}
comparison_name = sys.argv[1]
coarse_label, fine_label = comparisons[comparison_name]
coarse, fine = [B / f'combined_pair_{name}_20261104' for name in (coarse_label, fine_label)]
manifest = json.loads((fine / 'manifest.json').read_text())
for base in (coarse, fine):
    m = json.loads((base / 'manifest.json').read_text())
    for name, expected in m['input_hashes'].items():
        if sha256_file(name) != expected:
            raise ValueError('frozen input changed: ' + name)
    for key in ['final_time_s', 'tstep_code_units', 'field_relative_target', 'integral_relative_target', 'equil_accuracy']:
        if m[key] != manifest[key]:
            raise ValueError('comparison protocol differs: ' + key)
references = {}
for label in ['baseline48', 'perturbed48']:
    if physics_text(coarse / label / 'input') != physics_text(fine / label / 'input'):
        raise ValueError('non-spatial input controls differ')
    for base in (coarse, fine):
        result = json.loads((base / label / 'result.json').read_text())
        if result['status'] != 'completed':
            raise ValueError('qualified trajectory required')
        qualification = json.loads((R / 'cpu_combined_refined_qualification_20261103.json').read_text())
        if result['binary_sha256'] != qualification['binary_sha256']:
            raise ValueError('both trajectories must use the qualified combined executable')
        text = (base / label / 'stdout.log').read_text()
        errors = re.findall(r'MUMPS refinement steps and backward errors:\s*(\d+)\s+(\S+)\s+(\S+)', text)
        if len(errors) < 48 or any(int(n)>5 or not all(math.isfinite(float(x)) and float(x)>=0 for x in [a,b]) or float(a)+float(b)>1e-12 for n,a,b in errors):
            raise ValueError('complete bounded accurate linear solves required')
        for moment, name in [('initial', 'jorek000000.h5'), ('final', 'jorek_restart.h5')]:
            if sha256_file(base / label / name) != result[moment]['restart_sha256']:
                raise ValueError('trajectory output changed')
        if result['final']['health']['steps'] != 48 or abs(result['final']['health']['final_time_s']-manifest['final_time_s']) > 1e-15:
            raise ValueError('trajectory horizon differs')
    result = json.loads((coarse / label / 'result.json').read_text())
    references[label] = {'work': str(coarse / label), 'initial_sha256': result['initial']['restart_sha256'],
                         'final_sha256': result['final']['restart_sha256']}
manifest.update(references=references, spatial_refinement_scope=f'Qualified combined Newton/core/wall/MUMPS executable on both meshes; {coarse_label} to {fine_label}; physics and timestep fixed.')
path = fine / 'analysis_manifest.json'
with path.open('x') as stream:
    json.dump(manifest, stream, indent=2)
    stream.write('\n')
output = R / f'cpu_combined_{comparison_name}_convergence_20261104.json'
subprocess.run([sys.executable, str(R / 'cpu_spatial_directions_analysis_20261103.py'), str(fine), str(path), str(output)], check=True)
result = json.loads(output.read_text())
print(json.dumps({'directional_full_domain_targets_met': result['candidate_spatial_targets_met_on_covered_volume'] and result['coverage']['full_wall_coverage'],
                  'scientific_gates_passed': [], 'report': str(output)}), flush=True)
