"""Prepare one bounded modest-refinement CPU cost probe from a verified snapshot."""
import copy
import json
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from mastu_opsplit.common import config_hash, sha256_file
from mastu_opsplit.case import namelist_overrides
from mastu_opsplit.runner import render_slurm, run_plan


def main():
    template = 'energy_complete_baseline48'
    label = 'mesh22_probe3'
    old_config = ROOT/f'configs/cpu_{template}_20261102.yaml'
    old = ROOT/f'runs/prepared_cpu_{template}_20261102'
    config = yaml.safe_load(old_config.read_text())
    manifest = json.loads((old/'case.json').read_text())
    assert manifest['config_hash'] == config_hash(config)
    assert all(sha256_file(old/name) == value for name, value in manifest['snapshot_hashes'].items())
    mesh = {'n_R': 100, 'n_Z': 286, 'n_flux': 22, 'n_tht': 122, 'n_open': 2,
            'n_outer': 18, 'n_inner': 4, 'n_leg': 7, 'n_leg_out': 44, 'n_private': 4,
            'n_up_leg': 7, 'n_up_leg_out': 44, 'n_up_priv': 4,
            **{f'n_ext_block({i})': 42 if i == 3 else 22 for i in range(1, 12)}}
    assert all(value >= config['input_defaults'][name] for name, value in mesh.items())
    config['case_id'] = 'mastu-model303-'+label
    config['input_defaults'].update(mesh)
    original_horizon = config['acceptance']['final_time_s']
    config['evolution'].update(nstep_n=3, tstep_n=.0625, nout=1)
    config['acceptance'].pop('equilibrium_regression')
    config['acceptance']['final_time_s'] = original_horizon/16
    config['resource_authorization'].update(
        scope='One modest-refinement three-step CPU memory/runtime probe, not a reference acceptance or ensemble',
        approval_source='Explicit user instruction to continue', max_submissions=1)
    config['refinement'] = {'purpose': 'Bounded cost/health probe before full-horizon reference',
                            'base_case': template, 'mesh_counts': mesh,
                            'physical_parameters_unchanged': True, 'scientific_acceptance': False}
    path = ROOT/f'configs/cpu_{label}_20261102.yaml'
    prepared = ROOT/f'runs/prepared_cpu_{label}_20261102'
    report_path = ROOT/'reports/cpu_mesh22_continuation_20261102.json'
    script = ROOT/f'reports/cpu_{label}_20261102.sbatch'
    assert not any(p.exists() for p in [path, prepared, report_path, script])
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    shutil.copytree(old, prepared)
    target = prepared/'input'
    target.chmod(0o644)
    target.write_text(namelist_overrides((old/'input').read_text(), {**mesh, **config['evolution']}))
    target.chmod(0o444)
    derived = copy.deepcopy(manifest)
    derived.update(case_id=config['case_id'], config_hash=config_hash(config),
                   evolution=config['evolution'], output=str(prepared), parent_snapshot=str(old),
                   refinement=config['refinement'])
    derived.pop('unchanged_state_reference', None)
    derived['snapshot_hashes'] = {p.name: sha256_file(p) for p in prepared.iterdir()
                                 if p.is_file() and p.name != 'case.json'}
    target = prepared/'case.json'
    target.chmod(0o644)
    target.write_text(json.dumps(derived, indent=2)+'\n')
    bulk = Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102')
    output = bulk/f'cpu_{label}_20261102'
    (ROOT/f'runs/cpu_{label}_20261102').symlink_to(output, target_is_directory=True)
    script.write_text(render_slurm(config, prepared, output, path))
    record = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'preparation_source_sha256': sha256_file(__file__),
              'scope': config['resource_authorization']['scope'],
              'scientific_gates_passed': [], 'runs': [{'label': label, 'config': str(path),
              'prepared': str(prepared), 'bulk_output': str(output), 'script': str(script),
              'script_sha256': sha256_file(script), 'status': 'prepared',
              'plan': run_plan(config, prepared, output)}]}
    report_path.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'prepared': str(prepared), 'mesh': mesh,
                      'steps': 3, 'final_time_s': config['acceptance']['final_time_s']}, indent=2))


if __name__ == '__main__':
    main()
