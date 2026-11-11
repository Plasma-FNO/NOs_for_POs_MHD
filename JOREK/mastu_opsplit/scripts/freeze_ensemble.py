#!/usr/bin/env python3
"""Create an exclusive source/config snapshot and print reproducible Slurm commands."""
import argparse
import hashlib
from pathlib import Path
import shlex
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[1]
    snapshot = root/'study'
    snapshot.mkdir()
    for name in ('src', 'tests', 'configs', 'scripts', 'slurm', 'native'):
        shutil.copytree(source/name, snapshot/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    profile_manifest = source/'reports/physical_profiles_20261111.json'
    if profile_manifest.exists():
        (snapshot/'reports').mkdir()
        shutil.copy2(profile_manifest, snapshot/'reports'/profile_manifest.name)
    shutil.copy2(source/'ENSEMBLE_TRAINING.md', snapshot/'ENSEMBLE_TRAINING.md')
    records = []
    for path in sorted(snapshot.rglob('*')):
        if path.is_file():
            records.append(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(snapshot)}')
            path.chmod(0o444)
    (snapshot/'source.sha256').write_text('\n'.join(records)+'\n')
    (snapshot/'source.sha256').chmod(0o444)
    exports = shlex.quote(f'ALL,STUDY_ROOT={snapshot},OUTPUT_ROOT={root}')
    script = shlex.quote(str(snapshot/'slurm/ensemble.sbatch'))
    print(f'sbatch --partition=dcgp_fua_dbg --time=00:30:00 --export={exports},SMOKE=1 --output={root}/smoke-%j.log {script}')
    print(f'sbatch --array=0-8%3 --export={exports} --output={root}/ensemble-%A_%a.log {script}')


if __name__ == '__main__':
    main()
