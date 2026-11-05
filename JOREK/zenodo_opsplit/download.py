"""Checksum-verified, resumable downloads of the two public JOREK releases."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time
import urllib.request

RECORDS = {'es': '11099659', 'em': '11099527'}
DEFAULT_ROOT = '/pitagora_work/FUPB2_UKAEA_ML/vgopakum/data/jorek_zenodo'


def digest(path):
    h = hashlib.md5()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return 'md5:' + h.hexdigest()


def download(entry, destination):
    if destination.exists():
        if destination.stat().st_size == entry['size'] and digest(destination) == entry['checksum']:
            return
        raise ValueError(f'Existing file failed checksum: {destination}')
    partial = destination.with_suffix(destination.suffix + '.part')
    for attempt in range(5):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset != entry['size']:
                request = urllib.request.Request(entry['links']['self'], headers={'Range': f'bytes={offset}-'})
                with urllib.request.urlopen(request, timeout=120) as response:
                    append = offset > 0 and response.status == 206
                    if response.status == 206 and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        raise ValueError('Unexpected HTTP byte range')
                    with partial.open('ab' if append else 'wb') as out:
                        shutil.copyfileobj(response, out, 8 << 20)
            if partial.stat().st_size != entry['size'] or digest(partial) != entry['checksum']:
                partial.unlink()
                raise ValueError(f'Checksum/size mismatch: {destination.name}')
            partial.replace(destination)
            return
        except (OSError, ValueError):
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def extract(archive, destination):
    """Reject links, paths and duplicate members; only regular release files are accepted."""
    destination.mkdir(exist_ok=True)
    with tarfile.open(archive) as tar:
        seen = set()
        for member in tar.getmembers():
            if not member.isfile() or Path(member.name).name != member.name or member.name in seen:
                raise ValueError(f'Unsafe/duplicate archive member: {member.name}')
            seen.add(member.name)
        for member in tar.getmembers():
            target = destination / member.name
            if target.exists():
                # A repeated member must match; never silently replace scientific inputs.
                h = hashlib.md5(tar.extractfile(member).read()).hexdigest()
                if digest(target) != 'md5:' + h:
                    raise ValueError(f'Conflicting archive member: {target}')
                continue
            temporary = target.with_suffix(target.suffix + '.part')
            with tar.extractfile(member) as source, temporary.open('wb') as out:
                shutil.copyfileobj(source, out, 8 << 20)
            temporary.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(DEFAULT_ROOT))
    parser.add_argument('--archives', type=int, default=0, help='First N archives per record; 0 means all')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.archives < 0 or not 1 <= args.workers <= 4:
        parser.error('Nonnegative archive limit and 1..4 workers required')
    args.root.mkdir(parents=True, exist_ok=True)
    lock = (args.root / '.download.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for case, record in RECORDS.items():
        directory = args.root / case
        directory.mkdir(parents=True, exist_ok=True)
        metadata = directory / 'record.json'
        if not metadata.exists():
            with urllib.request.urlopen(f'https://zenodo.org/api/records/{record}', timeout=60) as response:
                metadata.write_text(json.dumps(json.load(response), indent=2) + '\n')
        entries = json.loads(metadata.read_text())['files']
        archives = sorted((f for f in entries if f['key'].endswith('.tar') and 'GIF' not in f['key']), key=lambda f: f['key'])
        archives = archives[:args.archives] if args.archives else archives
        selected = archives + [f for f in entries if f['key'] == 'README.txt']
        cache = directory / 'archives'
        cache.mkdir(exist_ok=True)
        needed = sum(f['size'] * 2 for f in selected if not (cache / (f['key'] + '.extracted')).exists())
        if shutil.disk_usage(directory).free < needed + (2 << 30):
            raise OSError(f'Need {needed / 1e9:.1f} GB plus 2 GiB free for {case}')
        def fetch(entry):
            if Path(entry['key']).name != entry['key']:
                raise ValueError('Unsafe filename from record')
            target = cache / entry['key']
            print(f'{case}: {target.name}', flush=True)
            download(entry, target)
            marker = cache / (entry['key'] + '.extracted')
            if target.suffix == '.tar' and not marker.exists():
                extract(target, directory / 'raw')
                marker.write_text(entry['checksum'] + '\n')
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(fetch, selected))
        print(f'{case}: {len(archives)} archives verified and extracted', flush=True)


if __name__ == '__main__':
    main()
