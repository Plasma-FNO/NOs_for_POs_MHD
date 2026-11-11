#!/usr/bin/env python3
"""Package the verified presentation assets, excluding large native trajectories."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import zipfile


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    summary = json.loads((root/'summary.json').read_text())
    if not summary['primary_complete'] or not summary['array_validation_performed'] or summary['missing'] or summary['pending']:
        raise ValueError('Require complete, array-verified primary and stress results')
    expected = [root/'videos'/case/f'{case}_{view}.mp4' for case in ('mastu_004', 'mastu_010')
                for view in ('fields', 'density_change', 'temperature_change')]
    expected.append(root/'videos/mastu_004_h384/mastu_004_density_change.mp4')
    for path in expected:
        if not path.is_file():
            raise FileNotFoundError(path)
        decoded = json.loads(path.with_name(path.stem+'_render.json').read_text())
        if decoded['sha256'] != sha(path) or decoded['decoded_frames'] != decoded['frames']:
            raise ValueError(f'Video verification mismatch: {path}')
    slides = root/'presentation/slides_manifest.json'
    manifest = json.loads(slides.read_text())
    if not manifest['original_text_content_streams_and_file_preserved']:
        raise ValueError('Original presentation preservation failed')
    for key in ('updated_pdf', 'insert_pdf'):
        if sha(Path(manifest[key]['path'])) != manifest[key]['sha256']:
            raise ValueError('Presentation artifact changed')
    rows = []
    for method in summary['methods']:
        rows.append(f'<tr><td>{html.escape(method["label"])}</td><td>{method["mean_rollout_seconds"]:.2f} s</td>'
                    f'<td>{method["mean_normalized_volume_mse"]:.6g}</td><td>{method["training_updates"]}</td></tr>')
    native = sum(row['wall_seconds'] for row in summary['native_ground_truth'])/len(summary['native_ground_truth'])
    sections = []
    for case in ('mastu_004', 'mastu_010'):
        videos = []
        for view, label in (('density_change', 'Density evolution'), ('temperature_change', 'Temperature evolution'), ('fields', 'Absolute fields')):
            path = f'videos/{case}/{case}_{view}.mp4'
            videos.append(f'<h3>{label}</h3><video controls preload="metadata" src="{path}"></video><p><a href="{path}">Download MP4</a></p>')
        sections.append(f'<h2>{case}</h2>'+''.join(videos))
    body = '<!doctype html><meta charset="utf-8"><title>MAST-U — SCML presentation assets</title>'
    body += '<style>body{font:18px system-ui;line-height:1.5;max-width:1100px;margin:35px auto;padding:0 20px;color:#17212a}h1,h2{color:#00558a}video{width:100%;background:#eee}td,th{padding:8px 18px;text-align:left;border-bottom:1px solid #ccc}a{color:#00558a}</style>'
    body += '<h1>MAST-U: native physics plus a learned implicit correction</h1><p>11 November 2026 · Two held-out cases · Seed 17 · Axisymmetric JOREK model303</p>'
    body += '<p><a href="presentation/SCML_2026_NOs4POs_with_MASTU.pdf">Updated presentation</a> · <a href="presentation/MASTU_implicit_OpsSplit_insert.pdf">Insert slides</a> · <a href="README.md">Equations and methodology</a> · <a href="summary.json">Measured results</a></p>'
    body += '<h2>Matched 96-interval forecasts</h2><p>1.405913 microseconds of physical evolution. Mean of two case medians, each from three warm synchronized H100 runs. Deployed guards included; setup, scoring and rendering excluded.</p>'
    body += '<table><thead><tr><th>Method</th><th>Rollout</th><th>Normalized MSE</th><th>Training updates</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>'
    body += f'<p>Native reference: {native:.1f} s ({native/60:.2f} min), historically on four CPU MPI ranks, including I/O and checks. Different hardware and timing scope.</p>'
    body += f'<p>The learned correction adds {manifest["matched_control"]["reduction_percent"]:.2f}% lower MSE over the matched full nonlinear physics control. Most of the large gain over AR/Neural ODE comes from retaining native physics. This is an exploratory two-case comparison with unequal training budgets.</p>'
    body += ''.join(sections)
    body += '<h2>384-step extrapolation: accuracy unavailable after step 96</h2><p>All three methods reached step 384 with guards active. Continued finite output is not evidence of accurate plasma physics. Implicit OpsSplit also applies additional native axis/X-point and Mach-stage checks.</p>'
    body += '<video controls preload="metadata" src="videos/mastu_004_h384/mastu_004_density_change.mp4"></video>'
    body += '<p>Missing ground truth is blank. Shared color scales stay fixed; saturation is reported. Fields use JOREK simulation units. Video error is instantaneous per-field normalized RMS; the table averages seven-field normalized MSE over the trajectory.</p>'
    (root/'index.html').write_text(body+'\n')
    readme = Path(__file__).with_name('README.md').read_text()
    (root/'README.md').write_text(readme.replace('[artifacts](artifacts)', 'this extracted package').replace('(artifacts/', '(') +
        '\nOpen `index.html` for the videos and presentation links. Insert the MP4 files into the original editable slide deck; the supplied PDF uses stills. Raw native trajectories and task-local dependencies are intentionally excluded from this portable package.\n')
    files = [root/name for name in ('index.html', 'README.md', 'summary.json', 'protocol.json', 'protocol_followups.json', 'video_metric_parity.json')]
    for folder in ('videos', 'presentation'):
        files.extend(path for path in (root/folder).rglob('*') if path.is_file() and path.suffix in ('.mp4', '.png', '.gif', '.html', '.json', '.pdf'))
    records = {str(path.relative_to(root)): dict(sha256=sha(path), bytes=path.stat().st_size) for path in sorted(set(files))}
    (root/'asset_manifest.json').write_text(json.dumps(dict(files=records, primary_video_count=6, extrapolation_video_count=1,
                                                           native_trajectory_arrays_included=False), indent=2)+'\n')
    files.append(root/'asset_manifest.json')
    archive = root/'MASTU_SCML_2026_presentation_assets.zip'
    temporary = archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as stream:
        for path in sorted(set(files)):
            stream.write(path, path.relative_to(root))
    with zipfile.ZipFile(temporary) as stream:
        if stream.testzip() is not None:
            raise ValueError('Corrupt archive entry')
        for name, record in records.items():
            if hashlib.sha256(stream.read(name)).hexdigest() != record['sha256']:
                raise ValueError(f'Archive content differs: {name}')
    temporary.replace(archive)
    result = dict(archive=str(archive), bytes=archive.stat().st_size, sha256=sha(archive), files=len(files), movies=len(expected))
    (root/'package_validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
