"""Check the MAST-U methods report: frozen evidence, honest pending cells, prose guards and the two-page limit."""
import copy
import json
import re
import subprocess

import pytest

import build as builder


def test_delivered_report_within_limit_and_describes_every_method():
    assert builder.page_count(builder.HERE/'results_one_page.pdf') <= builder.PAGE_LIMIT == 2
    text = subprocess.check_output(['pdftotext', str(builder.HERE/'results_one_page.pdf'), '-'], text=True)
    text = re.sub(r'\s+', ' ', text.replace('-\n', '-'))   # undo line-end hyphenation
    assert 'Explicit and Implicit Operator Splits' in text
    assert 'Train (h)' in text
    assert 'How each method is conducted' in text
    for method in ('AR.', 'Euler NODE.', 'Explicit physical OpsSplit.', 'Implicit native-tangent OpsSplit.',
                   'Coefficient physics for OOD.', 'Labels + rollout', 'Rollout only', 'Physics only'):
        assert method in text
    snapshot = builder.load_evidence()
    assert f"{snapshot['parameters']['ensemble']:,}" in text and f"{snapshot['parameters']['implicit']:,}" in text
    assert 'not compute-matched' in text


def test_overflow_preserves_existing_pdf(tmp_path):
    source = tmp_path/'overflow.tex'
    source.write_text((builder.HERE/'results_one_page.tex').read_text().replace(
        r'\end{document}', r'\clearpage A.\clearpage B.\clearpage C.\end{document}'))
    output = tmp_path/'previous.pdf'
    output.write_bytes(b'previous accepted report')
    with pytest.raises(RuntimeError, match='PAGE LIMIT'):
        builder.build(source, output)
    assert output.read_bytes() == b'previous accepted report'


def test_pdf_page_check_works_without_tex_guard(tmp_path):
    source = tmp_path/'three.tex'
    source.write_text(r'\documentclass{article}\begin{document}One\newpage Two\newpage Three\end{document}')
    with pytest.raises(RuntimeError, match='2-PAGE LIMIT'):
        builder.build(source, tmp_path/'three.pdf')
    assert not (tmp_path/'three.pdf').exists()


def test_changed_evidence_is_rejected(tmp_path, monkeypatch):
    snapshot = json.loads((builder.HERE/'results.json').read_text())
    evidence = tmp_path/'evaluation.json'
    evidence.write_text('changed')
    snapshot['evidence_sha256'] = {str(evidence): '0'*64}
    (tmp_path/'results.json').write_text(json.dumps(snapshot))
    monkeypatch.setattr(builder, 'HERE', tmp_path)
    with pytest.raises(ValueError, match='Evidence changed'):
        builder.load_evidence()


def test_incomplete_seeds_are_pending_not_averaged():
    complete = {'complete': True, 'mean': 1.0, 'trajectories': 20, 'expected': 20}
    partial = {'complete': False, 'mean': None, 'trajectories': 3, 'expected': 20}
    snapshot = {'comparison': {'test': {'x/seed17': {'16/1': complete}, 'x/seed18': {'16/1': partial}}}}
    assert builder.seed_values(snapshot, ('test', 'x'), (17, 18, 19), 16) == ([1.0], ['3/20', 'pending'])


def test_training_column_uses_only_finished_runs():
    snapshot = copy.deepcopy(builder.load_evidence())
    rows = {line.split(' & ')[0]: line.split(' & ')[1] for line in builder.table_rows(snapshot).splitlines()}
    assert rows[r'Implicit: physics only ($N\!=\!0$)'] == '0'
    assert rows[r'\quad$+\,D_\perp$ physics: rollout only'] == '--'
    seconds = list(snapshot['training']['node'].values())
    assert rows['Euler NODE'] == f'{sum(seconds)/len(seconds)/3600:.1f}'
    snapshot['training']['implicit_rollout'] = {}
    rows = {line.split(' & ')[0]: line.split(' & ')[1] for line in builder.table_rows(snapshot).splitlines()}
    assert rows['Implicit: rollout only'] == 'run.'


def test_prose_guard_rejects_contradicting_physics_only_result(tmp_path):
    snapshot = copy.deepcopy(builder.load_evidence())
    for cell in snapshot['comparison']['ood_coefficient']['physics/seed17'].values():
        if cell['complete']:
            cell['mean'] *= 10
    with pytest.raises(ValueError, match='contradicts the headline'):
        builder.make_values(tmp_path, snapshot)


def test_headline_requires_every_paired_comparison_to_improve(tmp_path):
    snapshot = copy.deepcopy(builder.load_evidence())
    key = next(k for k, row in snapshot['comparison']['paired'].items()
               if row and k.startswith('test/') and k.endswith('h96'))
    snapshot['comparison']['paired'][key]['left_better'] -= 1
    with pytest.raises(ValueError, match='every paired test trajectory'):
        builder.make_values(tmp_path, snapshot)


@pytest.mark.parametrize('target', [builder.ROOT/'2602.23113v2.pdf',
                                    builder.HERE.parent/'jorek_transfer_one_page/results_one_page.pdf',
                                    builder.HERE/'results_one_page.tex'])
def test_other_reports_and_sources_are_protected(target):
    before = builder.sha256(target)
    with pytest.raises(ValueError, match='protected'):
        builder.build(output=target)
    assert builder.sha256(target) == before
