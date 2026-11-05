import json

import pytest
import torch

from comparison import digest
from prepare_zscore import fit_zscore, prepare
from test_comparison import fixture
from train_comparison import Experiment


def test_fit_weights_unique_frames_and_cells_not_overlapping_windows():
    values = torch.zeros(2, 5, 2, 2, 2)
    values[0, :, 0] = torch.arange(5)[:, None, None]
    values[1, :, 0] = torch.arange(10, 15)[:, None, None]
    values[:, :, 1] = 7
    values[0, 4] = float('nan')  # Source-invalid/uncovered frame must not enter the fit.
    mean, scale, std, count = fit_zscore(values, [(0, 0), (0, 1), (1, 0)], 2)
    expected = torch.tensor([0, 1, 2, 3, 10, 11, 12], dtype=torch.float64)
    assert count == 28
    torch.testing.assert_close(mean, torch.tensor([expected.mean(), 7.], dtype=torch.float64))
    torch.testing.assert_close(std, torch.tensor([expected.std(correction=0), 0.], dtype=torch.float64))
    assert scale[1] == 1
    with pytest.raises(ValueError, match='Nonfinite'):
        fit_zscore(values, [(0, 2)], 2)
    with pytest.raises(ValueError, match='No eligible'):
        fit_zscore(values, [], 2)


def test_isolated_zscore_data_and_model_use_train_only_statistics(tmp_path):
    source = fixture(tmp_path, case='em', epochs=111)
    values = torch.load(source / 'data.pt', weights_only=False)
    # Make held-out values conspicuously different without changing training data.
    values['val']['exported'].fill_(1e6)
    torch.save(values, source / 'data.pt')
    meta = json.loads((source / 'comparison.json').read_text())
    meta['data_sha256'] = digest(source / 'data.pt')
    (source / 'comparison.json').write_text(json.dumps(meta))
    old_meta = (source / 'comparison.json').read_bytes()
    old_data = digest(source / 'data.pt')
    output = tmp_path / 'zscore'
    details = prepare(source, output)
    assert (source / 'comparison.json').read_bytes() == old_meta
    assert digest(source / 'data.pt') == old_data
    assert (output / 'terms').resolve() == source / 'terms'
    assert (output / 'data.pt').resolve() == source / 'data.pt'
    direct = values['train']['exported'].double()
    expected_mean = direct.mean((0, 1, 3, 4))
    expected_std = direct.std((0, 1, 3, 4), correction=0)
    torch.testing.assert_close(torch.tensor(details['mean'], dtype=torch.float64), expected_mean)
    torch.testing.assert_close(torch.tensor(details['population_std'], dtype=torch.float64), expected_std, atol=1e-14, rtol=1e-7)
    ex = Experiment(output, 'composite', 'cpu')
    assert ex.cfg['composite_stability_epochs'] == 110
    torch.testing.assert_close(ex.mean, expected_mean)
    torch.testing.assert_close(ex.scale, torch.where(expected_std > 0, expected_std, 1.))
    torch.testing.assert_close(ex.model.fluid_mean, ex.mean[:4])
    torch.testing.assert_close(ex.model.fluid_scale, ex.scale[:4])
    x = ex.values['train']['exported'][0, :1].double()
    y = ex.model(x)
    assert y.dtype == torch.float64 and torch.isfinite(y).all() and (y[:, :2] > 0).all()
    loss = ex.objective('train', 0, 0, 2)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in ex.model.parameters() if p.grad is not None)
