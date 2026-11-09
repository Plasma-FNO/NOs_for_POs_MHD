from pathlib import Path

import pytest
import torch
import yaml

from mastu_opsplit.learning import build_family, dependency_provenance, parameter_counts


def configuration():
    config = yaml.safe_load((Path(__file__).parents[1] / "configs/lsplinefno.yaml").read_text())
    config["architecture"].update(grid_shape=[8, 8], modes=2, width=4, n_layers=1)
    return config


@pytest.fixture(autouse=True)
def reproducible():
    torch.manual_seed(7)
    torch.set_num_threads(1)


def fixture():
    coords = torch.tensor([[1., -1.], [2., 1.], [1.2, -.6], [1.6, .3], [1.8, -.1], [1.35, .7]])
    return torch.randn(2, len(coords), 2), coords, torch.tensor([[.5], [1.5]])


def test_field_sensitivity_order_batch_query_and_fixed_bounds():
    model = build_family("whole_rhs", 2, [[1., 2.], [-1., 1.]], configuration()).eval()
    fields, coords, parameters = fixture()
    output = model(fields, coords, parameters)
    assert output.shape == fields.shape
    assert not torch.allclose(output, model(fields + 1, coords, parameters))
    order = torch.tensor([3, 0, 5, 2, 1, 4])
    assert torch.allclose(output[:, order], model(fields[:, order], coords[order], parameters), atol=2e-6)
    assert torch.allclose(output[1:], model(fields[1:], coords, parameters[1:]), atol=2e-6)
    query = coords[[2, 4, 1]]
    assert torch.allclose(output[:, [2, 4, 1]], model(fields, coords, parameters, query), atol=2e-6)
    assert torch.allclose(model.normalized_coordinates(torch.tensor([[1.5, 0.]]), 2), torch.full((2, 2, 1), .5))
    with pytest.raises(ValueError, match="fixed physical bounds"):
        model(fields, coords, parameters, torch.tensor([[2.1, 0.]]))
    with pytest.raises(ValueError, match="nonempty"):
        model(fields, coords, parameters, torch.empty(0, 2))
    with pytest.raises(ValueError, match="parameters"):
        model(fields, coords)
    with pytest.raises(ValueError, match="positive"):
        model(fields, coords, -parameters)


def test_film_plumbing_coordinates_unchanged_and_all_parameter_gradients():
    model = build_family("whole_rhs", 2, [[1., 2.], [-1., 1.]], configuration())
    fields, coords, parameters = fixture()
    fields.requires_grad_()
    with torch.no_grad():
        model.film.gamma_head.weight.fill_(.05)
    a = model(fields, coords, parameters)
    b = model(fields, coords, parameters + .5)
    assert not torch.allclose(a, b)
    a.square().mean().backward()
    assert torch.isfinite(fields.grad).all() and fields.grad.abs().sum() > 0
    groups = [model.film.parameters(), model.backbone.fno_blocks.parameters()]
    groups += [kernel.parameters() for kernel in model.backbone.scatter_kernels]
    for group in groups:
        gradients = [p.grad for p in group if p.requires_grad]
        assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum() for g in gradients) > 0
    epsilon = 1e-3
    high, low = fields.detach().clone(), fields.detach().clone()
    high[0, 2, 0] += epsilon
    low[0, 2, 0] -= epsilon
    finite_difference = ((model(high, coords, parameters).square().mean() -
                          model(low, coords, parameters).square().mean()) / (2 * epsilon))
    assert torch.allclose(finite_difference, fields.grad[0, 2, 0], atol=3e-5, rtol=.02)
    for kernel in model.backbone.scatter_kernels:
        parameter = kernel.coeffs
        index = 0
        original = parameter[index].item()
        with torch.no_grad():
            parameter[index] = original + epsilon
            high_loss = model(fields.detach(), coords, parameters).square().mean()
            parameter[index] = original - epsilon
            low_loss = model(fields.detach(), coords, parameters).square().mean()
            parameter[index] = original
        finite_difference = (high_loss - low_loss) / (2 * epsilon)
        assert torch.allclose(finite_difference, parameter.grad[index], atol=3e-5, rtol=.03)


def test_opsplit_independence_and_sparse_coverage_diagnostics():
    model = build_family("opsplit", 2, [[1., 2.], [-1., 1.]], configuration())
    assert not ({id(p) for p in model.density.parameters()} & {id(p) for p in model.pressure.parameters()})
    fields, coords, _ = fixture()
    out = model(fields, coords)
    assert out.shape == (2, len(coords), 2)
    out.square().mean().backward()
    for operator in (model.density, model.pressure):
        for kernel in operator.backbone.scatter_kernels:
            assert kernel.coeffs.grad is not None and kernel.coeffs.grad.abs().sum() > 0
        report = operator.safeguard_diagnostics(coords)
        assert report["uncovered_nodes"] > 0
        assert report["pou_floor_rows"] == 0
        assert report["signed_cancellation_floor_nodes"] == 0
        assert report["covered_density_floor_nodes"] == report["covered_absolute_epsilon_nodes"]
        assert torch.isfinite(out).all()
    assert parameter_counts(model)["total"] == sum(p.numel() for p in model.parameters())


def test_dependency_hash_mismatch_rejected():
    config = configuration()
    config["dependency"]["sha256"] = {"Models/learnable_spline_fno.py": "wrong"}
    with pytest.raises(ValueError, match="source hash mismatch"):
        dependency_provenance(config["dependency"])


def test_float64_mesh_boundary_precision_is_preserved():
    bounds = [[1.1234567890123, 2.2345678901234], [-1.3456789012345, .4567890123456]]
    model = build_family("whole_rhs", 2, bounds, configuration())
    coords = torch.tensor([[bounds[0][0], bounds[1][0]],
                           [bounds[0][1], bounds[1][1]]], dtype=torch.float64)
    fields = torch.randn(1, 2, 2)
    parameters = torch.ones(1, 1)
    normalized = model.normalized_coordinates(coords, 1)
    assert torch.equal(normalized, torch.tensor([[[0., 1.], [0., 1.]]], dtype=torch.float64))
    assert torch.isfinite(model(fields, coords, parameters)).all()
    model.double()
    assert torch.equal(model.normalized_coordinates(coords, 1), normalized)
    model.float()
    assert torch.isfinite(model(fields.float(), coords.float(), parameters.float())).all()
