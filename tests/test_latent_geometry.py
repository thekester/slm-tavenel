import importlib.util

import torch

spec = importlib.util.spec_from_file_location("geom", "scripts/latent_geometry.py")
geom = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geom)


def test_two_nn_recovers_dimension_of_a_linear_subspace():
    torch.manual_seed(0)
    basis = torch.linalg.qr(torch.randn(64, 5)).Q  # 5-dimensional subspace of R^64
    x = torch.randn(3000, 5) @ basis.T
    assert 4.0 < geom.two_nn(x) < 6.0
    assert 3.5 < geom.participation_ratio(x) < 5.5
