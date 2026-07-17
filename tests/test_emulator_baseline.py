import numpy as np
import pytest
import torch

from ml.data import get_recommended_bin_edges, load_histogram_dataset
from ml.baselines import (
    BaselineMLP,
    compute_binned_moments,
    compute_kl_divergence,
    compute_js_divergence,
    compute_wasserstein_distance
)

def test_recommended_bin_edges():
    edges = get_recommended_bin_edges()
    assert len(edges) == 101
    assert edges[0] == -0.5
    assert edges[-1] == 2.5
    assert np.all(np.diff(edges) > 0)

def test_mlp_emulator_forward():
    model = BaselineMLP(input_dim=4, hidden_dim=64, output_dim=100)
    # Batch size of 5
    x = torch.randn(5, 4)
    y = model(x)
    
    assert y.shape == (5, 100)
    # Check that outputs sum to 1.0 (probabilities) and are non-negative
    sums = torch.sum(y, dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)
    assert torch.all(y >= 0.0)

def test_metrics_self_identity():
    # Trivial cases: distance/divergence to self should be 0
    p = np.array([
        [0.1] * 10,
        [0.0] * 5 + [0.2] * 5
    ])
    
    kl = compute_kl_divergence(p, p)
    jsd = compute_js_divergence(p, p)
    
    assert np.allclose(kl, np.zeros_like(kl), atol=1e-10)
    assert np.allclose(jsd, np.zeros_like(jsd), atol=1e-10)
    
    bin_edges = np.linspace(0.0, 1.0, 11)
    was = compute_wasserstein_distance(p, p, bin_edges)
    assert np.allclose(was, np.zeros_like(was), atol=1e-10)

def test_binned_moments():
    # Symmetric distribution around 0.5
    p = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    bin_centers = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    
    mean, var, skew, kurt = compute_binned_moments(p, bin_centers)
    
    # Expected mean: 0.5
    assert pytest.approx(mean) == 0.5
    # Variance: sum p_i (x_i - 0.5)^2 = 0.1*0.16 + 0.2*0.04 + 0.4*0.0 + 0.2*0.04 + 0.1*0.16 = 0.016 + 0.008 + 0.0 + 0.008 + 0.016 = 0.048
    assert pytest.approx(var) == 0.048
    # Symmetric -> skewness should be 0.0
    assert pytest.approx(skew, abs=1e-7) == 0.0
    # Kurtosis: sum p_i (x_i - 0.5)^4 / var^2 - 3 = (0.1*0.0256 + 0.2*0.0016 + 0.4*0 + 0.2*0.0016 + 0.1*0.0256) / 0.048^2 - 3
    # = (0.00256 + 0.00032 + 0.00032 + 0.00256) / 0.002304 - 3 = 0.00576 / 0.002304 - 3 = 2.5 - 3 = -0.5
    assert pytest.approx(kurt) == -0.5
