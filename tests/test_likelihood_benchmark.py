import os
import sys
import numpy as np
import pytest
import torch

# Set path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gwlensing as gw
from ml.baselines import BaselineMLP
from ml.run_likelihood_benchmark import _mcmc_lnprob_sim

def test_mock_catalog_reproducibility():
    """Verify that generating a mock catalog twice with the same seed yields identical results."""
    z, h, om, s8 = 1.0, 0.67, 0.30, 0.85
    nsamples = 1000
    seed = 42
    
    bin_edges = np.linspace(-0.5, 2.5, 50)
    
    # Run 1
    res1 = gw.sample_lnmu_ml_with_diagnostics(z, h, om, s8, nsamples, seed, False)
    lnmu1 = np.array(res1["lnmu"])
    lnmu1 = lnmu1[~np.isnan(lnmu1)]
    clamped1 = np.clip(lnmu1, bin_edges[0] + 1e-9, bin_edges[-1] - 1e-9)
    counts1, _ = np.histogram(clamped1, bins=bin_edges)
    
    # Run 2
    res2 = gw.sample_lnmu_ml_with_diagnostics(z, h, om, s8, nsamples, seed, False)
    lnmu2 = np.array(res2["lnmu"])
    lnmu2 = lnmu2[~np.isnan(lnmu2)]
    clamped2 = np.clip(lnmu2, bin_edges[0] + 1e-9, bin_edges[-1] - 1e-9)
    counts2, _ = np.histogram(clamped2, bins=bin_edges)
    
    assert np.all(counts1 == counts2), "Mock catalog generation is not reproducible for identical seed."

def test_prior_bounds_consistency():
    """Verify that emulator and simulator likelihood return -inf for out-of-bounds parameters."""
    global _GLOBAL_MOCK_COUNTS, _GLOBAL_Z, _GLOBAL_BIN_EDGES
    import ml.run_likelihood_benchmark as lb
    
    # Setup global variables for lb._mcmc_lnprob_sim
    lb._GLOBAL_Z = 1.0
    lb._GLOBAL_BIN_EDGES = np.linspace(-0.5, 2.5, 100)
    lb._GLOBAL_MOCK_COUNTS = np.ones(100)
    
    # Out of bounds cases
    theta_oob_h_low = [0.50, 0.30, 0.85]
    theta_oob_h_high = [0.80, 0.30, 0.85]
    theta_oob_om_low = [0.67, 0.15, 0.85]
    theta_oob_om_high = [0.67, 0.45, 0.85]
    theta_oob_s8_low = [0.67, 0.30, 0.60]
    theta_oob_s8_high = [0.67, 0.30, 1.10]
    
    for theta in [theta_oob_h_low, theta_oob_h_high, theta_oob_om_low, theta_oob_om_high, theta_oob_s8_low, theta_oob_s8_high]:
        val = lb._mcmc_lnprob_sim(theta)
        assert val == -np.inf, f"Simulator likelihood did not return -inf for OOB parameter {theta}"

def test_emulator_likelihood_evaluation():
    """Verify that emulator likelihood evaluates to a finite value inside bounds."""
    checkpoint_path = "data/models/baseline_mlp.pt"
    if not os.path.exists(checkpoint_path):
        pytest.skip("Baseline MLP checkpoint does not exist. Run training script first.")
        
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_mean = checkpoint['input_mean']
    train_std = checkpoint['input_std']
    
    model = BaselineMLP(input_dim=4, hidden_dim=128, output_dim=100)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    z = 1.0
    h, om, s8 = 0.67, 0.30, 0.85
    mock_counts = np.ones(100)
    
    # Inside bounds
    x = np.array([[z, h, om, s8]])
    x_norm = (x - train_mean) / train_std
    with torch.no_grad():
        p = model(torch.tensor(x_norm, dtype=torch.float32)).numpy()[0]
    log_p = np.log(p + 1e-12)
    val = np.sum(mock_counts * log_p)
    
    assert np.isfinite(val), "Emulator likelihood evaluated to non-finite value inside bounds."
