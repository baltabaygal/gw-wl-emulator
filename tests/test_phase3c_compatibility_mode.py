import numpy as np
import pytest
import torch

from ml.phase3_common import load_nsf_model
from ml.nsf_likelihood import nsf_mixed_catalog_log_likelihood, load_bin_edges

def test_phase3c_compatibility_mode_evaluation():
    model = load_nsf_model()
    model.eval()
    
    # 1. Create a dummy catalog and grid point
    z = np.array([1.0, 1.0, 1.5, 2.0])
    lnmu = np.array([0.0, -0.5, 0.2, 0.9])
    theta_grid = np.array([[0.67, 0.30, 0.85]])
    
    # 2. Evaluate in both modes
    ll_continuous = nsf_mixed_catalog_log_likelihood(
        model, z, lnmu, theta_grid, likelihood_mode="continuous"
    )
    ll_compatible = nsf_mixed_catalog_log_likelihood(
        model, z, lnmu, theta_grid, likelihood_mode="simulator_compatible"
    )
    
    # 3. Assert results are finite and arrays of shape (1,)
    assert np.isfinite(ll_continuous).all()
    assert np.isfinite(ll_compatible).all()
    assert ll_continuous.shape == (1,)
    assert ll_compatible.shape == (1,)
    
    # 4. Check that incorrect mode raises error
    with pytest.raises(ValueError, match="Unknown likelihood_mode"):
        nsf_mixed_catalog_log_likelihood(
            model, z, lnmu, theta_grid, likelihood_mode="invalid_mode"
        )

def test_phase3c_compatibility_mode_nan_inf_rejection():
    model = load_nsf_model()
    model.eval()
    
    # Passing NaN to continuous mode should trigger ValueError (from flow or isfinite checks)
    z = np.array([1.0, np.nan])
    lnmu = np.array([0.0, 0.0])
    theta_grid = np.array([[0.67, 0.30, 0.85]])
    
    with pytest.raises(ValueError):
        nsf_mixed_catalog_log_likelihood(
            model, z, lnmu, theta_grid, likelihood_mode="continuous"
        )
        
    with pytest.raises(ValueError):
        nsf_mixed_catalog_log_likelihood(
            model, z, lnmu, theta_grid, likelihood_mode="simulator_compatible"
        )
