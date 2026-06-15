import os
# Workaround for macOS duplicate OpenMP runtime conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pytest
import torch

from ml.nsf_model import ConditionalNSF

def test_nsf_model_forward_and_shapes():
    # 1. Initialize
    model = ConditionalNSF(input_dim=1, context_dim=4, num_transforms=3, hidden_features=32, bins=8)
    
    # Set mock preprocessing stats
    context_mean = [0.5, 0.6, 0.3, 0.8]
    context_std = [0.2, 0.1, 0.1, 0.1]
    lnmu_mean = 0.05
    lnmu_std = 0.25
    model.set_preprocessing_stats(context_mean, context_std, lnmu_mean, lnmu_std)
    
    # 2. Test log_prob with torch tensors
    x_t = torch.randn(10, 1)
    context_t = torch.randn(10, 4)
    log_prob_t = model.log_prob(x_t, context_t)
    assert log_prob_t.shape == (10,)
    assert log_prob_t.dtype == torch.float32
    assert torch.all(torch.isfinite(log_prob_t))
    
    # 3. Test log_prob with numpy arrays
    x_np = np.random.randn(10, 1)
    context_np = np.random.randn(10, 4)
    log_prob_np = model.log_prob(x_np, context_np)
    assert isinstance(log_prob_np, np.ndarray)
    assert log_prob_np.shape == (10,)
    assert np.all(np.isfinite(log_prob_np))
    
    # 4. Test sample shapes
    samples_t = model.sample(context_t[0], nsamples=100)
    assert samples_t.shape == (100, 1)
    assert samples_t.dtype == torch.float32
    assert torch.all(torch.isfinite(samples_t))
    
    samples_np = model.sample(context_np[0], nsamples=100)
    assert isinstance(samples_np, np.ndarray)
    assert samples_np.shape == (100, 1)
    assert np.all(np.isfinite(samples_np))

def test_nsf_model_density_integration():
    model = ConditionalNSF(input_dim=1, context_dim=4, num_transforms=3, hidden_features=32, bins=8)
    
    # Use standard normal stats so that domain [-5, 5] matches physical [-5, 5]
    model.set_preprocessing_stats([0]*4, [1]*4, 0.0, 1.0)
    
    context = np.array([0.1, 0.2, 0.3, 0.4])
    x_grid = np.linspace(-5.0, 5.0, 1000)
    
    density = model.density_grid(context, x_grid)
    assert density.shape == (1000,)
    assert np.all(density >= 0.0)
    
    # Integrate using trapezoidal rule (NumPy 2.0 compatible)
    dx = x_grid[1] - x_grid[0]
    if hasattr(np, "trapezoid"):
        integral = np.trapezoid(density, dx=dx)
    else:
        integral = np.trapz(density, dx=dx)
    
    # Check that density integrates to approximately 1.0 (splines outside tail bound use linear tails)
    # The integral inside [-5, 5] tail bound should be close to 1.0
    assert pytest.approx(integral, abs=0.05) == 1.0

def test_nsf_model_checkpoint_save_load():
    checkpoint_path = "data/models/test_nsf_checkpoint.pt"
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        
    try:
        model1 = ConditionalNSF(input_dim=1, context_dim=4, num_transforms=3, hidden_features=32, bins=8)
        model1.set_preprocessing_stats([1.0]*4, [2.0]*4, 0.5, 1.5)
        
        # Save
        model1.save_checkpoint(checkpoint_path)
        assert os.path.exists(checkpoint_path)
        
        # Load into another model
        model2 = ConditionalNSF(input_dim=1, context_dim=4, num_transforms=3, hidden_features=32, bins=8)
        model2.load_checkpoint(checkpoint_path)
        
        # Compare statistics buffers
        np.testing.assert_allclose(model1.context_mean.cpu().numpy(), model2.context_mean.cpu().numpy())
        np.testing.assert_allclose(model1.context_std.cpu().numpy(), model2.context_std.cpu().numpy())
        assert float(model1.lnmu_mean) == float(model2.lnmu_mean)
        assert float(model1.lnmu_std) == float(model2.lnmu_std)
        
        # Compare predictions
        x = np.random.randn(5, 1)
        context = np.random.randn(5, 4)
        
        prob1 = model1.log_prob(x, context)
        prob2 = model2.log_prob(x, context)
        
        np.testing.assert_allclose(prob1, prob2, rtol=1e-5)
        
    finally:
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
