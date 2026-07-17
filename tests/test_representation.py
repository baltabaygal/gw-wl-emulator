import numpy as np
import torch
from ml.baselines import BaselineMLP
from ml.run_capacity_study import CustomMLP

def test_custom_mlp_sizes():
    # Small [64, 64]
    model_small = CustomMLP(input_dim=4, hidden_dims=[64, 64], output_dim=100)
    x = torch.randn(3, 4)
    y_small = model_small(x)
    assert y_small.shape == (3, 100)
    assert torch.allclose(torch.sum(y_small, dim=-1), torch.ones(3), atol=1e-6)

    # Large [256, 256, 256, 256]
    model_large = CustomMLP(input_dim=4, hidden_dims=[256, 256, 256, 256], output_dim=100)
    y_large = model_large(x)
    assert y_large.shape == (3, 100)
    assert torch.allclose(torch.sum(y_large, dim=-1), torch.ones(3), atol=1e-6)

def test_pca_svd_math():
    # Create random matrix with shape (10, 100)
    np.random.seed(42)
    Y = np.random.rand(10, 100)
    Y = Y / np.sum(Y, axis=-1, keepdims=True) # Normalize

    C = Y.shape[0]
    mean_pdf = np.mean(Y, axis=0)
    Y_centered = Y - mean_pdf

    # SVD
    U, S, Vt = np.linalg.svd(Y_centered, full_matrices=False)
    
    # Project with 5 components
    n_comp = 5
    V_sub = Vt[:n_comp].T
    scores = np.dot(Y_centered, V_sub)
    
    # Reconstruct
    Y_recon = np.dot(scores, Vt[:n_comp]) + mean_pdf
    
    assert Y_recon.shape == (10, 100)
    # Check that error is reasonable (with 5 components, we should capture substantial variance)
    reconstruction_err = np.mean((Y - Y_recon) ** 2)
    assert reconstruction_err < 0.1
