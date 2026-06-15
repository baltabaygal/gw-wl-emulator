import os
import json
import pytest

def test_benchmark_results_structure():
    path = "data/results/backend_current_benchmark_results.json"
    assert os.path.exists(path), f"Benchmark results file not found: {path}"
    
    with open(path, 'r') as f:
        data = json.load(f)
        
    assert "recovery" in data
    assert "statistics" in data
    assert "mcmc" in data
    
    stats = data["statistics"]
    assert "emu" in stats
    assert "sim" in stats
    assert "speedup_ratio" in stats
    
    mcmc = data["mcmc"]
    assert "jsd" in mcmc
    assert "h" in mcmc["jsd"]
    assert "OmegaM" in mcmc["jsd"]
    assert "sigma8" in mcmc["jsd"]

def test_validation_plots_and_reports():
    # Reports
    assert os.path.exists("docs/backend_current_recovery_benchmark.md")
    
    # Figures
    fig_dir = "plots/figures/phase2_backend_current_training"
    assert os.path.exists(os.path.join(fig_dir, "parameter_bias.png"))
    assert os.path.exists(os.path.join(fig_dir, "posterior_corner.png"))
    assert os.path.exists(os.path.join(fig_dir, "runtime_speedup.png"))

def test_likelihood_surface_validation_outputs():
    # Reports
    assert os.path.exists("docs/backend_current_posterior_validation.md")
    
    # Figures for z=0.5, 1.5, 2.5
    fig_dir = "plots/figures/phase2_backend_current_training"
    for z_suffix in ["5", "15", "25"]:
        assert os.path.exists(os.path.join(fig_dir, f"likelihood_grid_z{z_suffix}.png")), f"Missing likelihood grid plot for suffix {z_suffix}"
