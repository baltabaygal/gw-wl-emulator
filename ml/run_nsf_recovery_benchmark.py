import os
# Workaround for macOS duplicate OpenMP runtime conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import argparse
import time
import json
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from multiprocessing import Pool
import multiprocessing
import torch

# Set path for gwlensing C++ module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))
import gwlensing as gw

from ml.baselines import BaselineMLP
from ml.nsf_model import ConditionalNSF

# Global variables for worker processes
_NSF_MODEL = None
_MLP_MODEL = None
_MLP_MEAN = None
_MLP_STD = None
_BIN_EDGES = None

def init_worker(nsf_path, mlp_path):
    global _NSF_MODEL, _MLP_MODEL, _MLP_MEAN, _MLP_STD, _BIN_EDGES
    
    # Load NSF
    _NSF_MODEL = ConditionalNSF(input_dim=1, context_dim=4)
    _NSF_MODEL.load_checkpoint(nsf_path)
    _NSF_MODEL.eval()
    
    # Load MLP
    checkpoint = torch.load(mlp_path, map_location="cpu", weights_only=False)
    _BIN_EDGES = checkpoint['bin_edges']
    _MLP_MEAN = checkpoint['input_mean']
    _MLP_STD = checkpoint['input_std']
    _MLP_MODEL = BaselineMLP(input_dim=4, hidden_dim=128, output_dim=100)
    _MLP_MODEL.load_state_dict(checkpoint['model_state_dict'])
    _MLP_MODEL.eval()

def recovery_worker(args_tuple):
    idx, z_val, h_true, om_true, s8_true = args_tuple
    
    # Generate mock catalog
    mock_seed = idx + 2026
    mock_res = gw.sample_lnmu_ml_with_diagnostics(z_val, h_true, om_true, s8_true, 10000, mock_seed, False)
    lnmu = np.array(mock_res["lnmu"])
    lnmu_mock = lnmu[~np.isnan(lnmu)]
    
    # Clamp and bin for binned methods
    clamped = np.clip(lnmu_mock, _BIN_EDGES[0] + 1e-9, _BIN_EDGES[-1] - 1e-9)
    mock_counts, _ = np.histogram(clamped, bins=_BIN_EDGES)
    
    theta_true = np.array([h_true, om_true, s8_true])
    theta_init = np.array([0.67, 0.30, 0.85])  # midpoint init
    
    # 1. Simulator Recovery (Binned)
    def obj_sim(theta):
        h, om, s8 = theta
        if not (0.59 <= h <= 0.76 and 0.20 <= om <= 0.40 and 0.65 <= s8 <= 1.05):
            return 1e10
        res = gw.sample_lnmu_ml_with_diagnostics(z_val, h, om, s8, 10000, 100, False)
        lnmu_s = np.array(res["lnmu"])
        lnmu_s = lnmu_s[~np.isnan(lnmu_s)]
        clamped_s = np.clip(lnmu_s, _BIN_EDGES[0] + 1e-9, _BIN_EDGES[-1] - 1e-9)
        counts_s, _ = np.histogram(clamped_s, bins=_BIN_EDGES)
        sum_c = np.sum(counts_s)
        if sum_c > 0:
            p = counts_s / sum_c
        else:
            p = np.ones(len(_BIN_EDGES) - 1) / (len(_BIN_EDGES) - 1)
        log_p = np.log(p + 1e-12)
        return -float(np.sum(mock_counts * log_p))
        
    t0 = time.time()
    res_sim = minimize(obj_sim, theta_init, method='Nelder-Mead', options={'xatol': 1e-3, 'fatol': 1e-3})
    sim_time = time.time() - t0
    
    # 2. MLP Recovery (Binned)
    def obj_mlp(theta):
        h, om, s8 = theta
        if not (0.59 <= h <= 0.76 and 0.20 <= om <= 0.40 and 0.65 <= s8 <= 1.05):
            return 1e10
        x = np.array([[z_val, h, om, s8]])
        x_norm = (x - _MLP_MEAN) / _MLP_STD
        with torch.no_grad():
            p = _MLP_MODEL(torch.tensor(x_norm, dtype=torch.float32)).numpy()[0]
        log_p = np.log(p + 1e-12)
        return -float(np.sum(mock_counts * log_p))
        
    t0 = time.time()
    res_mlp = minimize(obj_mlp, theta_init, method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-4})
    mlp_time = time.time() - t0
    
    # 3. NSF Recovery (Continuous NLL)
    def obj_nsf(theta):
        h, om, s8 = theta
        if not (0.59 <= h <= 0.76 and 0.20 <= om <= 0.40 and 0.65 <= s8 <= 1.05):
            return 1e10
        context = np.array([[z_val, h, om, s8]], dtype=np.float32)
        log_probs = _NSF_MODEL.log_prob(lnmu_mock, context)
        return -float(np.sum(log_probs))
        
    t0 = time.time()
    res_nsf = minimize(obj_nsf, theta_init, method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-4})
    nsf_time = time.time() - t0
    
    return {
        'idx': idx,
        'true': theta_true.tolist(),
        'sim_fit': res_sim.x.tolist(),
        'sim_time': sim_time,
        'mlp_fit': res_mlp.x.tolist(),
        'mlp_time': mlp_time,
        'nsf_fit': res_nsf.x.tolist(),
        'nsf_time': nsf_time
    }

def main():
    parser = argparse.ArgumentParser(description="Perform Likelihood Replacement Inference Benchmark for NSF.")
    parser.add_argument("--nsf_model_path", type=str, default="data/models/conditional_nsf_backend_current.pt")
    parser.add_argument("--mlp_model_path", type=str, default="data/models/baseline_mlp_backend_current.pt")
    parser.add_argument("--output_json", type=str, default="data/results/phase2b_nsf_recovery_results.json")
    parser.add_argument("--plots_dir", type=str, default="plots/figures/phase2b_nsf_recovery")
    parser.add_argument("--output_report", type=str, default="docs/phase2b_nsf_recovery_benchmark.md")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    
    # Generate 20 random cosmologies within the In-Distribution bounds (reproducible seed)
    cosmo_rng = np.random.default_rng(20260615)
    z_vals = cosmo_rng.uniform(0.5, 2.5, 20)
    h_vals = cosmo_rng.uniform(0.59, 0.76, 20)
    om_vals = cosmo_rng.uniform(0.20, 0.40, 20)
    s8_vals = cosmo_rng.uniform(0.65, 1.05, 20)
    
    targets = []
    for idx in range(20):
        targets.append((idx, z_vals[idx], h_vals[idx], om_vals[idx], s8_vals[idx]))
        
    num_processes = min(multiprocessing.cpu_count(), 10)
    print(f"Using {num_processes} parallel processes for parameter recovery fits.")
    
    t0 = time.time()
    with Pool(processes=num_processes, initializer=init_worker, initargs=(args.nsf_model_path, args.mlp_model_path)) as pool:
        fit_results = pool.map(recovery_worker, targets)
    total_time = time.time() - t0
    print(f"All recovery fits completed in {total_time:.2f} s")
    
    # Save results to JSON
    with open(args.output_json, "w") as f:
        json.dump(fit_results, f, indent=2)
        
    # Analyze results
    true_vals = np.array([r['true'] for r in fit_results])
    sim_fits = np.array([r['sim_fit'] for r in fit_results])
    mlp_fits = np.array([r['mlp_fit'] for r in fit_results])
    nsf_fits = np.array([r['nsf_fit'] for r in fit_results])
    
    sim_times = [r['sim_time'] for r in fit_results]
    mlp_times = [r['mlp_time'] for r in fit_results]
    nsf_times = [r['nsf_time'] for r in fit_results]
    
    sim_bias = sim_fits - true_vals
    mlp_bias = mlp_fits - true_vals
    nsf_bias = nsf_fits - true_vals
    
    # Print metrics
    param_names = ['h', 'OmegaM', 'sigma8']
    
    report_lines = [
        "# Phase 2B — NSF Parameter Recovery Benchmark Report",
        "",
        f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        "",
        "## 1. Bias Statistics Comparison",
        "",
        "| Parameter | Method | Mean Bias &plusmn; Std Dev | Max Absolute Deviation |",
        "|---|---|---|---|",
    ]
    
    for i, name in enumerate(param_names):
        # Sim
        report_lines.append(f"| **{name}** | Simulator | {np.mean(sim_bias[:, i]):+.6f} &plusmn; {np.std(sim_bias[:, i]):.6f} | {np.max(np.abs(sim_bias[:, i])):.6f} |")
        # MLP
        report_lines.append(f"| **{name}** | Baseline MLP | {np.mean(mlp_bias[:, i]):+.6f} &plusmn; {np.std(mlp_bias[:, i]):.6f} | {np.max(np.abs(mlp_bias[:, i])):.6f} |")
        # NSF
        report_lines.append(f"| **{name}** | **Conditional NSF** | **{np.mean(nsf_bias[:, i]):+.6f} &plusmn; {np.std(nsf_bias[:, i]):.6f}** | **{np.max(np.abs(nsf_bias[:, i])):.6f}** |")
        report_lines.append("|---|---|---|---|")

    # Speedups
    avg_sim = np.mean(sim_times)
    avg_mlp = np.mean(mlp_times)
    avg_nsf = np.mean(nsf_times)
    
    report_lines.extend([
        "",
        "## 2. Optimization Speedup",
        "",
        f"- **Average Simulator Time**: {avg_sim:.4f} seconds",
        f"- **Average MLP Time**: {avg_mlp:.4f} seconds (Speedup: **{avg_sim / avg_mlp:.1f}x**)",
        f"- **Average NSF Time**: {avg_nsf:.4f} seconds (Speedup: **{avg_sim / avg_nsf:.1f}x**)",
        ""
    ])
    
    # Plot biases
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for i, name in enumerate(param_names):
        ax = axes[i]
        x_indices = np.arange(20)
        ax.errorbar(x_indices, mlp_bias[:, i], fmt='o', color='#2ca02c', label='MLP Bias')
        ax.errorbar(x_indices, nsf_bias[:, i], fmt='x', color='#1f77b4', label='NSF Bias')
        ax.axhline(0, color='black', ls='--', alpha=0.5)
        ax.set_title(f"Parameter: {name}")
        ax.set_xlabel("Cosmology Index")
        ax.set_ylabel("Recovery Bias")
        ax.legend()
        ax.grid(alpha=0.3)
        
    plt.tight_layout()
    plot_path = os.path.join(args.plots_dir, "parameter_recovery_biases.png")
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Recovery biases plot saved to {plot_path}")
    
    report_lines.extend([
        "## 3. Recovery Visualization",
        "",
        "![Parameter Recovery Biases](figures/phase2b_nsf_recovery/parameter_recovery_biases.png)"
    ])
    
    with open(args.output_report, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"Benchmark report written to {args.output_report}")

if __name__ == "__main__":
    main()
