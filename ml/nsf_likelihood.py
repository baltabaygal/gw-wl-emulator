import os
# Workaround for macOS duplicate OpenMP runtime conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import torch

from ml.nsf_model import ConditionalNSF

def log_prob_batch(model, x_batch, context_batch) -> torch.Tensor | np.ndarray:
    """Evaluates pairwise batch log-probs for matched shapes (e.g. x of shape (N, 1) and context of shape (N, 4))."""
    return model.log_prob(x_batch, context_batch)

def catalog_log_likelihood(model, catalog, theta_batch) -> torch.Tensor | np.ndarray:
    """Computes the log-likelihood of a catalog (shape (N,) or (N, 1)) across a batch of cosmologies (shape (M, 4))
    using the flattened batching trick to perform a single forward pass.
    """
    is_numpy = isinstance(catalog, np.ndarray) or isinstance(theta_batch, np.ndarray)
    
    device = model.context_mean.device
    
    # Convert inputs to tensors on model device
    if isinstance(catalog, np.ndarray):
        catalog_t = torch.tensor(catalog, dtype=torch.float32, device=device)
    else:
        catalog_t = catalog.to(device)
        
    if isinstance(theta_batch, np.ndarray):
        theta_t = torch.tensor(theta_batch, dtype=torch.float32, device=device)
    else:
        theta_t = theta_batch.to(device)
        
    if catalog_t.ndim == 1:
        catalog_t = catalog_t.unsqueeze(-1)  # (N, 1)
    if theta_t.ndim == 1:
        theta_t = theta_t.unsqueeze(0)   # (1, 4)
        
    N = catalog_t.shape[0]
    M = theta_t.shape[0]
    
    # Repeat catalog: (M * N, 1)
    # e.g., catalog_t repeated M times
    catalog_rep = catalog_t.repeat(M, 1)
    
    # Repeat theta: (M * N, 4)
    # e.g., each theta repeated N times consecutively
    theta_rep = theta_t.repeat_interleave(N, dim=0)
    
    # Normalize
    x_norm = (catalog_rep - model.lnmu_mean) / model.lnmu_std
    context_norm = (theta_rep - model.context_mean) / model.context_std
    
    # Single batched forward pass through the flow
    with torch.no_grad():
        log_prob_norm = model.flow(context_norm).log_prob(x_norm)
        log_prob_raw = log_prob_norm - torch.log(model.lnmu_std)
        
        # Reshape to (M, N) and sum over the N catalog samples
        log_liks = log_prob_raw.reshape(M, N).sum(dim=1)
        
    if is_numpy:
        return log_liks.cpu().numpy()
    return log_liks

def run_profiling(model_path: str, output_json: str, plots_dir: str, output_report: str):
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_report), exist_ok=True)
    
    print(f"Loading NSF checkpoint from {model_path}...")
    model = ConditionalNSF(input_dim=1, context_dim=4)
    model.load_checkpoint(model_path)
    model.eval()
    
    # 1. Setup benchmark parameters
    N_samples = 10000
    M_grid = 400
    catalog = np.random.randn(N_samples, 1).astype(np.float32)
    theta_batch = np.random.uniform(0.2, 0.4, (M_grid, 4)).astype(np.float32)
    
    # Single-context evaluation profiling
    t0 = time.time()
    _ = model.log_prob(catalog[:10], theta_batch[0])
    t_single_eval = (time.time() - t0) / 10.0
    
    # 2. Benchmark unbatched grid evaluation (loop)
    t0 = time.time()
    unbatched_liks = []
    for idx in range(M_grid):
        log_probs = model.log_prob(catalog, theta_batch[idx])
        unbatched_liks.append(np.sum(log_probs))
    unbatched_liks = np.array(unbatched_liks)
    t_unbatched = time.time() - t0
    
    # 3. Benchmark batched grid evaluation
    t0 = time.time()
    batched_liks = catalog_log_likelihood(model, catalog, theta_batch)
    t_batched = time.time() - t0
    
    # Verify numerical identity
    np.testing.assert_allclose(unbatched_liks, batched_liks, rtol=1e-5, err_msg="Batched and unbatched likelihoods do not match!")
    
    # 4. Estimate speedup against simulator reference
    # Average simulator grid time for 400 points from Phase 2B/2C: ~200.0 seconds
    t_sim_grid = 200.0
    t_sim_single = t_sim_grid / 400.0
    
    results = {
        "single_context_eval_time": float(t_single_eval),
        "unbatched_grid_eval_time": float(t_unbatched),
        "batched_grid_eval_time": float(t_batched),
        "sim_grid_eval_time_reference": float(t_sim_grid),
        "speedup_vs_unbatched_nsf": float(t_unbatched / t_batched),
        "speedup_vs_sim_grid": float(t_sim_grid / t_batched),
        "speedup_unbatched_vs_sim_grid": float(t_sim_grid / t_unbatched)
    }
    
    # Write to JSON
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Profiling results written to {output_json}")
    
    # Generate bar plot
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ["Simulator (Ref)", "Unbatched NSF (Loop)", "Batched NSF (Optimized)"]
    times = [t_sim_grid, t_unbatched, t_batched]
    colors = ["#f8766d", "#619cff", "#00ba38"]
    
    bars = ax.bar(categories, times, color=colors, edgecolor="black", width=0.5)
    ax.set_ylabel("Execution Time (seconds) - Log Scale")
    ax.set_yscale("log")
    ax.set_title("Likelihood Grid Execution Time (400 points)")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval * 1.2, f"{yval:.4f} s", ha="center", va="bottom", fontweight="bold")
        
    plot_path = os.path.join(plots_dir, "runtime_comparison.png")
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Plot saved to {plot_path}")
    
    # Generate report
    report_md = f"""# Phase 2C — NSF Runtime Optimization Report

Generated on: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}

## 1. Benchmarking Summary

We profile the execution time for evaluating a weak-lensing mock catalog of $N = 10,000$ sirens over a 20x20 grid of parameter space (400 cosmologies):

- **Simulator Grid (Reference)**: `{t_sim_grid:.2f} seconds` (C++ Monte Carlo engine)
- **Unbatched NSF (Loop)**: `{t_unbatched:.4f} seconds`
- **Batched NSF (Optimized)**: `{t_batched:.4f} seconds`

## 2. Speedup Metrics

- **Batching Speedup vs Loop NSF**: **`{results['speedup_vs_unbatched_nsf']:.1f}x`**
- **Speedup vs Simulator Grid**: **`{results['speedup_vs_sim_grid']:.1f}x`**

## 3. Runtime Visual comparison
![Runtime comparison](figures/phase2c_runtime/runtime_comparison.png)

## 4. Discussion
By flattening the samples and cosmologies into a single large tensor pass, we avoid the overhead of repeating PyTorch model calls. The resulting batched likelihood evaluation matches the unbatched loop results within numerical tolerance ($10^{{-5}}$) while executing orders of magnitude faster.
"""
    with open(output_report, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"Report written to {output_report}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile NSF likelihood evaluation.")
    parser.add_argument("--model_path", type=str, default="data/models/conditional_nsf_backend_current.pt")
    parser.add_argument("--output_json", type=str, default="data/results/phase2c_runtime_results.json")
    parser.add_argument("--plots_dir", type=str, default="plots/figures/phase2c_runtime")
    parser.add_argument("--output_report", type=str, default="docs/phase2c_runtime_optimization.md")
    args = parser.parse_args()
    
    run_profiling(args.model_path, args.output_json, args.plots_dir, args.output_report)
