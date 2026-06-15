import os
import sys
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.optimize import curve_fit

# Set path for gwlensing C++ module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))
import gwlensing as gw

from ml.data import load_histogram_dataset, get_recommended_bin_edges
from ml.baselines import (
    BaselineMLP,
    compute_kl_divergence,
    compute_js_divergence,
    compute_wasserstein_distance
)

def train_model(X_train, Y_train, X_val, Y_val, epochs=500, lr=1e-3, batch_size=32, patience=30):
    # Standardize inputs
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0.0] = 1.0
    
    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std
    
    train_ds = TensorDataset(torch.tensor(X_train_norm, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    model = BaselineMLP(input_dim=4, hidden_dim=128, output_dim=100)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.KLDivLoss(reduction='batchmean')
    
    best_loss = float('inf')
    best_weights = None
    no_improve = 0
    
    val_x_t = torch.tensor(X_val_norm, dtype=torch.float32)
    val_y_t = torch.tensor(Y_val, dtype=torch.float32)
    
    for epoch in range(1, epochs + 1):
        model.train()
        for x_b, y_b in train_loader:
            optimizer.zero_grad()
            pred = model(x_b)
            loss = loss_fn(torch.log(pred + 1e-12), y_b)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            pred_val = model(val_x_t)
            val_loss = loss_fn(torch.log(pred_val + 1e-12), val_y_t).item()
            
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
                
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    model.eval()
    with torch.no_grad():
        preds = model(val_x_t).numpy()
        
    return model, mean, std, preds

# Power law model: metric(N) = A * N^{-alpha} + B
def power_law(N, A, alpha, B):
    return A * (N ** (-alpha)) + B

def run_simulator_pdf(z, h, om, s8, nsamples, seed, bin_edges):
    res = gw.sample_lnmu_ml_with_diagnostics(z, h, om, s8, nsamples, seed, False)
    lnmu = np.array(res["lnmu"])
    lnmu = lnmu[~np.isnan(lnmu)]
    clamped = np.clip(lnmu, bin_edges[0] + 1e-9, bin_edges[-1] - 1e-9)
    counts, _ = np.histogram(clamped, bins=bin_edges)
    return counts / np.sum(counts)

def main() -> None:
    parser = argparse.ArgumentParser(description="Run interpolation ceiling study.")
    parser.add_argument("--dataset_dir", type=str, default="datasets_large_1k")
    parser.add_argument("--output_dir", type=str, default="plots/figures/phase2_interpolation")
    parser.add_argument("--artifact_dir", type=str, default="/Users/baltabay/.gemini/antigravity/brain/f3f9b801-1f10-4591-9e78-0368fe2e4a12")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.artifact_dir, "figures"), exist_ok=True)
    
    bin_edges = get_recommended_bin_edges()
    
    print(f"Loading dataset from {args.dataset_dir}...")
    dataset = load_histogram_dataset(args.dataset_dir, bin_edges)
    
    # Pool all In-Distribution configurations across splits
    all_x = []
    all_y = []
    for split_name, (X, Y, split_types) in dataset.items():
        for i in range(X.shape[0]):
            if split_types[i] in ["train", "interpolation"]:
                all_x.append(X[i])
                all_y.append(Y[i])
                
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    
    print(f"Total pooled In-Distribution configs: {all_x.shape[0]}")
    
    # Shuffle deterministically
    rng = np.random.default_rng(42)
    indices = np.arange(all_x.shape[0])
    rng.shuffle(indices)
    
    all_x = all_x[indices]
    all_y = all_y[indices]
    
    # Hold out 100 configurations for validation
    val_size = 100
    if all_x.shape[0] < val_size + 1000:
        print(f"WARNING: Dataset size ({all_x.shape[0]}) is smaller than 1100. Adjusting validation or train sizes.")
        val_size = min(100, all_x.shape[0] - 1000) if all_x.shape[0] > 1000 else 50
        
    X_val = all_x[:val_size]
    Y_val = all_y[:val_size]
    
    X_pool = all_x[val_size:]
    Y_pool = all_y[val_size:]
    
    print(f"Validation size: {X_val.shape[0]}")
    print(f"Training pool size: {X_pool.shape[0]}")
    
    eval_sizes = [50, 100, 200, 326, 500, 750, 1000]
    eval_sizes = [s for s in eval_sizes if s <= X_pool.shape[0]]
    if X_pool.shape[0] not in eval_sizes and X_pool.shape[0] > 0:
        eval_sizes.append(X_pool.shape[0])
        
    print(f"Evaluating subset sizes: {eval_sizes}")
    
    results = {}
    last_model = None
    last_mean = None
    last_std = None
    
    for N in eval_sizes:
        print(f"\n--- Training baseline MLP with N={N} configurations ---")
        X_train = X_pool[:N]
        Y_train = Y_pool[:N]
        
        model, mean, std, preds = train_model(X_train, Y_train, X_val, Y_val)
        
        if N == max(eval_sizes):
            last_model = model
            last_mean = mean
            last_std = std
            
        kls = compute_kl_divergence(Y_val, preds)
        jsds = compute_js_divergence(Y_val, preds)
        wasses = compute_wasserstein_distance(Y_val, preds, bin_edges)
        
        results[N] = {
            'kl': np.mean(kls),
            'jsd': np.mean(jsds),
            'was': np.mean(wasses)
        }
        print(f"Results N={N} | KL: {results[N]['kl']:.6f} | JSD: {results[N]['jsd']:.6f} | Wasserstein: {results[N]['was']:.6f}")
        
    # Fit curve fitting
    sizes = np.array(sorted(results.keys()))
    jsd_vals = np.array([results[s]['jsd'] for s in sizes])
    was_vals = np.array([results[s]['was'] for s in sizes])
    
    # Scipy curve fit for JSD
    # Initial guess: A=0.05, alpha=0.5, B=0.01
    try:
        popt_jsd, pcov_jsd = curve_fit(power_law, sizes, jsd_vals, p0=[0.05, 0.5, 0.01], bounds=(0, np.inf), maxfev=10000)
        A_jsd, alpha_jsd, B_jsd = popt_jsd
    except Exception as e:
        print(f"JSD curve fit failed: {e}")
        A_jsd, alpha_jsd, B_jsd = np.nan, np.nan, np.nan
        
    try:
        popt_was, pcov_was = curve_fit(power_law, sizes, was_vals, p0=[0.02, 0.5, 0.005], bounds=(0, np.inf), maxfev=10000)
        A_was, alpha_was, B_was = popt_was
    except Exception as e:
        print(f"Wasserstein curve fit failed: {e}")
        A_was, alpha_was, B_was = np.nan, np.nan, np.nan
        
    print(f"\n--- Curve Fitting Results (metric = A * N^-alpha + B) ---")
    print(f"JSD:         A={A_jsd:.6f}, alpha={alpha_jsd:.4f}, B (asymptotic floor)={B_jsd:.6f}")
    print(f"Wasserstein: A={A_was:.6f}, alpha={alpha_was:.4f}, B (asymptotic floor)={B_was:.6f}")
    
    # Plot Learning Curves
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # JSD Plot
    ax1.plot(sizes, jsd_vals, 'o', color='#1f77b4', ms=8, label='Validation JSD')
    if not np.isnan(B_jsd):
        N_dense = np.linspace(min(sizes), max(sizes)*1.5, 200)
        ax1.plot(N_dense, power_law(N_dense, *popt_jsd), '-', color='#1f77b4', lw=2, 
                 label=f'Fit: $A N^{{-\\alpha}} + B$\n$B={B_jsd:.5f}, \\alpha={alpha_jsd:.3f}$')
        ax1.axhline(B_jsd, color='#1f77b4', linestyle='--', alpha=0.7, label=f'Floor B = {B_jsd:.5f}')
    ax1.set_xlabel("Training Set Size (N)")
    ax1.set_ylabel("Average JSD")
    ax1.set_title("JSD scaling & Asymptotic Floor")
    ax1.legend(frameon=True)
    
    # Wasserstein Plot
    ax2.plot(sizes, was_vals, 's', color='#d62728', ms=8, label='Validation EMD')
    if not np.isnan(B_was):
        N_dense = np.linspace(min(sizes), max(sizes)*1.5, 200)
        ax2.plot(N_dense, power_law(N_dense, *popt_was), '-', color='#d62728', lw=2, 
                 label=f'Fit: $A N^{{-\\alpha}} + B$\n$B={B_was:.5f}, \\alpha={alpha_was:.3f}$')
        ax2.axhline(B_was, color='#d62728', linestyle='--', alpha=0.7, label=f'Floor B = {B_was:.5f}')
    ax2.set_xlabel("Training Set Size (N)")
    ax2.set_ylabel("Average Wasserstein Distance")
    ax2.set_title("Wasserstein scaling & Asymptotic Floor")
    ax2.legend(frameon=True)
    
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, "learning_curves.png")
    fig.savefig(plot_path, dpi=200)
    fig.savefig(os.path.join(args.artifact_dir, "figures", "learning_curves.png"), dpi=200)
    plt.close(fig)
    print(f"Learning curve plot saved to {plot_path}")
    
    # Sensitivity-preservation tests
    print(f"\n--- Running Sensitivity-Preservation Tests ---")
    z0, h0, om0, s80 = 1.0, 0.67, 0.30, 0.85
    ref_x = np.array([z0, h0, om0, s80])
    
    # Get true reference PDF
    print("Simulating reference configuration...")
    p_sim_ref = run_simulator_pdf(z0, h0, om0, s80, 10000, 42, bin_edges)
    
    # Get predicted reference PDF
    last_model.eval()
    ref_norm = (ref_x - last_mean) / last_std
    with torch.no_grad():
        p_emu_ref = last_model(torch.tensor(ref_norm, dtype=torch.float32).unsqueeze(0)).numpy()[0]
        
    perturbations = {
        'sigma8': {'param_idx': 3, 'val': s80, 'name': r'$\sigma_8$'},
        'OmegaM': {'param_idx': 2, 'val': om0, 'name': r'$\Omega_M$'},
        'h': {'param_idx': 1, 'val': h0, 'name': r'$h$'}
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    sensitivity_results = {}
    
    for ax_idx, (p_name, p_info) in enumerate(perturbations.items()):
        p_idx = p_info['param_idx']
        val0 = p_info['val']
        p_label = p_info['name']
        
        # +5% perturbation
        val_plus = val0 * 1.05
        x_plus = ref_x.copy()
        x_plus[p_idx] = val_plus
        
        # -5% perturbation
        val_minus = val0 * 0.95
        x_minus = ref_x.copy()
        x_minus[p_idx] = val_minus
        
        # Run simulator
        print(f"Simulating perturbations for {p_name}...")
        p_sim_plus = run_simulator_pdf(x_plus[0], x_plus[1], x_plus[2], x_plus[3], 10000, 42, bin_edges)
        p_sim_minus = run_simulator_pdf(x_minus[0], x_minus[1], x_minus[2], x_minus[3], 10000, 42, bin_edges)
        
        # Run emulator
        x_plus_norm = (x_plus - last_mean) / last_std
        x_minus_norm = (x_minus - last_mean) / last_std
        with torch.no_grad():
            p_emu_plus = last_model(torch.tensor(x_plus_norm, dtype=torch.float32).unsqueeze(0)).numpy()[0]
            p_emu_minus = last_model(torch.tensor(x_minus_norm, dtype=torch.float32).unsqueeze(0)).numpy()[0]
            
        # Compute difference profiles
        dp_sim_plus = p_sim_plus - p_sim_ref
        dp_sim_minus = p_sim_minus - p_sim_ref
        
        dp_emu_plus = p_emu_plus - p_emu_ref
        dp_emu_minus = p_emu_minus - p_emu_ref
        
        # Compute L2 norm and relative error
        norm_sim_plus = np.linalg.norm(dp_sim_plus)
        norm_sim_minus = np.linalg.norm(dp_sim_minus)
        
        err_plus = np.linalg.norm(dp_emu_plus - dp_sim_plus) / (norm_sim_plus + 1e-12)
        err_minus = np.linalg.norm(dp_emu_minus - dp_sim_minus) / (norm_sim_minus + 1e-12)
        
        sensitivity_results[p_name] = {
            'err_plus': err_plus,
            'err_minus': err_minus
        }
        
        print(f"{p_name} Sensitivity RelError: +5% = {err_plus:.4f} | -5% = {err_minus:.4f}")
        
        # Plot difference profiles
        ax = axes[ax_idx]
        ax.plot(bin_centers, dp_sim_plus, color='#1f77b4', lw=2, label=f'Sim +5% ({val_plus:.3f})')
        ax.plot(bin_centers, dp_emu_plus, color='#1f77b4', lw=2.5, linestyle='--', label=f'Emu +5% (error: {err_plus:.2%})')
        
        ax.plot(bin_centers, dp_sim_minus, color='#d62728', lw=2, label=f'Sim -5% ({val_minus:.3f})')
        ax.plot(bin_centers, dp_emu_minus, color='#d62728', lw=2.5, linestyle='--', label=f'Emu -5% (error: {err_minus:.2%})')
        
        ax.set_xlabel(r"$\ln\mu$")
        ax.set_ylabel(r"$\Delta$ Probability")
        ax.set_title(f"Sensitivity: {p_label} Perturbation")
        ax.set_xlim(-0.3, 1.0)
        ax.legend(frameon=True)
        
    plt.tight_layout()
    sens_path = os.path.join(args.output_dir, "sensitivity_test.png")
    fig.savefig(sens_path, dpi=200)
    fig.savefig(os.path.join(args.artifact_dir, "figures", "sensitivity_test.png"), dpi=200)
    plt.close(fig)
    print(f"Sensitivity plot saved to {sens_path}")
    
    # Save statistics for the report
    stats_file = os.path.join(args.artifact_dir, "interpolation_ceiling_stats.json")
    import json
    stats = {
        'sizes': sizes.tolist(),
        'jsd_vals': jsd_vals.tolist(),
        'was_vals': was_vals.tolist(),
        'fit_jsd': {
            'A': float(A_jsd),
            'alpha': float(alpha_jsd),
            'B': float(B_jsd)
        },
        'fit_was': {
            'A': float(A_was),
            'alpha': float(alpha_was),
            'B': float(B_was)
        },
        'sensitivity': {
            k: {
                'err_plus': float(v['err_plus']),
                'err_minus': float(v['err_minus'])
            } for k, v in sensitivity_results.items()
        }
    }
    with open(stats_file, 'w') as sf:
        json.dump(stats, sf, indent=2)
    print(f"Metrics stats saved to {stats_file}")

if __name__ == "__main__":
    main()
