import os
import sys
import argparse
import time
import json
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.optimize import minimize
from multiprocessing import Pool
import multiprocessing

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

# Helper function to run the simulator and bin the result
def run_simulator_pdf(z, h, om, s8, nsamples, seed, bin_edges):
    res = gw.sample_lnmu_ml_with_diagnostics(z, h, om, s8, nsamples, seed, False)
    lnmu = np.array(res["lnmu"])
    lnmu = lnmu[~np.isnan(lnmu)]
    clamped = np.clip(lnmu, bin_edges[0] + 1e-9, bin_edges[-1] - 1e-9)
    counts, _ = np.histogram(clamped, bins=bin_edges)
    sum_counts = np.sum(counts)
    if sum_counts > 0:
        pdf = counts / sum_counts
    else:
        pdf = np.ones(len(bin_edges) - 1, dtype=np.float32) / (len(bin_edges) - 1)
        counts = np.zeros(len(bin_edges) - 1, dtype=np.int32)
    return pdf, counts

# Training function
def train_model(X_train, Y_train, X_val, Y_val, epochs=500, lr=1e-3, batch_size=32, patience=30):
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
        
    return model, mean, std

# Worker function for parallel cosmology recovery optimization
def recovery_worker(args_tuple):
    (idx, z_true, h_true, om_true, s8_true, target_counts, target_pdf, 
     train_mean, train_std, model_weights, bin_edges) = args_tuple
    
    # Reload model weights in this worker process
    model = BaselineMLP(input_dim=4, hidden_dim=128, output_dim=100)
    model.load_state_dict(model_weights)
    model.eval()
    
    # Bounds bounds check
    bounds = {
        'h': (0.59, 0.76),
        'OmegaM': (0.20, 0.40),
        'sigma8': (0.65, 1.05)
    }
    
    theta_true = np.array([h_true, om_true, s8_true])
    theta_init = np.array([0.67, 0.30, 0.85]) # midpoint
    
    # Helper to predict PDF using emulator
    def emu_predict(h, om, s8):
        x = np.array([[z_true, h, om, s8]])
        x_norm = (x - train_mean) / train_std
        with torch.no_grad():
            p = model(torch.tensor(x_norm, dtype=torch.float32)).numpy()[0]
        return p
        
    # Helper to predict PDF using simulator (using seed 100 for inference simulations)
    def sim_predict(h, om, s8):
        p, _ = run_simulator_pdf(z_true, h, om, s8, 10000, 100, bin_edges)
        return p

    # --- Emulator Recovery A (JSD) ---
    def obj_emu_jsd(theta):
        h, om, s8 = theta
        if not (bounds['h'][0] <= h <= bounds['h'][1] and 
                bounds['OmegaM'][0] <= om <= bounds['OmegaM'][1] and 
                bounds['sigma8'][0] <= s8 <= bounds['sigma8'][1]):
            return 1e10
        p = emu_predict(h, om, s8)
        # JSD expects shape (N, Bins) or (Bins,)
        return float(compute_js_divergence(target_pdf, p))
        
    res_emu_jsd = minimize(obj_emu_jsd, theta_init, method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-4})
    theta_emu_jsd = res_emu_jsd.x
    
    # --- Emulator Recovery B (Likelihood) ---
    def obj_emu_lik(theta):
        h, om, s8 = theta
        if not (bounds['h'][0] <= h <= bounds['h'][1] and 
                bounds['OmegaM'][0] <= om <= bounds['OmegaM'][1] and 
                bounds['sigma8'][0] <= s8 <= bounds['sigma8'][1]):
            return 1e10
        p = emu_predict(h, om, s8)
        # Binned log-likelihood
        return -float(np.sum(target_counts * np.log(p + 1e-12)))
        
    res_emu_lik = minimize(obj_emu_lik, theta_init, method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-4})
    theta_emu_lik = res_emu_lik.x

    # --- Simulator Recovery A (JSD) ---
    def obj_sim_jsd(theta):
        h, om, s8 = theta
        if not (bounds['h'][0] <= h <= bounds['h'][1] and 
                bounds['OmegaM'][0] <= om <= bounds['OmegaM'][1] and 
                bounds['sigma8'][0] <= s8 <= bounds['sigma8'][1]):
            return 1e10
        p = sim_predict(h, om, s8)
        return float(compute_js_divergence(target_pdf, p))
        
    res_sim_jsd = minimize(obj_sim_jsd, theta_init, method='Nelder-Mead', options={'xatol': 1e-3, 'fatol': 1e-3})
    theta_sim_jsd = res_sim_jsd.x
    
    # --- Simulator Recovery B (Likelihood) ---
    def obj_sim_lik(theta):
        h, om, s8 = theta
        if not (bounds['h'][0] <= h <= bounds['h'][1] and 
                bounds['OmegaM'][0] <= om <= bounds['OmegaM'][1] and 
                bounds['sigma8'][0] <= s8 <= bounds['sigma8'][1]):
            return 1e10
        p = sim_predict(h, om, s8)
        return -float(np.sum(target_counts * np.log(p + 1e-12)))
        
    res_sim_lik = minimize(obj_sim_lik, theta_init, method='Nelder-Mead', options={'xatol': 1e-3, 'fatol': 1e-3})
    theta_sim_lik = res_sim_lik.x
    
    return {
        'idx': idx,
        'true': theta_true.tolist(),
        'emu_jsd': theta_emu_jsd.tolist(),
        'emu_lik': theta_emu_lik.tolist(),
        'sim_jsd': theta_sim_jsd.tolist(),
        'sim_lik': theta_sim_lik.tolist()
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Perform derivative failure audit.")
    parser.add_argument("--dataset_dir", type=str, default="datasets/large_1k")
    parser.add_argument("--output_dir", type=str, default="plots/figures/phase2_interpolation")
    parser.add_argument("--artifact_dir", type=str, default="/Users/baltabay/.gemini/antigravity/brain/f3f9b801-1f10-4591-9e78-0368fe2e4a12")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.artifact_dir, "figures"), exist_ok=True)
    
    bin_edges = get_recommended_bin_edges()
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    print(f"Loading dataset from {args.dataset_dir}...")
    dataset = load_histogram_dataset(args.dataset_dir, bin_edges)
    
    # Pool ID configurations
    all_x, all_y = [], []
    for split_name, (X, Y, split_types) in dataset.items():
        for i in range(X.shape[0]):
            if split_types[i] in ["train", "interpolation"]:
                all_x.append(X[i])
                all_y.append(Y[i])
                
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    
    # Shuffle
    rng = np.random.default_rng(42)
    indices = np.arange(all_x.shape[0])
    rng.shuffle(indices)
    all_x = all_x[indices]
    all_y = all_y[indices]
    
    val_size = 100
    X_val = all_x[:val_size]
    Y_val = all_y[:val_size]
    X_train = all_x[val_size:val_size+1000]
    Y_train = all_y[val_size:val_size+1000]
    
    # Train the MLP model
    print(f"Training baseline MLP on {X_train.shape[0]} configurations...")
    model, train_mean, train_std = train_model(X_train, Y_train, X_val, Y_val)
    model.eval()
    
    # --- 1 & 2 & 3. Sensitivity differences & additional diagnostics ---
    print("\n--- Running Sensitivity Audit ---")
    z0, h0, om0, s80 = 1.0, 0.67, 0.30, 0.85
    ref_x = np.array([z0, h0, om0, s80])
    
    # Simulator reference
    p_sim_ref, _ = run_simulator_pdf(z0, h0, om0, s80, 10000, 42, bin_edges)
    
    # Emulator reference
    ref_norm = (ref_x - train_mean) / train_std
    with torch.no_grad():
        p_emu_ref = model(torch.tensor(ref_norm, dtype=torch.float32).unsqueeze(0)).numpy()[0]
        
    perturbations = {
        'sigma8': {'idx': 3, 'val': s80, 'name': r'$\sigma_8$'},
        'OmegaM': {'idx': 2, 'val': om0, 'name': r'$\Omega_M$'},
        'h': {'idx': 1, 'val': h0, 'name': r'$h$'}
    }
    
    audit_results = {}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    for ax_idx, (p_name, p_info) in enumerate(perturbations.items()):
        p_idx = p_info['idx']
        val0 = p_info['val']
        p_label = p_info['name']
        
        # Perturbed inputs
        x_plus, x_minus = ref_x.copy(), ref_x.copy()
        x_plus[p_idx] = val0 * 1.05
        x_minus[p_idx] = val0 * 0.95
        
        # Simulator perturbed
        p_sim_plus, _ = run_simulator_pdf(x_plus[0], x_plus[1], x_plus[2], x_plus[3], 10000, 42, bin_edges)
        p_sim_minus, _ = run_simulator_pdf(x_minus[0], x_minus[1], x_minus[2], x_minus[3], 10000, 42, bin_edges)
        
        # Emulator perturbed
        with torch.no_grad():
            p_emu_plus = model(torch.tensor((x_plus - train_mean) / train_std, dtype=torch.float32).unsqueeze(0)).numpy()[0]
            p_emu_minus = model(torch.tensor((x_minus - train_mean) / train_std, dtype=torch.float32).unsqueeze(0)).numpy()[0]
            
        dp_sim_plus = p_sim_plus - p_sim_ref
        dp_sim_minus = p_sim_minus - p_sim_ref
        dp_emu_plus = p_emu_plus - p_emu_ref
        dp_emu_minus = p_emu_minus - p_emu_ref
        
        audit_results[p_name] = {}
        for direction, dp_sim, dp_emu in [('plus', dp_sim_plus, dp_emu_plus), ('minus', dp_sim_minus, dp_emu_minus)]:
            abs_err = float(np.linalg.norm(dp_emu - dp_sim))
            denom = float(np.linalg.norm(dp_sim))
            rel_err = abs_err / (denom + 1e-12)
            
            # Cosine similarity
            dot_prod = np.dot(dp_emu, dp_sim)
            norm_emu = np.linalg.norm(dp_emu)
            cosine_sim = float(dot_prod / (norm_emu * denom + 1e-12))
            
            # Pearson correlation
            pearson_corr = float(stats.pearsonr(dp_emu, dp_sim)[0]) if norm_emu > 0 and denom > 0 else 0.0
            
            # Sign agreement
            sign_sim = np.sign(dp_sim)
            sign_emu = np.sign(dp_emu)
            # Exclude bins where the true simulator derivative is basically zero to avoid noise
            val_mask = np.abs(dp_sim) > 1e-5
            if np.sum(val_mask) > 0:
                sign_agree = float(np.mean(sign_sim[val_mask] == sign_emu[val_mask]))
            else:
                sign_agree = 1.0
                
            audit_results[p_name][direction] = {
                'abs_err': abs_err,
                'denom': denom,
                'rel_err': rel_err,
                'cosine_sim': cosine_sim,
                'pearson_corr': pearson_corr,
                'sign_agree': sign_agree
            }
            
            print(f"{p_name:7s} | Dir: {direction:5s} | AbsErr: {abs_err:.6f} | Denom: {denom:.6f} | RelErr: {rel_err:.2%} | CosSim: {cosine_sim:.4f} | Pearson: {pearson_corr:.4f} | SignAgree: {sign_agree:.2%}")
            
        # Plot difference profiles for this parameter
        ax = axes[ax_idx]
        ax.plot(bin_centers, dp_sim_plus, color='#1f77b4', lw=2.0, label='Sim +5%')
        ax.plot(bin_centers, dp_emu_plus, color='#1f77b4', lw=2.2, linestyle='--', label=f'Emu +5% (Cos: {audit_results[p_name]["plus"]["cosine_sim"]:.2f})')
        ax.plot(bin_centers, dp_sim_minus, color='#d62728', lw=2.0, label='Sim -5%')
        ax.plot(bin_centers, dp_emu_minus, color='#d62728', lw=2.2, linestyle='--', label=f'Emu -5% (Cos: {audit_results[p_name]["minus"]["cosine_sim"]:.2f})')
        ax.set_xlabel(r"$\ln\mu$")
        ax.set_ylabel(r"$\Delta P$")
        ax.set_title(f"Sensitivity Profile: {p_label}")
        ax.set_xlim(-0.35, 1.1)
        ax.legend()
        
    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "derivative_audit_diagnostics.png"), dpi=200)
    fig.savefig(os.path.join(args.artifact_dir, "figures", "derivative_audit_diagnostics.png"), dpi=200)
    plt.close(fig)
    print("Diagnostics plot saved.")

    # --- 3D. PC-space sensitivity ---
    print("\n--- Running PC-Space Audit ---")
    C = Y_train.shape[0]
    mean_pdf = np.mean(Y_train, axis=0)
    Y_centered = Y_train - mean_pdf
    _, _, Vt = np.linalg.svd(Y_centered, full_matrices=False)
    
    pc_results = {}
    for p_name, p_info in perturbations.items():
        p_idx = p_info['idx']
        val0 = p_info['val']
        
        # Perturbed inputs
        x_plus, x_minus = ref_x.copy(), ref_x.copy()
        x_plus[p_idx] = val0 * 1.05
        x_minus[p_idx] = val0 * 0.95
        
        # Simulator difference
        p_sim_plus, _ = run_simulator_pdf(x_plus[0], x_plus[1], x_plus[2], x_plus[3], 10000, 42, bin_edges)
        p_sim_minus, _ = run_simulator_pdf(x_minus[0], x_minus[1], x_minus[2], x_minus[3], 10000, 42, bin_edges)
        dp_sim_plus = p_sim_plus - p_sim_ref
        dp_sim_minus = p_sim_minus - p_sim_ref
        
        # Emulator difference
        with torch.no_grad():
            p_emu_plus = model(torch.tensor((x_plus - train_mean) / train_std, dtype=torch.float32).unsqueeze(0)).numpy()[0]
            p_emu_minus = model(torch.tensor((x_minus - train_mean) / train_std, dtype=torch.float32).unsqueeze(0)).numpy()[0]
        dp_emu_plus = p_emu_plus - p_emu_ref
        dp_emu_minus = p_emu_minus - p_emu_ref
        
        pc_results[p_name] = {}
        for direction, dp_sim, dp_emu in [('plus', dp_sim_plus, dp_emu_plus), ('minus', dp_sim_minus, dp_emu_minus)]:
            pc_results[p_name][direction] = []
            for pc_idx in range(3):
                da_sim = np.dot(dp_sim, Vt[pc_idx])
                da_emu = np.dot(dp_emu, Vt[pc_idx])
                pc_results[p_name][direction].append({
                    'pc': pc_idx + 1,
                    'sim_coef_diff': float(da_sim),
                    'emu_coef_diff': float(da_emu)
                })
                print(f"{p_name:7s} | Dir: {direction:5s} | PC {pc_idx+1} | Sim coeff diff: {da_sim:.6f} | Emu coeff diff: {da_emu:.6f}")

    # --- 4. Cosmology recovery test ---
    print("\n--- Running Cosmology Recovery Test (10 random In-Distribution cosmologies) ---")
    
    # Generate 10 random cosmologies within the bounds
    cosmo_rng = np.random.default_rng(2026)
    
    z_vals = cosmo_rng.uniform(0.5, 2.5, 10) # Lensing source redshifts
    h_vals = cosmo_rng.uniform(0.59, 0.76, 10)
    om_vals = cosmo_rng.uniform(0.20, 0.40, 10)
    s8_vals = cosmo_rng.uniform(0.65, 1.05, 10)
    
    # Pre-simulate target counts and target PDFs for each cosmology
    targets_args = []
    for i in range(10):
        print(f"Simulating target cosmology {i+1}/10: z={z_vals[i]:.2f}, h={h_vals[i]:.3f}, om={om_vals[i]:.3f}, s8={s8_vals[i]:.3f}")
        pdf_true, counts_true = run_simulator_pdf(z_vals[i], h_vals[i], om_vals[i], s8_vals[i], 10000, 42, bin_edges)
        targets_args.append((
            i, z_vals[i], h_vals[i], om_vals[i], s8_vals[i],
            counts_true, pdf_true, train_mean, train_std, model.state_dict(), bin_edges
        ))
        
    # Execute optimization runs in parallel across 10 processes
    print(f"Executing recovery optimizations in parallel using multiprocessing...")
    num_processes = min(multiprocessing.cpu_count(), 10)
    with Pool(processes=num_processes) as pool:
        recovery_results = pool.map(recovery_worker, targets_args)
        
    recovery_results = sorted(recovery_results, key=lambda x: x['idx'])
    
    # Calculate recovery biases and report stats
    h_biases_emu_jsd, om_biases_emu_jsd, s8_biases_emu_jsd = [], [], []
    h_biases_emu_lik, om_biases_emu_lik, s8_biases_emu_lik = [], [], []
    h_biases_sim_jsd, om_biases_sim_jsd, s8_biases_sim_jsd = [], [], []
    h_biases_sim_lik, om_biases_sim_lik, s8_biases_sim_lik = [], [], []
    
    for res in recovery_results:
        idx = res['idx']
        h_true, om_true, s8_true = res['true']
        
        h_emu_jsd, om_emu_jsd, s8_emu_jsd = res['emu_jsd']
        h_emu_lik, om_emu_lik, s8_emu_lik = res['emu_lik']
        h_sim_jsd, om_sim_jsd, s8_sim_jsd = res['sim_jsd']
        h_sim_lik, om_sim_lik, s8_sim_lik = res['sim_lik']
        
        # Compute biases
        h_biases_emu_jsd.append(h_emu_jsd - h_true)
        om_biases_emu_jsd.append(om_emu_jsd - om_true)
        s8_biases_emu_jsd.append(s8_emu_jsd - s8_true)
        
        h_biases_emu_lik.append(h_emu_lik - h_true)
        om_biases_emu_lik.append(om_emu_lik - om_true)
        s8_biases_emu_lik.append(s8_emu_lik - s8_true)
        
        h_biases_sim_jsd.append(h_sim_jsd - h_true)
        om_biases_sim_jsd.append(om_sim_jsd - om_true)
        s8_biases_sim_jsd.append(s8_sim_jsd - s8_true)
        
        h_biases_sim_lik.append(h_sim_lik - h_true)
        om_biases_sim_lik.append(om_sim_lik - om_true)
        s8_biases_sim_lik.append(s8_sim_lik - s8_true)
        
    print("\n--- Cosmology Recovery Summary (Mean Bias & Standard Deviation) ---")
    
    def report_stat(name, biases):
        biases = np.array(biases)
        print(f"{name:25s} | Mean Bias: {np.mean(biases):+.6f} | Std Dev: {np.std(biases):.6f}")
        return float(np.mean(biases)), float(np.std(biases))
        
    stats_recovery = {}
    
    print("\n[Recovery A: JSD Objective]")
    m, s = report_stat("Emu h JSD bias", h_biases_emu_jsd)
    stats_recovery['emu_jsd_h'] = {'mean': m, 'std': s}
    m, s = report_stat("Emu om JSD bias", om_biases_emu_jsd)
    stats_recovery['emu_jsd_om'] = {'mean': m, 'std': s}
    m, s = report_stat("Emu s8 JSD bias", s8_biases_emu_jsd)
    stats_recovery['emu_jsd_s8'] = {'mean': m, 'std': s}
    
    m, s = report_stat("Sim h JSD bias", h_biases_sim_jsd)
    stats_recovery['sim_jsd_h'] = {'mean': m, 'std': s}
    m, s = report_stat("Sim om JSD bias", om_biases_sim_jsd)
    stats_recovery['sim_jsd_om'] = {'mean': m, 'std': s}
    m, s = report_stat("Sim s8 JSD bias", s8_biases_sim_jsd)
    stats_recovery['sim_jsd_s8'] = {'mean': m, 'std': s}
    
    print("\n[Recovery B: Likelihood Objective]")
    m, s = report_stat("Emu h Likelihood bias", h_biases_emu_lik)
    stats_recovery['emu_lik_h'] = {'mean': m, 'std': s}
    m, s = report_stat("Emu om Likelihood bias", om_biases_emu_lik)
    stats_recovery['emu_lik_om'] = {'mean': m, 'std': s}
    m, s = report_stat("Emu s8 Likelihood bias", s8_biases_emu_lik)
    stats_recovery['emu_lik_s8'] = {'mean': m, 'std': s}
    
    m, s = report_stat("Sim h Likelihood bias", h_biases_sim_lik)
    stats_recovery['sim_lik_h'] = {'mean': m, 'std': s}
    m, s = report_stat("Sim om Likelihood bias", om_biases_sim_lik)
    stats_recovery['sim_lik_om'] = {'mean': m, 'std': s}
    m, s = report_stat("Sim s8 Likelihood bias", s8_biases_sim_lik)
    stats_recovery['sim_lik_s8'] = {'mean': m, 'std': s}

    # Bounds dictionary for plotting limits
    bounds = {
        'h': (0.59, 0.76),
        'OmegaM': (0.20, 0.40),
        'sigma8': (0.65, 1.05)
    }

    # Plot parameter recovery results
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    params_info = [
        ('h', h_biases_sim_jsd, h_biases_emu_jsd, h_biases_sim_lik, h_biases_emu_lik, bounds['h']),
        ('OmegaM', om_biases_sim_jsd, om_biases_emu_jsd, om_biases_sim_lik, om_biases_emu_lik, bounds['OmegaM']),
        ('sigma8', s8_biases_sim_jsd, s8_biases_emu_jsd, s8_biases_sim_lik, s8_biases_emu_lik, bounds['sigma8'])
    ]
    
    for row_idx, (p_name, sim_jsd, emu_jsd, sim_lik, emu_lik, p_bounds) in enumerate(params_info):
        # Left plot: JSD Objective
        ax_left = axes[row_idx, 0]
        ax_left.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax_left.scatter(np.arange(10) + 1, sim_jsd, color='#2ca02c', marker='o', s=50, label='Sim JSD (stochastic)')
        ax_left.scatter(np.arange(10) + 1, emu_jsd, color='#1f77b4', marker='x', s=50, label='Emu JSD')
        ax_left.set_xticks(np.arange(10) + 1)
        ax_left.set_xlabel("Cosmology Realization Index")
        ax_left.set_ylabel(f"{p_name} bias")
        ax_left.set_title(f"{p_name} Recovery Bias: JSD Objective")
        ax_left.legend()
        
        # Right plot: Likelihood Objective
        ax_right = axes[row_idx, 1]
        ax_right.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax_right.scatter(np.arange(10) + 1, sim_lik, color='#ff7f0e', marker='o', s=50, label='Sim Likelihood')
        ax_right.scatter(np.arange(10) + 1, emu_lik, color='#d62728', marker='x', s=50, label='Emu Likelihood')
        ax_right.set_xticks(np.arange(10) + 1)
        ax_right.set_xlabel("Cosmology Realization Index")
        ax_right.set_ylabel(f"{p_name} bias")
        ax_right.set_title(f"{p_name} Recovery Bias: Likelihood Objective")
        ax_right.legend()
        
    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "parameter_recovery.png"), dpi=200)
    fig.savefig(os.path.join(args.artifact_dir, "figures", "parameter_recovery.png"), dpi=200)
    plt.close(fig)
    print("Parameter recovery plot saved.")

    # Save stats JSON
    stats_file = os.path.join(args.artifact_dir, "derivative_audit_stats.json")
    stats_data = {
        'audit': audit_results,
        'pc_space': pc_results,
        'recovery': stats_recovery,
        'realizations': recovery_results
    }
    with open(stats_file, 'w') as sf:
        json.dump(stats_data, sf, indent=2)
    print(f"Metrics stats saved to {stats_file}")

if __name__ == "__main__":
    main()
