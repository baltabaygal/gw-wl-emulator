import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

# Add build to path
sys.path.insert(0, "/Users/baltabay/Desktop/gw-wl-emulator/build")
import gwlensing as gw

COSMO = dict(OmegaM=0.315, sigma8=0.811, h=0.674)
N_TOTAL = 1000000
NUM_PROCESSES = 8
N_PER_PROC = N_TOTAL // NUM_PROCESSES

def run_proc_off(args):
    zs, seed_val = args
    raw = gw.sample_lnmu_ml(
        zs, COSMO['h'], COSMO['OmegaM'], COSMO['sigma8'], N_PER_PROC, seed_val,
        False, 1.0e7, False, 1.0e7
    )
    mu = np.exp(np.asarray(raw, dtype=float))
    return mu[np.isfinite(mu) & (mu > 0.0)]

def run_proc_on(args):
    zs, seed_val = args
    raw = gw.sample_lnmu_ml(
        zs, COSMO['h'], COSMO['OmegaM'], COSMO['sigma8'], N_PER_PROC, seed_val,
        False, 1.0e7, True, 1.0e7, subhalo_factor=1e-5
    )
    mu = np.exp(np.asarray(raw, dtype=float))
    return mu[np.isfinite(mu) & (mu > 0.0)]

if __name__ == "__main__":
    project_dir = "/Users/baltabay/Desktop/gw-wl-emulator"
    data_dir = os.path.join(project_dir, "data")
    plots_dir = os.path.join(project_dir, "plots", "figures")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    data_path_z1 = os.path.join(data_dir, "mu_data.npz")
    data_path_z2 = os.path.join(data_dir, "mu_data_z2.npz")
    
    # 1. Load z=1.0 data
    if os.path.exists(data_path_z1):
        print("Loading cached z=1.0 simulation data...")
        data_z1 = np.load(data_path_z1)
        mu_off_z1 = data_z1["mu_off"]
        mu_on_z1 = data_z1["mu_on"]
    else:
        print("Error: z=1.0 data cache not found. Please run the original script first.")
        sys.exit(1)
        
    # 2. Run/Load z=2.0 data
    if os.path.exists(data_path_z2):
        print("Loading cached z=2.0 simulation data...")
        data_z2 = np.load(data_path_z2)
        mu_off_z2 = data_z2["mu_off"]
        mu_on_z2 = data_z2["mu_on"]
    else:
        print(f"Running parallel simulations for z=2.0, N={N_TOTAL} realizations on {NUM_PROCESSES} cores...")
        
        seeds_off = [(2.0, 100 + i) for i in range(NUM_PROCESSES)]
        seeds_on = [(2.0, 200 + i) for i in range(NUM_PROCESSES)]
        
        print("Running Subhalo OFF at z=2.0 in parallel...")
        with Pool(processes=NUM_PROCESSES) as pool:
            results_off = pool.map(run_proc_off, seeds_off)
        mu_off_z2 = np.concatenate(results_off)
        
        print("Running Subhalo ON at z=2.0 in parallel...")
        with Pool(processes=NUM_PROCESSES) as pool:
            results_on = pool.map(run_proc_on, seeds_on)
        mu_on_z2 = np.concatenate(results_on)
        
        np.savez(data_path_z2, mu_off=mu_off_z2, mu_on=mu_on_z2)
        print(f"Saved z=2.0 simulation data to {data_path_z2}")

    # Set up binning limits
    hi = max(np.quantile(mu_off_z1, 0.99999), np.quantile(mu_on_z1, 0.99999),
             np.quantile(mu_off_z2, 0.99999), np.quantile(mu_on_z2, 0.99999))
    
    edges_lin = np.linspace(0.8, hi, 120)
    edges_log = np.logspace(np.log10(0.8), np.log10(hi), 120)

    def get_pdf(mu, edges):
        counts, _ = np.histogram(mu, bins=edges)
        widths = np.diff(edges)
        total = counts.sum()
        pdf = counts / (total * widths) if total > 0 else np.zeros_like(widths)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers, pdf

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # --- Panel 1: Log-Linear ---
    centers_lin, pdf_off_z1_lin = get_pdf(mu_off_z1, edges_lin)
    _, pdf_on_z1_lin = get_pdf(mu_on_z1, edges_lin)
    _, pdf_off_z2_lin = get_pdf(mu_off_z2, edges_lin)
    _, pdf_on_z2_lin = get_pdf(mu_on_z2, edges_lin)

    # z=1.0
    ax1.step(centers_lin, pdf_off_z1_lin, where="mid", lw=2, color="#2ca02c", alpha=0.5, label="z=1.0 Subhalo OFF")
    ax1.step(centers_lin, pdf_on_z1_lin, where="mid", lw=2, color="#d62728", alpha=0.5, label="z=1.0 Subhalo ON")
    # z=2.0
    ax1.step(centers_lin, pdf_off_z2_lin, where="mid", lw=2, color="#1f77b4", label="z=2.0 Subhalo OFF")
    ax1.step(centers_lin, pdf_on_z2_lin, where="mid", lw=2, color="#ff7f0e", label="z=2.0 Subhalo ON")
    
    ax1.set_yscale("log")
    ax1.set_xlim(0.8, 6.0)
    ax1.set_xlabel(r"Magnification $\mu$ (Linear)", fontsize=12, fontweight="bold")
    ax1.set_ylabel(r"$dP/d\mu$ (Logarithmic)", fontsize=12, fontweight="bold")
    ax1.set_title("Log-Linear scale", fontsize=13, fontweight="bold")
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend(frameon=True, facecolor="#f8fafc")

    # --- Panel 2: Log-Log ---
    centers_log, pdf_off_z1_log = get_pdf(mu_off_z1, edges_log)
    _, pdf_on_z1_log = get_pdf(mu_on_z1, edges_log)
    _, pdf_off_z2_log = get_pdf(mu_off_z2, edges_log)
    _, pdf_on_z2_log = get_pdf(mu_on_z2, edges_log)

    # z=1.0
    ax2.step(centers_log, pdf_off_z1_log, where="mid", lw=2, color="#2ca02c", alpha=0.5, label="z=1.0 Subhalo OFF")
    ax2.step(centers_log, pdf_on_z1_log, where="mid", lw=2, color="#d62728", alpha=0.5, label="z=1.0 Subhalo ON")
    # z=2.0
    ax2.step(centers_log, pdf_off_z2_log, where="mid", lw=2, color="#1f77b4", label="z=2.0 Subhalo OFF")
    ax2.step(centers_log, pdf_on_z2_log, where="mid", lw=2, color="#ff7f0e", label="z=2.0 Subhalo ON")

    # Fit C * mu^-3 to the z=2.0 Subhalo OFF PDF in range [3, 15]
    fit_mask = (centers_log >= 3.0) & (centers_log <= 15.0)
    C_fit = np.mean(pdf_off_z2_log[fit_mask] * (centers_log[fit_mask]**3))
    
    # Plot the fit line
    mu_fit_eval = np.logspace(np.log10(2.0), np.log10(hi), 100)
    pdf_fit_eval = C_fit * (mu_fit_eval**-3)
    ax2.plot(mu_fit_eval, pdf_fit_eval, color="black", linestyle="--", lw=1.5, label=r"Theory: $\propto \mu^{-3}$")

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"Magnification $\mu$ (Logarithmic)", fontsize=12, fontweight="bold")
    ax2.set_ylabel(r"$dP/d\mu$ (Logarithmic)", fontsize=12, fontweight="bold")
    ax2.set_title("Log-Log scale", fontsize=13, fontweight="bold")
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend(frameon=True, facecolor="#f8fafc")

    plt.suptitle(f"Magnification PDF Redshift Comparison (N={N_TOTAL} realizations)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    plot_path = os.path.join(plots_dir, "subhalo_pdf_redshift_comparison.png")
    plt.savefig(plot_path, dpi=300, facecolor="#ffffff")
    print(f"Saved plot to {plot_path}")
