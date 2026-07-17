import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

# Add build to path
sys.path.insert(0, "/Users/baltabay/Desktop/gw-wl-emulator/build")
import gwlensing as gw

ZS = 1.0
COSMO = dict(OmegaM=0.315, sigma8=0.811, h=0.674)
N_TOTAL = 1000000
NUM_PROCESSES = 8
N_PER_PROC = N_TOTAL // NUM_PROCESSES

def run_proc_off(seed_val):
    raw = gw.sample_lnmu_ml(
        ZS, COSMO['h'], COSMO['OmegaM'], COSMO['sigma8'], N_PER_PROC, seed_val,
        False, 1.0e7, False, 1.0e7
    )
    mu = np.exp(np.asarray(raw, dtype=float))
    return mu[np.isfinite(mu) & (mu > 0.0)]

def run_proc_on(seed_val):
    raw = gw.sample_lnmu_ml(
        ZS, COSMO['h'], COSMO['OmegaM'], COSMO['sigma8'], N_PER_PROC, seed_val,
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
    
    data_path = os.path.join(data_dir, "mu_data.npz")
    if os.path.exists(data_path):
        print("Loading cached simulation data...")
        data = np.load(data_path)
        mu_off = data["mu_off"]
        mu_on = data["mu_on"]
    else:
        print(f"Running parallel simulations for N={N_TOTAL} realizations on {NUM_PROCESSES} cores...")
        
        seeds_off = [100 + i for i in range(NUM_PROCESSES)]
        seeds_on = [200 + i for i in range(NUM_PROCESSES)]
        
        # 1. Subhalo OFF parallel run
        print("Running Subhalo OFF in parallel...")
        with Pool(processes=NUM_PROCESSES) as pool:
            results_off = pool.map(run_proc_off, seeds_off)
        mu_off = np.concatenate(results_off)
        
        # 2. Subhalo ON parallel run
        print("Running Subhalo ON in parallel...")
        with Pool(processes=NUM_PROCESSES) as pool:
            results_on = pool.map(run_proc_on, seeds_on)
        mu_on = np.concatenate(results_on)
        
        np.savez(data_path, mu_off=mu_off, mu_on=mu_on)
        print(f"Saved simulation data to {data_path}")

    # Set up bins for the tail (99.999th percentile)
    hi = max(np.quantile(mu_off, 0.99999), np.quantile(mu_on, 0.99999))
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

    # Panel 1: Log-Linear (Log y, Linear x)
    centers_lin, pdf_off_lin = get_pdf(mu_off, edges_lin)
    _, pdf_on_lin = get_pdf(mu_on, edges_lin)

    ax1.step(centers_lin, pdf_off_lin, where="mid", lw=2, color="#2ca02c", label="Subhalo OFF")
    ax1.step(centers_lin, pdf_on_lin, where="mid", lw=2, color="#d62728", label="Subhalo ON")
    ax1.set_yscale("log")
    ax1.set_xlim(0.8, 6.0)
    ax1.set_xlabel(r"Magnification $\mu$ (Linear)", fontsize=12, fontweight="bold")
    ax1.set_ylabel(r"$dP/d\mu$ (Logarithmic)", fontsize=12, fontweight="bold")
    ax1.set_title("Log-Linear scale", fontsize=13, fontweight="bold")
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend(frameon=True, facecolor="#f8fafc")

    # Panel 2: Log-Log (Log y, Log x)
    centers_log, pdf_off_log = get_pdf(mu_off, edges_log)
    _, pdf_on_log = get_pdf(mu_on, edges_log)

    ax2.step(centers_log, pdf_off_log, where="mid", lw=2, color="#2ca02c", label="Subhalo OFF")
    ax2.step(centers_log, pdf_on_log, where="mid", lw=2, color="#d62728", label="Subhalo ON")
    
    # Fit C * mu^-3 to the Subhalo OFF PDF in the range mu in [3, 15]
    fit_mask = (centers_log >= 3.0) & (centers_log <= 15.0)
    C_fit = np.mean(pdf_off_log[fit_mask] * (centers_log[fit_mask]**3))
    
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

    plt.suptitle(f"Magnification PDF Tail Comparison (N={N_TOTAL} realizations)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    plot_path = os.path.join(plots_dir, "subhalo_pdf_log_scales.png")
    plt.savefig(plot_path, dpi=300, facecolor="#ffffff")
    print(f"Saved plot to {plot_path}")
