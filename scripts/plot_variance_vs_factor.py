import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

# Add build to path
sys.path.insert(0, "/Users/baltabay/Desktop/gw-wl-emulator/build")
import gwlensing

# Parameters
N = 50000
seed = 42
factors = np.logspace(-8, 6, 15)

def run_one_factor(args):
    zs, f = args
    z, h, OmegaM, sigma8 = zs, 0.674, 0.315, 0.811
    res = gwlensing.sample_lensing_raw_ml(
        z=z, h=h, OmegaM=OmegaM, sigma8=sigma8,
        nsamples=N, seed=seed, filaments=False, bias=False, ell=False,
        subhalo=True, subhalo_factor=f, m_floor=1e7, subhalo_model=1
    )
    var_on = np.array(res["kappa"]).var()
    var_nosub = np.array(res["kappa_nosub"]).var()
    return f, var_on, var_nosub

if __name__ == "__main__":
    project_dir = "/Users/baltabay/Desktop/gw-wl-emulator"
    data_dir = os.path.join(project_dir, "data")
    plots_dir = os.path.join(project_dir, "plots", "figures")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    cache_path = os.path.join(data_dir, "variance_sweep_data_z1_N50k_to_1e-8_1e6.npz")
    
    if os.path.exists(cache_path):
        print("Loading cached N=50k sweep data from 10^-8 to 10^6...")
        data = np.load(cache_path)
        sweep_factors = data["factors"]
        var_on_paired = data["var_on_paired"]
        var_off = float(data["var_off"])
    else:
        print(f"\n--- Running Sweep for z=1.0 from 10^-8 to 10^6 (N={N} realizations) ---")
        print("Running baseline...")
        res_off = gwlensing.sample_lensing_raw_ml(
            z=1.0, h=0.674, OmegaM=0.315, sigma8=0.811,
            nsamples=N, seed=seed, filaments=False, bias=False, ell=False,
            subhalo=False
        )
        var_off = np.array(res_off["kappa"]).var()

        num_processes = 8
        print(f"Running parallel sweep in [10^-8, 10^6] on {num_processes} cores...")
        args_list = [(1.0, f) for f in factors]
        with Pool(processes=num_processes) as pool:
            results = pool.map(run_one_factor, args_list)

        results = sorted(results, key=lambda x: x[0])
        sweep_factors = np.array([r[0] for r in results])
        var_on = np.array([r[1] for r in results])
        var_nosub = np.array([r[2] for r in results])
        var_on_paired = var_off + (var_on - var_nosub)
        
        np.savez(cache_path, factors=sweep_factors, var_on_paired=var_on_paired, var_off=var_off)
        print(f"Saved sweep data to {cache_path}")

    # Plotting: single panel down to 10^-8
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(11, 7))

    ax.set_xlim(1e-8, 1e6)
    
    # Shade background floor and transition regimes
    ax.axvspan(1e-8, 1e-6, color="#f1f5f9", alpha=0.7, label="Physical Floor Plateau", zorder=0)
    ax.axvspan(1e-6, 1e2, color="#fffbeb", alpha=0.4, label="Transition Regime", zorder=0)

    # Shading the excess variance region
    ax.fill_between(sweep_factors, var_on_paired, var_off, where=(var_on_paired > var_off), 
                    color="#4f46e5", alpha=0.10, zorder=1)

    # Main plotting line
    ax.plot(sweep_factors, var_on_paired, color="#4f46e5", marker="o", markersize=7, 
            markerfacecolor="#4f46e5", markeredgecolor="white", markeredgewidth=1.5,
            linewidth=3.0, zorder=3)
            
    ax.axhline(y=var_off, color="#334155", linestyle="--", linewidth=2.0, zorder=2)

    ax.set_xscale("log")
    ax.set_xlabel(r"Subhalo Resolution Factor ($k_{\rm thr}$ scale factor)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel(r"Convergence Variance $\mathrm{Var}(\kappa)$", fontsize=12, fontweight="bold", labelpad=10)
    
    ax.grid(True, which="both", ls="--", color="#cbd5e1", alpha=0.6, zorder=0)
    ax.tick_params(axis='both', which='major', labelsize=10.5)
    ax.legend(frameon=True, facecolor="#f8fafc", loc="lower left", fontsize=11)
    plt.tight_layout()
    
    plot_path = os.path.join(plots_dir, "variance_vs_subhalo_factor.png")
    plt.savefig(plot_path, dpi=300, facecolor="#ffffff")
    print(f"Saved plot to {plot_path}")
