import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path

def power_law(N, A, alpha, B):
    return A * (N ** (-alpha)) + B

def fit_power_law(N_vals, y_vals):
    N_vals = np.array(N_vals, dtype=float)
    y_vals = np.array(y_vals, dtype=float)
    
    A_guess = y_vals[0] - y_vals[-1]
    if A_guess == 0:
        A_guess = 1.0
    B_guess = float(np.min(y_vals))
    p0 = [A_guess, 0.5, B_guess]
    
    # Bounds: alpha > 0.01, B >= 0 (JSD, TV are positive)
    bounds = ([-np.inf, 0.01, 0.0], [np.inf, 3.0, float(np.max(y_vals))])
    
    try:
        popt, _ = curve_fit(power_law, N_vals, y_vals, p0=p0, bounds=bounds, maxfev=10000)
        return popt
    except Exception:
        return [float(y_vals[0] - np.mean(y_vals)), 0.5, float(np.mean(y_vals))]

def plot_scaling_curves(results_by_type, fig_dir):
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # We want to plot JSD and TV scaling curves for each catalog type
    for metric_name in ["jsd", "tv"]:
        fig, ax = plt.subplots(figsize=(7.5, 5))
        colors = {"mixed_uniform": "#619cff", "mixed_low_z_dominated": "#f8766d", "mixed_high_z_dominated": "#00ba38"}
        labels = {"mixed_uniform": "Mixed Uniform", "mixed_low_z_dominated": "Low-z Dominated", "mixed_high_z_dominated": "High-z Dominated"}
        
        for cat_type, data in results_by_type.items():
            N_vals = sorted(data.keys())
            mean_vals = []
            std_vals = []
            
            for N in N_vals:
                vals = [run[metric_name] for run in data[N]]
                mean_vals.append(np.mean(vals))
                std_vals.append(np.std(vals))
                
            mean_vals = np.array(mean_vals)
            std_vals = np.array(std_vals)
            
            # Plot data points
            ax.errorbar(N_vals, mean_vals, yerr=std_vals, fmt="o", color=colors[cat_type], label=f"{labels[cat_type]} (Data)", capsize=4, elinewidth=1.5)
            
            # Fit and plot trend
            if len(N_vals) >= 3:
                popt = fit_power_law(N_vals, mean_vals)
                N_fit = np.linspace(min(N_vals), max(N_vals), 100)
                y_fit = power_law(N_fit, *popt)
                ax.plot(N_fit, y_fit, color=colors[cat_type], linestyle="--", alpha=0.8,
                        label=f"{labels[cat_type]} Fit (B={popt[2]:.4f})")
                        
        ax.set_xscale("log")
        ax.set_xlabel("Catalog Size (N_events)")
        ax.set_ylabel(metric_name.upper())
        ax.set_title(f"Posterior {metric_name.upper()} Scaling curves")
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / f"{metric_name}_scaling.png", dpi=200)
        plt.close(fig)
