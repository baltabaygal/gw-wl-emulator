#!/usr/bin/env python
"""
Fit power-law exponent P(mu) = A * mu^(-alpha) across candidate threshold ranges
for each source redshift z_s, finding the optimal transition threshold mu_min
that maximizes R^2 (minimized MSE/chi^2).

*** DIAGNOSTIC ONLY -- DO NOT QUOTE THE ALPHA IT PRINTS. ***
This is a count-weighted least-squares fit to the BINNED LOG-DENSITY, which is
biased shallow wherever the bins are sparse. It is the exact estimator
paper_writer.md sec 7 warns against. The symptom is visible in its own output:
z_s = 0.2/0.5/1 return alpha = 1.000 with R^2 = 1.0000, i.e. a non-normalizable
tail fitted to a handful of bins that hold one count each. Use the SURVIVAL
FUNCTION for tail exponents (docs/edge_tail_flux_note.md sec 2), and let
fig:magpdf_tail -- the compensated mu^2 dP_I/dmu plot -- carry the claim in the
paper. No number from this script appears in the draft (user, 2026-07-29).
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from paper_prod.plot_style import apply_style, FIGURE_SIZES

DATA_FILE = REPO / "paper_prod" / "plots" / "data" / "magpdf_combined_high.npz"
OUT_FIG = REPO / "paper_prod" / "plots" / "figures" / "fig_tail_fit_analysis.png"

def fit_power_law(centers, dpdmu, mu_min, mu_max=100.0, min_bins=5):
    mask = (centers >= mu_min) & (centers <= mu_max) & (dpdmu > 0) & np.isfinite(dpdmu)
    if np.count_nonzero(mask) < min_bins:
        return None
    x = np.log(centers[mask])
    y = np.log(dpdmu[mask])
    
    # Linear fit in log-log space: ln(dP/dmu) = ln(A) - alpha * ln(mu)
    slope, intercept = np.polyfit(x, y, 1)
    alpha = -slope
    y_pred = intercept + slope * x
    
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-12))
    mse = ss_res / len(y)
    
    return {
        "mu_min": mu_min,
        "alpha": alpha,
        "A": np.exp(intercept),
        "r2": r2,
        "mse": mse,
        "nbins": len(y),
        "x_fit": centers[mask],
        "y_fit": dpdmu[mask],
        "y_pred": np.exp(y_pred)
    }

def scan_fit_for_zs(centers, dpdmu, counts):
    candidates = np.geomspace(4.0, 25.0, 40)
    best_res = None
    all_results = []
    
    for mmin in candidates:
        res = fit_power_law(centers, dpdmu, mmin)
        if res is not None and res["nbins"] >= 5:
            all_results.append(res)
            if best_res is None or res["r2"] > best_res["r2"]:
                best_res = res
                
    return best_res, all_results

def main():
    apply_style()
    if not DATA_FILE.exists():
        print(f"Data file not found: {DATA_FILE}")
        sys.exit(1)
        
    z = np.load(DATA_FILE, allow_pickle=False)
    edges = z["edges"]
    centers = np.sqrt(edges[:-1] * edges[1:])
    width = np.diff(edges)
    
    zs_keys = [k for k in z.files if k.startswith("counts__full_z")]
    zs_vals = sorted([float(k.replace("counts__full_z", "")) for k in zs_keys])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    cmap = plt.get_cmap("plasma")
    colors = [cmap(t) for t in np.linspace(0.05, 0.85, len(zs_vals))]
    
    print("=" * 70)
    print(f"{'z_s':<6} | {'Opt mu_min':<11} | {'Fitted Exponent alpha':<22} | {'R^2':<8} | {'N_bins'}")
    print("=" * 70)
    
    for zs, col in zip(zs_vals, colors):
        tag = f"full_z{zs:g}"
        counts = z[f"counts__{tag}"]
        ntot = int(z[f"ntot__{tag}"])
        dpdmu = counts / (ntot * width)
        
        # Fit scan
        best, all_res = scan_fit_for_zs(centers, dpdmu, counts)
        
        if best is not None:
            print(f"{zs:<6.1f} | {best['mu_min']:<11.2f} | {best['alpha']:<22.3f} | {best['r2']:<8.4f} | {best['nbins']}")
            
            # Plot main density curve
            m_valid = (counts >= 5) & (centers >= 0.3) & (centers <= 100)
            ax1.plot(centers[m_valid], dpdmu[m_valid], color=col, lw=1.2, label=fr"$z_s = {zs:g}$")
            
            # Overlay optimal power-law fit line
            fit_x = np.geomspace(best["mu_min"], 100.0, 100)
            fit_y = best["A"] * (fit_x ** (-best["alpha"]))
            ax1.plot(fit_x, fit_y, color=col, ls="--", lw=1.4, alpha=0.9)
            
            # Plot R^2 vs mu_min scan on right panel
            scan_m = [r["mu_min"] for r in all_res]
            scan_r2 = [r["r2"] for r in all_res]
            ax2.plot(scan_m, scan_r2, color=col, lw=1.2, label=fr"$z_s = {zs:g}$")
            ax2.scatter([best["mu_min"]], [best["r2"]], color=col, s=25, zorder=4)

    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlim(0.3, 100); ax1.set_ylim(1e-5, 30)
    ax1.set_xlabel(r"$\mu$"); ax1.set_ylabel(r"$\mathrm{d}P_I/\mathrm{d}\mu$")
    ax1.set_title("PDF with Per-Redshift Optimal Power-Law Fits (dashed)")
    ax1.legend(frameon=False, fontsize=8)
    
    ax2.set_xscale("log")
    ax2.set_xlim(1.5, 25)
    ax2.set_xlabel(r"Candidate $\mu_{\rm min}$ threshold")
    ax2.set_ylabel(r"Fit Quality $R^2$")
    ax2.set_title(r"Fit Quality $R^2$ vs Transition Threshold $\mu_{\rm min}$")
    ax2.grid(True, ls=":", alpha=0.4)
    ax2.legend(frameon=False, fontsize=8)
    
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=300)
    print("=" * 70)
    print(f"Wrote analysis plot to {OUT_FIG}")

if __name__ == "__main__":
    main()
