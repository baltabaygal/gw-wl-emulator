import ace_lensing.model as m

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Ensure we can import gwlensing
sys.path.insert(0, "/Users/baltabay/Desktop/gw-wl-emulator/build")
sys.path.insert(0, "/Users/baltabay/Desktop/gw-wl-emulator")

import gwlensing as gw

# Parameters
Om = 0.3
h = 0.7
w = -1.0
s8 = 0.8
z = 1.0

Nreal = 100000
seed = 42

print(f"Generating best-fit Mmin=1e13 comparison plot...")

# 1. Run C++ samplers
# Base Mmin = 1e7
lnmu_m1e7 = gw.sample_lnmu_ml(z, h, Om, s8, Nreal, seed, False, 1.0e7)
mu_m1e7 = np.exp(lnmu_m1e7[np.isfinite(lnmu_m1e7)])
mu_prime_m1e7 = mu_m1e7 / np.mean(mu_m1e7)

# Best-fit Mmin = 1e13
lnmu_m1e13 = gw.sample_lnmu_ml(z, h, Om, s8, Nreal, seed, False, 1.0e13)
mu_m1e13 = np.exp(lnmu_m1e13[np.isfinite(lnmu_m1e13)])
mu_prime_m1e13 = mu_m1e13 / np.mean(mu_m1e13)

# 2. Get ACE prediction
mu_vec_ace, pdf_ace = m.predict_pdf(Om=Om, h=h, w=w, s8=s8, z=z)

# 3. Bin C++ simulations
bin_edges = np.zeros(len(mu_vec_ace) + 1)
bin_edges[1:-1] = 0.5 * (mu_vec_ace[:-1] + mu_vec_ace[1:])
bin_edges[0] = mu_vec_ace[0] - (mu_vec_ace[1] - mu_vec_ace[0])/2.0
bin_edges[-1] = mu_vec_ace[-1] + (mu_vec_ace[-1] - mu_vec_ace[-2])/2.0

pdf_cpp_m1e7 = np.histogram(mu_prime_m1e7, bins=bin_edges)[0] / (np.diff(bin_edges) * len(mu_prime_m1e7))
pdf_cpp_m1e13 = np.histogram(mu_prime_m1e13, bins=bin_edges)[0] / (np.diff(bin_edges) * len(mu_prime_m1e13))

# Plotting: Two panels side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# ---- Panel 1: Log-Lin (Tail focus) ----
ax1.plot(mu_vec_ace, pdf_ace, label="ACE Model", color="#1f77b4", lw=3.0)
ax1.step(mu_vec_ace, pdf_cpp_m1e13, where='mid', label="C++ Simulator (Best Fit: Mmin=1e13)", color="#e377c2", lw=2.5, ls='-')
ax1.step(mu_vec_ace, pdf_cpp_m1e7, where='mid', label="C++ Simulator (Baseline: Mmin=1e7)", color="#7f7f7f", alpha=0.5, ls='--')

ax1.set_yscale('log')
ax1.set_ylim(1e-3, 100.0)
ax1.set_xlim(0.6, 1.8)
ax1.set_xlabel(r"Normalized Magnification $\mu' = \mu / \langle\mu\rangle$")
ax1.set_ylabel("Probability Density (Log Scale)")
ax1.set_title("PDF Comparison: Log-Lin Plane (Tail & Caustics)")
ax1.legend(framealpha=0.9)
ax1.grid(True, which="both", ls="--", alpha=0.3)

# ---- Panel 2: Lin-Lin (Bulk/Peak focus) ----
ax2.plot(mu_vec_ace, pdf_ace, label="ACE Model", color="#1f77b4", lw=3.0)
ax2.step(mu_vec_ace, pdf_cpp_m1e13, where='mid', label="C++ Simulator (Best Fit: Mmin=1e13)", color="#e377c2", lw=2.5, ls='-')
ax2.step(mu_vec_ace, pdf_cpp_m1e7, where='mid', label="C++ Simulator (Baseline: Mmin=1e7)", color="#7f7f7f", alpha=0.5, ls='--')

ax2.set_ylim(0.0, 32.0)
ax2.set_xlim(0.6, 1.8)
ax2.set_xlabel(r"Normalized Magnification $\mu' = \mu / \langle\mu\rangle$")
ax2.set_ylabel("Probability Density (Linear Scale)")
ax2.set_title("PDF Comparison: Lin-Lin Plane (Bulk Distribution)")
ax2.legend(framealpha=0.9)
ax2.grid(True, which="both", ls="--", alpha=0.3)

plt.suptitle(f"ACE Model vs. C++ Lensing Best-Fit (z={z}, Om={Om}, s8={s8}, h={h})", fontsize=14, y=0.98)
plt.tight_layout()

os.makedirs("/Users/baltabay/Desktop/gw-wl-emulator/plots/figures", exist_ok=True)
plot_path = "/Users/baltabay/Desktop/gw-wl-emulator/plots/figures/compare_best_fit.png"
fig.savefig(plot_path, dpi=200)
plt.close(fig)

print(f"Saved best-fit comparison to: {plot_path}")
