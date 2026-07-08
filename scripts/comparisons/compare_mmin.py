import ace_lensing.model as m

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

# Ensure we can import gwlensing and ml
sys.path.insert(0, "/Users/baltabay/Desktop/gw-wl-emulator/build")
sys.path.insert(0, "/Users/baltabay/Desktop/gw-wl-emulator")

import gwlensing as gw

# Parameters
Om = 0.3
h = 0.7
w = -1.0
s8 = 0.8
z = 1.0

print(f"Comparing Mmin effects in log-lin plane at: Om={Om}, h={h}, s8={s8}, z={z}")

# Sampling params
Nreal = 100000
seed = 42

# 1. Run C++ sampler with halos = 100 and default Mmin = 1e7
lnmu_h100_m1e7 = gw.sample_lnmu_ml(z, h, Om, s8, Nreal, seed, False, 1.0e7)
valid_lnmu_h100_m1e7 = lnmu_h100_m1e7[np.isfinite(lnmu_h100_m1e7)]
mu_h100_m1e7 = np.exp(valid_lnmu_h100_m1e7)
mean_mu_h100_m1e7 = np.mean(mu_h100_m1e7)
mu_prime_h100_m1e7 = mu_h100_m1e7 / mean_mu_h100_m1e7

# 2. Run C++ sampler with halos = 100 and Mmin = 1e12
lnmu_h100_m1e12 = gw.sample_lnmu_ml(z, h, Om, s8, Nreal, seed, False, 1.0e12)
valid_lnmu_h100_m1e12 = lnmu_h100_m1e12[np.isfinite(lnmu_h100_m1e12)]
mu_h100_m1e12 = np.exp(valid_lnmu_h100_m1e12)
mean_mu_h100_m1e12 = np.mean(mu_h100_m1e12)
mu_prime_h100_m1e12 = mu_h100_m1e12 / mean_mu_h100_m1e12

# 3. Get ACE prediction
mu_vec_ace, pdf_ace = m.predict_pdf(Om=Om, h=h, w=w, s8=s8, z=z)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Construct bins for histograms using ACE's mu_vec_ace
bin_edges = np.zeros(len(mu_vec_ace) + 1)
bin_edges[1:-1] = 0.5 * (mu_vec_ace[:-1] + mu_vec_ace[1:])
bin_edges[0] = mu_vec_ace[0] - (mu_vec_ace[1] - mu_vec_ace[0])/2.0
bin_edges[-1] = mu_vec_ace[-1] + (mu_vec_ace[-1] - mu_vec_ace[-2])/2.0

counts_h100_m1e7, _ = np.histogram(mu_prime_h100_m1e7, bins=bin_edges)
pdf_cpp_h100_m1e7 = counts_h100_m1e7 / (np.diff(bin_edges) * len(mu_prime_h100_m1e7))

counts_h100_m1e12, _ = np.histogram(mu_prime_h100_m1e12, bins=bin_edges)
pdf_cpp_h100_m1e12 = counts_h100_m1e12 / (np.diff(bin_edges) * len(mu_prime_h100_m1e12))

# Plot lines in log-lin plane
ax.plot(mu_vec_ace, pdf_ace, label="ACE Model (Point Source Lensing)", color="#1f77b4", lw=2.5)
ax.step(mu_vec_ace, pdf_cpp_h100_m1e7, where='mid', label="C++ Simulator (Nhalos=100, Mmin=1e7)", color="#2ca02c", alpha=0.8, ls='-')
ax.step(mu_vec_ace, pdf_cpp_h100_m1e12, where='mid', label="C++ Simulator (Nhalos=100, Mmin=1e12)", color="#ff7f0e", alpha=0.8, ls='--')

# Log y scale
ax.set_yscale('log')
ax.set_ylim(1e-3, 100.0)
ax.set_xlim(0.6, 1.8)  # Vaskonen Fig 2 range

ax.set_xlabel(r"Normalized Magnification $\mu' = \mu / \langle\mu\rangle$")
ax.set_ylabel("Probability Density (Log Scale)")
ax.set_title(f"Lensing PDF Mmin Comparison (Log-Lin Plane) at z={z}, Om={Om}, s8={s8}, h={h}")
ax.legend()
ax.grid(True, which="both", ls="--", alpha=0.3)

plt.tight_layout()
os.makedirs("/Users/baltabay/Desktop/gw-wl-emulator/plots/figures", exist_ok=True)
plot_path = "/Users/baltabay/Desktop/gw-wl-emulator/plots/figures/compare_mmin.png"
fig.savefig(plot_path, dpi=200)
plt.close(fig)

print(f"Mmin comparison plot saved to {plot_path}")
