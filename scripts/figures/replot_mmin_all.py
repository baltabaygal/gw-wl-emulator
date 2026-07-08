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
from ml.nsf_model import ConditionalNSF

# Parameters
Om = 0.3
h = 0.7
w = -1.0
s8 = 0.8
z = 1.0

print(f"Generating full Mmin comparison plots at: Om={Om}, h={h}, s8={s8}, z={z}")

# Sampling params
Nreal = 100000
seed = 42

# 1. Run C++ samplers for different Mmin
print("Running C++ samplers...")
# Mmin = 1e7
lnmu_m1e7 = gw.sample_lnmu_ml(z, h, Om, s8, Nreal, seed, False, 1.0e7)
mu_m1e7 = np.exp(lnmu_m1e7[np.isfinite(lnmu_m1e7)])
mu_prime_m1e7 = mu_m1e7 / np.mean(mu_m1e7)

# Mmin = 1e10
lnmu_m1e10 = gw.sample_lnmu_ml(z, h, Om, s8, Nreal, seed, False, 1.0e10)
mu_m1e10 = np.exp(lnmu_m1e10[np.isfinite(lnmu_m1e10)])
mu_prime_m1e10 = mu_m1e10 / np.mean(mu_m1e10)

# Mmin = 1e12
lnmu_m1e12 = gw.sample_lnmu_ml(z, h, Om, s8, Nreal, seed, False, 1.0e12)
mu_m1e12 = np.exp(lnmu_m1e12[np.isfinite(lnmu_m1e12)])
mu_prime_m1e12 = mu_m1e12 / np.mean(mu_m1e12)

# Mmin = 1e13
lnmu_m1e13 = gw.sample_lnmu_ml(z, h, Om, s8, Nreal, seed, False, 1.0e13)
mu_m1e13 = np.exp(lnmu_m1e13[np.isfinite(lnmu_m1e13)])
mu_prime_m1e13 = mu_m1e13 / np.mean(mu_m1e13)

# Mmin = 1e14
lnmu_m1e14 = gw.sample_lnmu_ml(z, h, Om, s8, Nreal, seed, False, 1.0e14)
mu_m1e14 = np.exp(lnmu_m1e14[np.isfinite(lnmu_m1e14)])
mu_prime_m1e14 = mu_m1e14 / np.mean(mu_m1e14)

# 2. Get ACE prediction
print("Loading ACE model...")
mu_vec_ace, pdf_ace = m.predict_pdf(Om=Om, h=h, w=w, s8=s8, z=z)

# 3. Get NSF prediction
print("Loading NSF model...")
nsf_model = ConditionalNSF(input_dim=1, context_dim=4)
nsf_model.load_checkpoint("/Users/baltabay/Desktop/gw-wl-emulator/data/models/conditional_nsf_backend_current.pt")
nsf_model.eval()

context = np.array([z, h, Om, s8], dtype=np.float32)
lnmu_grid = np.log(mu_vec_ace * np.mean(mu_m1e7))
density_nsf_lnmu = nsf_model.density_grid(context, lnmu_grid)
pdf_nsf = density_nsf_lnmu / mu_vec_ace
dx = np.diff(mu_vec_ace)
pdf_nsf_norm = pdf_nsf / np.sum(pdf_nsf[:-1] * dx)

# 4. Construct bins for histograms
bin_edges = np.zeros(len(mu_vec_ace) + 1)
bin_edges[1:-1] = 0.5 * (mu_vec_ace[:-1] + mu_vec_ace[1:])
bin_edges[0] = mu_vec_ace[0] - (mu_vec_ace[1] - mu_vec_ace[0])/2.0
bin_edges[-1] = mu_vec_ace[-1] + (mu_vec_ace[-1] - mu_vec_ace[-2])/2.0

pdf_cpp_m1e7 = np.histogram(mu_prime_m1e7, bins=bin_edges)[0] / (np.diff(bin_edges) * len(mu_prime_m1e7))
pdf_cpp_m1e10 = np.histogram(mu_prime_m1e10, bins=bin_edges)[0] / (np.diff(bin_edges) * len(mu_prime_m1e10))
pdf_cpp_m1e12 = np.histogram(mu_prime_m1e12, bins=bin_edges)[0] / (np.diff(bin_edges) * len(mu_prime_m1e12))
pdf_cpp_m1e13 = np.histogram(mu_prime_m1e13, bins=bin_edges)[0] / (np.diff(bin_edges) * len(mu_prime_m1e13))
pdf_cpp_m1e14 = np.histogram(mu_prime_m1e14, bins=bin_edges)[0] / (np.diff(bin_edges) * len(mu_prime_m1e14))

# ----------------- PLOT 1: LOG-LIN -----------------
print("Generating log-lin plot...")
fig, ax = plt.subplots(figsize=(11, 7))

# Models
ax.plot(mu_vec_ace, pdf_ace, label="ACE Model (Point Source Lensing)", color="#1f77b4", lw=2.5)
ax.plot(mu_vec_ace, pdf_nsf_norm, label="NSF Model (GW Lensing - Mmin=1e7)", color="#ff7f0e", lw=2.5)

# Cpp Mmins
ax.step(mu_vec_ace, pdf_cpp_m1e7, where='mid', label="C++ Simulator (Mmin=1e7)", color="#2ca02c", alpha=0.8, ls='-')
ax.step(mu_vec_ace, pdf_cpp_m1e10, where='mid', label="C++ Simulator (Mmin=1e10)", color="#9467bd", alpha=0.8, ls='-.')
ax.step(mu_vec_ace, pdf_cpp_m1e12, where='mid', label="C++ Simulator (Mmin=1e12)", color="#d62728", alpha=0.8, ls='--')
ax.step(mu_vec_ace, pdf_cpp_m1e13, where='mid', label="C++ Simulator (Mmin=1e13)", color="#e377c2", alpha=0.8, ls=(0, (3, 1, 1, 1)))
ax.step(mu_vec_ace, pdf_cpp_m1e14, where='mid', label="C++ Simulator (Mmin=1e14)", color="#8c564b", alpha=0.8, ls=':')

ax.set_yscale('log')
ax.set_ylim(1e-3, 200.0)
ax.set_xlim(0.6, 1.8)

ax.set_xlabel(r"Normalized Magnification $\mu' = \mu / \langle\mu\rangle$")
ax.set_ylabel("Probability Density (Log Scale)")
ax.set_title(f"Lensing PDF Mmin Comparison (Log-Lin Plane) at z={z}, Om={Om}, s8={s8}, h={h}")
ax.legend(loc="upper right", framealpha=0.9)
ax.grid(True, which="both", ls="--", alpha=0.3)

plt.tight_layout()
os.makedirs("/Users/baltabay/Desktop/gw-wl-emulator/plots/figures", exist_ok=True)
plot_path_loglin = "/Users/baltabay/Desktop/gw-wl-emulator/plots/figures/compare_all_mmin_loglin.png"
fig.savefig(plot_path_loglin, dpi=200)
plt.close(fig)
print(f"Saved: {plot_path_loglin}")

# ----------------- PLOT 2: LIN-LIN -----------------
print("Generating lin-lin plot...")
fig, ax = plt.subplots(figsize=(11, 7))

# Models
ax.plot(mu_vec_ace, pdf_ace, label="ACE Model (Point Source Lensing)", color="#1f77b4", lw=2.5)
ax.plot(mu_vec_ace, pdf_nsf_norm, label="NSF Model (GW Lensing - Mmin=1e7)", color="#ff7f0e", lw=2.5)

# Cpp Mmins
ax.step(mu_vec_ace, pdf_cpp_m1e7, where='mid', label="C++ Simulator (Mmin=1e7)", color="#2ca02c", alpha=0.8, ls='-')
ax.step(mu_vec_ace, pdf_cpp_m1e10, where='mid', label="C++ Simulator (Mmin=1e10)", color="#9467bd", alpha=0.8, ls='-.')
ax.step(mu_vec_ace, pdf_cpp_m1e12, where='mid', label="C++ Simulator (Mmin=1e12)", color="#d62728", alpha=0.8, ls='--')
ax.step(mu_vec_ace, pdf_cpp_m1e13, where='mid', label="C++ Simulator (Mmin=1e13)", color="#e377c2", alpha=0.8, ls=(0, (3, 1, 1, 1)))
ax.step(mu_vec_ace, pdf_cpp_m1e14, where='mid', label="C++ Simulator (Mmin=1e14)", color="#8c564b", alpha=0.8, ls=':')

# Auto y-scale or max limit appropriate for the peak showing
ax.set_ylim(0.0, 100.0) # 1e14 peak is very sharp and goes high, let's limit to 100 to keep other details visible
ax.set_xlim(0.6, 1.8)

ax.set_xlabel(r"Normalized Magnification $\mu' = \mu / \langle\mu\rangle$")
ax.set_ylabel("Probability Density (Linear Scale)")
ax.set_title(f"Lensing PDF Mmin Comparison (Lin-Lin Plane) at z={z}, Om={Om}, s8={s8}, h={h}")
ax.legend(loc="upper right", framealpha=0.9)
ax.grid(True, which="both", ls="--", alpha=0.3)

plt.tight_layout()
plot_path_linlin = "/Users/baltabay/Desktop/gw-wl-emulator/plots/figures/compare_all_mmin_linlin.png"
fig.savefig(plot_path_linlin, dpi=200)
plt.close(fig)
print(f"Saved: {plot_path_linlin}")

print("All done!")
