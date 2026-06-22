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

print(f"Comparing in lin-lin plane at: Om={Om}, h={h}, s8={s8}, z={z}")

# 1. Run C++ sampler (halos = 100)
Nreal = 100000
seed = 42
lnmu_h100 = gw.sample_lnmu_ml(z, h, Om, s8, Nreal, seed)
valid_lnmu_h100 = lnmu_h100[np.isfinite(lnmu_h100)]
mu_h100 = np.exp(valid_lnmu_h100)
mean_mu_h100 = np.mean(mu_h100)
mu_prime_h100 = mu_h100 / mean_mu_h100

# 3. Get ACE prediction
mu_vec_ace, pdf_ace = m.predict_pdf(Om=Om, h=h, w=w, s8=s8, z=z)

# 4. Get NSF prediction
nsf_model = ConditionalNSF(input_dim=1, context_dim=4)
nsf_model.load_checkpoint("/Users/baltabay/Desktop/gw-wl-emulator/data/models/conditional_nsf_backend_current.pt")
nsf_model.eval()

context = np.array([z, h, Om, s8], dtype=np.float32)
# Evaluate NSF on a grid mapping from mu_vec_ace
lnmu_grid = np.log(mu_vec_ace * mean_mu_h100)
density_nsf_lnmu = nsf_model.density_grid(context, lnmu_grid)
pdf_nsf = density_nsf_lnmu / mu_vec_ace
dx = np.diff(mu_vec_ace)
pdf_nsf_norm = pdf_nsf / np.sum(pdf_nsf[:-1] * dx)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Construct bins for histograms
bin_edges = np.zeros(len(mu_vec_ace) + 1)
bin_edges[1:-1] = 0.5 * (mu_vec_ace[:-1] + mu_vec_ace[1:])
bin_edges[0] = mu_vec_ace[0] - (mu_vec_ace[1] - mu_vec_ace[0])/2.0
bin_edges[-1] = mu_vec_ace[-1] + (mu_vec_ace[-1] - mu_vec_ace[-2])/2.0

counts_h100, _ = np.histogram(mu_prime_h100, bins=bin_edges)
pdf_cpp_h100 = counts_h100 / (np.diff(bin_edges) * len(mu_prime_h100))

# Plot lines in lin-lin plane
ax.plot(mu_vec_ace, pdf_ace, label="ACE Model (Point Source Lensing)", color="#1f77b4", lw=2.5)
ax.plot(mu_vec_ace, pdf_nsf_norm, label="NSF Model (GW Lensing - Nhalos=100)", color="#ff7f0e", lw=2.5)
ax.step(mu_vec_ace, pdf_cpp_h100, where='mid', label="C++ Simulator (Nhalos=100)", color="#2ca02c", alpha=0.7, ls='--')

# Lin y scale with limits matching bulk
ax.set_ylim(0.0, 32.0)
ax.set_xlim(0.6, 1.8)  # Same x-axis range as Vaskonen Fig 2

ax.set_xlabel(r"Normalized Magnification $\mu' = \mu / \langle\mu\rangle$")
ax.set_ylabel("Probability Density (Linear Scale)")
ax.set_title(f"Lensing PDF Comparison (Lin-Lin Plane) at z={z}, Om={Om}, s8={s8}, h={h}")
ax.legend()
ax.grid(True, which="both", ls="--", alpha=0.3)

plt.tight_layout()
os.makedirs("/Users/baltabay/Desktop/gw-wl-emulator/plots/figures", exist_ok=True)
plot_path = "/Users/baltabay/Desktop/gw-wl-emulator/plots/figures/compare_all_models_linlin.png"
fig.savefig(plot_path, dpi=200)
plt.close(fig)

print(f"Comparison plot (linlin) saved to {plot_path}")
