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

# Cosmology benchmark (from Planck 2018 / Vaskonen Fig 2 & 3)
h_val = 0.674
om_val = 0.315
s8_val = 0.811
w_val = -1.0

# Redshift grid (log-spaced or typical redshifts up to 6.0)
redshifts = np.array([0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])

# Lists to store results
cpp_h100_y = []
nsf_y = []
ace_pdf_y = []
ace_sigma_y = []

# Load NSF model
nsf_model = ConditionalNSF(input_dim=1, context_dim=4)
nsf_model.load_checkpoint("/Users/baltabay/Desktop/gw-wl-emulator/data/models/conditional_nsf_backend_current.pt")
nsf_model.eval()

# Sampling params
Nreal = 10000
seed = 42

for z in redshifts:
    print(f"Calculating for redshift z = {z:.2f}...")
    
    # 1. C++ (Nhalos=100)
    try:
        lnmu_h100 = gw.sample_lnmu_ml(float(z), h_val, om_val, s8_val, Nreal, seed)
        valid_lnmu_h100 = lnmu_h100[np.isfinite(lnmu_h100)]
        mu_h100 = np.exp(valid_lnmu_h100)
        # std(1 / sqrt(mu))
        sigma_DL_h100 = np.std(1.0 / np.sqrt(mu_h100))
        cpp_h100_y.append(sigma_DL_h100)
    except Exception as e:
        print(f"Error C++ h100 at z={z}: {e}")
        cpp_h100_y.append(np.nan)
        
    # 3. NSF Model (our method)
    try:
        context = np.array([z, h_val, om_val, s8_val], dtype=np.float32)
        # Draw samples from NSF
        with torch.no_grad():
            lnmu_samples = nsf_model.sample(context, Nreal).flatten()
        mu_nsf = np.exp(lnmu_samples)
        sigma_DL_nsf = np.std(1.0 / np.sqrt(mu_nsf))
        nsf_y.append(sigma_DL_nsf)
    except Exception as e:
        print(f"Error NSF at z={z}: {e}")
        nsf_y.append(np.nan)
        
    # 4. ACE Model (from PDF integration)
    try:
        mu_vec_ace, pdf_ace = m.predict_pdf(Om=om_val, h=h_val, w=w_val, s8=s8_val, z=float(z), verbose=False)
        mu_c = 0.5 * (mu_vec_ace[:-1] + mu_vec_ace[1:])
        dx = np.diff(mu_vec_ace)
        probs = pdf_ace[:-1] * dx
        probs = probs / np.sum(probs) # ensure normalization
        
        mean_inv_mu = np.sum(probs / mu_c)
        mean_inv_sqrt_mu = np.sum(probs / np.sqrt(mu_c))
        var_inv_sqrt_mu = mean_inv_mu - (mean_inv_sqrt_mu ** 2)
        sigma_DL_ace_pdf = np.sqrt(max(0.0, var_inv_sqrt_mu))
        ace_pdf_y.append(sigma_DL_ace_pdf)
    except Exception as e:
        print(f"Error ACE PDF at z={z}: {e}")
        ace_pdf_y.append(np.nan)

    # 5. ACE Model (from predicted sigma/2)
    try:
        sigma_mu_ace = m.predict_sigma(Om=om_val, h=h_val, w=w_val, s8=s8_val, z=float(z), verbose=False)[0]
        # Approximation: sigma_DL / D_L = 0.5 * sigma_mu
        ace_sigma_y.append(0.5 * sigma_mu_ace)
    except Exception as e:
        print(f"Error ACE sigma at z={z}: {e}")
        ace_sigma_y.append(np.nan)

print("\nResults Summary:")
print("Redshifts:", redshifts)
print("C++ (Nhalos=100):", cpp_h100_y)
print("NSF Model:       ", nsf_y)
print("ACE (PDF Int):   ", ace_pdf_y)
print("ACE (0.5*sigma): ", ace_sigma_y)

# Plotting Vaskonen Fig 3 like plot
fig, ax = plt.subplots(figsize=(8.5, 6.5))

# Plot the curves
ax.plot(redshifts, cpp_h100_y, label="C++ Simulator (Nhalos=100)", color="#2ca02c", marker='o', ls='--', lw=2)
ax.plot(redshifts, nsf_y, label="NSF Model (GW Lensing - Nhalos=100)", color="#ff7f0e", marker='x', ls='-', lw=2)
ax.plot(redshifts, ace_pdf_y, label="ACE Model (from predicted PDF)", color="#1f77b4", marker='^', ls='-', lw=2)
ax.plot(redshifts, ace_sigma_y, label="ACE Model (0.5 * predicted sigma)", color="#9467bd", marker='d', ls='-.', lw=1.5)

# Style plot
ax.set_xscale('log')
ax.set_xlim(0.18, 6.5)
ax.set_ylim(0.0, 0.08) # Vaskonen Fig 3 y-range goes up to 0.10, but we go to 6.0 so 0.08 is perfect

# Custom tick formatters for log scale
from matplotlib.ticker import FormatStrFormatter
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.set_xticks([0.2, 0.5, 1.0, 2.0, 5.0, 6.0])

ax.set_xlabel("Redshift $z$ (Log Scale)")
ax.set_ylabel(r"Distance Scatter $\sigma_{D_L} / D_L$")
ax.set_title(f"Lensing-Induced Luminosity Distance Scatter\n(Planck 2018 Cosmology: $\Omega_M={om_val}$, $\sigma_8={s8_val}$, $h={h_val}$)")
ax.legend(loc="upper left")
ax.grid(True, which="both", ls="--", alpha=0.3)

plt.tight_layout()
os.makedirs("/Users/baltabay/Desktop/gw-wl-emulator/plots/figures", exist_ok=True)
plot_path = "/Users/baltabay/Desktop/gw-wl-emulator/plots/figures/compare_vaskonen_fig3.png"
fig.savefig(plot_path, dpi=200)
plt.close(fig)

print(f"\nFigure 3 comparison saved to {plot_path}")
