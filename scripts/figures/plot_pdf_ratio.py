import os
import numpy as np
import matplotlib.pyplot as plt

project_dir = "/Users/baltabay/Desktop/gw-wl-emulator"
data_dir = os.path.join(project_dir, "data")
plots_dir = os.path.join(project_dir, "plots", "figures")

os.makedirs(data_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

data_path_z1 = os.path.join(data_dir, "mu_data.npz")
data_path_z2 = os.path.join(data_dir, "mu_data_z2.npz")

# Load cached data
data_z1 = np.load(data_path_z1)
mu_off_z1 = data_z1["mu_off"]
mu_on_z1 = data_z1["mu_on"]

data_z2 = np.load(data_path_z2)
mu_off_z2 = data_z2["mu_off"]
mu_on_z2 = data_z2["mu_on"]

# Print overall statistics
print("=== z = 1.0 Statistics ===")
print(f"Subhalo OFF | Mean mu: {np.mean(mu_off_z1):.6f} | Std: {np.std(mu_off_z1):.6f}")
print(f"Subhalo ON  | Mean mu: {np.mean(mu_on_z1):.6f} | Std: {np.std(mu_on_z1):.6f}")
print(f"Std Ratio (ON/OFF): {np.std(mu_on_z1)/np.std(mu_off_z1):.4f}")

print("\n=== z = 2.0 Statistics ===")
print(f"Subhalo OFF | Mean mu: {np.mean(mu_off_z2):.6f} | Std: {np.std(mu_off_z2):.6f}")
print(f"Subhalo ON  | Mean mu: {np.mean(mu_on_z2):.6f} | Std: {np.std(mu_on_z2):.6f}")
print(f"Std Ratio (ON/OFF): {np.std(mu_on_z2)/np.std(mu_off_z2):.4f}")

# Compute PDF ratios
edges = np.linspace(0.8, 6.0, 50)
centers = 0.5 * (edges[:-1] + edges[1:])

def get_pdf(mu, edges):
    counts, _ = np.histogram(mu, bins=edges)
    widths = np.diff(edges)
    total = counts.sum()
    return counts / (total * widths) if total > 0 else np.zeros_like(widths)

pdf_off_z1 = get_pdf(mu_off_z1, edges)
pdf_on_z1 = get_pdf(mu_on_z1, edges)

pdf_off_z2 = get_pdf(mu_off_z2, edges)
pdf_on_z2 = get_pdf(mu_on_z2, edges)

# Avoid division by zero
mask_z1 = pdf_off_z1 > 0.0
ratio_z1 = np.where(mask_z1, pdf_on_z1 / pdf_off_z1, np.nan)

mask_z2 = pdf_off_z2 > 0.0
ratio_z2 = np.where(mask_z2, pdf_on_z2 / pdf_off_z2, np.nan)

# Plot PDF Ratio
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.figure(figsize=(9, 6))

plt.plot(centers, ratio_z1, color="#d62728", lw=2.5, label="z = 1.0 PDF Ratio (ON / OFF)")
plt.plot(centers, ratio_z2, color="#ff7f0e", lw=2.5, label="z = 2.0 PDF Ratio (ON / OFF)")
plt.axhline(1.0, color="black", linestyle="--", linewidth=1.5, label="No Difference (1.0)")

plt.xlabel(r"Magnification $\mu$", fontsize=12, fontweight="bold")
plt.ylabel(r"PDF Ratio $P_{\rm ON}(\mu) / P_{\rm OFF}(\mu)$", fontsize=12, fontweight="bold")
plt.title("Relative Magnification PDF Shift (Subhalo ON vs. OFF)", fontsize=13, fontweight="bold", pad=15)
plt.xlim(0.8, 5.0)
plt.ylim(0.7, 1.3)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(frameon=True, facecolor="#f8fafc", loc="upper right")
plt.tight_layout()

plot_path = os.path.join(plots_dir, "subhalo_pdf_ratio_comparison.png")
plt.savefig(plot_path, dpi=300, facecolor="#ffffff")
print(f"Saved ratio plot to {plot_path}")
