import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from ml.phase3_common import load_nsf_model

# Style setup for publication quality
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14,
    'text.usetex': False,
    'font.family': 'sans-serif'
})

model_path = 'data/models/conditional_nsf_backend_current.pt'
model = load_nsf_model(model_path)
model.eval()

# Select source redshift z_s = 5.0, fiducial Planck parameters
ctx = torch.tensor([[5.0, 0.674, 0.315, 0.811]], dtype=torch.float32)
ctx_norm = (ctx - model.context_mean) / model.context_std

n_samples = 2000000
torch.manual_seed(42)

flow_dist = model.flow(ctx_norm)
transform = flow_dist.transform

# Draw samples directly in standardized latent space z ~ N(0, 1)
z_base = flow_dist.base.sample((n_samples,))

# Collect generative intermediate realizations using layer.inv
samples_by_step = []

# Base Gaussian mapped to physical mu space
lnmu_base = (z_base * model.lnmu_std + model.lnmu_mean).squeeze().detach().numpy()
samples_by_step.append(('Base Gaussian ($z \\sim \\mathcal{N}(0,1)$)', np.exp(lnmu_base)))

curr = z_base
for k, layer in enumerate(transform.transforms):
    curr = layer.inv(curr)  # In Zuko, layer.inv maps latent z -> data x (generative direction!)
    lnmu_k = (curr * model.lnmu_std + model.lnmu_mean).squeeze().detach().numpy()
    samples_by_step.append((f'Layer {k+1}', np.exp(lnmu_k)))

# Create 8-panel figure (Base + 6 layers + summary overlay) - LIN-LIN SPACE
fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), sharex=True, sharey=True)
axes = axes.flatten()

colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(samples_by_step)))

# Fine linear grid in mu for z_s = 5.0
mu_bins = np.linspace(0.3, 3.5, 300)
centers = 0.5 * (mu_bins[:-1] + mu_bins[1:])

# Compute full model log_prob density grid (Hybrid Composite PDF)
mu_grid = np.linspace(0.3, 3.5, 500)
lnmu_grid = np.log(mu_grid)
with torch.no_grad():
    log_prob_lnmu = model.log_prob(lnmu_grid, ctx)
    prob_lnmu = np.exp(log_prob_lnmu)
    prob_mu_model = prob_lnmu / mu_grid

for i, (title, mu_samples) in enumerate(samples_by_step):
    ax = axes[i]
    hist, _ = np.histogram(mu_samples, bins=mu_bins, density=True)
    hist_smooth = gaussian_filter1d(hist, sigma=1.0)
    
    ax.plot(centers, hist_smooth, color=colors[i], linewidth=2.2)
    ax.fill_between(centers, 0, hist_smooth, color=colors[i], alpha=0.20)
    
    # Overlay full target density on final layer to demonstrate exact match!
    if i == 6:  # Layer 6
        ax.plot(mu_grid, prob_mu_model, 'k--', linewidth=1.8, label=r'Full $\mathrm{d}P/\mathrm{d}\mu$')
        ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
        
    ax.set_title(title, fontweight='bold', color='#111111')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.set_xlim(0.3, 3.2)
    ax.set_ylim(0.0, 2.8)
    if i >= 4:
        ax.set_xlabel(r'Magnification $\mu$')
    if i % 4 == 0:
        ax.set_ylabel(r'$\mathrm{d}P/\mathrm{d}\mu$')

# 8th panel: Summary Progression overlay
ax8 = axes[7]
for i, (title, mu_samples) in enumerate(samples_by_step):
    hist, _ = np.histogram(mu_samples, bins=mu_bins, density=True)
    hist_smooth = gaussian_filter1d(hist, sigma=1.0)
    label = 'Base' if i == 0 else (f'L{i}' if i < 6 else 'Raw Flow L6')
    lw = 1.2 if i < 6 else 2.2
    alpha = 0.55 if i < 6 else 0.85
    ax8.plot(centers, hist_smooth, label=label, color=colors[i], linewidth=lw, alpha=alpha)

# Plot full post-calibrated model density grid
ax8.plot(mu_grid, prob_mu_model, 'k--', linewidth=2.2, label=r'Full $\mathrm{d}P/\mathrm{d}\mu$ (Hybrid)')

ax8.set_title('Progression Overlay ($z_s=5.0$)', fontweight='bold')
ax8.grid(True, linestyle=':', alpha=0.5)
ax8.set_xlabel(r'Magnification $\mu$')
ax8.set_xlim(0.3, 3.2)
ax8.set_ylim(0.0, 2.8)
ax8.legend(fontsize=7.5, loc='upper right', framealpha=0.9, ncol=2)

plt.suptitle(r'Generative Layer-by-Layer Density Transformation in Lin-Lin Space ($z_s=5.0$)', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()

os.makedirs('paper_prod/plots', exist_ok=True)
pdf_path = 'paper_prod/plots/fig_flow_transformation_steps.pdf'
png_path = 'paper_prod/plots/fig_flow_transformation_steps.png'

plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
plt.savefig(png_path, dpi=200, bbox_inches='tight')
plt.close()

print(f"Generative flow transformation figure in lin-lin space for zs=5.0 successfully saved to {pdf_path} and {png_path}")
