import numpy as np
import matplotlib.pyplot as plt

# Host parameters (matching subhalo_demo.py)
M0 = 1e15
z0 = 0.5
cH = 2.89
r200H = 1756.0
rsH = r200H / cH

# Bias function model (SatGen Withering + Disruption fit)
# B(x) = 1 / sqrt( (x/0.537)**(-2.485) + 1 )
x0_fit = 0.537
alpha_fit = 2.485

def B_bias(x):
    raw = 1.0 / np.sqrt((x / x0_fit)**(-alpha_fit) + 1.0)
    norm = 1.0 / np.sqrt((1.0 / x0_fit)**(-alpha_fit) + 1.0)
    return raw / norm

# 1. Original NFW sampler (for comparison)
def sample_nfw_radii(n, rs, c, rng):
    mu = lambda x: np.log(1+x) - x/(1+x)
    u = rng.uniform(0, 1, n)
    target = u * mu(c)
    out = np.empty(n)
    from scipy.optimize import brentq
    for i, t in enumerate(target):
        out[i] = brentq(lambda x: mu(x)-t, 1e-6, c) * rs
    return out

# 2. New biased sampler (numerical CDF)
def sample_biased_radii(n, rs, c, rng):
    # Setup grid in ln(x) where x = r/rvir
    lx = np.linspace(np.log(1e-6), np.log(1.0), 8000)
    x_grid = np.exp(lx)
    
    # dP/dlnx = x^2 / (1 + c*x)^2 * B(x)
    integrand = (x_grid**2 / (1.0 + c * x_grid)**2) * B_bias(x_grid)
    
    # Cumulative distribution
    cdf = np.cumsum(integrand)
    cdf /= cdf[-1]
    
    # Sample
    u = rng.uniform(0, 1, n)
    x_sampled = np.exp(np.interp(u, cdf, lx))
    return x_sampled * (rs * c)

# Generate samples
rng = np.random.default_rng(42)
N_samples = 100000

print("Generating NFW samples...")
r_nfw = sample_nfw_radii(N_samples, rsH, cH, rng)

print("Generating biased samples...")
r_biased = sample_biased_radii(N_samples, rsH, cH, rng)

# Compute radial density profiles (number per unit volume)
bin_edges = np.logspace(-2.5, 0.0, 40) * r200H
bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
vol = 4.0/3.0 * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)

hist_nfw, _ = np.histogram(r_nfw, bins=bin_edges)
density_nfw = hist_nfw / vol

hist_biased, _ = np.histogram(r_biased, bins=bin_edges)
density_biased = hist_biased / vol

# Theoretical curves
r_eval = np.logspace(-2.5, 0.0, 200) * r200H
x_eval = r_eval / r200H

# NFW density: n(r) propto 1 / (r * (1 + r/rs)^2)
theory_nfw = 1.0 / (r_eval * (1.0 + r_eval/rsH)**2)
theory_nfw *= density_nfw[0] / theory_nfw[0] # normalize

theory_biased = theory_nfw * B_bias(x_eval)
theory_biased *= density_biased[0] / theory_biased[0] # normalize

# Plot profiles
fig, ax = plt.subplots(figsize=(8, 6))

ax.loglog(bin_centers/r200H, density_nfw, 'ko', alpha=0.5, label='NFW Samples')
ax.loglog(r_eval/r200H, theory_nfw, 'k-', lw=1.5, label='NFW Theory')

ax.loglog(bin_centers/r200H, density_biased, 'gs', alpha=0.5, label='Biased Samples (SatGen)')
ax.loglog(r_eval/r200H, theory_biased, 'g-', lw=2, label=r'Biased Theory: $[(x/0.54)^{-2.5} + 1]^{-1/2}$')

ax.set_xlabel(r'$r/r_{\rm vir}$')
ax.set_ylabel(r'Subhalo Number Density $n(r)\ [{\rm kpc}^{-3}]$')
ax.set_title('Subhalo Radial Number Density Profiles')
ax.grid(alpha=0.3, which='both')
ax.legend()

plt.tight_layout()
plt.savefig('plots/radial_sampling_test.png', dpi=150)
print("Wrote plots/radial_sampling_test.png")
