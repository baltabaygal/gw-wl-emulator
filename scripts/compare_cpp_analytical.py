# CAVEAT (2026-07-02): the "Delta" ratio printed below is NOT expected to be ~1.
# The screen's dK2c/dK3c is the unclustered-Poisson clump component only; the C++ MC delta
# additionally contains within-host clump clustering + host-clump covariance (dominant)
# and the (1-f_s)M host reduction. For the split-vs-brute validation use
# scripts/validate_split_vs_brute.py; see docs/subhalo_combining.md "Validation".
import sys
from pathlib import Path
import numpy as np

# Add the build directory to path for gwlensing import
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build"))

import gwlensing as gw

# Setup parameters matching the screen run
z_s = 5.0
h = 0.674
OmegaM = 0.315
sigma8 = 0.811
nsamples = 10_000
seed = 42

print(f"Running C++ sampler with {nsamples} realizations for z_s = {z_s}...")
print("Sampling host-only (no subhalos)...")
host_raw = gw.sample_lensing_raw_ml(
    z=z_s, h=h, OmegaM=OmegaM, sigma8=sigma8,
    nsamples=nsamples, seed=seed,
    filaments=False, bias=False, ell=False,
    Nhalos=100, subhalo=False, m_floor=1e7,
    subhalo_threads=8
)
k_host = np.asarray(host_raw["kappa"], dtype=float)

print("Sampling host + subhalos...")
sub_raw = gw.sample_lensing_raw_ml(
    z=z_s, h=h, OmegaM=OmegaM, sigma8=sigma8,
    nsamples=nsamples, seed=seed,
    filaments=False, bias=False, ell=False,
    Nhalos=100, subhalo=True, m_floor=1e7,
    subhalo_threads=8
)
k_sub = np.asarray(sub_raw["kappa"], dtype=float)

# Calculate sample moments from the C++ realizations
var_host_cpp = np.var(k_host)
var_sub_cpp = np.var(k_sub)
d_var_cpp = var_sub_cpp - var_host_cpp

# 3rd central moment: <(kappa - <kappa>)^3>
m3_host_cpp = np.mean((k_host - np.mean(k_host))**3)
m3_sub_cpp = np.mean((k_sub - np.mean(k_sub))**3)
d_m3_cpp = m3_sub_cpp - m3_host_cpp

# Load analytical Campbell screen results from scan
try:
    scan = np.load('plots/screen_zs_scan.npz')
    zs_arr = scan['zs']
    idx = np.where(np.isclose(zs_arr, z_s))[0][0]
    
    K2_model = scan['K2_model'][idx]
    dK2c = scan['dK2c'][idx]
    K3_model = scan['K3_model'][idx]
    dK3c = scan['dK3c'][idx]
except Exception as e:
    print(f"Could not load plots/screen_zs_scan.npz: {e}")
    K2_model = dK2c = K3_model = dK3c = 0.0

# Print comparison
print("\n" + "="*80)
print(f"Moment Comparison for z_s = {z_s} (C++ vs Campbell Analytical)")
print("="*80)

print(f"{'Moment / Quantity':<30} | {'C++ Sampler':<15} | {'Analytical':<15} | {'Ratio':<10}")
print("-"*80)

# Variance (2nd moment)
print(f"{'Variance <kappa^2> (Host)':<30} | {var_host_cpp:15.3e} | {K2_model:15.3e} | {var_host_cpp/K2_model:10.3f}")
print(f"{'Variance <kappa^2> (Host+Sub)':<30} | {var_sub_cpp:15.3e} | {K2_model+dK2c:15.3e} | {var_sub_cpp/(K2_model+dK2c):10.3f}")
print(f"{'Delta <kappa^2> (Subhalo)':<30} | {d_var_cpp:15.3e} | {dK2c:15.3e} | {d_var_cpp/dK2c:10.3f}")
print(f"{'Variance Boost %':<30} | {100*d_var_cpp/var_host_cpp:14.2f}% | {100*dK2c/K2_model:14.2f}% | {'-':<10}")

print("-"*80)

# 3rd moment
print(f"{'3rd Moment <kappa^3> (Host)':<30} | {m3_host_cpp:15.3e} | {K3_model:15.3e} | {m3_host_cpp/K3_model:10.3f}")
print(f"{'3rd Moment <kappa^3> (Host+Sub)':<30} | {m3_sub_cpp:15.3e} | {K3_model+dK3c:15.3e} | {m3_sub_cpp/(K3_model+dK3c):10.3f}")
print(f"{'Delta <kappa^3> (Subhalo)':<30} | {d_m3_cpp:15.3e} | {dK3c:15.3e} | {d_m3_cpp/dK3c:10.3f}")
print(f"{'3rd Moment Boost %':<30} | {100*d_m3_cpp/m3_host_cpp:14.2f}% | {100*dK3c/K3_model:14.2f}% | {'-':<10}")

print("="*80)
