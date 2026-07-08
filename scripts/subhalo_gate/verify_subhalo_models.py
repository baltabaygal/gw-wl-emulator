import sys
import os
import numpy as np
import time

sys.path.insert(0, 'build')
import gwlensing

# Parameters matching ace_gap.py
ZS = 1.0
COSMO = dict(OmegaM=0.315, sigma8=0.811, h=0.674)
N = 1000  # Modest N for fast timing

print("=== Running Subhalo Calibration Verification ===")
print(f"Nrealizations = {N}, zs = {ZS}")

# 1. Baseline: Subhalo OFF
t0 = time.time()
raw_off = gwlensing.sample_lensing_raw_ml(
    ZS, COSMO['h'], COSMO['OmegaM'], COSMO['sigma8'], N, seed=42,
    filaments=False, bias=False, ell=False, subhalo=False
)
t_off = time.time() - t0
k_off = np.asarray(raw_off['kappa'])
k_off = k_off - k_off.mean()
var_off = (k_off**2).mean()
skew_off = (k_off**3).mean() / var_off**1.5
print(f"Baseline (OFF): Var={var_off:.6e}, Skew={skew_off:.4f}, Time={t_off:.2f}s")

# Helper to run and compute statistics
def run_case(name, model, brute):
    t0 = time.time()
    raw = gwlensing.sample_lensing_raw_ml(
        ZS, COSMO['h'], COSMO['OmegaM'], COSMO['sigma8'], N, seed=42,
        filaments=False, bias=False, ell=False,
        subhalo=True, m_floor=1e7, subhalo_model=model, subhalo_brute=brute
    )
    dt = time.time() - t0
    k = np.asarray(raw['kappa'])
    k = k - k.mean()
    var = (k**2).mean()
    skew = (k**3).mean() / var**1.5
    shift_var = (var - var_off) / var_off * 100.0
    return var, skew, shift_var, dt

# 2. Option A (Dynamic Floor)
var_ad, skew_ad, shift_ad, t_ad = run_case("Option A (Dynamic)", model=0, brute=False)
print(f"Option A (Dyn): Var={var_ad:.6e} ({shift_ad:+.2f}%), Skew={skew_ad:.4f}, Time={t_ad:.2f}s")

# 3. Option A (Brute-Force Reference)
var_ab, skew_ab, shift_ab, t_ab = run_case("Option A (Brute)", model=0, brute=True)
print(f"Option A (Brut): Var={var_ab:.6e} ({shift_ab:+.2f}%), Skew={skew_ab:.4f}, Time={t_ab:.2f}s")

# 4. Option B (Dynamic Floor)
var_bd, skew_bd, shift_bd, t_bd = run_case("Option B (Dynamic)", model=1, brute=False)
print(f"Option B (Dyn): Var={var_bd:.6e} ({shift_bd:+.2f}%), Skew={skew_bd:.4f}, Time={t_bd:.2f}s")

# 5. Option B (Brute-Force Reference)
var_bb, skew_bb, shift_bb, t_bb = run_case("Option B (Brute)", model=1, brute=True)
print(f"Option B (Brut): Var={var_bb:.6e} ({shift_bb:+.2f}%), Skew={skew_bb:.4f}, Time={t_bb:.2f}s")

# Assertions to check correctness
# Dynamic floor should agree with brute force to within statistical error
err_var_a = abs(var_ad - var_ab) / var_ab * 100.0
err_var_b = abs(var_bd - var_bb) / var_bb * 100.0

print("\n=== Validation Summary ===")
print(f"Option A (Dyn vs Brut) Var discrepancy: {err_var_a:.4f}%")
print(f"Option B (Dyn vs Brut) Var discrepancy: {err_var_b:.4f}%")

if err_var_a < 0.5:
    print("SUCCESS: Option A dynamic floor is consistent with brute-force reference!")
else:
    print("WARNING: Option A dynamic floor discrepancy is high!")

if err_var_b < 0.5:
    print("SUCCESS: Option B dynamic floor is consistent with brute-force reference!")
else:
    print("WARNING: Option B dynamic floor discrepancy is high!")

print(f"\nShift in Variance: Option A = {shift_ad:+.2f}%, Option B = {shift_bd:+.2f}%")

