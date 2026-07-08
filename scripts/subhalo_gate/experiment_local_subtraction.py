import numpy as np
import os
import matplotlib.pyplot as plt

# 1. NFW profile functions
def Fg(x):
    x = np.atleast_1d(x)
    res = np.zeros_like(x)
    
    # x > 1
    idx = x > 1
    if np.any(idx):
        xx = x[idx]
        t = np.arctan(np.sqrt((xx-1)/(1+xx))) / np.sqrt(xx**2 - 1)
        res[idx] = (1.0 - 2.0*t) / (xx**2 - 1.0)
        
    # x < 1
    idx = x < 1
    if np.any(idx):
        xx = x[idx]
        t = np.arctanh(np.sqrt((1.0-xx)/(1+xx))) / np.sqrt(1.0 - xx**2)
        res[idx] = (1.0 - 2.0*t) / (xx**2 - 1.0)
        
    # x == 1
    idx = np.isclose(x, 1.0)
    if np.any(idx):
        res[idx] = 1.0 / 3.0
        
    return res

def get_rs_rhos(M, c):
    # Scale r200 with mass M^(1/3)
    r200 = 1750.0 * (M / 1e15)**(1/3)
    rs = r200 / c
    # rhos is proportional to the concentration-dependent NFW shape factor
    rhos = c**3 * (1.0 + c) / (3.0 * ((1.0 + c)*np.log(1.0 + c) - c))
    return rs, rhos

def kappa_NFW(d, M, c, scale=1.0):
    rs, rhos = get_rs_rhos(M, c)
    kappa0 = rs * rhos * scale
    x = d / rs
    return 2.0 * kappa0 * Fg(x)

# 2. Setup Experiment Parameters
M_total = 1e15      # Total halo mass
m_sub = 1e13        # Subhalo mass
c_host = 3.0        # Host concentration
c_sub = 6.0         # Subhalo concentration
scale_val = 1.3e-6  # Calibration scale to yield realistic kappa values (~0.05 at 100 kpc)

# Ray position: 100 kpc to the left of the host center
r_ray = 100.0

# Subhalo position R: varies from 100 kpc to 800 kpc to the right of the host center
R_vals = np.linspace(100.0, 800.0, 100)

# Calculate host convergences at the ray position (100 kpc)
k_host_full = kappa_NFW(r_ray, M_total, c_host, scale_val)[0]
k_host_reduced = kappa_NFW(r_ray, M_total - m_sub, c_host, scale_val)[0]

# Option A: per-clump subtraction slope (gslope = dkappa_host/dM)
# evaluated at the host mass M_total and ray distance r_ray
Me = 1.02 * M_total
k_host_e = kappa_NFW(r_ray, Me, c_host, scale_val)[0]
gslope = (k_host_e - k_host_full) / (Me - M_total)

print("=== Subhalo Mass Conservation Experiment ===")
print(f"Host Mass (M): {M_total:.2e} Msun")
print(f"Subhalo Mass (m): {m_sub:.2e} Msun")
print(f"Host Kappa (full): {k_host_full:.6f}")
print(f"Host Kappa (globally reduced - Option B): {k_host_reduced:.6f}")
print(f"Lensing ray is at 100 kpc (left)")

# Compute total convergence for each subhalo position R (right)
k_B_vals = []  # Option B: Host-reduced + Clump
k_A_vals = []  # Option A: Full Host + Clump - m * gslope
k_local_vals = []  # Local subtraction: perfect cancellation

for R in R_vals:
    # Distance from ray (at -100 kpc) to subhalo (at +R kpc)
    d_clump = r_ray + R
    k_clump = kappa_NFW(d_clump, m_sub, c_sub, scale_val)[0]
    
    # Option B: Globally reduced host + clump
    k_B = k_host_reduced + k_clump
    k_B_vals.append(k_B)
    
    # Option A: Full host + clump - m * gslope
    k_A = k_host_full + k_clump - m_sub * gslope
    k_A_vals.append(k_A)
    
    # Local Subtraction Model (perfect cancellation far away)
    # The clump and its local hole cancel out at the ray position
    k_local = k_host_full
    k_local_vals.append(k_local)

k_B_vals = np.array(k_B_vals)
k_A_vals = np.array(k_A_vals)
k_local_vals = np.array(k_local_vals)

# Output values for a few key subhalo distances R
indices = [0, 25, 50, 99]
for idx in indices:
    R = R_vals[idx]
    print(f"\nSubhalo position R = {R:.1f} kpc (right):")
    print(f"  Option B (Global): Kappa = {k_B_vals[idx]:.6f} (Mismatch: {(k_B_vals[idx] - k_host_full)/k_host_full*100.0:+.2f}%)")
    print(f"  Option A (Subtract): Kappa = {k_A_vals[idx]:.6f} (Mismatch: {(k_A_vals[idx] - k_host_full)/k_host_full*100.0:+.2f}%)")
    print(f"  Local model:         Kappa = {k_local_vals[idx]:.6f} (Mismatch: 0.00%)")

# 3. Create plot
plt.figure(figsize=(9, 6))
plt.axhline(k_host_full, color='black', linestyle='--', label='True Physical Baseline (Local Subtraction)')
plt.plot(R_vals, k_B_vals, color='red', label='Option B: Global Host Reduction')
plt.plot(R_vals, k_A_vals, color='blue', label='Option A: Constant Mass Subtraction')

plt.title('Ray Convergence (at -100 kpc) vs. Subhalo Position (on the right side)')
plt.xlabel('Subhalo Position R [kpc]')
plt.ylabel('Convergence $\kappa$ at Ray')
plt.grid(True, alpha=0.3)
plt.legend(loc='lower right')
plt.tight_layout()

# Save the plot
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/experiment_local_subtraction.png', dpi=150)
print("\nSaved plot to plots/experiment_local_subtraction.png")
