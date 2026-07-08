"""
Section-4 gate, baseline 3: the ACE - Vaskonen model gap.

Is the substructure shift (Delta<k^2>_c=+3.9%, Delta<k^3>_c=+1.8%) small or large
compared to how much two independent lensing-PDF predictions already disagree?

Also cross-validates the analytic screen (scripts/subhalo_screen.py) against the C++
raw-kappa moments with filaments/bias/ellipticity OFF (matching the screen's pure-NFW
host assumption).

Run with the test env:
  /Users/baltabay/miniforge3/envs/test/bin/python scripts/ace_gap.py
"""
import sys, numpy as np
sys.path.insert(0, 'build')
sys.path.insert(0, 'ace_lensing')
import gwlensing
from ace_lensing import predict_pdf

ZS = 1.0
COSMO = dict(OmegaM=0.315, sigma8=0.811, h=0.674)
N = 400000

def central_moments(x, w=None):
    if w is None:
        mu = x.mean(); m2 = ((x-mu)**2).mean(); m3 = ((x-mu)**3).mean()
    else:
        w = w/np.trapezoid(w, x); mu = np.trapezoid(w*x, x)
        m2 = np.trapezoid(w*(x-mu)**2, x); m3 = np.trapezoid(w*(x-mu)**3, x)
    return mu, m2, m3, m3/m2**1.5

# ---------- cross-validate analytic screen vs C++ raw kappa (features OFF) ----------
print("=== cross-validate analytic screen vs C++ (filaments/bias/ell OFF) ===")
raw = gwlensing.sample_lensing_raw_ml(ZS, COSMO['h'], COSMO['OmegaM'], COSMO['sigma8'],
                                      N, seed=1, filaments=False, bias=False, ell=False)
k = np.asarray(raw['kappa']); k = k - k.mean()       # ensemble-mean subtracted (as in code)
K2_cpp, K3_cpp = (k**2).mean(), (k**3).mean()
print(f"  C++   <k^2>={K2_cpp:.3e}  <k^3>={K3_cpp:.3e}  skew={K3_cpp/K2_cpp**1.5:.3f}")
print(f"  screen<k^2>=7.292e-04  <k^3>=7.434e-05  skew=3.776   (analytic, hosts+weak)")

# ---------- compare in CONVERGENCE (kappa moments converge; mu moments are
#            tail-dominated by strong lensing, which ACE smooths away) ----------
print("\n=== Vaskonen vs ACE CONVERGENCE moments (zs=%.1f) ===" % ZS)
# Vaskonen kappa, full model (filaments/bias/ell ON)
rawON = gwlensing.sample_lensing_raw_ml(ZS, COSMO['h'], COSMO['OmegaM'], COSMO['sigma8'],
                                        N, seed=3)
kON = np.asarray(rawON['kappa']); kON = kON - kON.mean()
K2V, K3V = (kON**2).mean(), (kON**3).mean(); skV = K3V/K2V**1.5

# ACE: convert magnification PDF -> convergence via kappa = (mu-1)/2 (weak regime)
# NB: predict_pdf returns (mu_axis, pdf) -- the README's "pdf, mu" naming is reversed
mu, pdf = predict_pdf(Om=COSMO['OmegaM'], h=COSMO['h'], w=-1.0, s8=COSMO['sigma8'],
                      z=ZS, verbose=False)
mu = np.asarray(mu, float); pdf = np.clip(np.asarray(pdf, float), 0, None)
ok = np.isfinite(mu) & np.isfinite(pdf); mu, pdf = mu[ok], pdf[ok]
kap = (mu - 1.0)/2.0
mAk, K2A, K3A, skA = central_moments(kap, pdf)

print(f"  {'':14s}{'sigma_kappa':>12s}{'<k^2>':>12s}{'<k^3>':>12s}{'skew':>9s}")
print(f"  Vaskonen (ON) {np.sqrt(K2V):>12.4f}{K2V:>12.3e}{K3V:>12.3e}{skV:>9.3f}")
print(f"  ACE           {np.sqrt(K2A):>12.4f}{K2A:>12.3e}{K3A:>12.3e}{skA:>9.3f}")

gap2 = abs(K2A-K2V)/K2V; gap3 = abs(K3A-K3V)/K3V
print("\n=== GATE baseline 3 (convergence) ===")
print(f"  ACE-Vaskonen gap  <k^2> : {gap2:.0%}      <k^3> : {gap3:.0%}")
print(f"  substructure shift      : <k^2> +3.9%%     <k^3> +1.8%%")
print(f"  substructure / model-gap: <k^2> {0.039/gap2:.2f}x      <k^3> {0.018/gap3:.2f}x")
np.save('plots/ace_gap.npy', np.array(dict(K2V=K2V,K3V=K3V,skV=skV,K2A=K2A,K3A=K3A,skA=skA,
        gap2=gap2,gap3=gap3), dtype=object), allow_pickle=True)
