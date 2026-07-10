"""PDF comparison: flat vs adaptive kappa_thr. Run from repo root with the test env:
   /Users/baltabay/miniforge3/envs/test/bin/python <this>
Compares magnification PDFs (subhalo OFF) to confirm flat 1e-4 is science-neutral."""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, 'build')
import numpy as np
import gwlensing

OM, S8, H = 0.315, 0.811, 0.674
RULES = {"adaptive": -1.0, "flat1e-4": 1e-4, "flat1e-3": 1e-3}
Z_LIST = [0.2, 1.0, 10.0]
NREAL  = {0.2: 1_000_000, 1.0: 1_000_000, 10.0: 300_000}  # dial down if slow

def samples(z, kthr, n):
    out = gwlensing.sample_lnmu(z=z, OmegaM=OM, sigma8=S8, h=H,
                                Nreal=n, seed=123, subhalo=False,
                                kappathr_flat=kthr)
    return np.asarray(out, float)          # lnmu

def kl_hist(a, b, bins):
    pa, _ = np.histogram(a, bins=bins, density=True)
    pb, _ = np.histogram(b, bins=bins, density=True)
    m = (pa > 0) & (pb > 0)
    w = np.diff(bins)[m]
    return float(np.sum(pa[m] * np.log(pa[m] / pb[m]) * w))

for z in Z_LIST:
    print(f"\n=== z_s = {z} ===")
    S = {k: samples(z, v, NREAL[z]) for k, v in RULES.items()}
    # shared bins from the union range of lnmu
    lo = min(s.min() for s in S.values()); hi = max(s.max() for s in S.values())
    bins = np.linspace(lo, hi, 400)
    ref = S["adaptive"]
    print(f"{'rule':>10} | {'<1/mu>':>9} {'sig(lnmu)':>10} {'KL_vs_adapt':>12} "
          f"{'q99':>8} {'q99.9':>9} {'q99.99':>9}")
    for k, s in S.items():
        mu = np.exp(s)
        inv = np.mean(1.0/mu)
        sig = np.std(s)
        kl  = kl_hist(s, ref, bins) if k != "adaptive" else 0.0
        q99, q999, q9999 = np.percentile(mu, [99, 99.9, 99.99])
        print(f"{k:>10} | {inv:>9.4f} {sig:>10.5f} {kl:>12.2e} "
              f"{q99:>8.4f} {q999:>9.4f} {q9999:>9.4f}")
