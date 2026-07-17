"""
Validate the dynamic resolved/unresolved split (subhalo_model=1, brute=False, conservative
floor at max(0, r - r200)) against brute force (floor = m_floor everywhere), which is the
ground truth the analytic screen was validated against.

Compares clump-induced Delta<kappa^2>, Delta<kappa^3> and tail quantiles, plus runtime.
"""
import sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build"))
import gwlensing as gw

z_s = 5.0
common = dict(z=z_s, h=0.674, OmegaM=0.315, sigma8=0.811,
              nsamples=40_000, seed=42,
              filaments=False, bias=False, ell=False,
              Nhalos=100, m_floor=1e7, subhalo_model=1)

def run(label, **kw):
    t0 = time.time()
    raw = gw.sample_lensing_raw_ml(**common, **kw)
    dt = time.time() - t0
    k = np.asarray(raw["kappa"], dtype=float)
    g = np.sqrt(np.asarray(raw["gamma1"], dtype=float)**2 +
                np.asarray(raw["gamma2"], dtype=float)**2)
    var = np.var(k)
    m3 = np.mean((k - k.mean())**3)
    q = np.quantile(k, [0.9, 0.99, 0.999])
    print(f"{label:<28} var={var:.4e}  m3={m3:.4e}  <g^2>={np.mean(g**2):.4e}  "
          f"q90/99/99.9={q[0]:.3e}/{q[1]:.3e}/{q[2]:.3e}  [{dt:.1f}s]")
    return var, m3

print(f"z_s = {z_s}, N = {common['nsamples']}, features off, Nhalos = 100\n")
v0, m0 = run("host only", subhalo=False)
vb, mb = run("subhalo BRUTE (truth)", subhalo=True, subhalo_brute=True)
vd, md = run("subhalo DYNAMIC split", subhalo=True, subhalo_brute=False)

print("\nclump-induced moment shifts (vs host-only):")
print(f"  Delta<k^2>: brute {vb-v0:+.4e} ({100*(vb-v0)/v0:+.2f}%)   "
      f"dynamic {vd-v0:+.4e} ({100*(vd-v0)/v0:+.2f}%)")
print(f"  Delta<k^3>: brute {mb-m0:+.4e} ({100*(mb-m0)/m0:+.2f}%)   "
      f"dynamic {md-m0:+.4e} ({100*(md-m0)/m0:+.2f}%)")
