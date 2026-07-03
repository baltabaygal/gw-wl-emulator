"""Fit the two physics ingredients for the low-mu cutoff + z-dependent tail (exp19).

1. EDGE: the simulator PDF has a physical lower edge (empty-beam demagnification
   bound); the flow leaks probability below it. Fit lnmu_edge(z,h,Om,s8) = ridge
   regression on the per-config 0.1% lnmu quantile of datasets_logz_1k (487 configs).
2. ALPHA(z): far-tail power-law index dP/dmu ~ mu^-alpha measured from the cached
   clean 10M reference (stress theta) over mu in [15,150], per z; production blend
   interpolates linearly in z (clamped). Theta-dependence checked at central theta.

Writes cache/edge_alpha_fit.json consumed by smooth_model.py.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json, sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np

AR = Path(__file__).resolve().parent; REPO = AR.parents[1]
OUT = AR / "cache" / "edge_alpha_fit.json"


def edge_features(z, h, om, s8):
    """Feature map for the edge fit (shared with smooth_model via json spec)."""
    lz = np.log1p(z)
    return np.stack([np.ones_like(lz), lz, lz**2, lz**3,
                     om*lz, h*lz, s8*lz, om, h, s8], axis=-1)


def fit_edge():
    from ml.data import load_dataset
    ds = load_dataset("datasets_logz_1k")["train"]
    lnmu, cnt = ds["lnmu"], ds["valid_counts"].astype(int)
    z, h, om, s8 = ds["z"], ds["h"], ds["OmegaM"], ds["sigma8"]
    q = np.array([np.quantile(lnmu[i, :cnt[i]], 1e-3) for i in range(len(cnt))])
    X = edge_features(z, h, om, s8)
    lam = 1e-3
    w = np.linalg.solve(X.T @ X + lam*np.eye(X.shape[1]), X.T @ q)
    resid = q - X @ w
    print(f"[edge] q0.1%% ridge fit: resid std={resid.std():.4f}  max|resid|={np.abs(resid).max():.4f}")
    return w.tolist(), float(resid.std())


def _alpha_from_hist(counts, edges, N, mu_lo=15.0, mu_hi=150.0, min_cnt=10):
    ctr = np.sqrt(edges[:-1]*edges[1:]); bw = np.diff(edges)
    m = (ctr >= mu_lo) & (ctr <= mu_hi) & (counts >= min_cnt)
    if m.sum() < 4:
        return np.nan
    x = np.log(ctr[m]); y = np.log(counts[m]/(N*bw[m]))
    A = np.stack([np.ones_like(x), x], axis=-1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(-coef[1])          # dP/dmu ~ mu^-alpha


def _worker(args):
    z, theta, seed, nper = args
    sys.path.insert(0, str(REPO/"build")); import gwlensing as gw
    r = gw.sample_lnmu_ml_with_diagnostics(float(z), *map(float, theta), int(nper), int(seed), False)
    x = np.asarray(r["lnmu"], float); return x[np.isfinite(x)]


def fit_alpha():
    ref = np.load(AR / "cache" / "clean_tail_ref.npz")
    edges = ref["edges"]
    zs = [2.0, 3.5, 5.0, 8.0]
    alphas = {z: _alpha_from_hist(ref[f"c_{z}"], edges, int(ref[f"N_{z}"])) for z in zs}
    print("[alpha] stress theta (10M):", {z: round(a, 3) for z, a in alphas.items()})
    # theta-independence spot check: central theta, 3M each at z=2 and 5
    for z in (2.0, 5.0):
        with Pool(6) as p:
            parts = p.map(_worker, [(z, (0.67, 0.30, 0.85), 300+j, 500_000) for j in range(6)])
        mu = np.exp(np.concatenate(parts))
        c, _ = np.histogram(mu, bins=edges)
        a = _alpha_from_hist(c, edges, mu.size)
        print(f"[alpha] central theta z={z}: alpha={a:.3f} (stress: {alphas[z]:.3f})")
    return {str(z): a for z, a in alphas.items()}


def main():
    w, sd = fit_edge()
    alph = fit_alpha()
    OUT.write_text(json.dumps(dict(edge_coef=w, edge_resid_std=sd, alpha_z=alph), indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
