"""Low-z data augmentation (exp20): the low-z shoulder overshoot is a data-sparsity
problem (shoulder density ~1e-3 => ~10 samples/config in datasets_logz_1k).
Generate 150 fresh configs with z ~ U[0.3, 2.5], theta ~ U(prior box), 40k samples
each -> cache/lowz_aug.npz (X: (N,4) [z,h,Om,s8], Y: (N,) lnmu).
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np

AR = Path(__file__).resolve().parent; REPO = AR.parents[1]
OUT = AR / "cache" / "lowz_aug.npz"
NCONF, NPER = 150, 40_000
PRIOR = dict(h=(0.59, 0.76), Om=(0.20, 0.40), s8=(0.65, 1.05))


def _worker(args):
    z, h, om, s8, seed = args
    sys.path.insert(0, str(REPO / "build"))
    import gwlensing as gw
    r = gw.sample_lnmu_ml_with_diagnostics(float(z), float(h), float(om), float(s8),
                                           int(NPER), int(seed), False)
    x = np.asarray(r["lnmu"], float)
    return x[np.isfinite(x)]


def main():
    rng = np.random.default_rng(2026)
    zs = rng.uniform(0.3, 2.5, NCONF)
    hs = rng.uniform(*PRIOR["h"], NCONF)
    oms = rng.uniform(*PRIOR["Om"], NCONF)
    s8s = rng.uniform(*PRIOR["s8"], NCONF)
    jobs = [(zs[i], hs[i], oms[i], s8s[i], 50_000 + i) for i in range(NCONF)]
    with Pool(10) as p:
        parts = p.map(_worker, jobs)
    X = np.concatenate([np.tile([zs[i], hs[i], oms[i], s8s[i]], (len(parts[i]), 1))
                        for i in range(NCONF)]).astype(np.float32)
    Y = np.concatenate(parts).astype(np.float32)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, X=X, Y=Y)
    print(f"wrote {OUT}: {Y.size} samples, z range [{zs.min():.2f}, {zs.max():.2f}]")


if __name__ == "__main__":
    main()
