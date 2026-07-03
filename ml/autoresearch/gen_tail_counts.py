"""exp24: counting-only sims to tighten the POT amplitude fit where exceedances are
rare (low z, low structure). 30 configs x 1.5M samples; only the exceedance counts
above THRESHOLDS are kept -> cache/tail_counts_extra.npz (ctx, N, counts).
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np

AR = Path(__file__).resolve().parent; REPO = AR.parents[1]
OUT = AR / "cache" / "tail_counts_extra.npz"
THRESHOLDS = (2.0, 3.0, 8.0)
NPER = 1_500_000
ZS = [0.4, 0.7, 1.0, 1.4, 1.9, 2.5]
THETAS = [(0.60, 0.22, 0.70),   # low-structure corner (the weak spot)
          (0.64, 0.25, 0.78),
          (0.62, 0.28, 0.72),
          (0.70, 0.24, 0.75),
          (0.67, 0.30, 0.85)]   # prior center (row was ~0.85 low)


def _worker(args):
    z, th, seed = args
    sys.path.insert(0, str(REPO / "build"))
    import gwlensing as gw
    r = gw.sample_lnmu_ml_with_diagnostics(float(z), *map(float, th), int(NPER), int(seed), False)
    x = np.asarray(r["lnmu"], float); x = x[np.isfinite(x)]
    return [int(np.sum(x > np.log(u))) for u in THRESHOLDS], x.size


def main():
    jobs = [(z, th, 90_000 + 7 * j) for j, (z, th) in
            enumerate((z, th) for th in THETAS for z in ZS)]
    with Pool(10) as p:
        res = p.map(_worker, jobs)
    ctx = np.array([[z, *th] for th in THETAS for z in ZS])
    cnts = np.array([r[0] for r in res], float)
    Ns = np.array([r[1] for r in res], float)
    np.savez(OUT, ctx=ctx, N=Ns, counts=cnts, thresholds=np.array(THRESHOLDS))
    for i in range(len(jobs)):
        print(f"z={ctx[i,0]:.1f} th=({ctx[i,1]:.2f},{ctx[i,2]:.2f},{ctx[i,3]:.2f})  "
              f"N={int(Ns[i])}  c2={int(cnts[i,0])} c3={int(cnts[i,1])} c8={int(cnts[i,2])}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
