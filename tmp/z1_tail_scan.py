"""
Close-the-gap scan: zs=1 low-factor tail (3.16e-6 ... 1e-7) + two extra brute seeds.

Same paired estimator as scripts/subhalo_factor_convergence.py:
  excess = var(kappa) - var(kappa_nosub), bootstrap errors over realizations.
Rows appended to data/subhalo_factor_z1_tail.csv as each job lands (resumable read).
"""
import csv, os, sys, time
import numpy as np
from multiprocessing import Pool

ROOT = "/Users/baltabay/Desktop/gw-wl-emulator"
sys.path.insert(0, os.path.join(ROOT, "build"))
import gwlensing

ZS, N, NBOOT = 1.0, 40_000, 200
CSV = os.path.join(ROOT, "data", "subhalo_factor_z1_tail.csv")

COMMON = dict(z=ZS, h=0.674, OmegaM=0.315, sigma8=0.811,
              nsamples=N, filaments=False, bias=False, ell=False,
              m_floor=1e7, subhalo_model=1)

def run_one(args):
    label, seed, kw = args
    t0 = time.time()
    res = gwlensing.sample_lensing_raw_ml(**COMMON, seed=seed, **kw)
    dt = time.time() - t0
    k = np.asarray(res["kappa"], float)
    kn = np.asarray(res["kappa_nosub"], float)
    ex = k.var() - kn.var()
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(k), size=(NBOOT, len(k)))
    err = (k[idx].var(axis=1) - kn[idx].var(axis=1)).std()
    return label, seed, ex, err, dt

if __name__ == "__main__":
    # expensive first; chunksize=1
    jobs = [("brute", 43, dict(subhalo=True, subhalo_brute=True)),
            ("brute", 44, dict(subhalo=True, subhalo_brute=True)),
            ("1e-07",     42, dict(subhalo=True, subhalo_factor=1.0e-7)),
            ("3.162e-07", 42, dict(subhalo=True, subhalo_factor=3.162e-7)),
            ("1e-06",     42, dict(subhalo=True, subhalo_factor=1.0e-6)),
            ("3.162e-06", 42, dict(subhalo=True, subhalo_factor=3.162e-6))]

    new_file = not os.path.exists(CSV)
    with open(CSV, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(["zs", "label", "seed", "N", "excess", "err", "runtime"])
            fh.flush()
        with Pool(processes=6) as pool:
            for label, seed, ex, err, dt in pool.imap_unordered(run_one, jobs, chunksize=1):
                w.writerow([ZS, label, seed, N, f"{ex:.8e}", f"{err:.3e}", f"{dt:.1f}"])
                fh.flush()
                print(f"done: {label:>10} seed={seed}  excess={ex:.4e} +- {err:.1e}  [{dt:.0f}s]",
                      flush=True)
    print("tail scan complete ->", CSV)
