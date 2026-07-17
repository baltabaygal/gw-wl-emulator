"""Resume the 4 missing brute-class jobs of the psi_max=1 rescan (append to same CSV)."""
import csv, os, sys, time
import numpy as np
from multiprocessing import Pool
ROOT = "/Users/baltabay/Desktop/gw-wl-emulator"
sys.path.insert(0, os.path.join(ROOT, "build"))
import gwlensing
N, NBOOT = 40_000, 200
CSV = os.path.join(ROOT, "data", "subhalo_rescan_psimax1.csv")
QS = [0.99, 0.999, 0.9999]
def run_one(args):
    label, zs, seed, nhalos, kw = args
    t0 = time.time()
    res = gwlensing.sample_lensing_raw_ml(z=zs, h=0.674, OmegaM=0.315, sigma8=0.811,
        nsamples=N, seed=seed, filaments=False, bias=False, ell=False, Nhalos=nhalos,
        m_floor=1e7, subhalo_model=1, **kw)
    dt = time.time()-t0
    k = np.asarray(res["kappa"], float); kn = np.asarray(res["kappa_nosub"], float)
    ex = k.var()-kn.var()
    rng = np.random.default_rng(0); idx = rng.integers(0, len(k), size=(NBOOT, len(k)))
    err = (k[idx].var(axis=1)-kn[idx].var(axis=1)).std()
    q = np.quantile(k, QS); qn = np.quantile(kn, QS)
    dqe = (np.quantile(k[idx], QS, axis=1)-np.quantile(kn[idx], QS, axis=1)).std(axis=1)
    return (label, zs, seed, nhalos, ex, err, *q, *qn, *(q-qn), *dqe, dt)
if __name__ == "__main__":
    B = dict(subhalo=True, subhalo_brute=True)
    jobs = [("brute_N300", 1.0, 42, 300, B), ("brute", 1.0, 42, 100, B),
            ("brute", 1.0, 43, 100, B), ("brute", 1.0, 44, 100, B)]
    with open(CSV, "a", newline="") as fh:
        w = csv.writer(fh)
        with Pool(processes=4) as pool:
            for out in pool.imap_unordered(run_one, jobs, chunksize=1):
                w.writerow([out[0], out[1], out[2], out[3]] + [f"{v:.6e}" for v in out[4:-1]] + [f"{out[-1]:.1f}"])
                fh.flush()
                print(f"done: {out[0]:>10} zs={out[1]} Nh={out[3]} excess={out[4]:.4e}+-{out[5]:.1e} dq99={out[12]:.4e} [{out[-1]:.0f}s]", flush=True)
    print("resume complete")
