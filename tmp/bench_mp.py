import sys, time, warnings, os
warnings.filterwarnings("ignore")
sys.path.insert(0, 'build')
import numpy as np
import multiprocessing as mp

OM, S8, H = 0.315, 0.811, 0.674
KTHR = 1e-4
Z = 1.0

def worker(args):
    seed, nreal = args
    import gwlensing  # imported per-process
    out = gwlensing.sample_lnmu(z=Z, OmegaM=OM, sigma8=S8, h=H, Nreal=nreal,
                                seed=seed, subhalo=True, subhalo_model=3,
                                subhalo_factor=1e-5, kappathr_flat=KTHR)
    return np.asarray(out)

def run_serial(total, seed0=1000):
    import gwlensing
    t0 = time.perf_counter()
    out = gwlensing.sample_lnmu(z=Z, OmegaM=OM, sigma8=S8, h=H, Nreal=total,
                                seed=seed0, subhalo=True, subhalo_model=3,
                                subhalo_factor=1e-5, kappathr_flat=KTHR)
    return time.perf_counter()-t0, np.asarray(out)

def run_mp(total, nproc):
    per = total // nproc
    tasks = [(1000 + i, per) for i in range(nproc)]
    t0 = time.perf_counter()
    with mp.Pool(nproc) as pool:
        res = pool.map(worker, tasks)
    dt = time.perf_counter()-t0
    return dt, np.concatenate(res)

if __name__ == "__main__":
    TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    ts, xs = run_serial(TOTAL)
    print(f"serial   : {TOTAL} real in {ts:6.2f}s  ({ts/TOTAL*1e4:6.2f} s/1e4)  mean={xs.mean():.5f}")
    for nproc in (4, 8, 10):
        tm, xm = run_mp(TOTAL, nproc)
        print(f"mp x{nproc:<2}   : {xm.size} real in {tm:6.2f}s  ({tm/xm.size*1e4:6.2f} s/1e4)  "
              f"speedup {ts/tm:4.2f}x  mean={xm.mean():.5f}")
