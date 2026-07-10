import sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, 'build')
import numpy as np
import gwlensing

OM, S8, H = 0.315, 0.811, 0.674
KTHR = 1e-4

def bench(z, nreal, threads, thr, factor=1e-5, seed=123):
    t0 = time.perf_counter()
    out = gwlensing.sample_lnmu(
        z=z, OmegaM=OM, sigma8=S8, h=H, Nreal=nreal, seed=seed,
        subhalo=True, subhalo_model=3, subhalo_factor=factor,
        kappathr_flat=KTHR,
        subhalo_threads=threads, subhalo_parallel_threshold=thr,
    )
    dt = time.perf_counter() - t0
    n = np.asarray(out).size
    return dt / n * 1e4, np.asarray(out)   # s per 1e4 real, samples

# warm up
bench(1.0, 100, 1, 200000)

def run(z, nreal, factor, label):
    print(f"\n### {label}: z_s={z}, subhalo_factor={factor:.0e}, Nreal={nreal}")
    print(f"{'config':>34} | {'s/1e4':>9} {'speedup':>8} {'mean(lnmu)':>11}")
    base = None
    configs = [
        ("threads=1 (baseline)",            1, 200000),
        ("threads=10, thr=200000 (default)",10, 200000),
        ("threads=10, thr=1000",            10, 1000),
        ("threads=10, thr=1 (force all)",   10, 1),
    ]
    for name, th, thr in configs:
        per, samp = bench(z, nreal, th, thr, factor)
        if base is None: base = per
        print(f"{name:>34} | {per:>9.2f} {base/per:>7.2f}x {np.mean(samp):>11.5f}")

# default production factor 1e-5
run(1.0, 2000, 1e-5, "PROD factor")
run(10.0, 800, 1e-5, "PROD factor")
# tiny factor -> inflates resolved-clump count Nc per host (threading's design case)
run(1.0, 500, 1e-8, "SMALL factor (large Nc)")
