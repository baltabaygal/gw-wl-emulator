"""Wall-clock cost of the PAPER-DEFAULT config, for the draft's runtime statement.

Separates the two terms that matter, because quoting a single number at small N is
how the model-4 cost got misreported by 100x in July (a ~4.5 s fixed precompute
swamped the per-ray work at N=300, reading "3.6x model 3" when the true marginal
ratio was a few hundred x):

    t(N) = t_fixed(z_s, cosmology) + N * t_ray

t_fixed is the per-(cosmology, z_s) table build: sigma(M)/HMF grids, the r_thr
reach tables, model-5's restricted-intensity bins, and the bias field's lnT/lnV
tables. It is paid ONCE per config, so it amortizes away in dataset generation
(1e4-1e6 rays per config) but dominates a single small call.

Two-point fit at N_lo and N_hi; run the arms in one process so the cosmology
tables are shared exactly as they are in production.

Usage:  $PY scripts/convergence/paper_config_timing.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "build"))

import gwlensing as gw                                    # noqa: E402
from ml.params import LEGACY_CONFIG, PRODUCTION_CONFIG     # noqa: E402

FID = dict(h=0.674, Om=0.315, sigma8=0.811)
N_LO, N_HI = 500, 5000
ZS = (0.5, 1.0, 5.0, 10.0)
OUT = ROOT / "data" / "results" / "paper_config_timing"


def timed(zs, n, cfg, seed):
    t0 = time.perf_counter()
    gw.sample_lnmu_ml_with_diagnostics(
        float(zs), FID["h"], FID["Om"], FID["sigma8"], int(n), int(seed), False, **cfg)
    return time.perf_counter() - t0


def fit(zs, cfg, reps=2):
    """Return (t_fixed_s, t_ray_ms). Takes the MIN over reps to suppress OS jitter."""
    lo = min(timed(zs, N_LO, cfg, 4000 + i) for i in range(reps))
    hi = min(timed(zs, N_HI, cfg, 5000 + i) for i in range(reps))
    t_ray = (hi - lo) / (N_HI - N_LO)
    return lo - N_LO * t_ray, t_ray * 1e3


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    arms = {"paper (draft defaults)": PRODUCTION_CONFIG,
            "legacy (Vaskonen-like)": LEGACY_CONFIG}
    res = {}
    print(f"two-point fit, N = {N_LO} / {N_HI}, min of 2 reps\n")
    print(f"{'arm':<24} {'z_s':>5} {'t_fixed [s]':>12} {'t_ray [ms]':>11} "
          f"{'4e5 rays':>12} {'1.6e6 rays':>12}")
    for name, cfg in arms.items():
        res[name] = {}
        for zs in ZS:
            tf, tr = fit(zs, cfg)
            t4 = tf + 4e5 * tr / 1e3
            t16 = tf + 1.6e6 * tr / 1e3
            res[name][zs] = dict(t_fixed_s=tf, t_ray_ms=tr,
                                 t_4e5_s=t4, t_1p6e6_s=t16)
            print(f"{name:<24} {zs:>5} {tf:>12.2f} {tr:>11.4f} "
                  f"{t4/60:>10.1f} m {t16/60:>10.1f} m")
    (OUT / "timing.json").write_text(json.dumps(res, indent=2))

    p, l = res["paper (draft defaults)"], res["legacy (Vaskonen-like)"]
    print("\nper-ray cost ratio paper/legacy:")
    for zs in ZS:
        print(f"  z_s={zs:<5} {p[zs]['t_ray_ms']/l[zs]['t_ray_ms']:.2f}x")
    print(f"\nwrote {OUT/'timing.json'}")


if __name__ == "__main__":
    main()
