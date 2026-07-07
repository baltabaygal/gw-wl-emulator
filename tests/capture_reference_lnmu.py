"""Capture reference lnmu vectors from the pre-A_s-change module (fixed seeds).

Run once BEFORE the 1+6d parameterization change; the output feeds the bitwise
backward-compat test in test_cosmology_params.py. Re-run only if the physics of
the sampler intentionally changes.

Usage (test env):  $PY tests/capture_reference_lnmu.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build"))
import gwlensing as gw  # noqa: E402

OUT = ROOT / "tests" / "data" / "reference_lnmu_pre6d.npz"

# (z, h, Om, s8) points spanning the box; small N keeps the test fast.
POINTS = [
    (1.0, 0.674, 0.315, 0.811),
    (3.0, 0.60, 0.22, 0.70),
    (5.0, 0.72, 0.38, 1.00),
]
NSAMP = 2000
SEED = 20260707


def main():
    arrs, meta = {}, []
    for i, (z, h, om, s8) in enumerate(POINTS):
        r = gw.sample_lnmu_ml_with_diagnostics(z, h, om, s8, NSAMP, SEED, False)
        arrs[f"lnmu_{i}"] = np.asarray(r["lnmu"], dtype=np.float64)
        meta.append((z, h, om, s8))
        print(f"point {i}: z={z} h={h} Om={om} s8={s8} -> {arrs[f'lnmu_{i}'].size} samples, "
              f"mean={arrs[f'lnmu_{i}'].mean():.6f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, points=np.array(meta), nsamp=NSAMP, seed=SEED, **arrs)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
