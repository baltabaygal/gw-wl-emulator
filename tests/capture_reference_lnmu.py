"""Capture reference lnmu vectors for the bitwise backward-compat test (fixed seeds).

The output feeds test_cosmology_params.py::test_backward_compat_bitwise. Re-run ONLY
when the physics of the DEFAULT sampler path intentionally changes.

Re-baseline history:
  2026-07-08  original capture (pre-1+6d parameterization).
  2026-07-29  re-captured after the PAPER-DEFAULT FLIP: the compiled-in defaults are
              now the config the draft describes (subhalo=true, subhalo_model=5,
              subhalo_virial=true, bias_model=1, bias_window=1, bias_Rperp=20000,
              bias_weak=true, fil_bias=true, kappa_anchor=1). Old file kept as
              `reference_lnmu_pre_paper_defaults.npz`, and it is still guarded --
              `test_legacy_physics_bitwise` pins it via ml.params.LEGACY_CONFIG.
              Verified before re-baselining: with the flip in place, passing
              LEGACY_CONFIG explicitly reproduced the old reference bit-for-bit at all
              three points, proving the flip moved DEFAULTS ONLY and no physics.
  2026-07-28  re-captured after `cosmology::halobias` q 0.75 -> 0.8 (b is now the
              peak-background split of pFC's own (0.3, 0.8) barrier). halobias is on
              the DEFAULT path (`samp.bias = 1` is hardcoded), so this shifts every
              ray. Old file kept as `reference_lnmu_pre_halobias.npz`.
              Verified before re-baselining: a build with ONLY halobias reverted
              reproduced the 2026-07-08 reference bit-for-bit at all three points, so
              halobias was the sole cause — the same-day subhalo radial-profile fix and
              the subhalo_virial code are both bitwise-clean on the default path.

BEFORE re-capturing, always run that check: revert the suspected change alone and
confirm the old reference still passes. Otherwise a re-baseline silently absorbs any
OTHER unintended drift that happens to be in the tree.

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
