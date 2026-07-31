"""Capture reference lnmu vectors for the bitwise backward-compat test (fixed seeds).

The output feeds test_cosmology_params.py::test_backward_compat_bitwise. Re-run ONLY
when the physics of the DEFAULT sampler path intentionally changes.

Captures BOTH guarded references:
  reference_lnmu_pre6d.npz              DEFAULT path (no kwargs) -> test_backward_compat_bitwise
  reference_lnmu_pre_paper_defaults.npz LEGACY path (ml.params.LEGACY_CONFIG splatted)
                                        -> test_legacy_physics_bitwise and
                                           test_sigma8_tophat::test_legacy_flag_is_bitwise

Re-baseline history:
  2026-07-08  original capture (pre-1+6d parameterization).
  2026-07-30  re-captured BOTH references after the z-GRID EXTENSION (zmax 10.01 ->
              12.341169644129371, Nz 100 -> 103, cpp/lnmu_wrapper.h ZMAX_DEFAULT /
              NZ_DEFAULT), which was needed because z_s above the old zmax was
              SILENTLY CLAMPED and the confirmed training range reaches z_s = 12.
              The pair was chosen so dlogz is the SAME double as the old grid
              (0.06978540181126486), i.e. nodes 0..99 are bitwise identical and three
              nodes are appended -- so this is an extension, not a rescaling. It still
              moves every ray, because sigma_W shifts at the ~1e-9 level and that
              reseeds the RNG stream; measured max |dlnmu| = 6.1e-7 on the legacy path
              and 3.5e-16 on the As/sigma8 round trip. The PDF is unchanged: sd ratios
              over 240k and 60k rays/arm scatter +-0.8% and FLIP SIGN between the two
              depths, i.e. MC noise, not a shift. The grid change affects ALL paths,
              which is why the legacy reference had to move too.
              Verified before re-baselining (in SEPARATE PROCESSES -- a C extension
              does NOT reload in-process, and deleting sys.modules['gwlensing'] silently
              leaves the first .so loaded, which made a first attempt at this check
              report a false pass): the pre-change build reproduced BOTH references
              bit-for-bit at all three points and passed 13/13 in
              test_cosmology_params.py, while the current build fails exactly the four
              stored-reference assertions -- so the grid change is the sole cause and
              no other tree drift is being absorbed. Old vectors kept as
              `reference_lnmu_pre_zmax.npz` and `reference_lnmu_legacy_pre_zmax.npz`.
  2026-07-30  re-captured after the sigma8 AMPLITUDE CONVENTION change ("option b",
              cosmology.h sigma8_tophat, default true): sigma8 is now anchored with
              the real-space top-hat at 8 Mpc/h instead of the smooth-k Ws, raising
              deltaH8 4.16% and P(k) 8.50%, so every ray moves.
              Verified before re-baselining: with the change in place, an explicit
              sigma8_tophat=false (now part of ml.params.LEGACY_CONFIG) reproduced
              `reference_lnmu_pre_paper_defaults.npz` bit-for-bit at all three points,
              proving only the amplitude ANCHOR moved and no other physics.
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
OUT_LEGACY = ROOT / "tests" / "data" / "reference_lnmu_pre_paper_defaults.npz"

# (z, h, Om, s8) points spanning the box; small N keeps the test fast.
POINTS = [
    (1.0, 0.674, 0.315, 0.811),
    (3.0, 0.60, 0.22, 0.70),
    (5.0, 0.72, 0.38, 1.00),
]
NSAMP = 2000
SEED = 20260707


def _capture(out, label, kwargs):
    arrs, meta = {}, []
    print(f"[{label}] {out.name}")
    for i, (z, h, om, s8) in enumerate(POINTS):
        r = gw.sample_lnmu_ml_with_diagnostics(z, h, om, s8, NSAMP, SEED, False, **kwargs)
        arrs[f"lnmu_{i}"] = np.asarray(r["lnmu"], dtype=np.float64)
        meta.append((z, h, om, s8))
        print(f"  point {i}: z={z} h={h} Om={om} s8={s8} -> {arrs[f'lnmu_{i}'].size} samples, "
              f"mean={arrs[f'lnmu_{i}'].mean():.6f}")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, points=np.array(meta), nsamp=NSAMP, seed=SEED, **arrs)
    print(f"  wrote {out}")


def main():
    # DEFAULT path: no physics kwargs, so this pins the compiled-in defaults themselves.
    _capture(OUT, "default", {})
    # LEGACY path: the pre-2026-07-29 physics, pinned explicitly. Both references sit on
    # the same compiled grid, so a grid change moves both and both must be re-captured
    # together -- capturing only the default one would leave the legacy tests red.
    from ml.params import LEGACY_CONFIG
    _capture(OUT_LEGACY, "legacy", LEGACY_CONFIG)


if __name__ == "__main__":
    main()
