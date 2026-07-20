"""Smoothed-field variance sigma^2(R) per bias_window, and the Gaussian radius
that matches the top-hat at the production R_perp (2026-07-20).

Three things, all analytic/quadrature (no MC, no C++):

1. sigma^2(R) = (1/2 pi^2) int dk k^2 P(k) W~^2(kR) for the isotropic windows
   (bias_window 1 = spherical top-hat, 2 = Gaussian), from the CODE's own
   linear P(k) (bias_field_prototype.Cosmo, the validated port of
   cpp/cosmology). Deliberately NOT the code's internal sigma(M), which uses a
   smooth-k window from a different family (+4.3% vs top-hat; CLAUDE.md).

2. The variance-matched Gaussian radius R_G: sigma^2_G(R_G) = sigma^2_TH(R_TH).
   lensing.h does NO internal rescaling — R_G is what you pass as bias_Rperp
   when running window 2 as a shape-robustness check against window 1.

3. Identity check for the isotropic windows: the pencil-projected field's POINT
   variance equals the 3D smoothed variance,
       sigma^2_point = (1/pi) int_0^inf dk_par P_1D(k_par) == sigma^2(R),
   because (1/4pi^2) dk_par (k_perp dk_perp) integrates to (1/2pi^2) k^2 dk over
   the same isotropic weight. This is an independent end-to-end check of the
   P_1D integrand: it fails if the window is applied to k_perp instead of |k|
   (that is exactly what window 0 does, and window 0 misses it by ~40x here).
   The comparison is quoted both untruncated and with production's mode cutoff
   k_max = 2 pi / R, which is the only reason the identity is not exact in the
   sampler.

Run: python3 scripts/convergence/bias_window_sigmaR.py [--zs 1.0]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "convergence"))
sys.path.insert(0, str(REPO / "playground" / "bias_field"))

from bias_field_prototype import Cosmo, PI                       # noqa: E402
from validate_field_covariance import Wtophat, Wgauss, Wdisk     # noqa: E402

RPERP_DEFAULT = 8441.0        # kpc, lensing.h bias_Rperp = R_L(1e14 Msun)
KPC2MPC = 1e-3


def sigma2_iso(C, R, window, nk=16384):
    """sigma^2(R) = (1/2pi^2) int dk k^2 P(k) W~^2(kR), trapezoid in ln k.

    Range: the integrand ~ k^3 P(k) at low k and (top-hat) ~ k^-5 at high k, so
    [1e-6/R, 300/R] brackets it to well below the quoted precision."""
    k = np.exp(np.linspace(np.log(1e-6 / R), np.log(300.0 / R), nk))
    W = Wgauss(k * R) if window == 2 else Wtophat(k * R)
    return float(np.trapezoid(k ** 3 * C.Pk(k) * W ** 2, dx=np.log(k[1] / k[0]))
                 / (2.0 * PI ** 2))


def sigma2_point_1D(C, R, window, kmax=None, nkpar=4096, nkperp=4096):
    """(1/pi) int_0^kmax dk_par P_1D(k_par), P_1D built exactly as
    BiasField1D::build does (window 0 on k_perp, 1/2 on |k|)."""
    kmax = kmax if kmax is not None else 300.0 / R
    kpar = np.exp(np.linspace(np.log(1e-6 / R), np.log(kmax), nkpar))
    kperp = np.exp(np.linspace(np.log(1e-9), np.log(300.0 / R), nkperp))
    dlnkp = np.log(kperp[1] / kperp[0])
    W2d = Wdisk(kperp * R) ** 2
    P1D = np.empty(nkpar)
    for i0 in range(0, nkpar, 64):
        i1 = min(i0 + 64, nkpar)
        kk = np.sqrt(kpar[None, i0:i1] ** 2 + kperp[:, None] ** 2)
        if window == 0:
            w2 = W2d[:, None]
        else:
            W = Wgauss(kk * R) if window == 2 else Wtophat(kk * R)
            w2 = W ** 2
        P1D[i0:i1] = np.trapezoid(kperp[:, None] ** 2 * C.Pk(kk) * w2,
                                  dx=dlnkp, axis=0) / (2.0 * PI)
    return float(np.trapezoid(P1D * kpar, dx=np.log(kpar[1] / kpar[0])) / PI)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=float, default=RPERP_DEFAULT, help="R_TH [kpc]")
    args = ap.parse_args()
    C = Cosmo(Nz=100)
    R = args.R

    print(f"P(k): code's own linear spectrum (bias_field_prototype.Cosmo), z = 0\n")
    s2_th = sigma2_iso(C, R, 1)
    s2_g_at_R = sigma2_iso(C, R, 2)
    print(f"R_TH = {R:.1f} kpc = {R*KPC2MPC:.3f} Mpc")
    print(f"  sigma_TH(R)      = {np.sqrt(s2_th):.6f}")
    print(f"  sigma_G(R)       = {np.sqrt(s2_g_at_R):.6f}   "
          f"(same R: the Gaussian is the more aggressive filter)")

    # ---- variance-matched Gaussian radius
    f = lambda RG: sigma2_iso(C, RG, 2) - s2_th
    R_G = brentq(f, 0.05 * R, 2.0 * R, xtol=1e-6 * R, rtol=1e-12)
    print(f"\nvariance-matched Gaussian: sigma^2_G(R_G) = sigma^2_TH({R:.0f})")
    print(f"  R_G = {R_G:.1f} kpc = {R_G*KPC2MPC:.3f} Mpc   "
          f"(R_G/R_TH = {R_G/R:.4f})")
    print(f"  check: sigma_G(R_G) = {np.sqrt(sigma2_iso(C, R_G, 2)):.6f} "
          f"vs sigma_TH(R_TH) = {np.sqrt(s2_th):.6f}")

    # ---- identity: pencil point variance == 3D smoothed variance
    print(f"\nidentity sigma^2_point(P_1D) == sigma^2(R), R = {R:.0f} kpc:")
    print(f"  {'window':8s} {'sig_point(untrunc)':>19s} {'sig(R) 3D':>11s} "
          f"{'ratio':>8s} {'sig_point(k<2pi/R)':>19s} {'ratio':>8s}")
    for window, name in ((1, "tophat"), (2, "gauss"), (0, "disk")):
        s3d = np.sqrt(sigma2_iso(C, R, window if window else 1))
        sp = np.sqrt(sigma2_point_1D(C, R, window))
        spc = np.sqrt(sigma2_point_1D(C, R, window, kmax=2.0 * PI / R))
        tag = "" if window else "   <- vs top-hat sigma(R); disk is not a 3D window"
        print(f"  {name:8s} {sp:19.6f} {s3d:11.6f} {sp/s3d:8.4f} "
              f"{spc:19.6f} {spc/s3d:8.4f}{tag}")


if __name__ == "__main__":
    main()
