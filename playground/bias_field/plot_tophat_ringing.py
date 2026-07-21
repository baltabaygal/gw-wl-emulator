"""Diagnostic: the top-hat window's ringing in P_1D(k_par) (2026-07-20).

Why the paper figure shows only ONE shelf: it stops at the sampler's mode
cutoff k_max = 2 pi / R_s (x = 6.28), which sits just past the FIRST zero of
the spherical top-hat (x = 4.4934). This script continues the curve past the
cutoff to expose the higher rings.

For an isotropic window the P_1D integral collapses to a radial one,
    P_1D(k_par) = (1/2pi) int_{k_par}^inf dk k P(k) W~^2(kR),
so  dP_1D/dk_par = -(1/2pi) k_par P(k_par) W~^2(k_par R):
the SLOPE is proportional to W~^2, and therefore vanishes at every zero of the
top-hat transform (tan x = x -> x = 4.4934, 7.7253, 10.9041, ...). Each zero is
a shelf in P_1D, never a bump: P_1D is monotonically decreasing throughout.

The radial form is used here because the production 2D k_perp quadrature
truncates its grid at k_perp = 60/R, which eats the tail once k_par R -> 60;
the two agree to 4e-5 over the sampler's range (verified, and re-checked in
panel (b) against the production `P1D_curve`).

Plot x-axis is the dimensionless x = k_par R, so the picture is universal (the
same for any R_s); the top axis gives k_par for the production R_s = 20 Mpc.

Run: python3 playground/bias_field/plot_tophat_ringing.py
Output: plots/tophat_ringing.png
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "convergence"))
sys.path.insert(0, str(REPO / "playground" / "bias_field"))
sys.path.insert(0, str(REPO / "paper_prod" / "scripts"))

from bias_field_prototype import Cosmo, PI                    # noqa: E402
from validate_field_covariance import Wtophat, Wgauss         # noqa: E402
from plot_fig_clustering_field import P1D_curve               # noqa: E402

RS = 20000.0          # kpc, production clustering scale (2026-07-20)
XMAX = 26.0           # plot out to x = k_par R
KPC2MPC = 1e-3


def tophat_zeros(n=6):
    """Roots of tan x = x (zeros of 3(sin x - x cos x)/x^3), bracketed between
    successive poles of tan."""
    # on each (j+1/2)pi < x < (j+3/2)pi, tan x sweeps -inf -> +inf, so
    # f = tan x - x has exactly one root there; j = 0 gives the first, 4.4934
    out = []
    f = lambda x: np.tan(x) - x
    for j in range(n):
        lo, hi = (j + 0.5) * PI + 1e-9, (j + 1.5) * PI - 1e-9
        try:
            out.append(brentq(f, lo, hi, xtol=1e-12))
        except ValueError:
            continue
    return np.array(out)


def P1D_radial(C, x, R, nk=400000):
    """(1/2pi) int_{k}^inf dk k P(k) W~^2(kR), evaluated at k = x/R."""
    out = np.empty(len(x))
    for i, xi in enumerate(x):
        k = np.exp(np.linspace(np.log(xi / R), np.log(4000.0 / R), nk))
        out[i] = np.trapezoid(k ** 2 * C.Pk(k) * Wtophat(k * R) ** 2,
                              dx=np.log(k[1] / k[0])) / (2.0 * PI)
    return out


def P1D_radial_gauss(C, x, R, nk=200000):
    out = np.empty(len(x))
    for i, xi in enumerate(x):
        k = np.exp(np.linspace(np.log(xi / R), np.log(200.0 / R), nk))
        out[i] = np.trapezoid(k ** 2 * C.Pk(k) * Wgauss(k * R) ** 2,
                              dx=np.log(k[1] / k[0])) / (2.0 * PI)
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C = Cosmo(Nz=100)
    zeros = tophat_zeros(6)
    zeros = zeros[zeros < XMAX]
    print("top-hat zeros (tan x = x):", np.round(zeros, 4))

    x = np.linspace(0.05, XMAX, 3000)
    P = P1D_radial(C, x, RS)
    Pg = P1D_radial_gauss(C, x, RS)
    W2 = Wtophat(x) ** 2
    slope = np.gradient(np.log(P), np.log(x))

    print(f"monotonically decreasing: {bool(np.all(np.diff(P) < 0))}")
    print(f"P_1D at x=2pi (cutoff) / at x=0.05 : {np.interp(2*PI, x, P)/P[0]:.3e}")
    for z in zeros:
        print(f"  x={z:8.4f}  W^2={Wtophat(np.array([z]))[0]**2:.2e}  "
              f"dlnP/dlnx={np.interp(z, x, slope):+.4f}  "
              f"P_1D={np.interp(z, x, P)*KPC2MPC:.3e} Mpc")

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.4), sharex=True)
    fig.subplots_adjust(hspace=0.12, top=0.93)

    def decorate(ax, first=False):
        for j, z in enumerate(zeros):
            ax.axvline(z, color="0.8", lw=0.8, zorder=0)
        ax.axvline(2 * PI, color="C3", lw=1.2, ls="--", zorder=1,
                   label="sampler cutoff $k_{\\max}=2\\pi/R_s$" if first else None)

    # (a) the window itself
    ax = axes[0]
    ax.semilogy(x, np.maximum(W2, 1e-16), color="C0", label=r"top-hat $\tilde W^2$")
    ax.semilogy(x, np.maximum(Wgauss(x) ** 2, 1e-16), color="C2", ls=":",
                label=r"Gaussian $\tilde W^2$ (no zeros)")
    decorate(ax, first=True)
    ax.set_ylim(1e-14, 3)
    ax.set_ylabel(r"$\tilde W^2(x)$")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_title("Spherical top-hat ringing: every zero of $\\tilde W$ is a shelf in "
                 "$P_{1D}$", fontsize=10)

    # (b) P_1D
    ax = axes[1]
    ax.semilogy(x, P * KPC2MPC, color="C0", lw=1.4, label="top-hat (radial form)")
    ax.semilogy(x, Pg * KPC2MPC, color="C2", ls=":", lw=1.2, label="Gaussian")
    # cross-check against the production 2D k_perp quadrature, where it is valid
    xv = x[x < 20.0]
    ax.semilogy(xv, P1D_curve(C, xv / RS, RS, window=1) * KPC2MPC, color="k",
                ls="--", lw=0.8, dashes=(6, 4),
                label="production 2D $k_\\perp$ quadrature")
    decorate(ax)
    # the Gaussian dies super-exponentially (~300 decades over this range) and
    # would flatten everything; clip to the top-hat's own dynamic range so its
    # staircase — one tread per zero — stays legible
    ax.set_ylim(2e-5, 3e2)
    ax.set_ylabel(r"$P_{\rm 1D}(k_\parallel)\ [{\rm Mpc}]$")
    ax.legend(fontsize=8, frameon=False, loc="upper right")

    # (c) the log-slope — the cleanest view of the rings
    ax = axes[2]
    ax.plot(x, slope, color="C0", lw=1.4)
    ax.axhline(0.0, color="0.4", lw=0.8, ls=":")
    decorate(ax)
    ax.set_ylim(-14, 1.5)
    ax.set_ylabel(r"$d\ln P_{\rm 1D}/d\ln k_\parallel$")
    ax.set_xlabel(r"$x = k_\parallel R_s$")
    ax.set_xlim(0, XMAX)
    for z in zeros:
        ax.annotate(f"{z:.2f}", (z, 0.6), fontsize=7, ha="center", color="0.35")

    secax = axes[0].secondary_xaxis(
        "top", functions=(lambda xx: xx / (RS * KPC2MPC),
                          lambda kk: kk * RS * KPC2MPC))
    secax.set_xlabel(r"$k_\parallel\ [{\rm Mpc}^{-1}]$ at $R_s = 20$ Mpc", fontsize=9)

    out = REPO / "plots" / "tophat_ringing.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
