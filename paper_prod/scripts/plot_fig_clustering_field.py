#!/usr/bin/env python3
"""
Paper figure (clustering subsection): the correlated 1D environment field.

Panel (a): P_1D(k_par; R_s) — the pencil-projected linear power through the
spherical top-hat window on the full modulus |k| (lensing.cpp BiasField1D,
bias_window = 1) — for the KP91 pencil limit R_s -> 0, the production default
R_s = 20 Mpc, and the signal-weighted scale R_s = 8.44 Mpc = R_L(1e14 Msun)
for comparison. Curves for R_s > 0 end at the mode cutoff k_max = 2 pi / R_s
used by the sampler (now essentially immaterial: the top-hat already suppresses
the power there, unlike the disk window whose only LOS cutoff was numerical).

Panel (b): three realizations of the count-modulation factor lambda(M, z) for
M = 1e13 Msun along a z_s = 3 line of sight, drawn exactly as production does:
per-shell segment averages of the field via the exact-covariance Cholesky
(playground/bias_field/validate_field_covariance.py::cpp_field — the verbatim
numpy replica of BiasField1D::build), then
lambda_i = exp(b Dg dbar_i - (b Dg)^2 sig2_i / 2)  (<lambda_i> = 1 per shell).

Run (no C++ build needed; numpy-only):
  python3 paper_prod/scripts/plot_fig_clustering_field.py
Outputs: plots/fig_clustering_field.{pdf,png}
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "convergence"))
sys.path.insert(0, str(REPO / "playground" / "bias_field"))

from bias_field_prototype import Cosmo, PI            # validated port of cpp/cosmology
from validate_field_covariance import (cpp_field, Wdisk,   # verbatim BiasField1D replica
                                       window2_iso)
from paper_prod.plot_style import apply_style
from plot_fig_subhalo_population import guard_broken_latex

KPC2MPC = 1.0e-3
RS_DEFAULT = 20000.0          # kpc; production clustering scale R_s = 20 Mpc (2026-07-20)
RS_SIGNAL = 8441.0            # kpc; signal-weighted scale R_L(1e14 Msun), for comparison
WINDOW = 1                    # spherical top-hat (lensing.h bias_window = 1)
ZS_FIELD = 3.0
M_LAMBDA = 1.0e13
SEEDS = (11, 21, 31)


def P1D_curve(C, kpar, Rperp, window=0):
    """P_1D(k_par) with the BiasField1D integrand/grids (nkperp=2048 log
    trapezoid; Rperp = 0 -> KP91 pencil).

    window (= lensing.h bias_window): 0 disk on k_perp only, 1 spherical
    top-hat, 2 Gaussian — the latter two on the full modulus |k|."""
    Rw = max(Rperp, 10.0)
    kperp = np.exp(np.linspace(np.log(1e-9), np.log(60.0 / Rw), 2048))
    dlnkp = np.log(kperp[1] / kperp[0])
    W2 = Wdisk(kperp * Rperp) ** 2 if Rperp > 0.0 else np.ones_like(kperp)
    iso = window != 0 and Rperp > 0.0
    out = np.empty_like(kpar)
    for i0 in range(0, len(kpar), 64):
        i1 = min(i0 + 64, len(kpar))
        kk = np.sqrt(kpar[None, i0:i1] ** 2 + kperp[:, None] ** 2)
        w2 = window2_iso(kk * Rperp, window) if iso else W2[:, None]
        f = kperp[:, None] ** 2 * C.Pk(kk) * w2
        out[i0:i1] = np.trapezoid(f, dx=dlnkp, axis=0) / (2.0 * PI)
    return out


def main():
    sizes = apply_style()
    guard_broken_latex()
    import matplotlib.pyplot as plt

    C = Cosmo(Nz=100)

    # ---------------- panel (a): P1D, spherical top-hat, three scales
    kpar = np.exp(np.linspace(np.log(3e-6), np.log(6e-3), 240))   # kpc^-1
    curves = [
        (0.0, "solid", "0.35", r"$R_s \to 0$ (pencil)"),
        (RS_DEFAULT, "solid", "C0", r"$R_s = 20\,{\rm Mpc}$ (default)"),
        (RS_SIGNAL, "dashed", "C1",
         r"$R_s = 8.44\,{\rm Mpc} = R_L(10^{14}M_\odot)$"),
    ]

    fig, (axa, axb) = plt.subplots(2, 1, figsize=(3.37, 4.7))
    fig.subplots_adjust(left=0.16, right=0.965, bottom=0.09, top=0.975,
                        hspace=0.33)

    for Rp, ls, col, lab in curves:
        kmaxR = 2.0 * PI / Rp if Rp > 0 else np.inf     # sampler mode cutoff
        sel = kpar <= kmaxR
        P = P1D_curve(C, kpar[sel], Rp, window=WINDOW)
        axa.plot(kpar[sel] / KPC2MPC, P * KPC2MPC, ls=ls, color=col, label=lab)
        if np.isfinite(kmaxR):
            axa.plot(kmaxR / KPC2MPC, P[-1] * KPC2MPC, "o", ms=3, color=col)
    axa.set_xscale("log")
    axa.set_yscale("log")
    axa.set_xlabel(r"$k_\parallel\ [{\rm Mpc}^{-1}]$")
    axa.set_ylabel(r"$P_{\rm 1D}(k_\parallel)\ [{\rm Mpc}]$")
    axa.legend(fontsize=6.5, frameon=False, loc="lower left")

    # ---------------- panel (b): lambda(M=1e13, z) realizations, zs = 3
    f = cpp_field(C, ZS_FIELD, RS_DEFAULT, window=WINDOW)
    n = f["n"]
    zsh = C.zlist[1:n + 1]                               # shell upper edges
    sigM = float(np.interp(M_LAMBDA, C.sig_M, C.sig_s))
    bDg = np.array([C.Dg(z) * C.halobias(z, sigM) for z in zsh])
    comp = 0.5 * bDg ** 2 * f["sig2"]

    for j, seed in enumerate(SEEDS):
        rng = np.random.default_rng(seed)
        dbar = f["chol"] @ rng.standard_normal(n)
        lam = np.exp(bDg * dbar - comp)
        axb.plot(zsh, lam, drawstyle="steps-mid", lw=0.9, color=f"C{j}")
    # per-shell +-1 sigma band of the mean-one lognormal
    lo = np.exp(-bDg * np.sqrt(f["sig2"]) - comp)
    hi = np.exp(+bDg * np.sqrt(f["sig2"]) - comp)
    axb.fill_between(zsh, lo, hi, step="mid", color="0.75", alpha=0.35, lw=0,
                     label=r"$\pm1\sigma$ band")
    axb.axhline(1.0, color="0.35", lw=0.7, ls=":")
    axb.set_xscale("log")
    axb.set_xlim(C.zlist[1], ZS_FIELD)
    axb.set_xlabel(r"$z$")
    axb.set_ylabel(r"$\lambda(M,z)$")
    axb.legend(fontsize=6.5, frameon=False, loc="upper left")
    axb.text(0.97, 0.93, r"$M = 10^{13}\,M_\odot,\ z_s = 3$",
             transform=axb.transAxes, ha="right", va="top", fontsize=7)

    for ext in ("pdf", "png"):
        fig.savefig(REPO / "plots" / f"fig_clustering_field.{ext}", dpi=300)
    print("wrote plots/fig_clustering_field.{pdf,png}")

    # ---- console diagnostics for the memo
    print(f"Nmax(zs=3, R=8441) = {f['Nmax']}, shells n = {n}, "
          f"L = {f['L']*KPC2MPC:.1f} Mpc")
    print(f"sigma(M_LAMBDA={M_LAMBDA:.0e}) = {sigM:.4f};  b(M_LAMBDA, z=0.5)"
          f" = {C.halobias(0.5, sigM):.3f}")
    print(f"max shell sigma_i = {np.sqrt(f['sig2'].max()):.4f}, "
          f"max bDg sigma_i = {(bDg*np.sqrt(f['sig2'])).max():.4f}")


if __name__ == "__main__":
    main()
