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

Panel (b): the count-modulation factor lambda(M, z) at three halo masses along a
single z_s = 3 line of sight, shown as the per-shell segment-average (the
binned field) -- exactly what the simulator applies to the halo counts
(mode_field: xi_q, eta_q ~ N(0,1), sinc(k_q L_i/2) reinstated, covariance
identical to validate_field_covariance.py::cpp_field / BiasField1D::build).
lambda = exp(b Dg delta - (b Dg)^2 sigma^2 / 2) with <lambda> = 1.

Run (no C++ build needed; numpy-only):
  python3 paper_prod/scripts/plot_fig_clustering_field.py
Outputs: paper_prod/plots/figures/fig_clustering_field.{pdf,png}
(the .tex includegraphics path stays `plots/...` — that is the Overleaf-side
folder these are copied into, as for the other paper figures)
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
                                       window2_iso, shells, PAD)
from paper_prod.plot_style import (apply_style, FIGURE_SIZES, SUBPLOTS_ADJUST)
from plot_fig_subhalo_population import guard_broken_latex

OUT_DIR = REPO / "paper_prod" / "plots" / "figures"   # paper figure output root

KPC2MPC = 1.0e-3
RS_DEFAULT = 20000.0          # kpc; production clustering scale R_s = 20 Mpc (2026-07-20)
RS_SIGNAL = 8441.0            # kpc; signal-weighted scale R_L(1e14 Msun), for comparison
WINDOW = 1                    # spherical top-hat (lensing.h bias_window = 1)
ZS_FIELD = 3.0
# Panel (b): one field, three halo masses. Mass is ORDINAL, so the encoding is a
# single-hue sequential ramp (light -> dark) rather than categorical hues: it
# reads as "more bias" at a glance and, unlike three hues, survives B/W print.
# Purple keeps it clear of panel (a), which owns blue/green/grey. Adjacent steps
# of any one-hue ramp sit below the normal-vision separation floor, so each mass
# also carries a distinct line style (the print-robust secondary encoding);
# widths increase with mass so the darkest, widest-swinging curve reads on top.
# Palette checked with the dataviz validator: contrast vs surface PASS (all
# >= 3:1, needed for thin lines), CVD separation PASS (dE 13.9 deutan).
M_LIST = (1.0e12, 1.0e13, 1.0e14)
M_STYLE = {1.0e12: ("#A569BD", ":", 0.8),
           1.0e13: ("#7D3C98", "--", 0.9),
           1.0e14: ("#4A235A", "-", 1.1)}
M_BAND = 1.0e14                     # mass whose +-1 sigma band is drawn (the widest)
SEED_FIELD = 7                      # the single delta_1D realization on display
PANELB_XSCALE = "log"                # panel (b) z axis: "linear" or "log"
PANELB_YSCALE = "linear"             # panel (b) lambda axis: "linear" or "log"
PANELB_ZMAX = None                   # None -> full z_s; else zoom panel (b) to z <= this


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


def mode_field(C, zs, Rperp, window, seed):
    """One realization of the CONTINUOUS 1D field via its exact Fourier-mode
    amplitudes -- the object BiasField1D discretizes into shells.

    delta(chi) = sum_q sqrt(w_q) [xi_q cos(chi k_q) + eta_q sin(chi k_q)],
    k_q = 2 pi q / L, w_q = 2 P_1D(k_q)/L (same P_1D table as cpp_field). The
    shell segment-averages delta_bar_i are formed from the SAME (xi_q, eta_q) by
    reinstating the sinc(k_q L_i/2) top-hat, so the returned steps are literally
    the binned version of the plotted curve (their covariance == cpp_field's)."""
    c, Lh = shells(C, zs)                         # shell centers, widths (comoving)
    L = PAD * float(C.dc(zs))
    kmin = 2.0 * PI / L
    Nmax = max(4, int(np.floor(L / Rperp)))
    kmax = 2.0 * PI * Nmax / L
    # P_1D(k) table -- identical construction to cpp_field / BiasField1D::build
    nkperp, nktab = 2048, 600
    Rw = max(Rperp, 10.0)
    kperp = np.exp(np.linspace(np.log(1e-9), np.log(60.0 / Rw), nkperp))
    dlnkp = np.log(kperp[1] / kperp[0])
    W2 = Wdisk(kperp * Rperp) ** 2
    lktab = np.linspace(np.log(0.5 * kmin), np.log(kmax), nktab)
    Ptab = np.empty(nktab)
    for t0 in range(0, nktab, 64):
        t1 = min(t0 + 64, nktab)
        kk = np.sqrt(np.exp(lktab[None, t0:t1]) ** 2 + kperp[:, None] ** 2)
        w2 = W2[:, None] if window == 0 else window2_iso(kk * Rperp, window)
        fP = kperp[:, None] ** 2 * C.Pk(kk) * w2
        Ptab[t0:t1] = np.trapezoid(fP, dx=dlnkp, axis=0) / (2.0 * PI)
    lPtab = np.log(np.maximum(Ptab, 1e-300))
    kq = 2.0 * PI * np.arange(1, Nmax + 1) / L
    wq = 2.0 * np.exp(np.interp(np.log(np.clip(kq, np.exp(lktab[0]),
                                               np.exp(lktab[-1]))),
                                lktab, lPtab)) / L
    rng = np.random.default_rng(seed)
    rw = np.sqrt(wq)
    xi = rw * rng.standard_normal(Nmax)           # cosine-mode amplitudes
    eta = rw * rng.standard_normal(Nmax)          # sine-mode amplitudes

    def delta_cont(chi):
        chi = np.atleast_1d(np.asarray(chi, float))
        return np.cos(np.outer(chi, kq)) @ xi + np.sin(np.outer(chi, kq)) @ eta

    sig2_point = float(wq.sum())                  # stationary point variance = sigma(R_s)^2
    snc = np.sinc(np.outer(Lh / 2.0, kq) / PI)    # sin(x)/x, x = k_q L_i/2 (segment avg)
    dbar = (snc * np.cos(np.outer(c, kq))) @ xi + (snc * np.sin(np.outer(c, kq))) @ eta
    sig2_shell = (wq[None, :] * snc ** 2).sum(1)  # == diag(cpp_field Cov)
    return dict(c=c, Lh=Lh, delta_cont=delta_cont, sig2_point=sig2_point,
                dbar=dbar, sig2_shell=sig2_shell, Nmax=Nmax, L=L, n=len(c))


def main():
    sizes = apply_style()
    guard_broken_latex()
    import matplotlib.pyplot as plt

    C = Cosmo(Nz=100)

    # ---------------- panel (a): P1D, spherical top-hat, three scales
    kpar = np.exp(np.linspace(np.log(3e-6), np.log(6e-3), 240))   # kpc^-1
    curves = [
        (0.0, "solid", "0.35", r"$R_s \to 0$"),
        (RS_DEFAULT, "solid", "C0", r"$R_s = 20\,{\rm Mpc}$"),
        (RS_SIGNAL, "dashed", "C1", r"$R_s = 8.44\,{\rm Mpc}$"),
    ]

    # Use the same single-column figure size as the other paper figures and
    # keep the stacked-panel spacing consistent with the shared style helper.
    fig, (axa, axb) = plt.subplots(
        2, 1,
        figsize=(FIGURE_SIZES["single"][0], 2 * FIGURE_SIZES["single"][1]),
    )
    fig.subplots_adjust(
        left=SUBPLOTS_ADJUST["single"]["left"],
        right=SUBPLOTS_ADJUST["single"]["right"],
        bottom=0.10,
        top=0.96,
        hspace=0.34,
    )

    for Rp, ls, col, lab in curves:
        kmaxR = 2.0 * PI / Rp if Rp > 0 else np.inf     # sampler mode cutoff
        sel = kpar <= kmaxR
        P = P1D_curve(C, kpar[sel], Rp, window=WINDOW)
        axa.plot(kpar[sel] / KPC2MPC, P * KPC2MPC, ls=ls, color=col,
                 label=lab)
        if np.isfinite(kmaxR):
            axa.plot(kmaxR / KPC2MPC, P[-1] * KPC2MPC, "o", ms=3, color=col)
    axa.set_xscale("log")
    axa.set_yscale("log")
    axa.set_xlabel(r"$k_\parallel\ [{\rm Mpc}^{-1}]$")
    axa.set_ylabel(r"$P_{\rm 1D}(k_\parallel)\ [{\rm Mpc}]$")
    axa.legend(fontsize=6.5, frameon=False, loc="lower left")

    # ---------------- panel (b): lambda(M,z) at three masses, ONE field
    # realization, zs = 3. Every mass rides the SAME delta_1D, so the curves
    # share their shape and differ only through btilde(M,z) = D(z) b(M,z) —
    # i.e. the panel shows the mass dependence of the halo bias the subsection
    # opens with. On the log axis this is exact:
    # ln lambda = btilde delta_1D - btilde^2 sig^2/2, so raising M rescales one
    # and the same field (up to the mean-one compensation).
    #
    # STEPS = the per-shell segment-average of the Fourier-mode field, i.e.
    # exactly what the simulator applies to the halo counts.
    fm = mode_field(C, ZS_FIELD, RS_DEFAULT, WINDOW, SEED_FIELD)
    n = fm["n"]
    zsh = C.zlist[1:n + 1]                               # shell upper edges
    sig_i = np.sqrt(fm["sig2_shell"])
    dbar = fm["dbar"]                                    # shell-averaged realization
    sig2_pt = fm["sig2_point"]                           # continuous point variance
    sig_pt = np.sqrt(sig2_pt)

    # dense LOS grid for the continuous +-1 sigma envelope (log-spaced to match the axis)
    z_c = np.exp(np.linspace(np.log(C.zlist[1]), np.log(ZS_FIELD), 600))

    def bDg_at(zarr, sigM):
        return np.array([C.Dg(z) * C.halobias(z, sigM) for z in zarr])

    def sigM_of(M):
        return float(np.interp(M, C.sig_M, C.sig_s))

    # +-1 sigma band of the mean-one lognormal (widest mass), CONTINUOUS envelope
    sigM_b = sigM_of(M_BAND)
    bDg_bc = bDg_at(z_c, sigM_b)
    comp_bc = 0.5 * bDg_bc ** 2 * sig2_pt
    axb.fill_between(z_c, np.exp(-bDg_bc * sig_pt - comp_bc),
                     np.exp(+bDg_bc * sig_pt - comp_bc),
                     color="0.75", alpha=0.35, lw=0,
                     label=r"$\pm1\sigma$ ($10^{14}M_\odot$)")

    for M in M_LIST:
        sigM = sigM_of(M)
        col, ls, lw = M_STYLE[M]
        # shell-averaged version (binned field) — what production applies
        bDg_s = bDg_at(zsh, sigM)
        lam_s = np.exp(bDg_s * dbar - 0.5 * bDg_s ** 2 * fm["sig2_shell"])
        axb.plot(zsh, lam_s, drawstyle="steps-mid", lw=lw, ls=ls, color=col,
                 label=rf"$M = 10^{{{int(round(np.log10(M)))}}}\,M_\odot$")
        print(f"  M={M:.0e}: sigma(M)={sigM:.4f}, b(z=0.5)="
              f"{C.halobias(0.5, sigM):.3f}, b(z=2)={C.halobias(2.0, sigM):.3f}, "
              f"lambda_shell in [{lam_s.min():.3f}, {lam_s.max():.3f}]")

    axb.axhline(1.0, color="0.35", lw=0.7, ls=":")
    axb.set_xscale(PANELB_XSCALE)
    axb.set_yscale(PANELB_YSCALE)
    axb.set_xlim(C.zlist[1], PANELB_ZMAX if PANELB_ZMAX else ZS_FIELD)
    if PANELB_YSCALE == "log":
        # lambda is lognormal: on a log axis the mean-one band is symmetric about
        # 1 and the mass scaling reads as a pure amplitude change
        axb.set_ylim(0.18, 11.0)        # M=1e14 point-field peaks clip above this
        axb.set_yticks([0.2, 0.5, 1.0, 2.0, 5.0])
        axb.get_yaxis().set_major_formatter(
            __import__("matplotlib").ticker.FuncFormatter(
                lambda v, _: f"{v:g}"))
    else:
        axb.set_ylim(0.0, 6.0)          # M=1e14 shell-average peaks clip above this
    axb.set_xlabel(r"$z$")
    axb.set_ylabel(r"$\lambda(M,z)$")
    axb.legend(fontsize=6.0, frameon=False, loc="upper left", ncol=2,
               columnspacing=0.9, handlelength=1.6, borderaxespad=0.3)
    axb.text(0.985, 0.955, r"$z_s = 3$", transform=axb.transAxes,
             ha="right", va="top", fontsize=7)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "fig_clustering_field.png"
    out_pdf = out_png.with_suffix(".pdf")
    fig.savefig(out_png, dpi=300, facecolor="white")
    fig.savefig(out_pdf, facecolor="white")
    print(f"wrote {out_pdf.relative_to(REPO)}")
    print(f"wrote {out_png.relative_to(REPO)}")

    # ---- console diagnostics for the memo
    print(f"Nmax(zs=3, R={RS_DEFAULT:.0f}) = {fm['Nmax']}, shells n = {n}, "
          f"L = {fm['L']*KPC2MPC:.1f} Mpc")
    print(f"sigma_point = {sig_pt:.4f}  (continuous field, = sqrt(sum w_q))")
    print(f"max shell sigma_i = {sig_i.max():.4f}  (field seed {SEED_FIELD})")


if __name__ == "__main__":
    main()
