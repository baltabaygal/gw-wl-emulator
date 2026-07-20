#!/usr/bin/env python3
"""
Paper figure (subhalo subsection): evolved SHMF + anti-biased radial profile.

Panel (a): the evolved subhalo mass function dN/dln(psi) = gamma psi^alpha
exp(-beta psi^omega) (JvdB14, (alpha,beta,omega) = (-0.82, 50, 4)) for hosts
M = 1e12..1e15 Msun at z_l = 0.5, with gamma anchored to the bound fraction
f_s(N_tau) exactly as in cpp/subhalo.cpp::precompute (verbatim numpy port:
Giocoli+2007 median formation weight wf, Bryan-Norman Delta_vir, JvdB14
eqs. 23-26). Dots mark psi = m_floor/M (m_floor = 1e7 Msun, the absolute
substructure floor).

Panel (b): the subhalo radial distribution dN/dx ~ x^2/(1+c200 x)^2 *
[1+(x/0.54)^{-5/2}]^{-1/2}, x = r/r200, with c200 = cons14(M, z) (Dutton &
Maccio 2014) for the same hosts; dashed: mass-follows-NFW comparator
(dN/dx ~ x/(1+cx)^2) for the 1e14 host, illustrating the central depletion.

Incomplete Gamma(s, x) is computed by two independent numpy methods (log-t
trapezoid + Nist series/continued fraction) and cross-checked at runtime.

Run: python3 paper_prod/scripts/plot_fig_subhalo_population.py
Outputs: plots/fig_subhalo_population.{pdf,png}
"""
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "convergence"))

from bias_field_prototype import Cosmo, PI
from paper_prod.plot_style import apply_style


def guard_broken_latex():
    """apply_style enables usetex when a `latex` binary exists; fall back to
    mathtext if the TeX tree is unusable (e.g. sandbox installs)."""
    import shutil
    import subprocess
    import matplotlib as mpl
    if not mpl.rcParams.get("text.usetex"):
        return
    ok = False
    if shutil.which("kpsewhich"):
        # type1ec.sty (cm-super) is matplotlib's hard usetex requirement
        r = subprocess.run(["kpsewhich", "type1ec.sty"], capture_output=True)
        ok = r.returncode == 0
    if not ok:
        mpl.rcParams["text.usetex"] = False
        mpl.rcParams["mathtext.fontset"] = "cm"

# --- SHMF constants (cpp/subhalo.h) ---
ALPHA, BETA, OMEGA = -0.82, 50.0, 4.0
PSI_RES, PSI_MAX = 1.0e-4, 1.0
M_FLOOR = 1.0e7
S_M = (1.0 + ALPHA) / OMEGA

AF = 0.815 * math.exp(-0.25) / 0.5 ** 0.707            # Giocoli+2007, f = 1/2
WF = math.sqrt(2.0 * math.log(AF + 1.0))

Z_L = 0.5
M_HOSTS = (1.0e12, 1.0e13, 1.0e14, 1.0e15)


# ---------------------------------------------------------------- Gamma(s,x)
def gamma_inc_quad(s, x):
    """Upper incomplete Gamma(s, x) by log-t trapezoid (x >= 0, s > -1)."""
    x = max(float(x), 1.0e-300)
    u = np.linspace(math.log(x), math.log(x + 400.0), 60_000)
    t = np.exp(u)
    return float(np.trapezoid(t ** s * np.exp(-t), u))


def gamma_inc_series(s, x):
    """Upper Gamma(s,x): NR gser/gcf (normalized), scaled by Gamma(s)."""
    if x <= 0.0:
        return math.gamma(s)
    if x < s + 1.0:                                    # series for P(s,x)
        ap, summ, delt = s, 1.0 / s, 1.0 / s
        for _ in range(10_000):
            ap += 1.0
            delt *= x / ap
            summ += delt
            if abs(delt) < abs(summ) * 1e-15:
                break
        P = summ * math.exp(-x + s * math.log(x) - math.lgamma(s))
        return (1.0 - P) * math.gamma(s)
    b, c, d, h = x + 1.0 - s, 1e300, 1.0 / (x + 1.0 - s), 1.0 / (x + 1.0 - s)
    for i in range(1, 10_000):
        an = -i * (i - s)
        b += 2.0
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        delt = d * c
        h *= delt
        if abs(delt - 1.0) < 1e-15:
            break
    Q = math.exp(-x + s * math.log(x) - math.lgamma(s)) * h
    return Q * math.gamma(s)


def gamma_inc(s, x):
    a, b = gamma_inc_quad(s, x), gamma_inc_series(s, x)
    scale = max(abs(a), abs(b), 1e-300)
    assert abs(a - b) / scale < 1e-6, (s, x, a, b)
    return b


# ------------------------------------------------- subhalo.cpp verbatim port
def fs_gamma(C, z, M):
    """(f_s, gamma_norm, zf, Ntau) exactly as Subhalo::precompute."""
    sigM = float(np.interp(M, C.sig_M, C.sig_s))
    sigH = float(np.interp(0.5 * M, C.sig_M, C.sig_s))
    dsig2 = sigH * sigH - sigM * sigM
    assert dsig2 > 0.0
    rhs = C.deltac(z) + WF * math.sqrt(dsig2)
    assert C.deltac(30.0) >= rhs, "host forms above the z grid"
    zlo, zhi = z, 30.0
    for _ in range(60):
        zm = 0.5 * (zlo + zhi)
        if C.deltac(zm) < rhs:
            zlo = zm
        else:
            zhi = zm
    zf = 0.5 * (zlo + zhi)

    Nstep = 200
    dz = (zf - z) / Nstep
    zz = z + (np.arange(Nstep) + 0.5) * dz
    d = C.OmegaM * (1 + zz) ** 3 / C.Az(zz) - 1.0
    Dvir = 18.0 * PI * PI + 82.0 * d - 39.0 * d * d
    Ntau = float(np.sum(6.006 * np.sqrt(Dvir / 178.0) / (1.0 + zz) * dz))

    fs = 0.3563 / Ntau ** 0.6 - 0.075                  # JvdB14 eq. 26
    assert 0.0 < fs < 0.95
    gden = gamma_inc(S_M, BETA * PSI_RES ** OMEGA) - gamma_inc(S_M, BETA)
    gam = OMEGA * BETA ** S_M / gden * fs              # JvdB14 eq. 23
    return fs, gam, zf, Ntau


def main():
    apply_style()
    guard_broken_latex()
    import matplotlib.pyplot as plt

    C = Cosmo(Nz=100)

    fig, (axa, axb) = plt.subplots(2, 1, figsize=(3.37, 4.7))
    fig.subplots_adjust(left=0.16, right=0.965, bottom=0.09, top=0.975,
                        hspace=0.33)

    psi = np.exp(np.linspace(np.log(1e-6), 0.0, 800))
    info = []
    for j, M in enumerate(M_HOSTS):
        fs, gam, zf, Ntau = fs_gamma(C, Z_L, M)
        dNdlnpsi = gam * psi ** ALPHA * np.exp(-BETA * psi ** OMEGA)
        lab = (rf"$M = 10^{{{int(round(math.log10(M)))}}} M_\odot$, "
               rf"$f_{{\rm s}} = {fs:.2f}$")
        axa.plot(psi, dNdlnpsi, color=f"C{j}", label=lab)
        pf = M_FLOOR / M
        if pf > psi[0]:
            axa.plot(pf, gam * pf ** ALPHA * math.exp(-BETA * pf ** OMEGA),
                     "o", ms=3, color=f"C{j}")
        # mean resolvable count above m_floor + bound fraction, for the memo
        band = np.exp(np.linspace(math.log(pf), 0.0, 4000))
        Nsub = np.trapezoid(gam * band ** ALPHA
                            * np.exp(-BETA * band ** OMEGA), np.log(band))
        fsb = gam / (OMEGA * BETA ** S_M) * (
            gamma_inc(S_M, BETA * pf ** OMEGA) - gamma_inc(S_M, BETA))
        info.append((M, fs, fsb, zf, Ntau, Nsub))
    axa.set_xscale("log")
    axa.set_yscale("log")
    axa.set_xlim(1e-6, 1.0)
    axa.set_ylim(1e-2, 3e4)
    axa.set_xlabel(r"$\psi = m/M$")
    axa.set_ylabel(r"${\rm d}N/{\rm d}\ln\psi$")
    axa.legend(fontsize=6.5, frameon=False, loc="lower left",
               title=rf"$z = {Z_L:g}$", title_fontsize=6.5)

    # ---------------- panel (b): radial profile
    x = np.linspace(1e-4, 1.0, 800)
    B = 1.0 / np.sqrt((x / 0.54) ** -2.5 + 1.0)
    for j, M in enumerate(M_HOSTS):
        c = float(C._cons14(Z_L, M))
        p = x ** 2 / (1.0 + c * x) ** 2 * B
        p /= np.trapezoid(p, x)
        axb.plot(x, p, color=f"C{j}",
                 label=rf"$M = 10^{{{int(round(math.log10(M)))}}} M_\odot$, "
                       rf"$c_{{200}} = {c:.1f}$")
    c14 = float(C._cons14(Z_L, 1.0e14))
    pn = x / (1.0 + c14 * x) ** 2
    pn /= np.trapezoid(pn, x)
    axb.plot(x, pn, ls="dashed", color="0.4", lw=0.9,
             label=r"mass follows NFW ($10^{14} M_\odot$)")
    axb.set_xlim(0.0, 1.0)
    axb.set_ylim(0.0, None)
    axb.set_xlabel(r"$x = r/r_{200}$")
    axb.set_ylabel(r"${\rm d}N/{\rm d}x$ (normalized)")
    axb.legend(fontsize=6.5, frameon=False, loc="upper left")

    for ext in ("pdf", "png"):
        fig.savefig(REPO / "plots" / f"fig_subhalo_population.{ext}", dpi=300)
    print("wrote plots/fig_subhalo_population.{pdf,png}")

    print(f"\nwf = {WF:.4f} (alpha_f = {AF:.4f})")
    print(f"{'M':>8} {'f_s':>7} {'f_sb(m_floor)':>13} {'z_f':>6} "
          f"{'N_tau':>6} {'<N>(>m_floor)':>13}")
    for M, fs, fsb, zf, Ntau, Nsub in info:
        print(f"{M:8.1e} {fs:7.3f} {fsb:13.3f} {zf:6.2f} {Ntau:6.2f} "
              f"{Nsub:13.3g}")
    g = math.gamma(S_M)
    print(f"\nGamma(s) sanity: quad(s,1e-300) = {gamma_inc_quad(S_M, 0):.6f} "
          f"vs math.gamma = {g:.6f}")


if __name__ == "__main__":
    main()
