#!/usr/bin/env python3
"""
Paper figure (subhalo subsection): evolved SHMF + radial bias function,
each compared against external subhalo models (2026-07-20 rework).

Panel (a): the evolved subhalo mass function dN/dln(psi) = gamma psi^alpha
exp(-beta psi^omega) (JvdB14, (alpha,beta,omega) = (-0.82, 50, 4)) for hosts
M = 1e12..1e15 Msun at z_l = 0.5, with gamma anchored to the bound fraction
f_s(N_tau) exactly as in cpp/subhalo.cpp::precompute (verbatim numpy port:
Giocoli+2007 median formation weight wf, Bryan-Norman Delta_vir, JvdB14
eqs. 23-26). Dots mark psi = m_floor/M (m_floor = 1e7 Msun, the absolute
substructure floor). Overlaid comparisons:
  - Diffhalos (Zacharegkas+26, arXiv:2607.10419, v0.2.0) conditional SHMF,
    exact numpy port of diffhalos.ccshmf (sig-slope kernel + default
    calibration, verified against the JAX package to <1e-6 in log10); note
    their masses are PEAK (unevolved) masses, tuned to Jiang & vdB 2016 -
    dashed lines, same colors as the matching host.
  - pyHalo (Gilman+, v0.2.8) BOUND masses after Galacticus stripping, from
    the saved matched-host run data/pyhalo_vs_ours.npz (host 1e13, z_l=0.5,
    z_s=2, m_infall up to the host mass, 2000 realizations, R<38 kpc aperture;
    deprojected to per-host counts by
    dividing by our Han+16 aperture fraction f_ap - see
    docs/subhalo/pyhalo_pipeline_comparison.md section 4 for caveats:
    position-conditioned stripping makes this a slight underestimate of
    their halo-average bound MF).

Panel (b): the radial bias function B(x) = n_sub/n_host (normalized to 1 at
x = r/r200 = 1) on log-log axes, in the style of plots/bias_fit_comparison.png:
  - adopted transition fit B(x) = [1+(x/0.54)^{-5/2}]^{-1/2} (fit to Green+21)
  - Green, van den Bosch & Jiang 2021 withering+disruption curve (digitized,
    from scripts/subhalo_gate/fit_bias_profile.py)
  - Bolshoi Fig. 7 points (Klypin+11, same digitization)
  - Springel+08 Aq-A-1: Einasto subhalo number density (alpha = 0.678,
    r_-2 = 0.81 r200, their sec. 3.2) over the host NFW at their c_NFW = 16.11
  - Han+16 power-law bias (R/R200)^1.3 (Aquarius A fit; their model gamma =
    alpha*beta ~ 1)

Incomplete Gamma(s, x) is computed by two independent numpy methods (log-t
trapezoid + Nist series/continued fraction) and cross-checked at runtime.

Run: python3 paper_prod/scripts/plot_fig_subhalo_population.py
Inputs: data/pyhalo_vs_ours.npz (copied from tmp/, made by
        tmp/pyhalo_vs_ours_plot.py in .venv_pyhalo)
Outputs: paper_prod/plots/figures/fig_subhalo_population.{pdf,png}
(the .tex includegraphics path stays `plots/...` — that is the Overleaf-side
folder these are copied into, as for the other paper figures)
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

OUT_DIR = REPO / "paper_prod" / "plots" / "figures"   # paper figure output root


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
M_HOSTS = (1.0e13,)                   # single fiducial host (matches pyHalo run)
M_PYHALO = 1.0e13                    # host of the saved pyHalo comparison run

# per-model colors (color no longer encodes host mass in the single-host figure)
COL_OURS, COL_DH, COL_PYH = "C0", "C1", "C2"


# ------------------------- Diffhalos CCSHMF (exact numpy port, v0.2.0)
# diffhalos/ccshmf/ccshmf_kernels.py + calibrations/ccshmf_cal/default_params.py
# <Nsub(>mu)|Mhost> in PEAK masses; kernel tuned to Jiang & vdB 2016 (Bolshoi,
# hosts 1e11-1e15, z 0-5). Port verified against the JAX package (<1e-6 log10).
DH_XTP, DH_K, DH_X0, DH_YHI = -1.0, 10.0, -0.35, -3.1


def _dh_sigmoid(x, x0, k, lo, hi):
    return lo + (hi - lo) / (1.0 + np.exp(-k * (x - x0)))


def _dh_sig_slope(x, xtp, ytp, x0, k, lo, hi):
    return ytp + _dh_sigmoid(x, x0, k, lo, hi) * (x - xtp)


def diffhalos_dndlnpsi(lgM, psi):
    """dN/dln(psi) of the Diffhalos default CCSHMF at host mass 10**lgM."""
    lgmu = np.log10(psi)
    ytp = _dh_sig_slope(lgM, 13.0, 0.06, 13.72, 1.95, 0.06, 0.15)
    ylo = _dh_sig_slope(lgM, 13.0, -0.98, 12.40, 1.67, 0.34, 0.09)
    f = _dh_sig_slope(lgmu, DH_XTP, ytp, DH_X0, DH_K, ylo, DH_YHI)   # lg N(>mu)
    s = _dh_sigmoid(lgmu, DH_X0, DH_K, ylo, DH_YHI)
    sp = DH_K * (s - ylo) * (1.0 - (s - ylo) / (DH_YHI - ylo))
    fp = s + (lgmu - DH_XTP) * sp                                    # df/dlgmu
    # dN/dln(psi) = -dN(>mu)/dln(mu) = -10**f * fp
    return -(10.0 ** f) * fp


# ------------------------------- panel (b) digitized curves + literature fits
# Bolshoi Fig. 7 points + Green+21 withering+disruption curve, digitized in
# scripts/subhalo_gate/fit_bias_profile.py (provenance: draft_comments_memo R6)
BOLSHOI_LOGX = np.array([-1.5, -1.4, -1.3, -1.25, -1.15, -1.1, -1.0, -0.9,
                         -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0])
BOLSHOI_LOGB = np.array([-1.52, -1.50, -1.33, -1.28, -1.30, -1.25, -1.10,
                         -0.97, -0.85, -0.73, -0.58, -0.48, -0.38, -0.25,
                         -0.15, -0.06, 0.0])
GREEN_LOGX = np.array([-1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7,
                       -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0])
GREEN_LOGB = np.array([-1.55, -1.43, -1.29, -1.15, -1.01, -0.88, -0.75,
                       -0.62, -0.50, -0.39, -0.29, -0.21, -0.13, -0.07,
                       -0.03, 0.0])

SP08_ALPHA, SP08_X2, SP08_CNFW = 0.678, 0.81, 16.11   # Springel+08 sec 3.2 + Tab 2
HAN16_GAMMA = 1.3                                     # Han+16 Fig. 1 (Aq-A)


def bias_adopted(x):
    """Adopted transition fit (subhalo.cpp), normalized to B(1)=1."""
    raw = 1.0 / np.sqrt((x / 0.54) ** -2.5 + 1.0)
    return raw / (1.0 / np.sqrt((1.0 / 0.54) ** -2.5 + 1.0))


def bias_springel08(x):
    """Aq-A-1 Einasto subhalo number density over the host NFW, B(1)=1."""
    def raw(y):
        n_ein = np.exp(-2.0 / SP08_ALPHA * ((y / SP08_X2) ** SP08_ALPHA - 1.0))
        rho_nfw = 1.0 / (SP08_CNFW * y * (1.0 + SP08_CNFW * y) ** 2)
        return n_ein / rho_nfw
    return raw(x) / raw(1.0)


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

    # height = 2 x single-panel (2.6") so each panel's axes box equals a
    # standalone single-panel figure; side/top/bottom margins match
    # SUBPLOTS_ADJUST["single"] (left=0.20, right=0.95, bottom/top -> 0.416"/0.208")
    fig, (axa, axb) = plt.subplots(2, 1, figsize=(3.37, 5.2))
    fig.subplots_adjust(left=0.20, right=0.95, bottom=0.08, top=0.96,
                        hspace=0.316)

    psi = np.exp(np.linspace(np.log(1e-6), 0.0, 800))
    info = []
    for j, M in enumerate(M_HOSTS):
        fs, gam, zf, Ntau = fs_gamma(C, Z_L, M)
        pf = M_FLOOR / M
        dNdlnpsi = gam * psi ** ALPHA * np.exp(-BETA * psi ** OMEGA)
        lab = (rf"This work (JvdB14, bound), "
               rf"$f_{{\rm s}} = {fs:.2f}$")
        axa.plot(psi, dNdlnpsi, color=COL_OURS, label=lab)

        # our bound subhalo mass fraction over [pf, 1] (JvdB14 eq. 23 integral)
        fsb = gam / (OMEGA * BETA ** S_M) * (
            gamma_inc(S_M, BETA * pf ** OMEGA) - gamma_inc(S_M, BETA))

        # Diffhalos CCSHMF is in PEAK masses. Convert to bound with the
        # EMERGENT mean stripping factor s_eff = f_s,b / f_peak, both integrated
        # over the same [pf, 1] range: this forces the total bound mass to match
        # ours, with NO free parameter (our f_s sets the conversion). dN/dlnpsi
        # is scale-free under psi -> s*psi (subhalo count conserved), so the
        # bound curve is the peak curve shifted: psi_bound = s_eff * psi_peak.
        lnb = np.linspace(math.log(pf), 0.0, 20_000)
        psib = np.exp(lnb)
        f_peak = np.trapezoid(psib * diffhalos_dndlnpsi(math.log10(M), psib),
                              lnb)
        s_eff = fsb / f_peak
        # show only where the underlying PEAK mass psi/s_eff <= 0.5; the
        # differential of their cumulative sig-slope fit develops a small (~5%)
        # rolloff wiggle near peak-psi ~ 0.2-0.3 (invisible on this log axis)
        # and becomes unreliable as peak-psi -> 1
        psi_dh = psi[psi <= 0.5 * s_eff]
        axa.plot(psi_dh, diffhalos_dndlnpsi(math.log10(M), psi_dh / s_eff),
                 color=COL_DH, ls="dashed", lw=1.0, alpha=0.9)

        if pf > psi[0]:
            axa.plot(pf, gam * pf ** ALPHA * math.exp(-BETA * pf ** OMEGA),
                     "o", ms=3, color=COL_OURS)
        # mean resolvable count above m_floor, for the memo
        band = np.exp(np.linspace(math.log(pf), 0.0, 4000))
        Nsub = np.trapezoid(gam * band ** ALPHA
                            * np.exp(-BETA * band ** OMEGA), np.log(band))
        info.append((M, fs, fsb, zf, Ntau, Nsub, s_eff))

    # pyHalo bound masses (evolved), matched 1e13 host, deprojected per-host
    pyh = np.load(REPO / "data" / "pyhalo_vs_ours.npz")
    a_ap = math.pi * float(pyh["R_ap"]) ** 2
    ok = pyh["h_bnd"] > 0
    psi_py = pyh["ctr"][ok] / M_PYHALO
    dn_py = pyh["h_bnd"][ok] * a_ap / float(pyh["f_ap"])
    axa.plot(psi_py, dn_py, "s", ms=2.4, mfc="none", mew=0.7, color=COL_PYH)

    # proxy artists for compact comparison legend
    from matplotlib.lines import Line2D
    proxies = [
        Line2D([], [], color=COL_DH, ls="dashed", lw=1.0,
               label=rf"Diffhalos (peak$\to$bound, $s_{{\rm eff}}={s_eff:.2f}$)"),
        Line2D([], [], color=COL_PYH, marker="s", ms=2.4, mfc="none", mew=0.7,
               ls="none", label=r"pyHalo (bound)"),
    ]
    leg1 = axa.legend(fontsize=5.8, frameon=False, loc="lower left",
                      title=rf"$M = 10^{{13}}\,M_\odot$, $z = {Z_L:g}$",
                      title_fontsize=5.8)
    axa.add_artist(leg1)
    axa.legend(handles=proxies, fontsize=5.8, frameon=False,
               loc="upper right")
    axa.set_xscale("log")
    axa.set_yscale("log")
    axa.set_xlim(1e-4, 1.0)
    axa.set_ylim(1e-2, 1e3)
    axa.set_xlabel(r"$\psi = m/M$")
    axa.set_ylabel(r"${\rm d}N/{\rm d}\ln\psi$")

    # ---------------- panel (b): radial bias function, log-log
    x = np.logspace(-1.6, 0.0, 400)
    axb.plot(x, bias_adopted(x), color="C0", lw=1.4,
             label=r"$[1+(x/0.54)^{-5/2}]^{-1/2}$ (this work)")
    axb.plot(10.0 ** GREEN_LOGX, 10.0 ** GREEN_LOGB, color="C2", lw=1.0,
             label="Green+21 (withering$+$disruption)")
    axb.plot(10.0 ** BOLSHOI_LOGX, 10.0 ** BOLSHOI_LOGB, "s", ms=2.2,
             color="k", ls="none", label="Bolshoi (Klypin+11)")
    axb.plot(x, bias_springel08(x), color="C4", ls="dashdot", lw=1.0,
             label=r"Springel+08 (Aq-A-1, Einasto/NFW)")
    axb.plot(x, x ** HAN16_GAMMA, color="C3", ls="dotted", lw=1.2,
             label=rf"Han+16: $x^{{{HAN16_GAMMA:g}}}$ (Aq-A)")
    axb.set_xscale("log")
    axb.set_yscale("log")
    axb.set_xlim(10 ** -1.6, 1.0)
    axb.set_ylim(10 ** -2.3, 1.6)
    axb.set_xlabel(r"$x = r/r_{200}$")
    axb.set_ylabel(r"$B(x) = n_{\rm sub}/n_{\rm host}$")
    axb.legend(fontsize=5.8, frameon=False, loc="upper left")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "fig_subhalo_population.png"
    out_pdf = out_png.with_suffix(".pdf")
    fig.savefig(out_png, dpi=300, facecolor="white")
    fig.savefig(out_pdf, facecolor="white")
    print(f"wrote {out_pdf.relative_to(REPO)}")
    print(f"wrote {out_png.relative_to(REPO)}")

    print(f"\nwf = {WF:.4f} (alpha_f = {AF:.4f})")
    print(f"{'M':>8} {'f_s':>7} {'f_sb(m_floor)':>13} {'z_f':>6} "
          f"{'N_tau':>6} {'<N>(>m_floor)':>13} {'s_eff':>6}")
    for M, fs, fsb, zf, Ntau, Nsub, s_eff in info:
        print(f"{M:8.1e} {fs:7.3f} {fsb:13.3f} {zf:6.2f} {Ntau:6.2f} "
              f"{Nsub:13.3g} {s_eff:6.3f}")
    g = math.gamma(S_M)
    print(f"\nGamma(s) sanity: quad(s,1e-300) = {gamma_inc_quad(S_M, 0):.6f} "
          f"vs math.gamma = {g:.6f}")

    # comparison cross-checks
    print("\nat psi=1e-3: ours(bound) vs Diffhalos peak and s_eff-converted bound:")
    for M, fs, fsb, zf, Ntau, Nsub, s_eff in info:
        _, gam, _, _ = fs_gamma(C, Z_L, M)
        ours = gam * 1e-3 ** ALPHA * math.exp(-BETA * 1e-3 ** OMEGA)
        dh = float(diffhalos_dndlnpsi(math.log10(M), np.array([1e-3]))[0])
        dhb = float(diffhalos_dndlnpsi(math.log10(M),
                                       np.array([1e-3 / s_eff]))[0])
        print(f"  M={M:.0e}: ours={ours:8.2f}  peak={dh:8.2f} "
              f" bound={dhb:8.2f}  bound/ours={dhb / ours:5.2f}")
    _, gam13, _, _ = fs_gamma(C, Z_L, M_PYHALO)
    ours13 = gam13 * psi_py ** ALPHA * np.exp(-BETA * psi_py ** OMEGA)
    print("pyHalo(bound)/ours per-host ratio (expect ~0.7, "
          "pyhalo_pipeline_comparison.md #4):")
    print("  " + " ".join(f"{r:.2f}" for r in dn_py / ours13))


if __name__ == "__main__":
    main()
