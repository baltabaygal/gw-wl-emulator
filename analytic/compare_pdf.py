#!/usr/bin/env python
"""Overlay the semi-analytic P(ln mu) on the halo-only C++ Monte Carlo.

The analytic chain and the engine are run in the SAME model scope (NFW halos
only; no filaments, bias, ellipticity or subhalos) and, with
``--mode engine``, on the same cosmology conventions (smooth-k sigma(M) window
+ full EH98 transfer function), so every input agrees to <0.3%.  What is left
in the comparison is the method itself:

  * the engine sums (kappa, gamma_1, gamma_2) over lenses and then forms
    mu = 1/[(1-kappa)^2 - gamma^2]; the analytic chain sums the per-lens
    xi_i = -ln[(1-kappa_i)^2 - gamma_i^2].  That is the "scalar reduction",
    exact for a single lens and approximate when lenses overlap;
  * the engine anchors <kappa> = 0 empirically per batch, the analytic chain
    compensates the Levy exponent so that <xi> = 0 exactly;
  * the engine renders lenses above kappa_thr (<N> = 100) explicitly and
    replaces the sub-threshold band by a Gaussian with the matching variance;
    the analytic chain integrates the whole band as explicit jumps.

Both curves are SOURCE-plane (weighted by 1/mu), which is the convention of
``lensing::Plnmuf`` and of the analytic Esscher tilt R_s = e^{-xi} R.

Usage:
  python analytic/compare_pdf.py --zs 1.0 --mode engine
  python analytic/compare_pdf.py --zs 1.0 --mode engine --all-z
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sgl

DATA = HERE / "data"
FIGS = HERE / "figures"

MODES = {"engine": dict(window="smoothk", transfer="eh98"),
         "spec": dict(window="tophat", transfer="nowiggle")}


# ------------------------------- MC side ------------------------------------
def mc_lnmu(path, anchor="robust", anchor_cut=1.0):
    """Reproduce ``lensing::sample_lnmu`` from the raw (kappa, gamma) rays.

    anchor='batch'  -> engine default (kappa_anchor=0): empirical mean over all
                       rays, i.e. one monster ray shifts the whole batch
    anchor='robust' -> engine kappa_anchor=1: mean over rays with kappa <= cut
    """
    d = np.load(path)
    k = d["kappa"]
    g = np.hypot(d["gamma1"], d["gamma2"])
    kbar = k.mean() if anchor == "batch" else k[k <= anchor_cut].mean()
    detA = (1.0 - (k - kbar))**2 - g * g
    ok = detA > 0
    lnmu = -np.log(detA[ok])
    return lnmu, dict(nray=k.size, nvalid=int(ok.sum()), kappa_bar=float(kbar),
                      zs=float(d["zs"]))


def source_plane_hist(lnmu, edges):
    """Source-plane pdf of ln mu: image-plane rays reweighted by 1/mu.

    Normalised by the total weight of ALL rays, not only those inside the
    plotting window, so the MC and the analytic curve are on the same absolute
    footing (both integrate to 1 over the full support).
    """
    w = np.exp(-lnmu)
    H, _ = np.histogram(lnmu, bins=edges, weights=w)
    H2, _ = np.histogram(lnmu, bins=edges, weights=w * w)
    width = np.diff(edges)
    norm = w.sum()
    return H / (norm * width), np.sqrt(H2) / (norm * width)


def weighted_stats(lnmu):
    w = np.exp(-lnmu)
    w = w / w.sum()
    m = float((w * lnmu).sum())
    v = float((w * (lnmu - m)**2).sum())
    s = float((w * (lnmu - m)**3).sum()) / v**1.5
    neff = 1.0 / np.sum(w**2)
    return dict(mean=m, var=v, skew=s, neff=float(neff))


def sigma_DL_mc(lnmu):
    """Source-plane fractional scatter of D_L ~ mu^{-1/2}, i.e. of e^{-xi/2},
    with the 1/mu source-plane weights."""
    w = np.exp(-lnmu)
    w = w / w.sum()
    x = np.exp(-0.5 * lnmu)
    m = float((w * x).sum())
    v = float((w * (x - m)**2).sum())
    return np.sqrt(v) / m


# ----------------------------- analytic side --------------------------------
def analytic_curve(zs, mode="engine", Mmin=1e7, Mmax=1e16, Nz=40, NM=48,
                   cache=True):
    tag = f"R_{mode}_zs{zs:g}_Mmin{Mmin:g}_Nz{Nz}_NM{NM}.npy"
    f = DATA / tag
    if cache and f.exists():
        R = np.load(f)
    else:
        cos = sgl.Cosmology(**MODES[mode])
        R = sgl.R_of_xi(cos, zs, Mmin=Mmin, Mmax=Mmax, Nz=Nz, NM=NM)
        DATA.mkdir(parents=True, exist_ok=True)
        np.save(f, R)
    return R


# --------------------------------- report -----------------------------------
def microscopic_checks(zs, R, h=0.674, Om=0.315, s8=0.811, Nhalos=100):
    """Two engine internals the analytic R(xi) must reproduce independently of
    the PDF -- the expected explicit-lens count and the variance the engine
    assigns to its sub-threshold Gaussian background.

    The engine slices in kappa at kappa_thr; the analytic slices in xi.  In the
    far field xi = 2 kappa + O(kappa^2) so xi_thr = 2 kappa_thr, accurate to
    ~1e-3 at these thresholds.
    """
    sys.path.insert(0, str(HERE.parent / "build"))
    import gwlensing as gw
    kt = gw.get_kappa_threshold(z=zs, h=h, OmegaM=Om, sigma8=s8, Nhalos=Nhalos)
    N_eng = gw.get_expected_halo_count(z=zs, kappathr=kt, h=h, OmegaM=Om,
                                       sigma8=s8)
    sW = gw.get_sigma_background(z=zs, kappathr=kt, h=h, OmegaM=Om, sigma8=s8)
    xi, lo = sgl.XI, sgl.XI < 2 * kt
    return dict(
        kappa_thr=float(kt),
        N_engine=float(N_eng),
        N_analytic=float(np.trapezoid(R[~lo], xi[~lo])),
        weakvar_engine=float(4 * sW**2),
        weakvar_analytic=float(np.trapezoid((R * xi**2)[lo], xi[lo])),
    )


def jsd(p, q, w):
    """Jensen-Shannon divergence of two binned pdfs (bin width w)."""
    p, q = np.clip(p, 0, None), np.clip(q, 0, None)
    p = p / (p * w).sum()
    q = q / (q * w).sum()
    m = 0.5 * (p + q)
    def kl(a, b):
        s = a > 0
        return float(np.sum(a[s] * np.log(a[s] / b[s]) * w[s]))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def compare(zs, mode="engine", anchor="robust", nbins=160,
            xi_lo=-0.35, xi_hi=0.55):
    mcf = DATA / f"mc_halo_only_zs{zs:g}.npz"
    lnmu, meta = mc_lnmu(mcf, anchor=anchor)
    R = analytic_curve(zs, mode=mode)
    Rs = sgl.tilt_source(R)

    edges = np.linspace(xi_lo, xi_hi, nbins + 1)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    width = np.diff(edges)
    pdf_mc, err_mc = source_plane_hist(lnmu, edges)

    _, P_a = sgl.P_of_xi(Rs, xi_out=ctr)

    st_mc = weighted_stats(lnmu)
    st_an = sgl.moments(Rs)
    # analytic curve shifted onto the MC's mean: isolates SHAPE agreement from
    # the known mean-subtraction convention difference (<kappa>=0 in the engine
    # vs <xi>=0 in the compensated Levy exponent)
    _, P_shift = sgl.P_of_xi(Rs, xi_out=ctr - st_mc["mean"])

    out = dict(
        zs=zs, mode=mode, anchor=anchor,
        **{f"mc_{k}": v for k, v in meta.items()},
        mc_mean=st_mc["mean"], mc_var=st_mc["var"], mc_skew=st_mc["skew"],
        an_var=st_an["var"], an_skew=st_an["skew"],
        mc_sigmaDL=float(sigma_DL_mc(lnmu)),
        an_sigmaDL=float(sgl.sigma_DL_over_DL(Rs)),
        **microscopic_checks(zs, R),
    )
    # probability-weighted |ratio - 1|: the deviation an actual event sample
    # would feel, rather than one dominated by empty tail bins
    good = (P_a > 0) & (pdf_mc > 0)
    wgt = (pdf_mc * width)[good]
    out["body_reldiff"] = float(np.sum(wgt * np.abs(pdf_mc[good] / P_a[good]
                                                    - 1)) / wgt.sum())
    out["body_reldiff_meanmatched"] = float(
        np.sum(wgt * np.abs(pdf_mc[good] / P_shift[good] - 1)) / wgt.sum())
    out["jsd"] = jsd(pdf_mc, P_a, width)
    out["jsd_meanmatched"] = jsd(pdf_mc, P_shift, width)
    out["window_prob_mc"] = float((pdf_mc * width).sum())
    return ctr, pdf_mc, err_mc, P_a, P_shift, out


def figure(zs_list, mode="engine", anchor="robust", fname=None):
    n = len(zs_list)
    fig, axes = plt.subplots(2, n, figsize=(4.1 * n, 6.4), sharex="col",
                             gridspec_kw=dict(height_ratios=[2.4, 1]))
    if n == 1:
        axes = axes.reshape(2, 1)
    rows = []
    for j, zs in enumerate(zs_list):
        ctr, pdf, err, Pa, Pshift, out = compare(zs, mode=mode, anchor=anchor)
        rows.append(out)
        ax, ar = axes[0, j], axes[1, j]
        ax.errorbar(ctr, pdf, yerr=err, fmt="o", ms=2.2, lw=0.7, color="0.35",
                    label="C++ MC (halos only)", zorder=2)
        ax.plot(ctr, Pa, "-", lw=2.0, color="crimson",
                label="semi-analytic", zorder=3)
        ax.plot(ctr, Pshift, "--", lw=1.2, color="steelblue",
                label="semi-analytic, mean-matched", zorder=3)
        ax.set_yscale("log")
        ax.set_ylim(max(1e-4, 0.5 * min(pdf[pdf > 0].min(), Pa[Pa > 0].min())),
                    3 * max(pdf.max(), Pa.max()))
        ax.set_title(rf"$z_s = {zs:g}$", fontsize=11)
        ax.grid(alpha=0.25, lw=0.5)
        if j == 0:
            ax.set_ylabel(r"$P(\ln\mu)$  (source plane)")
            ax.legend(fontsize=8, frameon=False)

        good = (pdf > 0) & (Pa > 0)
        ar.axhline(1.0, color="crimson", lw=1.2)
        ar.errorbar(ctr[good], pdf[good] / Pa[good],
                    yerr=err[good] / Pa[good], fmt="o", ms=2.2, lw=0.7,
                    color="0.35")
        gs = (pdf > 0) & (Pshift > 0)
        ar.plot(ctr[gs], pdf[gs] / Pshift[gs], ".", ms=2.0, color="steelblue",
                alpha=0.8)
        ar.set_ylim(0.6, 1.4)
        ar.set_xlabel(r"$\ln\mu$")
        ar.grid(alpha=0.25, lw=0.5)
        if j == 0:
            ar.set_ylabel("MC / analytic")
    fig.suptitle(f"Halo-only sGL: semi-analytic vs Monte Carlo "
                 f"(cosmology mode: {mode}, anchor: {anchor})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    FIGS.mkdir(parents=True, exist_ok=True)
    fname = fname or FIGS / f"pdf_overlay_{mode}_{anchor}.png"
    fig.savefig(fname, dpi=160)
    print(f"[fig] {fname}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zs", type=float, nargs="+", default=[1.0])
    ap.add_argument("--mode", choices=list(MODES), default="engine")
    ap.add_argument("--anchor", choices=["robust", "batch"], default="robust")
    a = ap.parse_args()
    rows = figure(a.zs, mode=a.mode, anchor=a.anchor)
    print()
    print("  zs   Var MC     Var an   ratio  sDL MC  sDL an  ratio  "
          " N_an/N_eng  weakVar an/eng")
    for r in rows:
        print(f" {r['zs']:4.1f}  {r['mc_var']:.5f}  {r['an_var']:.5f}  "
              f"{r['mc_var']/r['an_var']:.3f}  {r['mc_sigmaDL']:.4f}  "
              f"{r['an_sigmaDL']:.4f}  {r['mc_sigmaDL']/r['an_sigmaDL']:.3f}"
              f"   {r['N_analytic']/r['N_engine']:6.3f}     "
              f"{r['weakvar_analytic']/r['weakvar_engine']:6.3f}")
    print()
    print("  zs   JSD       JSD(mean-matched)  body|d|  body|d|(mm)")
    for r in rows:
        print(f" {r['zs']:4.1f}  {r['jsd']:.2e}  {r['jsd_meanmatched']:.2e}"
              f"           {100*r['body_reldiff']:5.1f}%  "
              f"{100*r['body_reldiff_meanmatched']:5.1f}%")
    (DATA / f"compare_{a.mode}_{a.anchor}.json").write_text(
        json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
