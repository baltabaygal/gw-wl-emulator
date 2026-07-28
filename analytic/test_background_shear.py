#!/usr/bin/env python
"""Is the high-z_s analytic-vs-MC gap caused by the engine's shear-free
sub-threshold background?

The engine fills its Gaussian weak arm with convergence only
(``lensing.cpp:779-781``: kappa = PkappaW(mt), gamma1 = gamma2 = 0), while the
analytic jump measure gives every lens in that band its shear too.  Rather than
argue about whether that matters, we ADD the missing shear back into the MC
rays and re-derive the PDF.

The missing amount is measured, not guessed:

    <gamma_sub^2> = [int gamma^2 dR over the full engine kappa coverage]
                  - [int gamma^2 dR over kappa > kappa_thr]

and it is injected as a 2D Gaussian (the same statistical stand-in the engine
already uses for the band's convergence), so <g1^2 + g2^2> matches exactly and
the position angles stay random.

Ladder:  MC as-is  ->  MC + background shear  ->  analytic.
Whatever the injection does not close is the scalar reduction + the mean
convention.

Usage:  python analytic/test_background_shear.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "build"))
import sgl
import gwlensing as gw
from compare_pdf import (source_plane_hist, analytic_curve, weighted_stats,
                         sigma_DL_mc, jsd, MODES)

P = dict(h=0.674, OmegaM=0.315, sigma8=0.811)
EDGES = np.linspace(-0.45, 0.55, 201)
CTR = 0.5 * (EDGES[1:] + EDGES[:-1])
WID = np.diff(EDGES)


def lnmu_from(k, g1, g2, anchor_cut=1.0):
    kbar = k[k <= anchor_cut].mean()
    detA = (1.0 - (k - kbar))**2 - (g1 * g1 + g2 * g2)
    ok = detA > 0
    return -np.log(detA[ok])


def arm(lnmu, Pa):
    pdf, _ = source_plane_hist(lnmu, EDGES)
    st = weighted_stats(lnmu)
    g = (pdf > 0) & (Pa > 0)
    wt = (pdf * WID)[g]
    return dict(pdf=pdf, var=st["var"], mean=st["mean"],
                sdl=float(sigma_DL_mc(lnmu)), jsd=jsd(pdf, Pa, WID),
                body=float(np.sum(wt * np.abs(pdf[g] / Pa[g] - 1)) / wt.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zs", type=float, nargs="+", default=[1.0, 5.0, 10.0])
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    cos = sgl.Cosmology(**MODES["engine"])
    rng = np.random.default_rng(a.seed)

    fig, axes = plt.subplots(1, len(a.zs), figsize=(4.3 * len(a.zs), 4.0),
                             squeeze=False)
    for j, zs in enumerate(a.zs):
        d = np.load(HERE / "data" / f"mc_halo_only_zs{zs:g}.npz")
        k, g1, g2 = d["kappa"], d["gamma1"], d["gamma2"]
        kt = gw.get_kappa_threshold(z=zs, Nhalos=100, **P)

        full = sgl.campbell_moments(cos, zs, 1e-3 * kt)["gamma2"]
        expl = sgl.campbell_moments(cos, zs, kt)["gamma2"]
        miss = full - expl                       # <gamma_sub^2> the engine omits

        Rs = sgl.tilt_source(analytic_curve(zs))
        _, Pa = sgl.P_of_xi(Rs, xi_out=CTR)
        an_var = float(sgl.moments(Rs)["var"])
        an_sdl = float(sgl.sigma_DL_over_DL(Rs))

        s = np.sqrt(miss / 2.0)                  # per component
        gb1 = rng.normal(0.0, s, k.size)
        gb2 = rng.normal(0.0, s, k.size)

        base = arm(lnmu_from(k, g1, g2), Pa)
        shear = arm(lnmu_from(k, g1 + gb1, g2 + gb2), Pa)

        print(f"=== z_s = {zs:g}   kappa_thr = {kt:.3e}")
        print(f"    <gamma^2>: explicit {expl:.4e}  full {full:.4e}"
              f"   MISSING {miss:.4e}  ({100*miss/full:.0f}% of the band)")
        print(f"    {'arm':<22} {'Var(lnmu)':>10} {'/an':>7} "
              f"{'<lnmu>':>10} {'sigma_DL':>9} {'/an':>7} {'JSD':>9} "
              f"{'body|d|':>8}")
        for nm, r in (("MC as-is", base), ("MC + bkgd shear", shear)):
            print(f"    {nm:<22} {r['var']:10.5f} {r['var']/an_var:7.3f} "
                  f"{r['mean']:+10.5f} {r['sdl']:9.5f} "
                  f"{r['sdl']/an_sdl:7.3f} {r['jsd']:9.2e} "
                  f"{100*r['body']:7.1f}%")
        print(f"    {'analytic':<22} {an_var:10.5f} {1.0:7.3f} "
              f"{0.0:+10.5f} {an_sdl:9.5f} {1.0:7.3f}")
        closed = ((shear["var"] - base["var"]) / (an_var - base["var"])
                  if an_var != base["var"] else np.nan)
        print(f"    -> background shear closes {100*closed:.1f}% of the "
              f"Var(lnmu) gap, and moves <lnmu> by "
              f"{shear['mean']-base['mean']:+.5f}\n")

        ax = axes[0, j]
        ax.axhline(1.0, color="crimson", lw=1.2)
        gg = (base["pdf"] > 0) & (Pa > 0)
        ax.plot(CTR[gg], base["pdf"][gg] / Pa[gg], "o", ms=2.4, color="0.35",
                label="MC as-is / analytic")
        gg = (shear["pdf"] > 0) & (Pa > 0)
        ax.plot(CTR[gg], shear["pdf"][gg] / Pa[gg], "o", ms=2.4,
                color="tab:purple", label="MC + bkgd shear / analytic")
        ax.set_ylim(0.6, 1.4)
        ax.set_xlabel(r"$\ln\mu$")
        ax.set_title(rf"$z_s={zs:g}$", fontsize=11)
        ax.grid(alpha=0.25, lw=0.5)
        if j == 0:
            ax.set_ylabel("MC / analytic")
            ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Adding the engine's missing background shear barely moves "
                 "P(ln$\\mu$)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = HERE / "figures" / "background_shear_test.png"
    fig.savefig(out, dpi=160)
    print(f"[fig] {out}")


if __name__ == "__main__":
    main()
