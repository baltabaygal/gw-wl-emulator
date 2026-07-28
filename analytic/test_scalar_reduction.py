#!/usr/bin/env python
"""Isolate the SCALAR REDUCTION, with everything else held identical.

The analytic chain sums per-lens jumps

    xi_scalar = sum_i  -ln[(1-k_i)^2 - g_i^2]

while the engine sums the fields first and forms the jump once

    xi_vector = -ln[(1 - sum_i k_i)^2 - |sum_i g_i|^2].

Every other difference (mass function, profile, geometry, counts, thresholds,
grids) is removed here by drawing BOTH arms from the SAME realizations of the
SAME lens population -- the analytic chain's own jump measure.  The gap between
the two arms is therefore the scalar reduction and nothing else.

Lenses with kappa > kappa_low are drawn exactly (Poisson over the (z, M, b)
cells).  The faint remainder is added as a Gaussian with the Campbell variance,
identically to both arms, so it cancels in the comparison.

Usage:  python analytic/test_scalar_reduction.py --zs 1 5 10
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
import sgl
from compare_pdf import source_plane_hist, weighted_stats, jsd, MODES

EDGES = np.linspace(-0.45, 0.55, 201)
CTR = 0.5 * (EDGES[1:] + EDGES[:-1])
WID = np.diff(EDGES)


def build_population(cos, zs, kappa_low, Mmin=1e7, Mmax=1e16, Nz=40, NM=48):
    """Flatten the lens population into (kappa, gamma, expected-count) cells.

    Returns the resolved cells (kappa > kappa_low) plus the Campbell variances
    of the unresolved remainder, which is handed to both arms as a Gaussian.
    """
    zgrid = np.linspace(1e-3, zs - 1e-3, Nz)
    dz = zgrid[1] - zgrid[0]
    Mgrid = np.logspace(np.log10(Mmin), np.log10(Mmax), NM)
    dlnM = np.log(Mgrid[1]) - np.log(Mgrid[0])
    K, G, W = [], [], []
    faint_k2 = faint_g2 = 0.0
    xa, xb = sgl._XG[:-1], sgl._XG[1:]
    xm = np.sqrt(xa * xb)
    for z in zgrid:
        wz = (1 + z)**2 * sgl.CKMS / cos.Hz(z) * dz
        for M in Mgrid:
            C, rs, ks, fC = cos.nfw_params(M, z, zs)
            k, g = sgl.kappa_gamma(xm, ks)
            w = (wz * cos.dndlnM(M, z) * dlnM
                 * np.pi * rs * rs * (xb**2 - xa**2))     # expected count
            hi = k > kappa_low
            K.append(k[hi]); G.append(g[hi]); W.append(w[hi])
            faint_k2 += float(np.sum(w[~hi] * k[~hi]**2))
            faint_g2 += float(np.sum(w[~hi] * g[~hi]**2))
    return (np.concatenate(K), np.concatenate(G), np.concatenate(W),
            faint_k2, faint_g2)


def simulate(K, G, W, faint_k2, faint_g2, nray, rng, chunk=20000):
    """Draw rays and return (xi_scalar, xi_vector) for the same realizations."""
    lam = float(W.sum())
    p = np.cumsum(W / lam)
    xi_i = -np.log(np.clip((1 - K)**2 - G * G, 1e-300, None))
    out_s, out_v = [], []
    done = 0
    while done < nray:
        n = min(chunk, nray - done)
        counts = rng.poisson(lam, n)
        tot = int(counts.sum())
        idx = np.searchsorted(p, rng.random(tot))
        seg = np.repeat(np.arange(n), counts)
        phi = rng.uniform(0, 2 * np.pi, tot)
        # scalar arm: sum the per-lens jumps
        s = np.bincount(seg, weights=xi_i[idx], minlength=n)
        # vector arm: sum the fields, then form one jump
        kt = np.bincount(seg, weights=K[idx], minlength=n)
        g1 = np.bincount(seg, weights=G[idx] * np.cos(phi), minlength=n)
        g2 = np.bincount(seg, weights=G[idx] * np.sin(phi), minlength=n)
        # unresolved remainder: identical draw handed to both arms
        kf = rng.normal(0.0, np.sqrt(faint_k2), n)
        gf1 = rng.normal(0.0, np.sqrt(faint_g2 / 2), n)
        gf2 = rng.normal(0.0, np.sqrt(faint_g2 / 2), n)
        s = s + 2 * kf + (gf1**2 + gf2**2)      # linear-regime equivalent
        kt, g1, g2 = kt + kf, g1 + gf1, g2 + gf2
        out_s.append(s)
        out_v.append(np.stack([kt, g1, g2]))
        done += n
    xs = np.concatenate(out_s)
    kv, gv1, gv2 = np.concatenate(out_v, axis=1)
    # each arm anchored the way its own framework does
    xs = xs - xs.mean()                          # analytic: <xi> = 0
    detA = (1 - (kv - kv.mean()))**2 - (gv1**2 + gv2**2)   # engine: <kappa> = 0
    ok = detA > 0
    return xs, -np.log(detA[ok])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zs", type=float, nargs="+", default=[1.0, 5.0, 10.0])
    ap.add_argument("--nray", type=int, default=200000)
    ap.add_argument("--kappa-low", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    cos = sgl.Cosmology(**MODES["engine"])
    rng = np.random.default_rng(a.seed)

    fig, axes = plt.subplots(1, len(a.zs), figsize=(4.3 * len(a.zs), 4.0),
                             squeeze=False)
    for j, zs in enumerate(a.zs):
        K, G, W, fk2, fg2 = build_population(cos, zs, a.kappa_low)
        xs, xv = simulate(K, G, W, fk2, fg2, a.nray, rng)
        ps, _ = source_plane_hist(xs, EDGES)
        pv, _ = source_plane_hist(xv, EDGES)
        ss, sv = weighted_stats(xs), weighted_stats(xv)
        print(f"=== z_s = {zs:g}   <N>_resolved = {W.sum():.0f} "
              f"(kappa > {a.kappa_low:g}),  {a.nray} rays")
        print(f"    scalar arm (analytic composition): Var = {ss['var']:.5f}"
              f"   <lnmu>_src = {ss['mean']:+.5f}")
        print(f"    vector arm (engine composition):   Var = {sv['var']:.5f}"
              f"   <lnmu>_src = {sv['mean']:+.5f}")
        print(f"    -> scalar/vector Var ratio = {ss['var']/sv['var']:.4f}"
              f"   JSD(scalar, vector) = {jsd(ps, pv, WID):.2e}")
        print(f"    -> mean offset introduced by the two anchors = "
              f"{ss['mean']-sv['mean']:+.5f}\n")

        ax = axes[0, j]
        ax.axhline(1.0, color="crimson", lw=1.2)
        g = (ps > 0) & (pv > 0)
        ax.plot(CTR[g], ps[g] / pv[g], "o", ms=2.4, color="tab:red")
        ax.set_ylim(0.6, 1.4)
        ax.set_xlabel(r"$\ln\mu$")
        ax.set_title(rf"$z_s={zs:g}$", fontsize=11)
        ax.grid(alpha=0.25, lw=0.5)
        if j == 0:
            ax.set_ylabel("scalar / vector composition")
    fig.suptitle("Scalar reduction in isolation: same lenses, same "
                 "realizations, two compositions", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = HERE / "figures" / "scalar_reduction_isolated.png"
    fig.savefig(out, dpi=160)
    print(f"[fig] {out}")


if __name__ == "__main__":
    main()
