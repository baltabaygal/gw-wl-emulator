#!/usr/bin/env python
"""Mode analysis of the cascade corrections (understanding phase, analysis-only).

Per ingredient i∈{sub,bias} and z_s, build the correction matrix
  M_i[θ, lnμ] = logP_{base+i}(lnμ;θ) − logP_base(lnμ;θ)
on a shared lnμ grid, then SVD to test:
  H1 subhalo ≈ 1 width mode (weak 2nd tail mode by z_s=5)
  H2 clustering ≈ 1D but a SKEW/shoulder mode, ORTHOGONAL to the subhalo width mode
  H3 SVD modes ≈ analytic ∂P/∂σ (width, Hermite H2) and ∂P/∂skew (H3)

Traps handled: (1) MASK the moving edge (body+shoulder only; edge kept as a separate
scalar); (2) MC noise floor on the singular-value spectrum from the multi-seed block
(same light smoothing applied to data and noise); (3) both UNCENTERED (correction shape)
and θ-CENTERED (cosmology-dependence) SVD reported.

Usage: mode_analysis.py [--zs 0.5,1.0,5.0] [--tag lowz]
Env: `test` (numpy only for this stage; cascade-operator cross-check imports the -ar model).
"""
import os, sys, glob, json, argparse, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

D = ("/private/tmp/claude-501/-Users-baltabay-Desktop-gw-wl-emulator/"
     "ac612839-196b-4778-8b02-eab1b572a0e1/scratchpad/cascade_data")
OUT = "data/results/cascade_residual"; PLOTS = os.path.join(OUT, "plots")
os.makedirs(PLOTS, exist_ok=True)
idx = json.load(open(os.path.join(D, "index.json")))
CMIN = 20; NBINS = 160
LOWZ = {0.5, 1.0, 5.0}


def suffix(z, arm):
    if z in LOWZ:
        return {"base": "sub0", "sub": "sub1", "bias": "bias", "full": "full"}[arm]
    return arm                                   # z∈{6,8,10} use base/sub/bias


def load_grid(z, arm):
    """list of (theta=(h,om,s8), lnmu-array) over the Om/σ8/h grid rows."""
    rows = []
    for f in sorted(glob.glob(os.path.join(D, f"grid_*_zs{z}_{suffix(z, arm)}.npz"))):
        d = np.load(f)
        rows.append(((float(d["h"]), float(d["Om"]), float(d["s8"])),
                     d["lnmu"].astype(float)))
    return rows


def load_noise(z, arm):
    fs = sorted(glob.glob(os.path.join(D, f"noise_zs{z}_seed*_{suffix(z, arm)}.npz")))
    return [np.load(f)["lnmu"].astype(float) for f in fs]


def logP(x, edges):
    c, _ = np.histogram(x, bins=edges); w = np.diff(edges)
    dens = c / (x.size * w)
    lp = np.where(c >= CMIN, np.log(np.where(dens > 0, dens, 1e-300)), np.nan)
    return lp, c


def smooth(v, k=3):
    """light Hann smoothing along lnμ (reduce noise-mode inflation)."""
    w = np.hanning(k + 2)[1:-1]; w /= w.sum()
    return np.array([np.convolve(row, w, mode="same") for row in np.atleast_2d(v)])


def build_matrix(z, ing):
    """Return (cen, M[θ,lnμ], noise_bin, base_pool_logP, edge_scalar, keep_mask)."""
    base_rows = load_grid(z, "base"); ing_rows = load_grid(z, ing)
    pool = np.concatenate([r[1] for r in base_rows])
    lo = np.percentile(pool, 1.0); hi = np.percentile(pool, 99.0)     # body+shoulder; edge & far tail cut
    edges = np.linspace(lo, hi, NBINS + 1); cen = 0.5 * (edges[:-1] + edges[1:])
    edge_scalar = np.percentile(pool, 0.5)                            # edge kept SEPARATE
    M, ok = [], np.ones(NBINS, bool)
    thetas = []
    for (th, xb), (_, xi) in zip(base_rows, ing_rows):
        lpb, cb = logP(xb, edges); lpi, ci = logP(xi, edges)
        m = lpb - lpi if False else lpi - lpb
        M.append(m); thetas.append(th)
        ok &= (cb >= CMIN) & (ci >= CMIN)
    M = np.array(M)
    # noise: per-bin std of M over the multi-seed fiducial block
    nb = load_noise(z, "base"); ni = load_noise(z, ing)
    ns = min(len(nb), len(ni))
    Mn = np.array([logP(ni[k], edges)[0] - logP(nb[k], edges)[0] for k in range(ns)])
    noise_bin = np.nanstd(Mn, axis=0)
    keep = ok & np.isfinite(noise_bin) & (noise_bin > 0)
    return cen[keep], M[:, keep], noise_bin[keep], np.array(thetas), edge_scalar, ns


def svd_with_floor(M, noise_bin, ncomp=6, nmc=200, smooth_k=3, center=False):
    """Light-smooth, SVD; MC noise floor via matrices ~N(0,noise_bin²) same smoothing."""
    Ms = smooth(M, smooth_k)
    if center:
        Ms = Ms - Ms.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Ms, full_matrices=False)
    floor = np.zeros(ncomp)
    tops = np.zeros((nmc, ncomp))
    rng = np.random.default_rng(0)
    for j in range(nmc):
        Nn = rng.standard_normal(M.shape) * noise_bin[None, :]
        Nn = smooth(Nn, smooth_k)
        if center:
            Nn = Nn - Nn.mean(0, keepdims=True)
        s = np.linalg.svd(Nn, compute_uv=False)
        tops[j, :ncomp] = s[:ncomp]
    floor = np.percentile(tops, 95, axis=0)
    return U, S, Vt, floor


def sym_score(cen, v):
    """+1 = symmetric (width-like), −1 = antisymmetric (skew-like), about the centroid."""
    c0 = cen[np.argmax(np.abs(v))] if False else cen.mean()
    vi = np.interp(2 * c0 - cen, cen, v)          # mirror about c0
    a = float(np.dot(v, vi) / (np.linalg.norm(v) * np.linalg.norm(vi) + 1e-12))
    return a


def hermite_shapes(cen, base_pool):
    """analytic ΔlogP width (H2) and skew (H3) modes from the base body moments."""
    m = base_pool[(base_pool > np.percentile(base_pool, 1)) &
                  (base_pool < np.percentile(base_pool, 99))]
    mu, s = m.mean(), m.std()
    x = (cen - mu) / s
    H2 = x ** 2 - 1.0                    # ∂logP/∂σ (Gaussian)
    H3 = x ** 3 - 3.0 * x                # skew perturbation
    return H2 / np.linalg.norm(H2), H3 / np.linalg.norm(H3)


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zs", default="0.5,1.0,5.0")
    ap.add_argument("--tag", default="lowz")
    a = ap.parse_args()
    ZS = [float(x) for x in a.zs.split(",")]
    ings = ["sub", "bias"]

    res = {}          # res[(z,ing)] = dict
    rows = [f"# Cascade correction — mode (SVD) analysis [{a.tag}]\n",
            "M_i[θ,lnμ] = logP_base+i − logP_base on body+shoulder (edge MASKED, kept as "
            "separate scalar; far tail >q99 cut). Light Hann-smoothed; MC noise floor from "
            "the multi-seed block (95th pct). Sym: +1 symmetric/width, −1 antisym/skew.\n"]
    rows.append("## A. Per-ingredient SVD — modes above the noise floor (UNCENTERED)\n")
    rows.append("| z_s | ing | S1 | S2 | S3 | floor(S1..) | n>floor | PC1 sym | PC1·H2 | PC1·H3 |")
    rows.append("|----|----|----|----|----|----|----|----|----|----|")

    for z in ZS:
        base_pool = np.concatenate([r[1] for r in load_grid(z, "base")])
        for ing in ings:
            if not load_grid(z, ing):
                continue
            cen, M, noise_bin, thetas, edge_sc, ns = build_matrix(z, ing)
            U, S, Vt, floor = svd_with_floor(M, noise_bin, center=False)
            Uc, Sc, Vtc, floorc = svd_with_floor(M, noise_bin, center=True)
            H2, H3 = hermite_shapes(cen, base_pool)
            pc1 = Vt[0] * np.sign(np.dot(Vt[0], H2) + 1e-9)   # sign so width-positive
            n_above = int(np.sum(S[:6] > floor[0]))            # clears the TOP noise mode
            res[(z, ing)] = dict(cen=cen, S=S, floor=floor, Vt=Vt, U=U, thetas=thetas,
                                 Sc=Sc, floorc=floorc, Vtc=Vtc, H2=H2, H3=H3,
                                 pc1=pc1, edge=edge_sc, M=M, noise=noise_bin, ns=ns,
                                 nabove=n_above, nabove_c=int(np.sum(Sc[:6] > floorc[:6])))
            rows.append("| %.1f | %s | %.3f | %.3f | %.3f | %.3f | %d | %+.2f | %.2f | %.2f |" % (
                z, ing, S[0], S[1], S[2], floor[0], n_above,
                sym_score(cen, pc1), abs(cos(pc1, H2)), abs(cos(pc1, H3))))

    # B. mean-correction shapes + unique-deformation decomposition
    rows.append("\n## B. Shared vs unique deformation (the honest orthogonality test)\n")
    rows.append("u_i = unit MEAN-ΔlogP shape of ingredient i. cos(u_sub,u_bias) = how "
                "parallel the leading corrections are (both broaden ⇒ expected high). The "
                "real H2 test: remove the shared subhalo direction from clustering, "
                "u_bias⊥ = u_bias − (u_bias·u_sub)u_sub; report ‖u_bias⊥‖ (fraction of "
                "clustering NOT shared with subhalos) and cos(u_bias⊥, H3) (is that unique "
                "part SKEW). Symmetrically u_sub⊥ vs H2. noise‖ = MC ‖·‖ of a noise-only "
                "mean shape (floor for the residual norms).\n")
    rows.append("| z_s | cos(u_sub,u_bias) | ‖u_bias⊥‖ | cos(u_bias⊥,H3) | ‖u_sub⊥‖ | cos(u_sub⊥,H2) | noise‖ |")
    rows.append("|----|----|----|----|----|----|----|")
    for z in ZS:
        if (z, "sub") not in res or (z, "bias") not in res:
            continue
        rs, rb = res[(z, "sub")], res[(z, "bias")]
        cs = rs["cen"]
        us = rs["M"].mean(0)
        ub = np.interp(cs, rb["cen"], rb["M"].mean(0))    # align bias grid -> sub grid
        us /= np.linalg.norm(us); ub /= np.linalg.norm(ub)
        H2, H3 = rs["H2"], rs["H3"]
        ubp = ub - np.dot(ub, us) * us
        usp = us - np.dot(us, ub) * ub
        # noise floor on residual norm: mean of ns noise ΔlogP shapes, normalized scale
        rng = np.random.default_rng(1)
        nf = np.mean([np.linalg.norm(smooth(rng.standard_normal(rb["M"].shape) *
                     rb["noise"][None, :]).mean(0)) for _ in range(50)])
        scale = np.linalg.norm(rb["M"].mean(0))     # so ‖u_bias⊥‖ and noise‖ comparable
        rows.append("| %.1f | %+.3f | %.3f | %.2f | %.3f | %.2f | %.3f |" % (
            z, cos(us, ub), np.linalg.norm(ubp), abs(cos(ubp, H3)),
            np.linalg.norm(usp), abs(cos(usp, H2)), nf / scale))

    # C. parametric cross-check summary already in table A (PC1·H2, PC1·H3)
    rows.append("\n## C. SVD-vs-parametric (H3): PC1·H2 (width) and PC1·H3 (skew) — see table A.\n")

    open(os.path.join(OUT, f"report_modes_{a.tag}.md"), "w").write("\n".join(rows) + "\n")

    # ---- figures ----
    nz = len(ZS)
    # (1) SV spectra + floor
    fig, axs = plt.subplots(len(ings), nz, figsize=(4.2 * nz, 3.6 * len(ings)), squeeze=False)
    for r, ing in enumerate(ings):
        for cci, z in enumerate(ZS):
            ax = axs[r][cci]
            if (z, ing) not in res:
                ax.axis("off"); continue
            R = res[(z, ing)]
            k = np.arange(1, 7)
            ax.semilogy(k, R["S"][:6], "o-", label="singular values")
            ax.semilogy(k, R["floor"][:6], "x--", color="r", label="noise floor (95%)")
            ax.set_title(f"{ing}  z_s={z}  (n>floor={R['nabove']})")
            ax.set_xlabel("mode"); ax.set_ylabel("S")
            if r == 0 and cci == 0:
                ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, f"mode_sv_spectra_{a.tag}.png"), dpi=120); plt.close(fig)

    # (2) leading mode shapes vs Hermite
    fig, axs = plt.subplots(len(ings), nz, figsize=(4.2 * nz, 3.6 * len(ings)), squeeze=False)
    for r, ing in enumerate(ings):
        for cci, z in enumerate(ZS):
            ax = axs[r][cci]
            if (z, ing) not in res:
                ax.axis("off"); continue
            R = res[(z, ing)]
            ax.plot(R["cen"], R["pc1"], "b-", label="PC1")
            ax.plot(R["cen"], R["Vt"][1] * np.sign(np.dot(R["Vt"][1], R["H3"])), "g-", alpha=0.7, label="PC2")
            ax.plot(R["cen"], R["H2"] * np.sign(np.dot(R["pc1"], R["H2"])), "k--", alpha=0.5, label="H2 (width)")
            ax.plot(R["cen"], R["H3"], "m:", alpha=0.6, label="H3 (skew)")
            ax.set_title(f"{ing}  z_s={z}"); ax.set_xlabel("lnμ")
            if r == 0 and cci == 0:
                ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, f"mode_shapes_{a.tag}.png"), dpi=120); plt.close(fig)

    # (3) ΔlogP mean curves
    fig, axs = plt.subplots(1, nz, figsize=(4.6 * nz, 3.8), squeeze=False)
    for cci, z in enumerate(ZS):
        ax = axs[0][cci]
        for ing, col in [("sub", "C0"), ("bias", "C1")]:
            if (z, ing) in res:
                R = res[(z, ing)]
                ax.plot(R["cen"], R["M"].mean(0), col, label=f"ΔlogP_{ing}")
        ax.axhline(0, color="gray", lw=0.5); ax.set_title(f"z_s={z}"); ax.set_xlabel("lnμ")
        if cci == 0:
            ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, f"mode_dlogp_curves_{a.tag}.png"), dpi=120); plt.close(fig)

    print("\n".join(rows))
    return res


if __name__ == "__main__":
    main()
