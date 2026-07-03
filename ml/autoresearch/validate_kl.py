"""ACE-style validation battery (Turker et al. 2026, arXiv:2512.01607) for the
production model (smooth_model.load_smooth_fn):

1. KL divergence D_KL(sim || model) per (cosmology, z) panel, on the ACE support
   (mu in [edge, 6]) and on our full window (mu in [edge, 100]); image plane and
   source plane (P_S ~ P_I/mu, renormalized — ACE works in the source plane).
   Plug-in KL from an N-sample histogram is biased by ~(K_bins-1)/(2N); we report
   the bias-corrected value (correction printed).
2. Moment recovery: mean, sigma, skew of lnmu (sim vs model) per panel.
3. Flux conservation: <1/mu>_image-plane = 1 (equivalent to <mu>_source = 1).

Uses the cached 15-panel 400k reference (param_space_ref.npz).
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
from pathlib import Path
from ml.autoresearch import smooth_model
from ml.autoresearch.param_space_check import COSMOS, ZS, EDGES

AR = Path(__file__).resolve().parent
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def panel_kl(fn, cnt, N, z, th, mu_hi, source_plane):
    """Bias-corrected plug-in KL(sim || model) over bins within [edge, mu_hi]."""
    ctr = np.sqrt(EDGES[:-1] * EDGES[1:])
    lw = np.diff(np.log(EDGES))
    e = smooth_model.edge_lnmu(z, *th)
    m = (np.log(ctr) > e - 0.03) & (ctr <= mu_hi) & (cnt > 0)
    if m.sum() < 5:
        return np.nan, 0.0
    p_sim = cnt[m].astype(float)
    lp = np.asarray(fn(np.log(ctr[m]), z, th), np.float64)
    q = np.exp(lp) * lw[m]                      # model probability per bin
    if source_plane:                            # P_S ~ P_I / mu
        p_sim = p_sim / ctr[m]
        q = q / ctr[m]
    p_sim = p_sim / p_sim.sum()
    q = np.maximum(q / q.sum(), 1e-300)
    kl = float(np.sum(p_sim * np.log(p_sim / q)))
    bias = (m.sum() - 1) / (2.0 * cnt[m].sum())  # plug-in estimator bias
    return max(kl - bias, 0.0), bias


def panel_moments(fn, cnt, N, z, th):
    """lnmu moments: sim (from histogram) vs model (from dense grid)."""
    ctr = np.sqrt(EDGES[:-1] * EDGES[1:]); lc = np.log(ctr)
    w = cnt / cnt.sum()
    m1s = np.sum(w * lc); s_s = np.sqrt(np.sum(w * (lc - m1s) ** 2))
    sk_s = np.sum(w * (lc - m1s) ** 3) / s_s ** 3
    g = np.linspace(-2.5, np.log(150), 4000)
    p = np.exp(np.asarray(fn(g, z, th), np.float64)); p /= _trapz(p, g)
    m1m = _trapz(g * p, g); s_m = np.sqrt(_trapz((g - m1m) ** 2 * p, g))
    sk_m = _trapz((g - m1m) ** 3 * p, g) / s_m ** 3
    # flux conservation: <1/mu> in the image plane should be 1
    flux = float(_trapz(np.exp(-g) * p, g))
    return (m1s, m1m), (s_s, s_m), (sk_s, sk_m), flux


def main():
    ref = np.load(AR / "cache" / "param_space_ref.npz")
    counts, sizes = ref["counts"], ref["sizes"]
    fn = smooth_model.load_smooth_fn()

    print(f"{'panel':26s} {'KL_img[e,6]':>11} {'KL_src[e,6]':>11} {'KL_img[e,100]':>13} "
          f"{'mean(s/m)':>15} {'sig(s/m)':>13} {'skew(s/m)':>12} {'<1/mu>':>7}")
    kls6, kls6s, kls100 = [], [], []
    k = 0
    for cname, th in COSMOS:
        for z in ZS:
            cnt, N = counts[k], int(sizes[k]); k += 1
            kl6, b6 = panel_kl(fn, cnt, N, z, th, 6.0, False)
            kl6s, _ = panel_kl(fn, cnt, N, z, th, 6.0, True)
            kl100, _ = panel_kl(fn, cnt, N, z, th, 100.0, False)
            (m1s, m1m), (ss, sm), (sks, skm), flux = panel_moments(fn, cnt, N, z, th)
            kls6.append(kl6); kls6s.append(kl6s); kls100.append(kl100)
            print(f"{cname:14s} z={z:<4}   {kl6:11.4f} {kl6s:11.4f} {kl100:13.4f} "
                  f"{m1s:+7.4f}/{m1m:+7.4f} {ss:6.4f}/{sm:6.4f} {sks:+5.2f}/{skm:+5.2f} {flux:7.4f}")
    print(f"\nMEDIAN KL: image[edge,6] = {np.nanmedian(kls6):.4f}   "
          f"source[edge,6] = {np.nanmedian(kls6s):.4f}   image[edge,100] = {np.nanmedian(kls100):.4f}")
    print(f"(ACE-Lensing reports median KL = 0.007, source plane, mu in [0.1,6], z <= 6, "
          f"vs their own N-body test PDFs)")


if __name__ == "__main__":
    main()
