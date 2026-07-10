"""
Stratified Monte-Carlo test of the Wsub partition: sigma_sub vs subhalo_factor.

The per-host analog of plots/sigma_partition_panels.pdf (field level, sigma vs
kappa_thr): generate random subhalo populations for host encounters, split the
clumps at each subhalo_factor into resolved (kept discretely, MC) and unresolved
(replaced by the analytic mu_unres(y) + N(0, sigma_unres^2(y)) term), and check
that   strong (MC)  (+)  weak (analytic)   sits flat on the brute plateau.

Estimator: doubly stratified.
(i) The encounter impact parameter y is stratified (deterministic log-y strata
    with exact area weights 2y dy / rmax^2): the host profile given y is
    deterministic, and Monte-Carloing y only injects the host's heavy small-y
    tail into the noise (measured S/N ~ 0.1).
(ii) The clump MASS axis is stratified into log-psi bands. Disjoint mass bands
    of a Poisson process are independent, so per-y means and variances add
    across bands EXACTLY; each band gets its own replica count, boosting the
    rare top-mass bands (Nbar ~ 0.05/host) to ~1e5 replicas at negligible cost.
    Without this, Var(S) at small y is dominated by rare near-ray passes of the
    few most massive clumps and a plain population MC (incl. the C++ brute
    mode) underestimates it at any affordable size (measured -15% at
    y = 0.15 rmax with 2e4 populations).
Within a band the randomness is honest MC: Poisson counts with the exact JvdB14
intensity (power-law proposal thinned by exp(-beta psi^omega), rejects dropped,
as in the C++), anti-biased 3D positions isotropically projected. Gate: clump
resolved iff kappa_m(y) >= factor*kappa_thr (identical to reach(m) >= y by
monotonicity); the same continuous rule feeds the MC masks and the analytic
keep tables.

Schemes (per stratum y; D = deterministic part, S = MC clump sum, v+ = added
Gaussian variance):
  ns        D = host(M)                                     [paired baseline]
  brute     D = host((1-f_b)M),           S = S_all
  strong    D = host((1-f_b)M) + mu_U(y), S = S_res(f)
  total_new = strong with v += sigma_U^2(y)                 [proposed scheme]
  old       D = host((1-f_res(f,y))M),    S = S_res(f)      [current production]
  old_zm    = old with v += sigma_U^2(y)                    [zero-mean-only fix]

Var(kappa_X) = sum_y a_y [ v_X(y) + g_X(y)^2 ] - (sum_y a_y g_X(y))^2,
g = D + mean_MC(S), v = var_MC(S) (both summed over mass bands; squares
bias-corrected); excess = Var - Var_ns. Analytic counterparts of every scheme
from the same Campbell tables are overlaid; errors from K disjoint replica
groups (mean +/- std/sqrt(K)).

Run:  /Users/baltabay/miniforge3/envs/test/bin/python playground/analytic/wsub_mc_partition.py
      [--quick]
Writes playground/analytic/wsub_mc_partition.json and
plots/wsub_partition_panels_mc.{png,pdf}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/Users/baltabay/Desktop/gw-wl-emulator")
sys.path.insert(0, str(ROOT))

from scripts.subhalo_gate.subhalo_factor_proxy_check import (  # noqa: E402
    ALPHA, BETA, OMEGA, PSI_MAX,
    fg_kappa, gamma_norm, n_tau, nfw_params, sigma_crit,
)
from playground.dgate.subhalo_factor_dgate_area_scan import host_rmax  # noqa: E402
from playground.analytic.subhalo_factor_analytic_deficit import (  # noqa: E402
    distance_kernel, projected_profile,
)

INK, MUTED, BASE, GRID = "#0b0b0b", "#898781", "#c3c2b7", "#e1e0d9"
BLUE, GREEN, RED = "#2a78d6", "#2e9e62", "#d64545"
ORANGE, PURPLE = "#d97706", "#7c3aed"


def host_kappa(mass_eff, z_lens, sigmac, y):
    rs, rhos, _, _ = nfw_params(mass_eff, z_lens)
    return 2.0 * (rs * rhos / sigmac) * fg_kappa(np.maximum(np.asarray(y) / rs, 1e-12))


# ----------------------------------------------------------------- analytic side
def analytic_tables(host_mass, z_lens, z_source, kappa_thr_host, m_floor, factors,
                    n_mass=100, n_y=200, n_d=240, n_theta=256):
    """Campbell tables on the y strata: brute mu/C2, per-factor unresolved mu_U /
    sigma2_U and resolved fraction f_res; host profiles; strata weights a_y."""
    rs_h, rhos_h, c_host, r200_h = nfw_params(host_mass, z_lens)
    rmax = host_rmax(host_mass, z_lens, z_source, kappa_thr_host)
    nt, _ = n_tau(host_mass, z_lens)
    fs = 0.3563 / nt**0.6 - 0.075
    gamma = gamma_norm(fs)
    sigmac = sigma_crit(z_source, z_lens)

    psi_floor = m_floor / host_mass
    lpsi = np.linspace(np.log(psi_floor), np.log(PSI_MAX), n_mass)
    psi = np.exp(lpsi)
    mass = psi * host_mass
    dN = gamma * psi**ALPHA * np.exp(-BETA * psi**OMEGA)      # dN/dlnpsi

    d_grid = np.logspace(-3, np.log10(rmax + r200_h), n_d)
    rs_c, rhos_c, _, _ = nfw_params(mass, z_lens)             # vectorized
    kappa0_c = rs_c * rhos_c / sigmac
    k1 = 2.0 * kappa0_c[:, None] * fg_kappa(np.maximum(d_grid[None, :] / rs_c[:, None], 1e-12))
    k2 = k1**2

    sigma_interp, _ = projected_profile(c_host, r200_h)
    y_grid = np.logspace(np.log10(0.05), np.log10(rmax), n_y)
    f_dy = distance_kernel(y_grid, d_grid, sigma_interp, n_theta)
    fd = f_dy * (d_grid * np.gradient(np.log(d_grid)))[None, :]
    J1 = np.einsum("yd,md->my", fd, k1, optimize=True)
    J2 = np.einsum("yd,md->my", fd, k2, optimize=True)

    # exact strata weights, normalized on [y_grid[0], rmax]
    a_y = 2.0 * y_grid**2 * np.gradient(np.log(y_grid)) / rmax**2
    a_y /= a_y.sum()

    # clump kappa at HOST-CENTER distance y (gate variable), (n_mass, n_y)
    k_at_y = 2.0 * kappa0_c[:, None] * fg_kappa(np.maximum(y_grid[None, :] / rs_c[:, None], 1e-12))

    mu_b = np.trapezoid(dN[:, None] * J1, lpsi, axis=0)
    C2_b = np.trapezoid(dN[:, None] * J2, lpsi, axis=0)
    f_b = float(np.trapezoid(dN * psi, lpsi))

    T = {"y_grid": y_grid, "a_y": a_y, "rmax": rmax, "r200": r200_h,
         "c_host": c_host, "sigmac": sigmac, "fs": fs, "gamma": gamma,
         "f_b": f_b, "mu_b": mu_b, "C2_b": C2_b,
         "k_ns": host_kappa(host_mass, z_lens, sigmac, y_grid),
         "k_red_b": host_kappa(max((1.0 - f_b) * host_mass, 1.0), z_lens, sigmac, y_grid),
         "mu_U": {}, "s2_U": {}, "k_red_res": {}}
    for f in factors:
        unres = k_at_y < f * kappa_thr_host                   # (n_mass, n_y)
        T["mu_U"][f] = np.trapezoid(dN[:, None] * J1 * unres, lpsi, axis=0)
        T["s2_U"][f] = np.trapezoid(dN[:, None] * J2 * unres, lpsi, axis=0)
        f_res = np.minimum(np.trapezoid((dN * psi)[:, None] * ~unres, lpsi, axis=0), 0.95)
        T["k_red_res"][f] = np.array([
            host_kappa(max((1.0 - fr) * host_mass, 1.0), z_lens, sigmac, np.array([yy]))[0]
            for fr, yy in zip(f_res, y_grid)])
    return T


def analytic_excess(T, factors, sel=None):
    """Exact per-scheme excesses from the tables (no MC). `sel` restricts to a
    subset of y strata (renormalized) to match the MC's thinned grid."""
    if sel is None:
        sel = np.arange(len(T["y_grid"]))
    T = dict(T, a_y=T["a_y"][sel] / T["a_y"][sel].sum(),
             k_ns=T["k_ns"][sel], k_red_b=T["k_red_b"][sel],
             mu_b=T["mu_b"][sel], C2_b=T["C2_b"][sel],
             mu_U={f: T["mu_U"][f][sel] for f in T["mu_U"]},
             s2_U={f: T["s2_U"][f][sel] for f in T["s2_U"]},
             k_red_res={f: T["k_red_res"][f][sel] for f in T["k_red_res"]})
    a, kns = T["a_y"], T["k_ns"]
    var_ns = float(np.sum(a * kns**2) - np.sum(a * kns) ** 2)

    def var_of(g, v):
        return float(np.sum(a * (v + g**2)) - np.sum(a * g) ** 2)

    g_b = T["k_red_b"] + T["mu_b"]
    out = {"brute": var_of(g_b, T["C2_b"]) - var_ns, "rows": []}
    for f in factors:
        s2U, muU = T["s2_U"][f], T["mu_U"][f]
        C2_res, mu_res = T["C2_b"] - s2U, T["mu_b"] - muU
        weak = float(np.sum(a * s2U))
        strong = var_of(g_b, C2_res) - var_ns                  # host_b + S_res + mu_U
        old = var_of(T["k_red_res"][f] + mu_res, C2_res) - var_ns
        out["rows"].append({"factor": float(f), "strong": strong, "weak": weak,
                            "total": strong + weak, "old": old,
                            "old_zm": old + weak})
    return out


# ----------------------------------------------------------------- MC side
def run_mc(T, host_mass, z_lens, kappa_thr_host, m_floor, factors, seed,
           n_ms=6, draw_budget=5.0e5, n_rep_min=500, n_rep_max=200_000,
           thin_y=2, k_groups=10):
    """Doubly stratified clump MC (y strata x log-psi mass bands).

    Per (y, band): n_rep independent Poisson populations of that band only,
    n_rep = clip(draw_budget / Nbar_band, n_rep_min, n_rep_max) — rare heavy
    bands get many replicas cheaply. Means/variances add across bands (disjoint
    Poisson). Returns per-scheme excesses (mean over K groups, std/sqrt(K)).
    """
    factors = np.asarray(factors)
    nf, K = len(factors), k_groups
    sel = np.arange(0, len(T["y_grid"]), thin_y)
    y_sel = T["y_grid"][sel]
    a_y = T["a_y"][sel] / T["a_y"][sel].sum()
    n_y = len(y_sel)
    sigmac = T["sigmac"]
    thr = factors * kappa_thr_host

    # mass bands (equal log-psi)
    edges = np.exp(np.linspace(np.log(m_floor / host_mass), np.log(PSI_MAX), n_ms + 1))
    nbar_b = T["gamma"] / ALPHA * (edges[1:]**ALPHA - edges[:-1]**ALPHA)
    n_rep_b = np.clip((draw_budget / np.maximum(nbar_b, 0.5)).astype(int),
                      n_rep_min, n_rep_max)
    n_rep_b = (n_rep_b // K) * K

    # 3D anti-biased radial inverse CDF (same profile as the C++ / validated python)
    xg = np.linspace(1e-9, 1.0, 5000)
    w3 = xg**2 / (1.0 + T["c_host"] * xg)**2 / np.sqrt((xg / 0.54)**-2.5 + 1.0)
    cdf3 = np.concatenate([[0.0], np.cumsum(0.5 * (w3[1:] + w3[:-1]) * np.diff(xg))])
    cdf3 /= cdf3[-1]

    rng = np.random.default_rng(seed)
    # accumulators over bands, per K-group: mean, var, and var/n (bias) per y
    m_all = np.zeros((K, n_y)); v_all = np.zeros((K, n_y)); b_all = np.zeros((K, n_y))
    m_res = np.zeros((nf, K, n_y)); v_res = np.zeros((nf, K, n_y)); b_res = np.zeros((nf, K, n_y))

    for ib in range(n_ms):
        n_rep = int(n_rep_b[ib])
        ng = n_rep // K
        pa_lo, pa_hi = edges[ib]**ALPHA, edges[ib + 1]**ALPHA
        for iy, y in enumerate(y_sel):
            counts = rng.poisson(nbar_b[ib], n_rep)
            tot = int(counts.sum())
            rid = np.repeat(np.arange(n_rep), counts)
            u = rng.uniform(0.0, 1.0, tot)
            psi = (pa_lo + u * (pa_hi - pa_lo)) ** (1.0 / ALPHA)
            keep = rng.uniform(0.0, 1.0, tot) <= np.exp(-BETA * psi**OMEGA)  # thin: DROP
            psi, rid = psi[keep], rid[keep]
            m = psi * host_mass

            r3d = np.interp(rng.uniform(0.0, 1.0, m.size), cdf3, xg) * T["r200"]
            cth = rng.uniform(-1.0, 1.0, m.size)
            az = rng.uniform(0.0, 2.0 * np.pi, m.size)
            R2d = r3d * np.sqrt(1.0 - cth**2)
            d = np.hypot(y - R2d * np.cos(az), R2d * np.sin(az))

            rs_c, rhos_c, _, _ = nfw_params(m, z_lens)
            kappa0 = rs_c * rhos_c / sigmac
            kap = 2.0 * kappa0 * fg_kappa(np.maximum(d / rs_c, 1e-12))
            kap_at_y = 2.0 * kappa0 * fg_kappa(np.maximum(y / rs_c, 1e-12))

            # resolved at factor f  <=>  kappa_m(y) >= f*kappa_thr
            j = np.searchsorted(thr, kap_at_y, side="right")
            bc = np.bincount(rid * (nf + 1) + j, weights=kap,
                             minlength=n_rep * (nf + 1)).reshape(n_rep, nf + 1)
            cum = np.cumsum(bc[:, ::-1], axis=1)[:, ::-1]      # suffix sums over j
            Sa = cum[:, 0].reshape(K, ng)                      # band S_all, per group
            m_all[:, iy] += Sa.mean(axis=1)
            vb = Sa.var(axis=1, ddof=1)
            v_all[:, iy] += vb
            b_all[:, iy] += vb / ng
            Sr = cum[:, 1:].reshape(K, ng, nf)                 # band S_res(f)
            m_res[:, :, iy] += Sr.mean(axis=1).T
            vr = Sr.var(axis=1, ddof=1)
            v_res[:, :, iy] += vr.T
            b_res[:, :, iy] += vr.T / ng

    kns = T["k_ns"][sel]
    var_ns = np.sum(a_y * kns**2) - np.sum(a_y * kns) ** 2

    def excess(D, mS, vS, bS, extra_v=0.0):
        """per-group excess for kappa = D(y) + S [+ N(0, extra_v(y))]; unbiased in
        the squared-mean terms (per-stratum and pooled)."""
        vals = []
        for k in range(K):
            gg = D + mS[k]
            g2 = gg**2 - bS[k]
            v = (np.sum(a_y * (vS[k] + extra_v + g2))
                 - (np.sum(a_y * gg) ** 2 - np.sum(a_y**2 * bS[k])))
            vals.append(v - var_ns)
        vals = np.asarray(vals)
        return float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(K))

    out = {"brute": excess(T["k_red_b"][sel], m_all, v_all, b_all),
           "sel": sel.tolist(), "n_rep_bands": n_rep_b.tolist(),
           "nbar_bands": nbar_b.tolist(), "rows": []}
    for i, f in enumerate(factors):
        s2U = T["s2_U"][f][sel]
        row = {"factor": float(f), "weak": float(np.sum(a_y * s2U))}
        row["strong"] = excess(T["k_red_b"][sel] + T["mu_U"][f][sel],
                               m_res[i], v_res[i], b_res[i])
        row["total"] = excess(T["k_red_b"][sel] + T["mu_U"][f][sel],
                              m_res[i], v_res[i], b_res[i], extra_v=s2U)
        row["old"] = excess(T["k_red_res"][f][sel], m_res[i], v_res[i], b_res[i])
        row["old_zm"] = excess(T["k_red_res"][f][sel], m_res[i], v_res[i], b_res[i],
                               extra_v=s2U)
        out["rows"].append(row)
    return out


def run_config(host_mass, z_lens, z_source, kappa_thr_host, m_floor, factors,
               seed, **mc_kw):
    T = analytic_tables(host_mass, z_lens, z_source, kappa_thr_host, m_floor, factors)
    mc = run_mc(T, host_mass, z_lens, kappa_thr_host, m_floor, factors, seed, **mc_kw)
    ana = analytic_excess(T, factors, sel=np.asarray(mc["sel"]))
    print(f"  analytic brute excess = {ana['brute']:.5g}   "
          f"MC = {mc['brute'][0]:.5g} ± {mc['brute'][1]:.2g}")
    for ra, rm in zip(ana["rows"], mc["rows"]):
        print(f"  f={ra['factor']:9.3g}  total(MC)={rm['total'][0]:.4g}±{rm['total'][1]:.2g}"
              f"  [ana {ra['total']:.4g}]  strong={rm['strong'][0]:.4g}"
              f"  weak={rm['weak']:.4g}  old_zm={rm['old_zm'][0]:.4g} [ana {ra['old_zm']:.4g}]")
    return {"host_mass": host_mass, "z_lens": z_lens, "m_floor": m_floor,
            "nbar": float(T["gamma"] / ALPHA * (PSI_MAX**ALPHA - (m_floor / host_mass)**ALPHA)),
            "f_b": T["f_b"], "rmax": T["rmax"], "analytic": ana, "mc": mc}


def make_proof_plot(results, z_source, out_stem):
    """Proof-style summary focused on one representative MC configuration."""
    if not results:
        return None

    target = min(
        results,
        key=lambda r: (abs(np.log10(r["host_mass"]) - 13.0), abs(r["z_lens"] - 0.5)),
    )
    mc, ana = target["mc"], target["analytic"]
    f = np.array([row["factor"] for row in mc["rows"]], dtype=float)

    def mc_mean(key):
        return np.array([row[key][0] for row in mc["rows"]], dtype=float)

    def mc_err(key):
        return np.array([row[key][1] for row in mc["rows"]], dtype=float)

    def ana_val(key):
        return np.array([row[key] for row in ana["rows"]], dtype=float)

    brute_mc = float(mc["brute"][0])
    brute_mc_err = float(mc["brute"][1])
    brute_ana = float(ana["brute"])
    current = mc_mean("old")
    gauss = mc_mean("old_zm")
    meanfix = mc_mean("strong")
    both = mc_mean("total")

    def ratio_with_err(num, num_err):
        ratio = num / brute_mc
        frac = np.sqrt((num_err / np.maximum(np.abs(num), 1e-30)) ** 2 +
                       (brute_mc_err / max(abs(brute_mc), 1e-30)) ** 2)
        return ratio, np.abs(ratio) * frac

    r_current, e_current = ratio_with_err(current, mc_err("old"))
    r_gauss, e_gauss = ratio_with_err(gauss, mc_err("old_zm"))
    r_meanfix, e_meanfix = ratio_with_err(meanfix, mc_err("strong"))
    r_both, e_both = ratio_with_err(both, mc_err("total"))

    deficit = np.maximum(brute_ana - ana_val("old"), 1e-30)
    frac_mean = (ana_val("strong") - ana_val("old")) / deficit
    frac_gauss = (ana_val("total") - ana_val("strong")) / deficit
    frac_left = np.clip(1.0 - frac_mean - frac_gauss, 0.0, 1.0)

    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "text.color": INK,
        "axes.edgecolor": BASE,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.size": 10.5,
        "axes.titlesize": 11.5,
    })

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.6, 4.8))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.14, top=0.84, wspace=0.28)

    axA.axhline(1.0, color=MUTED, lw=1.4, ls="--", zorder=1)
    axA.plot(f, ana_val("old") / brute_ana, "-", color=BLUE, lw=2.0,
             label="current: resolved only")
    axA.plot(f, ana_val("old_zm") / brute_ana, "-", color=RED, lw=2.0,
             label=r"+ Gaussian $\sigma^2_{W,\rm sub}$ only")
    axA.plot(f, ana_val("strong") / brute_ana, "-", color=ORANGE, lw=2.0,
             label=r"+ mean profile $\mu_{\rm unres}(y)$ only")
    axA.plot(f, ana_val("total") / brute_ana, "-", color=GREEN, lw=2.0,
             label="both (MC total)")
    axA.errorbar(f, r_current, yerr=e_current, fmt="o", ms=4.0, color=BLUE,
                 lw=1.0, capsize=2.0, zorder=4)
    axA.errorbar(f, r_gauss, yerr=e_gauss, fmt="s", ms=4.0, color=RED,
                 lw=1.0, capsize=2.0, zorder=4)
    axA.errorbar(f, r_meanfix, yerr=e_meanfix, fmt="^", ms=4.4, color=ORANGE,
                 lw=1.0, capsize=2.0, zorder=4)
    axA.errorbar(f, r_both, yerr=e_both, fmt="D", ms=4.1, color=GREEN,
                 lw=1.0, capsize=2.0, zorder=5)
    axA.set_xscale("log")
    axA.set_ylim(0.08, 1.06)
    axA.set_xlabel("subhalo_factor")
    axA.set_ylabel(r"paired $\kappa^2$ excess / brute")
    axA.set_title("A — bookkeeping closure against brute", loc="left")
    axA.grid(color=GRID, lw=0.6, alpha=0.7, which="both")
    axA.set_axisbelow(True)
    axA.legend(frameon=False, fontsize=8.8, loc="lower left")

    width = np.diff(np.log10(f)).mean()
    axB.bar(f, frac_mean, width=10**(np.log10(f) + 0.5 * width) - 10**(np.log10(f) - 0.5 * width),
            color=ORANGE, alpha=0.8, label="share recovered by mean profile")
    axB.bar(f, frac_gauss, bottom=frac_mean,
            width=10**(np.log10(f) + 0.5 * width) - 10**(np.log10(f) - 0.5 * width),
            color=PURPLE, alpha=0.8, label=r"share recovered by Gaussian variance")
    axB.plot(f, frac_left, "o--", color=MUTED, ms=3.8, lw=1.2,
             label="residual after both (analytic)")
    axB.set_xscale("log")
    axB.set_ylim(0.0, 1.02)
    axB.set_xlabel("subhalo_factor")
    axB.set_ylabel("fraction of resolved-only deficit")
    axB.set_title("B — what fixes the variance deficit?", loc="left")
    axB.grid(color=GRID, lw=0.6, alpha=0.7, axis="y")
    axB.set_axisbelow(True)
    axB.legend(frameon=False, fontsize=8.8, loc="upper left")

    fig.suptitle(
        rf"Which unresolved-subhalo term closes the variance?   "
        rf"($M=10^{{{int(np.log10(target['host_mass']))}}},\ z_l={target['z_lens']:g},\ z_s={z_source:g}$)",
        x=0.08, ha="left", fontsize=12, color=INK,
    )

    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", dpi=200, facecolor="#fcfcfb")
    plt.close(fig)
    return target


def make_panel_plot(results, z_source):
    """Original multi-panel sweep figure from already-computed results."""
    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                         "axes.edgecolor": BASE, "font.size": 9.5})
    fig, axes = plt.subplots(1, len(results), figsize=(3.5 * len(results), 3.7))
    axes = np.atleast_1d(axes)
    sq = lambda v: np.sign(v) * np.sqrt(np.abs(v))             # signed sqrt for safety

    for ax, r in zip(axes, results):
        f = np.array([row["factor"] for row in r["mc"]["rows"]])
        mc, ana = r["mc"], r["analytic"]
        get = lambda key: np.array([row[key][0] for row in mc["rows"]])
        err = lambda key: np.array([row[key][1] for row in mc["rows"]])
        aget = lambda key: np.array([row[key] for row in ana["rows"]])

        plateau = sq(mc["brute"][0])
        serr = lambda key: err(key) / np.maximum(2.0 * np.abs(sq(get(key))), 1e-30)

        ax.axhspan(sq(mc["brute"][0] - mc["brute"][1]), sq(mc["brute"][0] + mc["brute"][1]),
                   color=MUTED, alpha=0.3, lw=0)
        ax.axhline(plateau, color=MUTED, ls=":", lw=1.0, zorder=1)
        ax.plot(f, sq(aget("weak")), "-", color=BLUE, lw=1.6, zorder=3,
                label=r"$\sigma_{W,\rm sub}$ (analytic)")
        ax.plot(f, sq(aget("strong")), "-", color=GREEN, lw=1.0, alpha=0.6, zorder=2)
        ax.errorbar(f, sq(get("strong")), yerr=serr("strong"), fmt="^", color=GREEN,
                    ms=3.8, lw=0.8, capsize=1.2, zorder=4,
                    label=r"$\sigma_{\rm strong}$ (MC)")
        ax.errorbar(f, sq(get("total")), yerr=serr("total"), fmt="o", color=INK,
                    ms=3.6, lw=0.8, capsize=1.2, zorder=5,
                    label=r"$\sigma_{\rm total}$ (MC)")
        ax.plot(f, sq(aget("old_zm")), "--", color=RED, lw=1.0, alpha=0.6, zorder=2)
        ax.errorbar(f, sq(get("old_zm")), yerr=serr("old_zm"), fmt="s", mfc="none",
                    color=RED, ms=3.6, lw=0.8, capsize=1.2, zorder=4,
                    label="old bookkeeping + Gaussian")
        ax.set_xscale("log")
        lo = min(0.0, 1.15 * sq(aget("old_zm")).min())
        ax.set_ylim(lo - 0.06 * abs(plateau), 1.30 * abs(plateau))
        ax.text(0.04, 0.95, rf"$M=10^{{{int(np.log10(r['host_mass']))}}},\ "
                            rf"z_l={r['z_lens']:g}$",
                transform=ax.transAxes, fontsize=10, va="top", fontweight="bold")
        ax.text(0.04, 0.84, rf"$\sigma_{{\rm brute}}={plateau:.4g}$",
                transform=ax.transAxes, fontsize=8.5, va="top", color=MUTED)
        ax.grid(color=GRID, lw=.5, which="both")
        ax.set_axisbelow(True)
        ax.set_xlabel("subhalo_factor")
    axes[0].set_ylabel(r"$\sigma_{\kappa,\rm sub}$ (per-host excess)")
    axes[0].legend(frameon=False, fontsize=7.5, loc="center left")
    fig.suptitle(rf"Per-host substructure variance partition, $z_s={z_source:g}$"
                 "  —  strong (resolved-clump MC) $\\oplus$ weak (analytic) vs brute",
                 fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "plots" / f"wsub_partition_panels_mc.{ext}", dpi=200,
                    facecolor="#fcfcfb")
    plt.close(fig)


def render_saved_results(payload):
    """Render figures from an existing JSON payload without rerunning MC."""
    results = payload["results"]
    z_source = payload["z_source"]
    make_panel_plot(results, z_source)
    proof_target = make_proof_plot(
        results,
        z_source,
        ROOT / "plots" / "wsub_partition_variance_proof",
    )
    if proof_target is not None:
        print("saved plots/wsub_partition_variance_proof.png/.pdf "
              f"(fiducial M={proof_target['host_mass']:.0e}, z_l={proof_target['z_lens']:g})")
    print("saved plots/wsub_partition_panels_mc.png/.pdf from cached JSON")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--plot-only", action="store_true",
                        help="Render plots from an existing JSON file without rerunning MC.")
    parser.add_argument("--json-in", type=Path,
                        default=ROOT / "playground" / "analytic" / "wsub_mc_partition.json")
    parser.add_argument("--z-source", type=float, default=1.0)
    parser.add_argument("--kappa-thr-host", type=float, default=1.276e-4)
    parser.add_argument("--m-floor", type=float, default=1.0e7)
    parser.add_argument("--seed", type=int, default=20260709)
    args = parser.parse_args()

    if args.plot_only:
        payload = json.loads(args.json_in.read_text())
        render_saved_results(payload)
        return

    factors = list(np.logspace(-5, 0, 11))
    configs = [  # (host_mass, z_lens, draw_budget / 1e3)
        (1.0e12, 0.5, 400), (1.0e13, 0.2, 200),
        (1.0e13, 0.5, 200), (1.0e14, 0.5, 60),
    ]
    if args.quick:
        configs = [(m, z, max(n // 10, 20)) for m, z, n in configs]

    results = []
    for k, (M, zl, draw_k) in enumerate(configs):
        draw_budget = 1.0e3 * draw_k
        print(f"config M={M:.0e} z_l={zl} draw_budget={draw_budget:.0f}")
        results.append(run_config(M, zl, args.z_source, args.kappa_thr_host,
                                  args.m_floor, factors, seed=args.seed + k,
                                  draw_budget=draw_budget))

    out_json = ROOT / "playground" / "analytic" / "wsub_mc_partition.json"
    out_json.write_text(json.dumps(
        {"z_source": args.z_source, "kappa_thr_host": args.kappa_thr_host,
         "m_floor": args.m_floor, "factors": factors, "results": results},
        indent=2, default=float))
    render_saved_results({"results": results, "z_source": args.z_source})
    print(f"saved plots/wsub_partition_panels_mc.png/.pdf and {out_json.name}")


if __name__ == "__main__":
    main()
