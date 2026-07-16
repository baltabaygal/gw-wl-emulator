"""Joint-field sizing: clustered weak background from the same LOS environment field.

Context (docs/bias_field_joint_framework.md, design note §9): in the excursion-set
reading of the bias-layer redesign, the weak background is the below-threshold halo
continuum, so the SAME realized field delta(chi) that modulates explicit-halo counts
must also source a clustered convergence component:

    kappa_W = kappa_W,shot (current sigma_W, kept)  +  sum_shells a_w[jz]*delta_jz

This script sizes the variance budget analytically (no MC, no gwlensing):

    S_ee = a_e' Cov a_e   count-modulation clustering (prototype's object)
    S_ww = a_w' Cov a_w   clustered weak background   (the NEW term)
    S_ew = a_e' Cov a_w   count-background covariance (what turboGL drops)

against the Poisson denominators Var_e = sum barN*k2bar and Var_W = sigma_W^2.

The weak amplitude a_w is computed DIRECTLY as the bias-weighted first moment of the
sub-threshold annuli (same loop as the validated sigmakappaW port, kappa^1 instead of
kappa^2; self-checked against sigma_W to machine precision), PLUS the sub-Mmin halo
continuum from the extended mass integral. The sum-rule subtraction route
a_w = Dg*(dktot*I_b - Web) is kept as a DIAGNOSTIC only: it fails at low z_s
(f_exp > 1) because the code's NFW halos are UNTRUNCATED - an explicit halo with
rmax >~ r200 carries more projected mass than its M200, so the mass budget cannot
close by subtraction. That overshoot is itself a reported finding.

Both correlation conventions are reported for the explicit arm:
  - pencil:   per-shell amplitude = Dg*Web with the raw segment covariance (upper);
  - windowed: per-cell cylinder-window sigma (new_model_cell_sigma), the prototype's
    convention (realistic).
The weak arm always uses the pencil segment covariance (sub-threshold encounters are
transversely compact vs P_1D scales).

Machinery imported from scripts/convergence/bias_field_prototype.py (validated vs the
C++ helpers to <=5e-9). Run with the test env python; no build/ needed:

  /Users/baltabay/miniforge3/envs/test/bin/python \
      scripts/convergence/bias_field_joint.py all

Outputs: data/results/bias_field_joint/{sizing.npz, report.md}.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bias_field_prototype import (Cosmo, LOSField, new_model_cell_sigma,
                                  DELTAC0, CLIGHT, PI)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "results" / "bias_field_joint"
VARK_REF = REPO / "data" / "results" / "vark_nz" / "vark_vs_nz.npz"

ZS_LIST = [0.2, 1.0, 5.0, 10.0]
MLO_EXT = 1e-3          # extended mass integral lower edge (sum-rule closure)
NM_EXT = 400
MLO_REPORT = [1e-3, 1e0, 1e3]   # closure trend cuts


def halobias_q(C, z, sigma, q, p=0.3):
    """SMT01-family bias with free q (q=0.75 = the code's, q=0.8 = PBS-consistent
    with pFC)."""
    qnu2 = q * (C.deltac(z) / sigma) ** 2
    return 1.0 + (qnu2 - 1.0) / DELTAC0 + 2.0 * p / (DELTAC0 * (1.0 + qnu2 ** p))


def extended_mass_integrals(C):
    """I_m, I_b(q) on C.zlist over the extended grid Mlo..1e17, plus the
    sub-Mmin (M < C.Mmin) bias-weighted piece used for the weak-arm extension.

    Returns dict with (Nz,) arrays; integrands cached on the extended M grid."""
    Mext = np.exp(np.linspace(np.log(MLO_EXT), np.log(1e17), NM_EXT))
    dlogMext = (np.log(1e17) - np.log(MLO_EXT)) / (NM_EXT - 1)
    sig = np.empty(NM_EXT)
    dsig = np.empty(NM_EXT)
    for j, M in enumerate(Mext):
        sig[j], dsig[j] = C._sigma_smoothk(M, C.deltaH8)
    z = C.zlist[:, None]
    s = sig[None, :]
    # dn/dlnM > 0 (ds/dM < 0), same recipe as Cosmo._build_hmf_bias_nfw
    dn = -C.rhoM0 * Cosmo._pFC(C.deltac(z), s ** 2) * 2.0 * s * dsig[None, :]
    b075 = halobias_q(C, z, s, 0.75)
    b080 = halobias_q(C, z, s, 0.80)
    w = Mext[None, :] * dn * dlogMext / C.rhoM0        # mass-fraction integrand
    out = dict(Mext=Mext, sig_ext=sig)
    for mlo in MLO_REPORT:
        m = Mext >= mlo
        tag = f"{mlo:.0e}"
        out[f"I_m_{tag}"] = w[:, m].sum(axis=1)
        out[f"I_b075_{tag}"] = (w * b075)[:, m].sum(axis=1)
        out[f"I_b080_{tag}"] = (w * b080)[:, m].sum(axis=1)
    low = Mext < C.Mmin                                # sub-Mmin continuum
    out["I_b075_low"] = (w * b075)[:, low].sum(axis=1)
    out["I_m_low"] = w[:, low].sum(axis=1)
    # full-range (Mlo=MLO_EXT) aliases used by the subtraction diagnostic
    out["I_m"] = out[f"I_m_{MLO_EXT:.0e}"]
    out["I_b075"] = out[f"I_b075_{MLO_EXT:.0e}"]
    out["I_b080"] = out[f"I_b080_{MLO_EXT:.0e}"]
    return out


def weak_first_moments(C, zs, kappathr, eps_floor=0.001):
    """Per-cell first and second moments of the SUB-threshold annuli — the same
    log-annulus loop as Cosmo.sigmakappaW (validated port), accumulating kappa^1
    per (jz,jM) alongside kappa^2 as a self-check.

    Returns (m1, mr2, m2_total) with
      m1[jz,jM]  = contribution of cell (jz,jM) to the mean weak convergence rate,
      mr2[jz,jM] = same, weighted by r^2 (PHYSICAL) -> kappa-weighted RMS beam
                   radius R_w = sqrt(mr2/m1) per cell,
      m2_total   = sum pref r^2 k^2 == sigmakappaW(zs,kappathr)^2 (assert outside)."""
    rmax, kappa0 = C.rmax_grid(zs, kappathr)
    zl = C.zlist
    dz = np.concatenate([[0.0], np.diff(zl)])
    mask = np.broadcast_to(zl[:, None] < zs, (C.Nz, C.NM)).copy()
    mask[0, :] = False
    mask[:, 0] = False
    r = np.where(rmax > 0.0, rmax, 1.0e-6)
    pref = (CLIGHT * 2.0 * PI * (1 + zl[:, None]) ** 2
            / C.Hz(zl)[:, None] * C.HMF0 * 0.01 * C.dlogM * dz[:, None])
    m1 = np.zeros((C.Nz, C.NM))
    mr2 = np.zeros((C.Nz, C.NM))
    m2 = 0.0
    alive = mask.copy()
    Edlnr = np.exp(0.01)
    while alive.any():
        k = 2.0 * kappa0 * C.Fg0(r / C.rs)
        c1 = pref * r ** 2 * k
        m1[alive] += c1[alive]
        mr2[alive] += (c1 * r ** 2)[alive]
        m2 += np.sum((c1 * k)[alive])
        r = np.where(alive, r * Edlnr, r)
        alive &= k > eps_floor * kappathr
    return m1, mr2, m2


def stage_sizing(zs_list):
    t0 = time.time()
    C = Cosmo(Nz=100)
    fld = LOSField(C)

    print("extended mass integrals (sum-rule closure + sub-Mmin piece)...",
          flush=True)
    ext = extended_mass_integrals(C)
    save = {k: v for k, v in ext.items()}
    save["zlist"] = C.zlist
    for zt in ZS_LIST:
        i = int(np.argmin(np.abs(C.zlist - zt)))
        row = " ".join(f"I_b075({m:.0e})={ext[f'I_b075_{m:.0e}'][i]:.4f}"
                       for m in MLO_REPORT)
        print(f"[closure] z={C.zlist[i]:.3f}: I_m(1e-3)={ext['I_m'][i]:.4f} "
              f"{row} I_b080(1e-3)={ext['I_b080'][i]:.4f}", flush=True)

    for zs in zs_list:
        print(f"--- zs = {zs:g}", flush=True)
        kt = C.find_kappathr(zs, 100)
        T = C.cell_tables(zs, kt)

        jz_arr = np.where((C.zlist < zs) & (np.arange(C.Nz) >= 1))[0]
        zl = C.zlist[jz_arr]
        dz = C.zlist[jz_arr] - C.zlist[jz_arr - 1]
        dchi = CLIGHT * dz / C.Hz(zl)
        dktot = dchi * C.rhoM0 * (1.0 + zl) ** 2 / C.Sigmacf(zs, zl)
        Dg = C.Dg(zl)

        # explicit arm
        w = T["barN"] * T["kbar"]
        b = C.biaslist[T["jz"], T["jM"]]
        We = np.bincount(T["jz"], weights=w, minlength=C.Nz)[jz_arr]
        Web = np.bincount(T["jz"], weights=w * b, minlength=C.Nz)[jz_arr]
        a_e = Dg * Web                                    # pencil convention
        sig_win = new_model_cell_sigma(fld, C, T)         # per-cell windowed std
        s_e_win = np.bincount(T["jz"], weights=w * sig_win,
                              minlength=C.Nz)[jz_arr]

        # weak arm, DIRECT: bias-weighted first moment of sub-threshold annuli
        m1, mr2, m2 = weak_first_moments(C, zs, kt)
        sigW = C.sigmakappaW(zs, kt)
        m2dev = abs(m2 - sigW ** 2) / sigW ** 2
        assert m2dev < 1e-9, f"annulus-loop self-check failed: {m2dev:.2e}"
        m1_shell = m1.sum(axis=1)[jz_arr]
        m1b_shell = (m1 * C.biaslist).sum(axis=1)[jz_arr]
        ext_low = dktot * np.interp(zl, C.zlist, ext["I_b075_low"])
        a_w = Dg * (m1b_shell + ext_low)
        assert np.all(a_w >= 0.0)

        # weak arm, WINDOWED: per-cell disc window at the kappa-weighted RMS
        # beam radius (comoving), R_L(M) floor, same cell_sigma2 machinery as
        # the explicit arm. The pencil a_w keeps ALL transversely-aliased 3D
        # power; sub-threshold annuli extend to ~Mpc, so this is first-order.
        cj, cM = np.nonzero(m1 > 1e-10 * m1.sum())
        keep = np.isin(cj, jz_arr)
        cj, cM = cj[keep], cM[keep]
        Rw = np.sqrt(mr2[cj, cM] / m1[cj, cM]) * (1 + C.zlist[cj])  # comoving
        RL = (3.0 * C.Mlist[cM] / (4.0 * PI * C.rhoM0)) ** (1.0 / 3.0)
        chi_all = C.dc(C.zlist)
        Lseg_c = chi_all[cj] - chi_all[cj - 1]
        sig_w_cell = np.empty(len(cj))
        for s0 in range(0, len(cj), 2000):
            s1 = min(s0 + 2000, len(cj))
            sig_w_cell[s0:s1] = np.sqrt(fld.cell_sigma2(
                Rw[s0:s1], Lseg_c[s0:s1], RL[s0:s1]))
        amp_w = (m1[cj, cM] * C.biaslist[cj, cM] * C.Dg(C.zlist[cj])
                 * sig_w_cell)
        s_w_win = np.bincount(cj, weights=amp_w, minlength=C.Nz)[jz_arr]
        # sub-Mmin continuum: pencil-diag amplitude (small; upper for its share)

        # weak arm, SUBTRACTION (diagnostic of the untruncated-NFW overshoot)
        a_w_sub_raw = Dg * (dktot * np.interp(zl, C.zlist, ext["I_b075"]) - Web)
        overshoot = -a_w_sub_raw[a_w_sub_raw < 0].sum()
        f_exp = We.sum() / dktot.sum()

        # covariance and budget
        corr, Cov = fld.segment_corr(C.dc(C.zlist[jz_arr - 1]),
                                     C.dc(C.zlist[jz_arr]))
        sqd = np.sqrt(np.diag(Cov))
        s_w_win = s_w_win + Dg * ext_low * sqd            # sub-Mmin (pencil) part
        s_w = a_w * sqd                                   # pencil weak amp
        S_ee = a_e @ Cov @ a_e                            # pencil brackets
        S_ww = a_w @ Cov @ a_w
        S_ew = a_e @ Cov @ a_w
        S_ee_win = s_e_win @ corr @ s_e_win               # windowed (realistic)
        S_ww_win = s_w_win @ corr @ s_w_win
        S_ew_win = s_e_win @ corr @ s_w_win
        S_ww_chk = s_w @ corr @ s_w
        assert abs(S_ww_chk - S_ww) / S_ww < 1e-8
        S_tot = S_ee + 2 * S_ew + S_ww
        assert abs(S_tot - (a_e + a_w) @ Cov @ (a_e + a_w)) / S_tot < 1e-10
        S_tot_win = S_ee_win + 2 * S_ew_win + S_ww_win

        Var_e = np.sum(T["barN"] * T["k2bar"])
        Var_w = sigW ** 2
        Var_tot = Var_e + Var_w

        print(f"[budget] zs={zs:g} kt={kt:.3e} <N>={T['barN'].sum():.1f} "
              f"f_exp={f_exp:.3f} overshoot={overshoot:.2e} "
              f"sum(m1)={m1_shell.sum():.4f} sum(m1b)={m1b_shell.sum():.4f} "
              f"sum(ext_low)={ext_low.sum():.4f}", flush=True)
        print(f"[budget] pencil: S_ee={S_ee:.3e} S_ww={S_ww:.3e} "
              f"2S_ew={2*S_ew:.3e} | windowed: S_ee={S_ee_win:.3e} "
              f"S_ww={S_ww_win:.3e} 2S_ew={2*S_ew_win:.3e} | Var_e={Var_e:.3e} "
              f"Var_w={Var_w:.3e}", flush=True)

        p = f"z{zs:g}_"
        save.update({
            p + "kt": kt, p + "Nhalos": T["barN"].sum(),
            p + "a_e": a_e, p + "a_w": a_w, p + "s_e_win": s_e_win,
            p + "s_w_win": s_w_win,
            p + "dktot": dktot, p + "We": We, p + "Web": Web,
            p + "m1_shell": m1_shell, p + "m1b_shell": m1b_shell,
            p + "ext_low": ext_low, p + "zl": zl,
            p + "f_exp": f_exp, p + "overshoot": overshoot,
            p + "a_w_sub_raw": a_w_sub_raw,
            p + "S_ee": S_ee, p + "S_ww": S_ww, p + "S_ew": S_ew,
            p + "S_ee_win": S_ee_win, p + "S_ew_win": S_ew_win,
            p + "S_ww_win": S_ww_win, p + "S_tot_win": S_tot_win,
            p + "S_tot": S_tot, p + "Var_e_pois": Var_e,
            p + "Var_w_pois": Var_w, p + "Var_tot": Var_tot,
            p + "sigW": sigW,
        })

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "sizing.npz", **save)
    print(f"sizing done in {time.time()-t0:.0f}s -> {OUT/'sizing.npz'}",
          flush=True)


def stage_report():
    d = np.load(OUT / "sizing.npz")
    zs_avail = sorted({float(k.split("_")[0][1:]) for k in d.files
                       if k.endswith("_S_ee")})
    # measured full-C++ body variance reference (Nz=100), if present
    ref = {}
    if VARK_REF.exists():
        vr = np.load(VARK_REF, allow_pickle=True)
        i100 = list(vr["nz_grid"]).index(100)
        for zs in zs_avail:
            key = f"K2fix0.5_z{zs:g}"
            if key in vr.files:
                ref[zs] = vr[key][i100].mean()

    L = ["# Joint-field sizing: clustered weak background\n",
         "Weak arm a_w = Dg * (bias-weighted first moment of sub-threshold "
         "annuli + sub-Mmin continuum), computed DIRECTLY (the sum-rule "
         "subtraction is diagnostic-only: untruncated-NFW overshoot). Two "
         "correlation conventions bracket the transverse window: PENCIL "
         "(R=0.5 kpc, keeps all aliased 3D power - hard upper bound) and "
         "WINDOWED (explicit: counting cylinder; weak: disc at the kappa-"
         "weighted RMS beam radius, R_L(M) floor - realistic). Denominator "
         "Var_tot = Sigma barN k2bar + sigma_W^2 (analytic, unclipped); "
         "meas.body is the full-C++ CLIPPED Var(|kappa|<0.5) at Nz=100 "
         "(definitions differ at high zs where the tail dominates).\n",
         "## 1. Variance budget\n",
         "| zs | k_thr | sigma_W | Var_e_pois | Var_w_pois | S_ee pen/win | "
         "S_ww pen/win | 2S_ew pen/win | rho_ew | Var_tot | meas.body |",
         "|--:|--:|--:|--:|--:|:--|:--|:--|--:|--:|--:|"]
    for zs in zs_avail:
        p = f"z{zs:g}_"
        g = lambda k: float(d[p + k])
        rho = g("S_ew_win") / np.sqrt(g("S_ee_win") * g("S_ww_win"))
        L.append(f"| {zs:g} | {g('kt'):.3e} | {g('sigW'):.4f} | "
                 f"{g('Var_e_pois'):.3e} | {g('Var_w_pois'):.3e} | "
                 f"{g('S_ee'):.2e} / {g('S_ee_win'):.2e} | "
                 f"{g('S_ww'):.2e} / {g('S_ww_win'):.2e} | "
                 f"{2*g('S_ew'):.2e} / {2*g('S_ew_win'):.2e} | {rho:.3f} | "
                 f"{g('Var_tot'):.3e} | "
                 + (f"{ref[zs]:.3e} |" if zs in ref else "- |"))
    L += ["\n## 2. Headline ratios (windowed = realistic; pencil in parens "
          "= upper bracket)\n",
          "| zs | S_ww/Var_w_pois | S_ww/Var_tot | 2S_ew/Var_tot | "
          "(S_ww+2S_ew)/Var_tot | S_ee/Var_tot | new/meas.body |",
          "|--:|:--|:--|:--|:--|:--|:--|"]
    for zs in zs_avail:
        p = f"z{zs:g}_"
        g = lambda k: float(d[p + k])
        new_w = g("S_ww_win") + 2 * g("S_ew_win")
        new_p = g("S_ww") + 2 * g("S_ew")
        L.append(f"| {zs:g} | {g('S_ww_win')/g('Var_w_pois'):.3f} "
                 f"({g('S_ww')/g('Var_w_pois'):.1f}) | "
                 f"{g('S_ww_win')/g('Var_tot'):.2e} "
                 f"({g('S_ww')/g('Var_tot'):.2e}) | "
                 f"{2*g('S_ew_win')/g('Var_tot'):.2e} "
                 f"({2*g('S_ew')/g('Var_tot'):.2e}) | "
                 f"{new_w/g('Var_tot'):.2e} ({new_p/g('Var_tot'):.2e}) | "
                 f"{g('S_ee_win')/g('Var_tot'):.4f} "
                 f"({g('S_ee')/g('Var_tot'):.4f}) | "
                 + (f"{new_w/ref[zs]:.2e} |" if zs in ref else "- |"))
    L += ["\n## 3. Mass bookkeeping\n",
          "sum-rule closure: I_b(Mlo) -> 1 as Mlo -> 0 for a PBS-consistent "
          "(HMF, b) pair. f_exp = kappa-weighted explicit capture fraction; "
          "> 1 exposes the untruncated-NFW overshoot (projected mass within "
          "rmax exceeds M200 when rmax >~ r200), which is why the subtraction "
          "route for a_w is diagnostic-only. overshoot = summed negative part "
          "of the subtraction a_w.\n",
          "| z | I_m(1e-3) | I_b075: 1e3 / 1e0 / 1e-3 | I_b080(1e-3) | "
          "I_b075_low(<Mmin) | f_exp | overshoot |",
          "|--:|--:|:--|--:|--:|--:|--:|"]
    zlist = d["zlist"]
    for zs in zs_avail:
        i = int(np.argmin(np.abs(zlist - zs)))
        p = f"z{zs:g}_"
        tr = " / ".join(f"{d[f'I_b075_{m:.0e}'][i]:.4f}" for m in MLO_REPORT[::-1])
        L.append(f"| {zs:g} | {d['I_m'][i]:.4f} | {tr} | "
                 f"{d['I_b080'][i]:.4f} | {d['I_b075_low'][i]:.4f} | "
                 f"{float(d[p+'f_exp']):.3f} | {float(d[p+'overshoot']):.2e} |")
    (OUT / "report.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT / "report.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sizing", "report", "all"])
    ap.add_argument("--zs", type=float, default=None)
    a = ap.parse_args()
    zs_run = ZS_LIST if a.zs is None else [a.zs]
    if a.stage in ("sizing", "all"):
        stage_sizing(zs_run)
    if a.stage in ("report", "all"):
        stage_report()
