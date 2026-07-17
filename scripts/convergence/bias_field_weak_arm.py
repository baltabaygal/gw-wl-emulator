"""Weak-arm MC: PDF-level effect of the clustered weak background (joint field).

Context (docs/bias_field_joint_framework.md §7.3, memory bias-field-joint-sizing):
the analytic sizing showed the clustered weak background S_ww plus the
count-background covariance 2S_ew is 12-25% of Var(kappa) at zs<=1 (windowed
bracket). The shipped C++ bias_model=1 modulates ONLY explicit counts and
filaments; kappa_W stays an independent N(0, sigma_W) (lensing.cpp:573). This
script measures what the emulator actually cares about — the P(kappa)-level /
JSD effect — in the validated Python prototype BEFORE any C++ change:

    kappa_W = kappa_W,shot (sigma_W, kept)  +  sum_jz s_w[jz] * v_jz

with v the SAME realized unit-variance shell field that modulates the counts,
and s_w from the sizing's direct bookkeeping (bias-weighted first moment of the
sub-threshold annuli + sub-Mmin continuum; windowed = disc at the kappa-weighted
RMS beam radius with the R_L(M) floor, pencil = upper bracket).

Arms (base/weakw/weakp are PAIRED — identical random draws; the weak terms are
deterministic given the realized field, so differences are pure weak-arm effect):
  old    : legacy iid per-cell lognormal layer (bias_model=0 replica, context)
  base   : correlated-field counts only (= shipped bias_model=1)
  weakw  : base + windowed weak arm (realistic)
  weakp  : base + pencil weak arm (hard upper bracket)

Same deliberate prototype deviations as bias_field_prototype.py (kappa only,
pure Poisson counts, no filaments, clip-centered stats). Run with the test env
python; no build/ needed:

  /Users/baltabay/miniforge3/envs/test/bin/python \
      scripts/convergence/bias_field_weak_arm.py all

Outputs: data/results/bias_field_weak_arm/{mc/*.npz, report.md},
plots/bias_field_weak_arm.png. Seed namespace 9.7e8 (disjoint: vark 8.0-8.2e8,
prototype 9.0e8, rperp scan 9.5e8).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bias_field_prototype import (Cosmo, LOSField, new_model_cell_sigma,
                                  _localize_jz, CLIGHT, PI)
from bias_field_joint import weak_first_moments, extended_mass_integrals

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "results" / "bias_field_weak_arm"
MCDIR = OUT / "mc"
PLOTS = REPO / "plots"
SIZING = REPO / "data" / "results" / "bias_field_joint" / "sizing.npz"

ZS_LIST = [0.2, 1.0, 5.0]     # zs>=5 sizing ratios are indicative only; z<=1
                              # is where the GW events (and clean numbers) are
NREAL = 100_000               # per seed; two seeds per config -> 200k pooled
NSEEDS = 2
SEED0 = 970_000_000
CHUNK = 500
NBINS = 120


def weak_amp_tables(C, fld, zs, kt, ext):
    """Per-shell weak-arm amplitudes over ALL shells jz>=1 with zl<zs.

    Returns dict with jz_arr, s_w_win, s_w_pen, corr (over jz_arr), and the
    bookkeeping pieces for the record. Replicates bias_field_joint.stage_sizing
    (validated 2026-07-15) cell-for-cell."""
    jz_arr = np.where((C.zlist < zs) & (np.arange(C.Nz) >= 1))[0]
    zl = C.zlist[jz_arr]
    dz = C.zlist[jz_arr] - C.zlist[jz_arr - 1]
    dchi = CLIGHT * dz / C.Hz(zl)
    dktot = dchi * C.rhoM0 * (1.0 + zl) ** 2 / C.Sigmacf(zs, zl)
    Dg = C.Dg(zl)

    m1, mr2, m2 = weak_first_moments(C, zs, kt)
    sigW = C.sigmakappaW(zs, kt)
    m2dev = abs(m2 - sigW ** 2) / sigW ** 2
    assert m2dev < 1e-9, f"annulus-loop self-check failed: {m2dev:.2e}"

    ext_low = dktot * np.interp(zl, C.zlist, ext["I_b075_low"])
    m1b_shell = (m1 * C.biaslist).sum(axis=1)[jz_arr]
    a_w = Dg * (m1b_shell + ext_low)
    assert np.all(a_w >= 0.0)

    corr, Cov = fld.segment_corr(C.dc(C.zlist[jz_arr - 1]),
                                 C.dc(C.zlist[jz_arr]))
    sqd = np.sqrt(np.diag(Cov))
    s_w_pen = a_w * sqd

    # windowed: disc at the kappa-weighted RMS beam radius, R_L(M) floor
    cj, cM = np.nonzero(m1 > 1e-10 * m1.sum())
    keep = np.isin(cj, jz_arr)
    cj, cM = cj[keep], cM[keep]
    Rw = np.sqrt(mr2[cj, cM] / m1[cj, cM]) * (1 + C.zlist[cj])   # comoving
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
    s_w_win = (np.bincount(cj, weights=amp_w, minlength=C.Nz)[jz_arr]
               + Dg * ext_low * sqd)
    return dict(jz_arr=jz_arr, s_w_win=s_w_win, s_w_pen=s_w_pen, corr=corr,
                sigW=sigW, a_w=a_w)


def mc_path(zs, arm, iseed):
    return MCDIR / f"z{zs:g}_{arm}_s{iseed}.npz"


def run_new_arm(C, fld, zs, kt, W, seed, nreal):
    """One correlated-field MC: returns (kappa_counts_with_shot, wwin, wpen)."""
    T = C.cell_tables(zs, kt)
    sig = new_model_cell_sigma(fld, C, T)
    jz_arr = W["jz_arr"]
    jloc = np.searchsorted(jz_arr, T["jz"])
    assert np.all(jz_arr[jloc] == T["jz"])
    cholT = np.linalg.cholesky(
        W["corr"] + 1e-10 * np.eye(len(W["corr"]))).T
    barN, rmax, rs, k0 = T["barN"], T["rmax"], T["rs"], T["kappa0"]
    sigW, s_w_win, s_w_pen = W["sigW"], W["s_w_win"], W["s_w_pen"]
    nsh, nc = len(jz_arr), len(barN)
    halved = sig ** 2 / 2.0
    rng = np.random.default_rng(seed)
    kap = np.empty(nreal, np.float32)
    wwin = np.empty(nreal, np.float32)
    wpen = np.empty(nreal, np.float32)
    for s0 in range(0, nreal, CHUNK):
        s1 = min(s0 + CHUNK, nreal)
        nch = s1 - s0
        v = rng.standard_normal((nch, nsh)) @ cholT
        g = sig[None, :] * v[:, jloc]
        rate = np.exp(g - halved[None, :]) * barN[None, :]
        Ncnt = rng.poisson(rate)
        rr, cc = np.nonzero(Ncnt)
        reps = Ncnt[rr, cc]
        rid = np.repeat(rr, reps)
        cid = np.repeat(cc, reps)
        u = rng.random(len(cid))
        r = rmax[cid] * np.sqrt(u)
        kk = 2.0 * k0[cid] * C.Fg0(r / rs[cid])
        kap[s0:s1] = (np.bincount(rid, weights=kk, minlength=nch)
                      + sigW * rng.standard_normal(nch))
        wwin[s0:s1] = v @ s_w_win
        wpen[s0:s1] = v @ s_w_pen
    return kap, wwin, wpen


def run_old_arm(C, fld, zs, kt, seed, nreal):
    """Legacy iid per-cell lognormal replica (context arm)."""
    T = C.cell_tables(zs, kt)
    sigW = C.sigmakappaW(zs, kt)
    sig = T["sigma_old"]
    barN, rmax, rs, k0 = T["barN"], T["rmax"], T["rs"], T["kappa0"]
    nc = len(barN)
    halved = sig ** 2 / 2.0
    rng = np.random.default_rng(seed)
    kap = np.empty(nreal, np.float32)
    for s0 in range(0, nreal, CHUNK):
        s1 = min(s0 + CHUNK, nreal)
        nch = s1 - s0
        g = sig[None, :] * rng.standard_normal((nch, nc))
        rate = np.exp(g - halved[None, :]) * barN[None, :]
        Ncnt = rng.poisson(rate)
        rr, cc = np.nonzero(Ncnt)
        reps = Ncnt[rr, cc]
        rid = np.repeat(rr, reps)
        cid = np.repeat(cc, reps)
        u = rng.random(len(cid))
        r = rmax[cid] * np.sqrt(u)
        kk = 2.0 * k0[cid] * C.Fg0(r / rs[cid])
        kap[s0:s1] = (np.bincount(rid, weights=kk, minlength=nch)
                      + sigW * rng.standard_normal(nch))
    return kap


def stage_mc(nreal=NREAL):
    MCDIR.mkdir(parents=True, exist_ok=True)
    C = Cosmo(Nz=100)
    fld = LOSField(C)
    print("extended mass integrals (sub-Mmin continuum)...", flush=True)
    ext = extended_mass_integrals(C)
    for iz, zs in enumerate(ZS_LIST):
        kt = C.find_kappathr(zs, 100)
        W = None
        for iseed in range(NSEEDS):
            pn = mc_path(zs, "new", iseed)
            po = mc_path(zs, "old", iseed)
            if pn.exists() and po.exists():
                continue
            if W is None:
                W = weak_amp_tables(C, fld, zs, kt, ext)
                print(f"[amp] zs={zs:g} kt={kt:.3e} sigW={W['sigW']:.4f} "
                      f"nsh={len(W['jz_arr'])} |s_w_win|={np.linalg.norm(W['s_w_win']):.3e} "
                      f"|s_w_pen|={np.linalg.norm(W['s_w_pen']):.3e}", flush=True)
            seed = SEED0 + 1_000_000 * iz + 10_000 * iseed
            if not pn.exists():
                t0 = time.time()
                kap, wwin, wpen = run_new_arm(C, fld, zs, kt, W, seed, nreal)
                np.savez(pn, kappa=kap, wwin=wwin, wpen=wpen, kappathr=kt,
                         sigW=W["sigW"], nreal=nreal, seed=seed,
                         s_w_win=W["s_w_win"], s_w_pen=W["s_w_pen"])
                print(f"[mc] zs={zs:g} new s{iseed}: Var(w_win)="
                      f"{wwin.astype(np.float64).var():.3e} Var(w_pen)="
                      f"{wpen.astype(np.float64).var():.3e} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            if not po.exists():
                t0 = time.time()
                kap = run_old_arm(C, fld, zs, kt, seed + 100, nreal)
                np.savez(po, kappa=kap, kappathr=kt, nreal=nreal,
                         seed=seed + 100)
                print(f"[mc] zs={zs:g} old s{iseed}: ({time.time()-t0:.0f}s)",
                      flush=True)


# ------------------------------------------------------------------ analysis
def _jsd(h1, h2):
    """Jensen-Shannon divergence (nats) between two histograms."""
    p = h1 / h1.sum()
    q = h2 / h2.sum()
    m = 0.5 * (p + q)

    def kl(x, y):
        s = x > 0
        return np.sum(x[s] * np.log(x[s] / y[s]))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _hist(x, edges):
    h, _ = np.histogram(x, bins=edges)
    under = (x < edges[0]).sum()
    over = (x >= edges[-1]).sum()
    return np.concatenate([[under], h, [over]]).astype(np.float64)


def _stats(k):
    x = k[np.abs(k) < 0.5]
    x = x - x.mean()
    return dict(var05=(x ** 2).mean(), std05=np.sqrt((x ** 2).mean()),
                q99=np.quantile(k, 0.99), q999=np.quantile(k, 0.999),
                f05=(k > 0.5).mean(), f1=(k > 1).mean())


def stage_report():
    siz = np.load(SIZING) if SIZING.exists() else None
    arms = {}
    for zs in ZS_LIST:
        new = [np.load(mc_path(zs, "new", i)) for i in range(NSEEDS)]
        old = [np.load(mc_path(zs, "old", i)) for i in range(NSEEDS)]
        kc = [d["kappa"].astype(np.float64) for d in new]
        ww = [d["wwin"].astype(np.float64) for d in new]
        wp = [d["wpen"].astype(np.float64) for d in new]
        ko = [d["kappa"].astype(np.float64) for d in old]
        arms[zs] = dict(
            base=[kc[0], kc[1]],
            weakw=[kc[0] + ww[0], kc[1] + ww[1]],
            weakp=[kc[0] + wp[0], kc[1] + wp[1]],
            old=[ko[0], ko[1]],
            kc=kc, ww=ww, wp=wp, sigW=float(new[0]["sigW"]))

    L = ["# Weak-arm MC: PDF-level effect of the clustered weak background\n",
         f"Prototype MC (kappa only, Nz=100, fixed-<N>=100 rule), {NSEEDS}x"
         f"{NREAL} realizations per arm per zs. base = shipped bias_model=1 "
         "(counts-only field); weakw/weakp = same draws + windowed/pencil "
         "clustered weak term; old = legacy iid layer. JSD in nats on "
         f"{NBINS} body bins (pooled-base quantile range 1e-4..1-1e-4) + "
         "under/overflow; floor = JSD(baseA, baseB) at N/2 vs N/2 (the "
         "pooled arm-vs-base comparisons have ~half that floor).\n",
         "## 1. Body statistics (pooled seeds)\n",
         "| zs | arm | Var(|k|<0.5) | dVar vs base | q99 | q99.9 | f(k>1) |",
         "|--:|:--|--:|:--|--:|--:|--:|"]
    jsd_rows = []
    var_rows = []
    for zs in ZS_LIST:
        A = arms[zs]
        pooled = {a: np.concatenate(A[a]) for a in
                  ("base", "weakw", "weakp", "old")}
        vb = _stats(pooled["base"])["var05"]
        for arm in ("base", "weakw", "weakp", "old"):
            st = _stats(pooled[arm])
            dv = (st["var05"] / vb - 1.0) * 100.0
            L.append(f"| {zs:g} | {arm} | {st['var05']:.4e} | "
                     f"{dv:+.1f}% | {st['q99']:.4f} | {st['q999']:.4f} | "
                     f"{st['f1']:.2e} |")
        # JSD vs base
        lo, hi = np.quantile(pooled["base"], [1e-4, 1 - 1e-4])
        edges = np.linspace(lo, hi, NBINS + 1)
        hb = _hist(pooled["base"], edges)
        floor = _jsd(_hist(A["base"][0], edges), _hist(A["base"][1], edges))
        for arm in ("weakw", "weakp", "old"):
            j = _jsd(_hist(pooled[arm], edges), hb)
            jhalves = [_jsd(_hist(A[arm][i], edges),
                            _hist(A["base"][i], edges)) for i in range(2)]
            jsd_rows.append((zs, arm, j, floor, jhalves))
        # measured vs analytic covariance budget
        kc = np.concatenate(A["kc"])
        ww = np.concatenate(A["ww"])
        wp = np.concatenate(A["wp"])
        m = np.abs(kc) < 0.5           # body-clipped, matches sizing "meas"
        Sww_m = ww.var()
        Sew_m = np.cov(kc[m], ww[m])[0, 1]
        Swwp_m = wp.var()
        Sewp_m = np.cov(kc[m], wp[m])[0, 1]
        row = [zs, Sww_m, 2 * Sew_m, Swwp_m, 2 * Sewp_m]
        if siz is not None and f"z{zs:g}_S_ww_win" in siz.files:
            row += [float(siz[f"z{zs:g}_S_ww_win"]),
                    2 * float(siz[f"z{zs:g}_S_ew_win"]),
                    float(siz[f"z{zs:g}_S_ww"]),
                    2 * float(siz[f"z{zs:g}_S_ew"])]
        else:
            row += [np.nan] * 4
        var_rows.append(row)

    L += ["\n## 2. JSD vs base (nats)\n",
          "| zs | arm | JSD(pooled) | floor(baseA,baseB) | per-seed-half |",
          "|--:|:--|--:|--:|:--|"]
    for zs, arm, j, floor, jh in jsd_rows:
        L.append(f"| {zs:g} | {arm} | {j:.3e} | {floor:.3e} | "
                 f"{jh[0]:.3e} / {jh[1]:.3e} |")

    L += ["\n## 3. Measured weak-arm variance budget vs analytic sizing\n",
          "(S_ew measured on the body-clipped counts arm; analytic from "
          "bias_field_joint sizing.npz)\n",
          "| zs | S_ww win MC/an | 2S_ew win MC/an | S_ww pen MC/an | "
          "2S_ew pen MC/an |", "|--:|:--|:--|:--|:--|"]
    for r in var_rows:
        L.append(f"| {r[0]:g} | {r[1]:.2e} / {r[5]:.2e} | "
                 f"{r[2]:.2e} / {r[6]:.2e} | {r[3]:.2e} / {r[7]:.2e} | "
                 f"{r[4]:.2e} / {r[8]:.2e} |")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT / "report.md")
    _plot(arms)


def _plot(arms):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink = "#334155"
    grid_c = "#e2e8f0"
    c_base = "#2a78d6"    # blue: base (shipped counts-only field)
    c_ww = "#1baf7a"      # aqua: + windowed weak arm
    c_wp = "#eb6834"      # orange: + pencil weak arm (upper bracket)
    c_old = "#8a8776"     # muted: legacy iid layer
    plt.rcParams.update({
        "text.color": ink, "axes.labelcolor": ink, "axes.edgecolor": grid_c,
        "xtick.color": ink, "ytick.color": ink, "font.size": 10,
        "axes.grid": True, "grid.color": grid_c, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white"})

    fig, axes = plt.subplots(1, len(ZS_LIST), figsize=(4.0 * len(ZS_LIST), 3.6),
                             sharey=False)
    for ax, zs in zip(np.atleast_1d(axes), ZS_LIST):
        A = arms[zs]
        pooled = {a: np.concatenate(A[a]) for a in
                  ("base", "weakw", "weakp", "old")}
        lo, hi = np.quantile(pooled["base"], [5e-4, 1 - 5e-4])
        pad = 0.35 * (hi - lo)
        edges = np.linspace(lo - pad, hi + pad, 160)
        ctr = 0.5 * (edges[1:] + edges[:-1])
        # legacy arm = neutral dashed reference line, not a categorical series
        for arm, c, ls, lw, lab in (
                ("old", c_old, "--", 1.1, "legacy iid (bias_model=0)"),
                ("base", c_base, "-", 1.4, "field, counts-only (shipped)"),
                ("weakw", c_ww, "-", 1.4, "+ clustered weak (windowed)"),
                ("weakp", c_wp, "-", 1.4, "+ clustered weak (pencil)")):
            h, _ = np.histogram(pooled[arm], bins=edges, density=True)
            ax.plot(ctr, h, color=c, ls=ls, lw=lw, label=lab)
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-3)
        ax.set_xlabel("kappa")
        ax.set_title(f"z_s = {zs:g}", fontsize=10)
    np.atleast_1d(axes)[0].set_ylabel("P(kappa)")
    np.atleast_1d(axes)[0].legend(fontsize=7.5, frameon=False)
    fig.suptitle("Clustered weak background: P(kappa), prototype MC "
                 f"(2x{NREAL//1000}k per arm)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    PLOTS.mkdir(exist_ok=True)
    fig.savefig(PLOTS / "bias_field_weak_arm.png", dpi=160)
    print("wrote", PLOTS / "bias_field_weak_arm.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["mc", "report", "all"])
    ap.add_argument("--nreal", type=int, default=NREAL)
    a = ap.parse_args()
    if a.stage in ("mc", "all"):
        stage_mc(a.nreal)
    if a.stage in ("report", "all"):
        stage_report()
