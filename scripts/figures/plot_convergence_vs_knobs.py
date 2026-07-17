"""Cross-knob convergence summary (2026-07-13).

Two one-row figures over the five audited numerical knobs
(Mmin, Nz, kappa_thr rule, subhalo_factor, eps_floor):

  plots/convergence_excess_vs_knobs.png
      D_sys = max(0, JSD - floor), the resolved systematic divergence above
      the finite-sampling floor, vs each knob. MC panels reuse the cached
      study samples (no new sampling); eps_floor has no MC arm (the knob is
      C++-internal, not exposed in the bindings) so its panel shows the
      analytic propagated bound instead.

  plots/vark_vs_knobs.png
      Variance response vs each knob, each panel normalized to its own
      converged/truth value so the row shares one reading: "how far is the
      default's variance from the converged limit".

Reads only cached results:
  data/results/{mmin,mmin_pd,nz}_convergence/, kappathr_subhalo_jsd/,
  subhalo_factor_jsd/ (summary.json + per-config lnmu sample .npy),
  data/results/convergence_analytic/analytic_arms.npz,
  data/results/mmin_convergence/ace_gap_vs_mmin.json,
  data/sigma_partition_vs_kthr_z1.npz,
  data/sigma_partition_vs_subhalo_factor_z1.npz,
  playground/k2_vs_floor_zs{1,5}.txt,
  playground/sigmaW_vs_kthr_zs{1,5}.txt, playground/sigma_total_vs_kthr_zs{1,5}.txt

Caveat carried on the figure: mmin_pd floors are provisional until the
batch-mean kappa anchor is fixed and the flagged shards regenerated
(data/results/floor_permutation_null/report.md, 2026-07-13).
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "data" / "results"
PLOTS = REPO / "plots"
PLAY = REPO / "playground"

sys.path.insert(0, str(REPO / "scripts" / "figures"))
import plot_convergence_studies as pcs   # palette, loaders, bootstrap machinery

C, GRAY, EMU_KL = pcs.C, pcs.GRAY, pcs.EMU_KL
C5 = "#eda100"   # z_s=5 slot (same as pcs)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
})


# ---------------------------------------------------------------- helpers
def boot_excess(outdir, edges, truth_files, cfg_file, floor_pred, jsd,
                B=300, seed=20260713):
    """(excess, lo68, hi68) for one config vs its truth, multinomial bootstrap."""
    rng = np.random.default_rng(seed)
    w = np.diff(edges)
    tr = np.concatenate([np.load(f) for f in truth_files])
    tr = tr[np.isfinite(tr)]
    ct = np.histogram(tr, bins=edges)[0]
    s = np.load(cfg_file); s = s[np.isfinite(s)]
    cc = np.histogram(s, bins=edges)[0]
    pt, pc = ct / ct.sum(), cc / cc.sum()
    bs = np.empty(B)
    for b in range(B):
        rc = rng.multinomial(cc.sum(), pc)
        rt = rng.multinomial(ct.sum(), pt)
        bs[b] = pcs._jsd_from_counts(rc, rt, w)
    e = bs - floor_pred
    return max(jsd - floor_pred, 0.0), np.percentile(e, 16), np.percentile(e, 84)


def style_excess_axis(ax, xlabel, default_x=None, budget_label=False):
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.axhline(EMU_KL, color=GRAY, lw=1.0)
    if budget_label:
        ax.text(0.98, EMU_KL * 0.72, "emulator median KL (budget)",
                color=GRAY, fontsize=7, ha="right",
                transform=ax.get_yaxis_transform())
    if default_x is not None:
        ax.axvline(default_x, color=GRAY, lw=0.8, alpha=0.5)
        ax.text(default_x, 0.97, " default", color=GRAY, fontsize=7,
                rotation=90, transform=ax.get_xaxis_transform(), va="top")
    ax.set_yscale("symlog", linthresh=1e-4, linscale=0.6)
    ax.set_ylim(-5e-5, 1.2e-2)
    ax.set(xscale="log", xlabel=xlabel)


# ================================================== Figure 1: D_sys vs knobs
def fig_excess():
    fig, axes = plt.subplots(1, 5, figsize=(19, 3.6))

    # ---- panel 1: Mmin (plain solid, per-decade-matched dashed)
    ax = axes[0]
    plain = pcs.load("mmin_convergence")
    pd_ = pcs.load("mmin_pd_convergence")

    def xval_m(cname, d):
        if cname == "truth" or cname.startswith("nm"):
            return None
        return d["kwargs"]["Mmin"]

    for src, name, ls, m, lw in ((plain, "mmin_convergence", "-", "o", 1.6),
                                 (pd_, "mmin_pd_convergence", "--", "s", 1.2)):
        if src is None:
            continue
        for z, (xs, ex, lo, hi) in pcs.excess_series(name, src, xval_m).items():
            xs_e = np.concatenate([[1e4], xs])
            ex_e = np.concatenate([[0.0], ex])
            ax.plot(xs_e, ex_e, ls, marker=m, ms=4, lw=lw, color=C[z],
                    label=(f"z_s={z}" if ls == "-" else None),
                    markevery=list(range(1, len(xs_e))))
            ax.plot([1e4], [0.0], m, ms=5.5, mfc="white", mec=C[z], mew=1.2)
            if ls == "-":
                ax.fill_between(xs, lo, hi, color=C[z], alpha=0.15, lw=0)
    style_excess_axis(ax, r"$M_{\min}$  [$M_\odot$]", 1e7, budget_label=True)
    ax.set_ylabel(r"$D_{\rm sys}=\max(0,\,{\rm JSD}-{\rm floor})$  [nats]")
    ax.set_title("$M_{\\min}$ (HMF lower limit)\nsolid NM=100, dashed 10/decade;"
                 " open = truth", fontsize=8.5)
    ax.text(0.03, 0.02, "pd (dashed) floors provisional\n(batch-anchor bug,"
            " 2026-07-13)", fontsize=6.5, color=GRAY, transform=ax.transAxes)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")

    # ---- panel 2: Nz
    ax = axes[1]
    nz = pcs.load("nz_convergence")

    def xval_n(cname, d):
        return None if cname == "truth" else d["kwargs"]["Nz"]

    for z, (xs, ex, lo, hi) in pcs.excess_series("nz_convergence", nz,
                                                 xval_n).items():
        xs_e = np.concatenate([xs, [400]])
        ex_e = np.concatenate([ex, [0.0]])
        ax.plot(xs_e, ex_e, "-", marker="o", ms=4, lw=1.6, color=C[z],
                label=f"z_s={z}", markevery=list(range(len(xs))))
        ax.plot([400], [0.0], "o", ms=5.5, mfc="white", mec=C[z], mew=1.2)
        ax.fill_between(xs, lo, hi, color=C[z], alpha=0.15, lw=0)
    style_excess_axis(ax, r"$N_z$", 100)
    ax.set_title("$N_z$ (LOS z-grid)\nopen = truth ($N_z$=400)", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")

    # ---- panel 3: kappa_thr rule (x = effective kappa_thr; subhalo on/off)
    ax = axes[2]
    kd = json.loads((RES / "kappathr_subhalo_jsd" / "summary.json").read_text())
    for gk, ls, m in (("z1_on", "-", "o"), ("z1_off", "--", "s"),
                      ("z10_on", "-", "o"), ("z10_off", "--", "s")):
        g = kd[gk]
        z, arm = gk.split("_")[0][1:], gk.split("_")[1]
        edges = np.array(g["edges"])
        tfiles = [RES / "kappathr_subhalo_jsd" / f"z{z}_{arm}_truth{h}.npy"
                  for h in "AB"]
        xs, ex, lo, hi, names = [], [], [], [], []
        for rule, d in g["rules"].items():
            if rule == "truth":
                continue
            cfg = RES / "kappathr_subhalo_jsd" / f"z{z}_{arm}_{rule}.npy"
            e, l, h = boot_excess(RES / "kappathr_subhalo_jsd", edges, tfiles,
                                  cfg, d["floor_pred"], d["jsd"])
            xs.append(d["kthr_eff"]); ex.append(e); lo.append(l); hi.append(h)
            names.append(rule)
        o = np.argsort(xs)
        xs, ex, lo, hi = (np.array(v)[o] for v in (xs, ex, lo, hi))
        names = [names[i] for i in o]
        kt_truth = g["rules"]["truth"]["kthr_eff"]
        ax.plot(np.concatenate([[kt_truth], xs]), np.concatenate([[0.0], ex]),
                ls, marker=m, ms=4, lw=1.4 if ls == "-" else 1.0, color=C[z],
                label=(f"z_s={z} subhalo-{arm}"),
                markevery=list(range(1, len(xs) + 1)))
        ax.plot([kt_truth], [0.0], m, ms=5.5, mfc="white", mec=C[z], mew=1.2)
        if arm == "on":
            ax.fill_between(xs, lo, hi, color=C[z], alpha=0.15, lw=0)
        i = names.index("fixedN")
        ax.plot([xs[i]], [ex[i]], "*", ms=11, mfc="none", mec=C[z], mew=1.3)
    ax.plot([], [], "*", ms=10, mfc="none", mec=GRAY, mew=1.2,
            label=r"fixed-$\langle N\rangle$=100 (default rule)")
    style_excess_axis(ax, r"$\kappa_{\rm thr}$ (effective)")
    ax.set_title("$\\kappa_{\\rm thr}$ rule (explicit/Gaussian split)\n"
                 "open = truth (flat 3e-5); default floats with $z_s$",
                 fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.8, loc="upper left")

    # ---- panel 4: subhalo_factor (model 3 vs brute truth; model-1 contrast)
    ax = axes[3]
    sd = json.loads((RES / "subhalo_factor_jsd" / "summary.json").read_text())
    for gk, g in sd.items():
        z = gk[1:]
        edges = np.array(g["edges"])
        tfiles = [RES / "subhalo_factor_jsd" / f"z{z}_truth{h}.npy" for h in "AB"]
        xs, ex, lo, hi = [], [], [], []
        for cname, d in g["configs"].items():
            if cname == "truth":
                continue
            cfg = RES / "subhalo_factor_jsd" / f"z{z}_{cname}.npy"
            e, l, h = boot_excess(RES / "subhalo_factor_jsd", edges, tfiles,
                                  cfg, d["floor_pred"], d["jsd"])
            x = d["kwargs"].get("subhalo_factor")
            if cname.startswith("m1_"):
                ax.plot([x], [e], "D", ms=6, mfc="none", mec=C[z], mew=1.3)
                dy = 4 if z == "5" else -16
                ax.annotate("model 1\n(no Wsub)", (x, e), xytext=(6, dy),
                            textcoords="offset points", fontsize=6.5,
                            color=C[z])
                continue
            xs.append(x); ex.append(e); lo.append(l); hi.append(h)
        o = np.argsort(xs)
        xs, ex, lo, hi = (np.array(v)[o] for v in (xs, ex, lo, hi))
        ax.plot(xs, ex, "-", marker="o", ms=4, lw=1.6, color=C[z],
                label=f"z_s={z}, model 3")
        ax.fill_between(xs, lo, hi, color=C[z], alpha=0.15, lw=0)
    style_excess_axis(ax, "subhalo_factor", 1e-2)
    ax.set_title("subhalo_factor (clump split, model 3)\n"
                 "truth = brute clump resolution", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")

    # ---- panel 5: eps_floor (analytic propagated bound; no MC arm)
    ax = axes[4]
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    # default kappa_thr_eff per z from the mmin study (fixed-<N>=100 rule)
    mm = pcs.load("mmin_convergence")
    kdef = {z: mm[f"z{z}_main"]["configs"]["m1e7"]["kthr_eff"] for z in ("1", "5")}
    for z in ("1", "5"):
        eps, K2 = np.loadtxt(PLAY / f"k2_vs_floor_zs{z}.txt", unpack=True)
        o = np.argsort(eps); eps, K2 = eps[o], K2[o]
        deficit = 1.0 - K2 / K2[0]              # relative sigma_W^2 truncation
        kt, sw, _ = np.loadtxt(PLAY / f"sigmaW_vs_kthr_zs{z}.txt", unpack=True)
        kt2, st = np.loadtxt(PLAY / f"sigma_total_vs_kthr_zs{z}.txt",
                             unpack=True, usecols=(0, 1))
        fW = (np.interp(kdef[z], kt, sw) / np.interp(kdef[z], kt2, st)) ** 2
        bound = (deficit * fW) ** 2 / 16.0      # Gaussian-variance JSD bound
        ax.plot(eps[1:], bound[1:], "-", marker="o", ms=3.5, lw=1.6, color=C[z],
                label=f"z_s={z}  ($f_W$={fW:.1e})")
    ax.axhline(1.1e-4, color=GRAY, ls=":", lw=1.0)
    ax.text(0.03, 1.5e-4, "typical MC floor (240k)", color=GRAY, fontsize=7,
            transform=ax.get_yaxis_transform())
    ax.axhline(EMU_KL, color=GRAY, lw=1.0)
    ax.axvline(1e-3, color=GRAY, lw=0.8, alpha=0.5)
    ax.text(1e-3, 0.97, " default", color=GRAY, fontsize=7, rotation=90,
            transform=ax.get_xaxis_transform(), va="top")
    ax.set(xscale="log", yscale="log", xlabel=r"$\epsilon_{\rm floor}$",
           ylim=(1e-17, 3e-2))
    ax.set_title("$\\epsilon_{\\rm floor}$ ($\\sigma_W$ radial cutoff)\n"
                 "analytic bound $(\\epsilon$-deficit $\\times f_W)^2/16$ — "
                 "no MC arm needed", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7.5, loc="center left")

    fig.suptitle("Resolved systematic divergence $D_{\\rm sys}$ of P(ln$\\mu$) "
                 "vs each numerical knob — defaults marked; truth = open marker; "
                 "bands = 68% bootstrap", y=1.04, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(PLOTS / "convergence_excess_vs_knobs.png", bbox_inches="tight")
    print("wrote plots/convergence_excess_vs_knobs.png")


# ================================================== Figure 2: Var(kappa) vs knobs
def fig_var():
    fig, axes = plt.subplots(1, 5, figsize=(19, 3.6))
    ana = np.load(RES / "convergence_analytic" / "analytic_arms.npz")

    # ---- panel 1: Mmin — MC total (ACE sweep, clipped) + analytic sigma_W^2
    ax = axes[0]
    d = json.loads((RES / "mmin_convergence" / "ace_gap_vs_mmin.json").read_text())
    mg = np.array(d["mmin_grid"])
    for z in d["zs"]:
        lab = f"{z:g}"
        arr = np.array(d[f"K2clip_z{z:g}"], float)      # (nMmin, nseeds)
        m, s = arr.mean(1), arr.std(1)
        ax.errorbar(mg, m / m[0], yerr=s / m[0], fmt="-o", ms=4, lw=1.6,
                    capsize=2, color=C[lab],
                    label=f"z_s={lab}  MC total (clipped)")
    amg = ana["mmin_grid"]
    for z in ("1.0", "5.0"):
        lab = z.rstrip("0").rstrip(".")
        v = ana[f"sigW_mmin_z{z}"] ** 2
        ref = v[np.searchsorted(amg, 1e4)]
        ax.plot(amg, v / ref, "--", lw=1.1, color=C[lab])
    ax.plot([], [], "--", lw=1.1, color=GRAY,
            label=r"analytic $\sigma_W^2$ (rule's own $\kappa_{\rm thr}$)")
    ax.axhline(1, color=GRAY, lw=0.8)
    ax.axvline(1e7, color=GRAY, lw=0.8, alpha=0.5)
    ax.text(1e7, 0.97, " default", color=GRAY, fontsize=7, rotation=90,
            transform=ax.get_xaxis_transform(), va="top")
    ax.set(xscale="log", xlabel=r"$M_{\min}$  [$M_\odot$]",
           ylabel=r"variance / converged value", ylim=(0.84, 1.06),
           xlim=(5e3, 2.5e9))
    ax.set_title("$M_{\\min}$: clipped total Var($\\kappa$) $-$5.4/$-$6.6% at"
                 " default ($z_s$=1/5);\nbackground $\\sigma_W^2$ converges"
                 " slower ($-$6.8/$-$11%)", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7, loc="lower left")

    # ---- panel 2: Nz — analytic sigma_W^2(Nz), rule vs frozen kappa_thr
    # (extended arm 2026-07-13: Nz up to 6400, sigW_nz_extended.npz)
    ax = axes[1]
    ext = RES / "convergence_analytic" / "sigW_nz_extended.npz"
    if ext.exists():
        d = np.load(ext)
        nzg = d["nz_grid"]
        nz_ref = int(nzg[-1])
        for zl in ("0.2", "1", "5", "10"):
            r = (d[f"sigW_rule_z{zl}"] / d[f"sigW_rule_z{zl}"][-1]) ** 2
            f = (d[f"sigW_fix_z{zl}"] / d[f"sigW_fix_z{zl}"][-1]) ** 2
            ax.plot(nzg, r, "-", marker="o", ms=3.5, lw=1.6, color=C[zl],
                    label=f"z_s={zl}")
            ax.plot(nzg, f, "--", marker="s", ms=3, lw=1.0, color=C[zl])
    else:
        nzg = ana["nz_grid"]
        nz_ref = int(nzg[-1])
        for z in ("0.2", "1.0", "5.0", "10.0"):
            lab = z.rstrip("0").rstrip(".") if z != "10.0" else "10"
            r = (ana[f"sigW_nz_rulekthr_z{z}"]
                 / ana[f"sigW_nz_rulekthr_z{z}"][-1]) ** 2
            f = (ana[f"sigW_nz_fixkthr_z{z}"]
                 / ana[f"sigW_nz_fixkthr_z{z}"][-1]) ** 2
            ax.plot(nzg, r, "-", marker="o", ms=3.5, lw=1.6, color=C[lab],
                    label=f"z_s={lab}")
            ax.plot(nzg, f, "--", marker="s", ms=3, lw=1.0, color=C[lab])
    ax.axhline(1, color=GRAY, lw=0.8)
    ax.axvline(100, color=GRAY, lw=0.8, alpha=0.5)
    ax.text(100, 0.05, " default", color=GRAY, fontsize=7, rotation=90,
            transform=ax.get_xaxis_transform(), va="bottom")
    ax.set(xscale="log", xlabel=r"$N_z$",
           ylabel=f"$\\sigma_W^2\\,/\\,\\sigma_W^2(N_z{{=}}{nz_ref})$")
    ax.set_title("$N_z$: $\\sigma_W^2$ (solid: rule's own $\\kappa_{\\rm thr}$,"
                 " dashed: frozen)\nconverges to a true limit — unlike the "
                 "bias tail", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7, loc="lower right")

    # ---- panel 3: kappa_thr — variance partition (z=1, full LOS MC)
    ax = axes[2]
    p = np.load(REPO / "data" / "sigma_partition_vs_kthr_z1.npz")
    plateau = np.median(p["sigma_total"]) ** 2
    ax.plot(p["kthr_total"], p["sigma_total"] ** 2 / plateau, "-o", ms=3.5,
            lw=1.6, color=C["1"], label=r"total (MC)")
    ax.plot(p["kthr_explicit"], p["sigma_explicit"] ** 2 / plateau, "--", lw=1.1,
            color=C["1"], alpha=0.7, label="explicit part")
    ax.plot(p["kthr_background"], p["sigma_background"] ** 2 / plateau, ":",
            lw=1.4, color=C["1"], alpha=0.9, label=r"Gaussian bg $\sigma_W^2$")
    kdef1 = json.loads((RES / "mmin_convergence" / "summary.json").read_text()
                       )["z1_main"]["configs"]["m1e7"]["kthr_eff"]
    ax.axhline(1, color=GRAY, lw=0.8)
    ax.axvline(kdef1, color=GRAY, lw=0.8, alpha=0.5)
    ax.text(kdef1, 0.03, r" default (fixed-$\langle N\rangle$, $z_s$=1)",
            color=GRAY, fontsize=7, rotation=90,
            transform=ax.get_xaxis_transform(), va="bottom")
    ax.set(xscale="log", yscale="log", xlabel=r"$\kappa_{\rm thr}$",
           ylabel=r"variance / total plateau", ylim=(1e-5, 3))
    ax.set_title("$\\kappa_{\\rm thr}$: partition trades explicit $\\leftrightarrow$"
                 " Gaussian,\ntotal Var($\\kappa$) stays flat ($z_s$=1)",
                 fontsize=8.5)
    ax.legend(frameon=False, fontsize=7, loc="center left")

    # ---- panel 4: subhalo_factor — single-host partition (exactness theorem)
    ax = axes[3]
    q = np.load(REPO / "data" / "sigma_partition_vs_subhalo_factor_z1.npz")
    tot2 = float(q["sigma_total"]) ** 2
    fac = q["subhalo_factor"]
    ax.plot(fac, (q["sigma_strong"] ** 2 + q["sigma_weak"] ** 2) / tot2, "-o",
            ms=3.5, lw=1.6, color=C["1"], label="resolved + Wsub (total)")
    ax.plot(fac, q["sigma_strong"] ** 2 / tot2, "--", lw=1.1, color=C["1"],
            alpha=0.7, label="resolved clumps")
    ax.plot(fac, q["sigma_weak"] ** 2 / tot2, ":", lw=1.4, color=C["1"],
            alpha=0.9, label="unresolved (Wsub)")
    ax.axhline(1, color=GRAY, lw=0.8)
    ax.axvline(1e-2, color=GRAY, lw=0.8, alpha=0.5)
    ax.text(1e-2, 0.03, " default", color=GRAY, fontsize=7, rotation=90,
            transform=ax.get_xaxis_transform(), va="bottom")
    ax.set(xscale="log", yscale="log", xlabel="subhalo_factor",
           ylabel=r"clump variance / brute total", ylim=(1e-4, 3))
    ax.set_title("subhalo_factor: clump-variance partition, single host\n"
                 "(M=1e13, $z_l$=0.5, $z_s$=1) — total flat = the Wsub theorem",
                 fontsize=8.5)
    ax.legend(frameon=False, fontsize=7, loc="center left")

    # ---- panel 5: eps_floor — K2 truncation
    ax = axes[4]
    for z in ("1", "5"):
        eps, K2 = np.loadtxt(PLAY / f"k2_vs_floor_zs{z}.txt", unpack=True)
        o = np.argsort(eps); eps, K2 = eps[o], K2[o]
        ax.plot(eps[1:], (K2 / K2[0])[1:], "-", marker="o", ms=3.5, lw=1.6,
                color=C[z], label=f"z_s={z}")
    ax.axhline(1, color=GRAY, lw=0.8)
    ax.axvline(1e-3, color=GRAY, lw=0.8, alpha=0.5)
    ax.text(1e-3, 0.05, " default", color=GRAY, fontsize=7, rotation=90,
            transform=ax.get_xaxis_transform(), va="bottom")
    ax.set(xscale="log", xlabel=r"$\epsilon_{\rm floor}$",
           ylabel=r"$K_2(\epsilon)\,/\,K_2(\epsilon{\to}0)$", ylim=(0.9, 1.005))
    ax.set_title("$\\epsilon_{\\rm floor}$: $\\sigma_W^2$ truncation, deficit"
                 " $\\propto\\epsilon$\n(0.10% at the default)", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")

    fig.suptitle(r"Variance response vs each numerical knob — each panel"
                 " normalized to its converged/truth value", y=1.04,
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(PLOTS / "vark_vs_knobs.png", bbox_inches="tight")
    print("wrote plots/vark_vs_knobs.png")


# ================================== Figure 3: total observable variance vs knobs
# Clipped Var(lnmu) from the cached study samples: clip = the study's own
# shared histogram range (the JSD protocol's quantile clip), variance of the
# clipped samples, ratio config/truth. SE via the moment formula
# Var[s^2] ~ (m4 - m2^2)/n, propagated to the ratio. NOTE: batch-anchor
# coupling makes iid SEs slightly optimistic (exchangeable unit = shard).
def _clipvar(path, lo, hi):
    x = np.load(path); x = x[np.isfinite(x) & (x >= lo) & (x <= hi)]
    x = x - x.mean()
    m2 = float((x ** 2).mean()); m4 = float((x ** 4).mean())
    return m2, np.sqrt(max(m4 - m2 ** 2, 0.0) / x.size)


def _ratio_series(outdir, edges, truth_files, cfg_files):
    lo, hi = edges[0], edges[-1]
    m2s = [_clipvar(f, lo, hi) for f in truth_files]
    wt = np.array([1.0 / se ** 2 for _, se in m2s])
    vt = np.average([m for m, _ in m2s], weights=wt)
    st = 1.0 / np.sqrt(wt.sum())
    out = []
    for f in cfg_files:
        v, s = _clipvar(f, lo, hi)
        r = v / vt
        out.append((r, r * np.hypot(s / v, st / vt)))
    return out


def fig_var_total():
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.6), sharey=True)

    def style(ax, xlabel, default_x=None):
        ax.axhline(1, color=GRAY, lw=0.8)
        if default_x is not None:
            ax.axvline(default_x, color=GRAY, lw=0.8, alpha=0.5)
            ax.text(default_x, 0.97, " default", color=GRAY, fontsize=7,
                    rotation=90, transform=ax.get_xaxis_transform(), va="top")
        ax.set(xscale="log", xlabel=xlabel)

    # ---- Mmin (plain + pd)
    ax = axes[0]
    for name, ls, m, lw in (("mmin_convergence", "-", "o", 1.6),
                            ("mmin_pd_convergence", "--", "s", 1.2)):
        src = pcs.load(name)
        if src is None:
            continue
        for gk, g in src.items():
            if not gk.endswith("_main"):
                continue
            z = gk.split("_")[0][1:]
            edges = np.array(g["edges"])
            od = RES / name
            tf = [od / f"z{z}_main_truth{h}.npy" for h in "AB"]
            xs, cf = [], []
            for cname, d in g["configs"].items():
                if cname == "truth" or cname.startswith("nm"):
                    continue
                xs.append(d["kwargs"]["Mmin"])
                cf.append(od / f"z{z}_main_{cname}.npy")
            o = np.argsort(xs)
            rs = _ratio_series(od, edges, tf, [cf[i] for i in o])
            xs = np.array(xs)[o]
            ax.errorbar(xs, [r for r, _ in rs], yerr=[e for _, e in rs],
                        fmt=ls, marker=m, ms=4, lw=lw, capsize=2, color=C[z],
                        label=(f"z_s={z}" if ls == "-" else None))
    ax.plot([1e4], [1.0], "o", ms=5.5, mfc="white", mec=GRAY, mew=1.2)
    style(ax, r"$M_{\min}$  [$M_\odot$]", 1e7)
    ax.set_ylabel("clipped Var(ln$\\mu$) / truth")
    ax.set_title("$M_{\\min}$ (solid NM=100, dashed 10/decade)", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7, loc="upper right")

    # ---- Nz
    ax = axes[1]
    src = pcs.load("nz_convergence")
    for gk, g in src.items():
        if not gk.endswith("_main"):
            continue
        z = gk.split("_")[0][1:]
        edges = np.array(g["edges"])
        od = RES / "nz_convergence"
        tf = [od / f"z{z}_main_truth{h}.npy" for h in "AB"]
        xs, cf = [], []
        for cname, d in g["configs"].items():
            if cname == "truth":
                continue
            xs.append(d["kwargs"]["Nz"])
            cf.append(od / f"z{z}_main_{cname}.npy")
        o = np.argsort(xs)
        rs = _ratio_series(od, edges, tf, [cf[i] for i in o])
        xs = np.array(xs)[o]
        ax.errorbar(xs, [r for r, _ in rs], yerr=[e for _, e in rs],
                    fmt="-", marker="o", ms=4, lw=1.6, capsize=2, color=C[z],
                    label=f"z_s={z}")
    ax.plot([400], [1.0], "o", ms=5.5, mfc="white", mec=GRAY, mew=1.2)
    style(ax, r"$N_z$", 100)
    ax.set_title("$N_z$", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7, loc="lower right")

    # ---- kappa_thr rule
    ax = axes[2]
    kd = json.loads((RES / "kappathr_subhalo_jsd" / "summary.json").read_text())
    for gk, g in kd.items():
        z, arm = gk.split("_")[0][1:], gk.split("_")[1]
        ls, m = ("-", "o") if arm == "on" else ("--", "s")
        edges = np.array(g["edges"])
        od = RES / "kappathr_subhalo_jsd"
        tf = [od / f"z{z}_{arm}_truth{h}.npy" for h in "AB"]
        xs, cf, names = [], [], []
        for rule, d in g["rules"].items():
            if rule == "truth":
                continue
            xs.append(d["kthr_eff"])
            cf.append(od / f"z{z}_{arm}_{rule}.npy")
            names.append(rule)
        o = np.argsort(xs)
        rs = _ratio_series(od, edges, tf, [cf[i] for i in o])
        xs = np.array(xs)[o]; names = [names[i] for i in o]
        ax.errorbar(xs, [r for r, _ in rs], yerr=[e for _, e in rs],
                    fmt=ls, marker=m, ms=4, lw=1.4 if arm == "on" else 1.0,
                    capsize=2, color=C[z], label=f"z_s={z} subhalo-{arm}")
        i = names.index("fixedN")
        ax.plot([xs[i]], [rs[i][0]], "*", ms=11, mfc="none", mec=C[z], mew=1.3)
    ax.plot([3e-5], [1.0], "o", ms=5.5, mfc="white", mec=GRAY, mew=1.2)
    style(ax, r"$\kappa_{\rm thr}$ (effective)")
    ax.set_title("$\\kappa_{\\rm thr}$ rule ($\\star$ = fixed-$\\langle N"
                 "\\rangle$ default)", fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.8, loc="lower left")

    # ---- subhalo_factor
    ax = axes[3]
    sd = json.loads((RES / "subhalo_factor_jsd" / "summary.json").read_text())
    for gk, g in sd.items():
        z = gk[1:]
        edges = np.array(g["edges"])
        od = RES / "subhalo_factor_jsd"
        tf = [od / f"z{z}_truth{h}.npy" for h in "AB"]
        xs, cf = [], []
        for cname, d in g["configs"].items():
            if cname == "truth":
                continue
            f = od / f"z{z}_{cname}.npy"
            if cname.startswith("m1_"):
                r, e = _ratio_series(od, edges, tf, [f])[0]
                ax.errorbar([d["kwargs"]["subhalo_factor"]], [r], yerr=[e],
                            fmt="D", ms=6, mfc="none", mec=C[z], color=C[z],
                            capsize=2, mew=1.3)
                continue
            xs.append(d["kwargs"]["subhalo_factor"]); cf.append(f)
        o = np.argsort(xs)
        rs = _ratio_series(od, edges, tf, [cf[i] for i in o])
        xs = np.array(xs)[o]
        ax.errorbar(xs, [r for r, _ in rs], yerr=[e for _, e in rs],
                    fmt="-", marker="o", ms=4, lw=1.6, capsize=2, color=C[z],
                    label=f"z_s={z}, model 3")
    ax.plot([], [], "D", ms=6, mfc="none", mec=GRAY, mew=1.2,
            label="model 1 (no Wsub)")
    style(ax, "subhalo_factor", 1e-2)
    ax.set_title("subhalo_factor (truth = brute $\\equiv$ 1)", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7, loc="lower left")

    axes[0].set_ylim(0.82, 1.09)
    fig.suptitle("Total observable variance vs each MC-audited knob — "
                 "clipped Var(ln$\\mu$) ratio to the refined truth "
                 "(clip = each study's shared histogram range; "
                 "$\\epsilon_{\\rm floor}$ has no MC arm, effect $<10^{-5}$)",
                 y=1.04, fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOTS / "vark_total_vs_knobs.png", bbox_inches="tight")
    print("wrote plots/vark_total_vs_knobs.png")


# ================================== Figure 4: Nz total-variance deep dive
def fig_var_nz():
    """Var(kappa) vs Nz (raw-kappa MC to Nz=1600, seed namespace 8e8):
    support decomposition. The quantile clip floats with the deepest grid's
    strong-lensing tail (kappa>1 population grows with Nz, unsaturated), so
    'the variance' is support-dependent — fixed |kappa| windows separate the
    converging body from the migrating tail."""
    p = RES / "vark_nz" / "vark_vs_nz.npz"
    if not p.exists():
        print("skip fig_var_nz: no vark_vs_nz.npz yet")
        return
    d = np.load(p)
    nzg = d["nz_grid"]
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.6))

    ax = axes[0]
    for z in d["zs"]:
        lab = f"{z:g}"
        arr = d[f"K2clip_z{z:g}"]          # (nNz, nseeds)
        m, s = arr.mean(1), arr.std(1)
        ax.errorbar(nzg, m / m[-1], yerr=s / m[-1], fmt="-o", ms=4, lw=1.6,
                    capsize=2, color=C[lab], label=f"z_s={lab}")
    ax.set_ylabel("variance / $N_z$=1600 value")
    ax.set_title("quantile clip [$10^{-4}$, $1{-}10^{-4}$] of the deepest "
                 "grid:\nno saturation — the window admits the growing "
                 "strong tail", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.set_ylim(0.55, 1.35)

    ax = axes[1]
    for z in d["zs"]:
        if f"{z:g}" == "0.2":
            continue                       # |kappa| windows single-ray noisy
        lab = f"{z:g}"
        for b, ls, m_ in ((1.0, "-", "o"), (0.5, "--", "s")):
            arr = d[f"K2fix{b:g}_z{z:g}"]
            m, s = arr.mean(1), arr.std(1)
            ax.errorbar(nzg, m / m[-1], yerr=s / m[-1], fmt=ls, marker=m_,
                        ms=4, lw=1.6 if b == 1.0 else 1.0, capsize=2,
                        color=C[lab],
                        label=(f"z_s={lab}" if b == 1.0 else None))
    ax.plot([], [], "-o", ms=4, color=GRAY, label=r"$|\kappa|<1$")
    ax.plot([], [], "--s", ms=3.5, lw=1.0, color=GRAY, label=r"$|\kappa|<0.5$")
    ax.set_title("fixed supports: the $|\\kappa|<0.5$ body converges from "
                 "ABOVE,\nthe shoulder mass migrates outward as $N_z$ refines",
                 fontsize=8.5)
    ax.legend(frameon=False, fontsize=7, loc="upper right", ncols=2)
    ax.set_ylim(0.82, 1.35)

    ax = axes[2]
    for z in d["zs"]:
        lab = f"{z:g}"
        arr = d[f"frac_kgt1_z{z:g}"]
        m, s = arr.mean(1), arr.std(1)
        if m.max() <= 0:
            continue
        ax.errorbar(nzg, m, yerr=s, fmt="-o", ms=4, lw=1.6, capsize=2,
                    color=C[lab], label=f"z_s={lab}")
    ax.set(yscale="log", ylabel=r"fraction of rays with $\kappa>1$")
    ax.set_title("the strong-lensing population (multi-image regime,\n"
                 "outside weak-lensing validity) grows unsaturated",
                 fontsize=8.5)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")

    for ax in axes:
        if ax is not axes[2]:
            ax.axhline(1, color=GRAY, lw=0.8)
        ax.axvline(100, color=GRAY, lw=0.8, alpha=0.5)
        ax.text(100, 0.05, " default", color=GRAY, fontsize=7, rotation=90,
                transform=ax.get_xaxis_transform(), va="bottom")
        ax.set(xscale="log", xlabel=r"$N_z$")
    fig.suptitle("Var($\\kappa$) vs $N_z$ (raw-$\\kappa$ MC, 4 seeds "
                 "$\\times$ 50k, $N_z$ up to 1600) — variance convergence is "
                 "SUPPORT-DEPENDENT: the body converges, the $\\kappa>1$ tail "
                 "does not", y=1.05, fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOTS / "vark_total_vs_nz.png", bbox_inches="tight")
    print("wrote plots/vark_total_vs_nz.png")


# ============================ Figure 5: Nz strong-tail mechanism (bias layer)
# Analytic count-quadrature convergence, printed by the get_expected_halo_count
# scan 2026-07-13 (kc=1; identical shape for kc=0.5 and all z_s):
# ratio to Nz=3200 at Nz = 25,50,100,200,400,800,1600.
ANA_NZ = np.array([25, 50, 100, 200, 400, 800, 1600])
ANA_RATIO = np.array([0.870, 0.934, 0.967, 0.984, 0.992, 0.997, 0.999]) / 0.999


def fig_nz_mechanism():
    arms = {}
    for name, f in (("rule", "vark_vs_nz.npz"),
                    ("frozen", "vark_vs_nz_frozen.npz"),
                    ("nobias", "vark_vs_nz_nobias.npz")):
        p = RES / "vark_nz" / f
        if p.exists():
            arms[name] = np.load(p)
    if "nobias" not in arms:
        print("skip fig_nz_mechanism: arms missing")
        return
    STYLE = {"rule": ("-", "o", 1.8, 1.0, "bias ON, rule $\\kappa_{\\rm thr}$"),
             "frozen": ("--", "s", 1.2, 0.85, "bias ON, frozen $\\kappa_{\\rm thr}$"),
             "nobias": (":", "^", 1.8, 1.0, "bias OFF")}
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))

    ax = axes[0]
    for name, d in arms.items():
        ls, m, lw, al, lab = STYLE[name]
        for z in d["zs"]:
            zl = f"{z:g}"
            if zl not in ("5", "10"):
                continue
            arr = d[f"frac_kgt1_z{z:g}"]
            mm, ss = arr.mean(1), arr.std(1)
            ax.errorbar(d["nz_grid"], mm, yerr=ss, fmt=ls, marker=m, ms=4.5,
                        lw=lw, alpha=al, capsize=2, color=C[zl])
    for name in STYLE:
        ax.plot([], [], STYLE[name][0], marker=STYLE[name][1], color=GRAY,
                label=STYLE[name][4])
    for zl in ("5", "10"):
        ax.plot([], [], "s", ms=6, color=C[zl], label=f"z_s={zl}")
    ax.set(xscale="log", yscale="log", xlabel=r"$N_z$",
           ylabel=r"fraction of rays with $\kappa>1$")
    ax.set_title("the growth needs the bias layer:\nfreezing "
                 "$\\kappa_{\\rm thr}$ changes nothing, bias OFF is flat",
                 fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, loc="center left", ncols=2)

    ax = axes[1]
    for name, d in arms.items():
        ls, m, lw, al, lab = STYLE[name]
        for z in d["zs"]:
            zl = f"{z:g}"
            if zl != "10":
                continue
            arr = d[f"K2fix1_z{z:g}"]
            mm, ss = arr.mean(1), arr.std(1)
            ax.errorbar(d["nz_grid"], mm / mm[-1], yerr=ss / mm[-1], fmt=ls,
                        marker=m, ms=4.5, lw=lw, alpha=al, capsize=2,
                        color=C["10"], label=lab)
    ax.plot(ANA_NZ, ANA_RATIO, "-", lw=2.2, color=GRAY, alpha=0.8,
            label="analytic count quadrature\n(no MC, no bias)")
    ax.axhline(1, color=GRAY, lw=0.8)
    ax.set(xscale="log", xlabel=r"$N_z$",
           ylabel=r"Var($\kappa$; $|\kappa|<1$) / $N_z$=1600 value")
    ax.set_title("$z_s$=10, fixed support $|\\kappa|<1$: bias OFF lands on\n"
                 "the analytic quadrature curve (0.966 vs 0.967 at "
                 "$N_z$=100)", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")

    for ax in axes:
        ax.axvline(100, color=GRAY, lw=0.8, alpha=0.5)
        ax.text(100, 0.05, " default", color=GRAY, fontsize=7, rotation=90,
                transform=ax.get_xaxis_transform(), va="bottom")
    fig.suptitle("Nz strong-tail mechanism test — three MC arms + the "
                 "analytic prediction", y=1.03, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(PLOTS / "nz_tail_mechanism.png", bbox_inches="tight")
    print("wrote plots/nz_tail_mechanism.png")


if __name__ == "__main__":
    fig_excess()
    fig_var()
    fig_var_total()
    fig_var_nz()
    fig_nz_mechanism()
