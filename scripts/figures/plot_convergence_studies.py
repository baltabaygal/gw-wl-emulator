"""Figures for the Mmin / Nz grid-convergence studies (2026-07-12).

Reads:
  data/results/mmin_convergence/summary.json        (plain axis, NM=100 fixed)
  data/results/mmin_pd_convergence/summary.json     (per-decade-matched NM)
  data/results/nz_convergence/summary.json
  data/results/convergence_analytic/analytic_arms.npz
  data/results/mmin_convergence/ace_gap_vs_mmin.json (optional)

Writes plots/mmin_convergence.png, plots/nz_convergence.png,
plots/ace_gap_vs_mmin.png (skips whichever inputs are missing).

Design notes: JSD is plotted RAW (not floor-subtracted) with each z-group's
finite-sample floor as a dotted line in the matching hue — floor-subtraction
hides how close converged points sit to the noise. One axis per panel.
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

# categorical slots 1-4 (dataviz reference palette, light mode), fixed order
C = {"0.2": "#2a78d6", "1": "#1baf7a", "5": "#eda100", "10": "#008300"}
GRAY = "#52514e"
EMU_KL = 7.3e-3   # production emulator's own median KL (REPORT.md 2026-07-03)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load(name):
    p = RES / name / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def zkey(gk):        # "z0.2_main" -> "0.2"
    return gk.split("_")[0][1:]


def jsd_series(summary, xval_of):
    """{z: (x[], jsd[], floor[], floor_half)} for main-arm groups."""
    out = {}
    for gk, g in summary.items():
        if not gk.endswith("_main"):
            continue
        xs, js, fl = [], [], []
        for cname, d in g["configs"].items():
            x = xval_of(cname, d)
            if x is None:
                continue
            xs.append(x); js.append(d["jsd"]); fl.append(d["floor_pred"])
        order = np.argsort(xs)
        out[zkey(gk)] = (np.array(xs)[order], np.array(js)[order],
                         np.array(fl)[order])
    return out


# -------------------------------------------------- bootstrap excess (2026-07-13)
# Added so the MC panels *show* convergence: (a) each curve is extended to the
# truth's own x, where JSD = the finite-sample floor by construction (open
# marker), and (b) a floor-subtracted excess panel with multinomial-bootstrap
# CIs makes "consistent with zero" visible. Uses the cached per-config lnmu
# samples + the group's stored shared bin edges — no new MC.
def _jsd_from_counts(cp, cq, w):
    eps = 1e-300
    p = np.clip(cp / np.maximum(cp.sum(), 1), eps, None) / w
    q = np.clip(cq / np.maximum(cq.sum(), 1), eps, None) / w
    p = p / (p * w).sum(); q = q / (q * w).sum()
    m = 0.5 * (p + q)
    return 0.5 * float((p * np.log(p / m) * w).sum()
                       + (q * np.log(q / m) * w).sum())


def excess_series(name, summary, xval_of, B=300, seed=20260713):
    """{z: (x[], excess[], lo[], hi[])}: floor-subtracted JSD excess with
    68% multinomial-bootstrap CIs, from the cached samples."""
    rng = np.random.default_rng(seed)
    outdir = RES / name
    out = {}
    for gk, g in summary.items():
        if not gk.endswith("_main"):
            continue
        z = gk.split("_")[0][1:]
        edges = np.array(g["edges"])
        w = np.diff(edges)
        tA = outdir / f"z{z}_main_truthA.npy"
        tB = outdir / f"z{z}_main_truthB.npy"
        if not (tA.exists() and tB.exists()):
            continue
        tr = np.concatenate([np.load(tA), np.load(tB)])
        tr = tr[np.isfinite(tr)]
        ct = np.histogram(tr, bins=edges)[0]
        pt = ct / ct.sum()
        xs, ex, lo, hi = [], [], [], []
        for cname, d in g["configs"].items():
            x = xval_of(cname, d)
            if x is None:
                continue
            f = outdir / f"z{z}_main_{cname}.npy"
            if not f.exists():
                continue
            s = np.load(f); s = s[np.isfinite(s)]
            cc = np.histogram(s, bins=edges)[0]
            pc = cc / cc.sum()
            bs = np.empty(B)
            for b in range(B):
                rc = rng.multinomial(cc.sum(), pc)
                rt = rng.multinomial(ct.sum(), pt)
                bs[b] = _jsd_from_counts(rc, rt, w)
            e = bs - d["floor_pred"]
            xs.append(x)
            ex.append(max(d["jsd"] - d["floor_pred"], 0.0))
            lo.append(np.percentile(e, 16)); hi.append(np.percentile(e, 84))
        order = np.argsort(xs)
        out[z] = tuple(np.array(v)[order] for v in (xs, ex, lo, hi))
    return out


# ---------------------------------------------------------------- Mmin figure
def fig_mmin():
    plain = load("mmin_convergence")
    pd_ = load("mmin_pd_convergence")
    ana = np.load(RES / "convergence_analytic" / "analytic_arms.npz")
    if plain is None:
        return

    def xval(cname, d):
        if cname == "truth" or cname.startswith("nm"):
            return None
        return d["kwargs"]["Mmin"]

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.6))

    ax = axes[0]
    for src, ls, m, lw in ((plain, "-", "o", 1.6), (pd_, "--", "s", 1.2)):
        if src is None:
            continue
        for z, (xs, js, fl) in jsd_series(src, xval).items():
            # extend the curve to the truth's own Mmin (1e4), where
            # JSD = floor by construction (open marker)
            xs_e = np.concatenate([[1e4], xs])
            js_e = np.concatenate([[fl[0]], js])
            ax.plot(xs_e, js_e, ls, marker=m, ms=4, lw=lw, color=C[z],
                    label=(f"z_s={z}" if ls == "-" else None),
                    markevery=list(range(1, len(xs_e))))
            ax.plot([1e4], [fl[0]], m, ms=5.5, mfc="white", mec=C[z], mew=1.2)
            if ls == "-":
                ax.hlines(fl[0], 1e4, xs[-1], color=C[z], ls=":", lw=0.9)
    ax.axhline(EMU_KL, color=GRAY, lw=1.0)
    ax.text(1.3e4, EMU_KL * 1.15, "emulator median KL", color=GRAY, fontsize=7.5)
    ax.axvline(1e7, color=GRAY, lw=0.8, alpha=0.5)
    ax.text(1e7, 0.03, " default", color=GRAY, fontsize=7.5, rotation=90,
            transform=ax.get_xaxis_transform(), va="bottom")
    ax.set(xscale="log", yscale="log", xlabel=r"$M_{\min}$  [$M_\odot$]",
           ylabel="JSD to Mmin=1e4 truth  [nats]",
           title="P(ln$\\mu$) error vs $M_{\\min}$  (solid NM=100, dashed 10/decade;\n"
                 "dotted = floor; open marker at 1e4 = truth's own floor)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax = axes[1]
    for src, name, ls, m, lw in ((plain, "mmin_convergence", "-", "o", 1.6),
                                 (pd_, "mmin_pd_convergence", "--", "s", 1.2)):
        if src is None:
            continue
        for z, (xs, ex, lo, hi) in excess_series(name, src, xval).items():
            xs_e = np.concatenate([[1e4], xs])
            ex_e = np.concatenate([[0.0], ex])
            ax.plot(xs_e, ex_e, ls, marker=m, ms=4, lw=lw, color=C[z],
                    label=(f"z_s={z}" if ls == "-" else None),
                    markevery=list(range(1, len(xs_e))))
            ax.plot([1e4], [0.0], m, ms=5.5, mfc="white", mec=C[z], mew=1.2)
            if ls == "-":
                ax.fill_between(xs, lo, hi, color=C[z], alpha=0.15, lw=0)
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.axhline(EMU_KL, color=GRAY, lw=1.0)
    ax.text(1.3e4, EMU_KL * 1.3, "emulator median KL", color=GRAY, fontsize=7.5)
    ax.axvline(1e7, color=GRAY, lw=0.8, alpha=0.5)
    ax.set(xscale="log", yscale="symlog", xlabel=r"$M_{\min}$  [$M_\odot$]",
           ylabel="floor-subtracted JSD excess  [nats]",
           title="Convergence: excess $\\to$ 0 at the truth\n"
                 "(band: 68% bootstrap, solid axis only)")
    ax.set_yscale("symlog", linthresh=1e-4, linscale=0.6)
    ax.set_ylim(-5e-5, 1.2e-2)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax = axes[2]
    mg = ana["mmin_grid"]
    for z in ("0.2", "1.0", "5.0", "10.0"):
        v = ana[f"sigW_mmin_z{z}"] ** 2
        ref = v[np.searchsorted(mg, 1e4)]
        lab = z.rstrip("0").rstrip(".") if z != "10.0" else "10"
        ax.plot(mg, 1 - v / ref, "-", lw=1.6, color=C[lab], marker="o", ms=3)
        ax.annotate(f"z_s={lab}", (mg[-1], 1 - v[-1] / ref), xytext=(4, 0),
                    textcoords="offset points", color=C[lab], fontsize=7.5,
                    va="center")
    ax.axvline(1e7, color=GRAY, lw=0.8, alpha=0.5)
    ax.text(1e7, 0.55, " default", color=GRAY, fontsize=7.5, rotation=90,
            transform=ax.get_xaxis_transform(), va="bottom")
    ax.axvspan(mg[0], 1e6, color=GRAY, alpha=0.08, lw=0)
    ax.text(1.5e3, 0.02, "c(M) extrapolated\n(model territory)", fontsize=7,
            color=GRAY)
    ax.set(xscale="log", xlabel=r"$M_{\min}$  [$M_\odot$]",
           ylabel=r"missing $\sigma^2_W$ fraction vs $M_{\min}$=1e4",
           title="Analytic arm: background-variance deficit")
    fig.tight_layout()
    fig.savefig(PLOTS / "mmin_convergence.png", bbox_inches="tight")
    print("wrote plots/mmin_convergence.png")


# ---------------------------------------------------------------- Nz figure
def fig_nz():
    ana = np.load(RES / "convergence_analytic" / "analytic_arms.npz")
    mc = load("nz_convergence")

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.6))

    ax = axes[0]
    if mc is not None:
        def xval(cname, d):
            return None if cname == "truth" else d["kwargs"]["Nz"]
        for z, (xs, js, fl) in jsd_series(mc, xval).items():
            xs_e = np.concatenate([xs, [400]])
            js_e = np.concatenate([js, [fl[0]]])
            ax.plot(xs_e, js_e, "-", marker="o", ms=4, lw=1.6, color=C[z],
                    label=f"z_s={z}", markevery=list(range(len(xs))))
            ax.plot([400], [fl[0]], "o", ms=5.5, mfc="white", mec=C[z], mew=1.2)
            ax.hlines(fl[0], xs[0], 400, color=C[z], ls=":", lw=0.9)
        ax.axhline(EMU_KL, color=GRAY, lw=1.0)
        ax.text(26, EMU_KL * 1.15, "emulator median KL", color=GRAY, fontsize=7.5)
        ax.axvline(100, color=GRAY, lw=0.8, alpha=0.5)
        ax.text(100, 0.03, " default", color=GRAY, fontsize=7.5, rotation=90,
                transform=ax.get_xaxis_transform(), va="bottom")
        ax.set(xscale="log", yscale="log", xlabel=r"$N_z$",
               ylabel=f"JSD to Nz=400 truth  [nats]",
               title="P(ln$\\mu$) error vs $N_z$  (dotted = floor;\n"
                     "open marker at 400 = truth's own floor)")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    else:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "MC arm pending", ha="center", transform=ax.transAxes)

    ax = axes[1]
    if mc is not None:
        for z, (xs, ex, lo, hi) in excess_series("nz_convergence", mc,
                                                 xval).items():
            xs_e = np.concatenate([xs, [400]])
            ex_e = np.concatenate([ex, [0.0]])
            ax.plot(xs_e, ex_e, "-", marker="o", ms=4, lw=1.6, color=C[z],
                    label=f"z_s={z}", markevery=list(range(len(xs))))
            ax.plot([400], [0.0], "o", ms=5.5, mfc="white", mec=C[z], mew=1.2)
            ax.fill_between(xs, lo, hi, color=C[z], alpha=0.15, lw=0)
        ax.axhline(0, color=GRAY, lw=0.8)
        ax.axhline(EMU_KL, color=GRAY, lw=1.0)
        ax.text(26, EMU_KL * 1.3, "emulator median KL", color=GRAY, fontsize=7.5)
        ax.axvline(100, color=GRAY, lw=0.8, alpha=0.5)
        ax.set(xscale="log", xlabel=r"$N_z$",
               ylabel="floor-subtracted JSD excess  [nats]",
               title="Convergence: excess $\\to$ 0 at the truth\n"
                     "(band: 68% bootstrap)")
        ax.set_yscale("symlog", linthresh=1e-4, linscale=0.6)
        ax.set_ylim(-5e-5, 1.2e-2)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    else:
        ax.set_axis_off()

    ax = axes[2]
    nz = ana["nz_grid"]
    for z in ("0.2", "1.0", "5.0", "10.0"):
        lab = z.rstrip("0").rstrip(".") if z != "10.0" else "10"
        s = ana[f"sigW_nz_rulekthr_z{z}"]
        f = ana[f"sigW_nz_fixkthr_z{z}"]
        ax.plot(nz, np.abs(s / s[-1] - 1), "-", marker="o", ms=3.5, lw=1.6,
                color=C[lab], label=f"z_s={lab}")
        ax.plot(nz, np.abs(f / f[-1] - 1), "--", marker="s", ms=3, lw=1.0,
                color=C[lab])
    ax.axvline(100, color=GRAY, lw=0.8, alpha=0.5)
    ax.text(100, 0.03, " default", color=GRAY, fontsize=7.5, rotation=90,
            transform=ax.get_xaxis_transform(), va="bottom")
    ax.set(xscale="log", yscale="log", xlabel=r"$N_z$",
           ylabel=r"$|\sigma_W/\sigma_W(N_z{=}1600) - 1|$",
           title="Quadrature arm: solid = rule's own $\\kappa_{thr}(N_z)$,\n"
                 "dashed = frozen $\\kappa_{thr}$")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(PLOTS / "nz_convergence.png", bbox_inches="tight")
    print("wrote plots/nz_convergence.png")


# ---------------------------------------------------------------- ACE figure
def fig_ace():
    p = RES / "mmin_convergence" / "ace_gap_vs_mmin.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    mg = np.array(d["mmin_grid"])
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharex=True)
    for ax, mom, name in ((axes[0], "K2clip", r"$\langle\kappa^2\rangle$ (ACE-support clipped)"),
                          (axes[1], "K3clip", r"$\langle\kappa^3\rangle$ (ACE-support clipped)")):
        for z in d["zs"]:
            arr = np.array(d[f"{mom}_z{z:g}"], float)   # (nMmin, nseeds)
            m, s = arr.mean(1), arr.std(1)
            lab = f"{z:g}"
            amom = mom.replace("clip", "")
            ax.errorbar(mg, m, yerr=s, fmt="-o", ms=4, lw=1.6, capsize=2,
                        color=C[lab], label=f"Vaskonen z_s={lab}")
            ax.axhline(d[f"ace_{amom}_z{z:g}"], color=C[lab], ls="--", lw=1.2)
            ax.text(mg[-1], d[f"ace_{amom}_z{z:g}"] * 0.88, f"ACE z_s={lab}",
                    color=C[lab], fontsize=7, ha="right")
        ax.axvline(1e7, color=GRAY, lw=0.8, alpha=0.5)
        ax.set(xscale="log", yscale="log", xlabel=r"$M_{\min}$  [$M_\odot$]",
               ylabel=name, title=f"{name}: halo model vs ACE (dashed)",
               xlim=(5e3, 2.5e9))
        ax.legend(frameon=False, fontsize=7.5, loc="center left")
    fig.tight_layout()
    fig.savefig(PLOTS / "ace_gap_vs_mmin.png", bbox_inches="tight")
    print("wrote plots/ace_gap_vs_mmin.png")


if __name__ == "__main__":
    fig_mmin()
    fig_nz()
    fig_ace()
