#!/usr/bin/env python3
"""Convergence + cost study to choose flat-kappa_thr vs fixed-<N> threshold rule.

For z_s in {1, 5}: sweep several flat kappa_thr, measure
  - JSD(ln mu) to the lowest-threshold reference run (convergence toward truth),
  - wall-clock sampling time,
  - expected explicit-halo count <N>.
Overlays where the legacy fixed-<N>=100 threshold falls for each z_s.

Outputs plots/kappathr_decision.{png,pdf} and data/results/kappathr_decision.npz.
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo / "build"))
import gwlensing as gw  # noqa: E402

H, OM, S8 = 0.674, 0.315, 0.811
SEED = 12345
NS = 80000

# (z_s, [kappa_thr grid, first entry = reference/truth], nsamples)
CASES = {
    1.0: [1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3],
    5.0: [1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3],
}


def density(lnmu, edges):
    c, _ = np.histogram(lnmu, bins=edges)
    w = np.diff(edges); t = c.sum()
    return c / (t * w) if t else np.zeros_like(w, float)


def jsd(p, q, w):
    eps = 1e-300
    p = np.clip(p, eps, None); p /= (p * w).sum()
    q = np.clip(q, eps, None); q /= (q * w).sum()
    m = 0.5 * (p + q)
    return 0.5 * ((p * np.log(p / m) * w).sum() + (q * np.log(q / m) * w).sum())


def run(z, kthr, ns):
    t0 = time.perf_counter()
    d = gw.sample_lnmu_ml_with_diagnostics(
        z=z, h=H, OmegaM=OM, sigma8=S8, nsamples=ns, seed=SEED,
        strict_weak_lensing=False, subhalo=False, kappathr_flat=kthr)
    dt = time.perf_counter() - t0
    return np.asarray(d["lnmu"]), dt


results = {}
for z, grid in CASES.items():
    print(f"=== z_s={z} ===", flush=True)
    samples, times = {}, {}
    for k in grid:
        lnmu, dt = run(z, k, NS)
        samples[k] = lnmu; times[k] = dt
        print(f"  kthr={k:.1e}  <N>={gw.get_expected_halo_count(z,H,OM,S8,k):7.1f}"
              f"  t={dt:6.2f}s", flush=True)
    ref = grid[0]
    all_ln = np.concatenate(list(samples.values()))
    lo, hi = np.quantile(all_ln, 1e-4), np.quantile(all_ln, 1 - 1e-4)
    pad = 0.08 * (hi - lo)
    edges = np.linspace(lo - pad, hi + pad, 121)
    w = np.diff(edges)
    pref = density(samples[ref], edges)
    js = {k: jsd(pref, density(v, edges), w) for k, v in samples.items()}
    q999 = {k: float(np.quantile(v, 0.999)) for k, v in samples.items()}
    Nexp = {k: gw.get_expected_halo_count(z, H, OM, S8, k) for k in grid}
    legacy_kthr = gw.get_kappa_threshold(z, H, OM, S8, 100)
    results[z] = dict(grid=grid, ref=ref, js=js, q999=q999, Nexp=Nexp,
                      times=times, legacy_kthr=legacy_kthr)

# ---------- plot ----------
ink, muted, base, grid_c = "#0b0b0b", "#6d6a61", "#c9c5b8", "#e6e2d7"
blue, red, green = "#2a78d6", "#d94b3d", "#2e8b57"
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "axes.edgecolor": base, "font.size": 10.5})
zorder = sorted(results)
colz = {zorder[0]: blue, zorder[1]: red}

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

# (1) <N> vs z_s for the three rules across the full z range
axN = axes[0]
zline = np.array([0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
for kf, lab, c in [(1e-4, r"flat $\kappa_{\rm thr}=10^{-4}$", green),
                   (1e-3, r"flat $\kappa_{\rm thr}=10^{-3}$", red)]:
    Ns = [gw.get_expected_halo_count(z, H, OM, S8, kf) for z in zline]
    axN.plot(zline, Ns, "o-", color=c, lw=1.9, label=lab)
axN.axhline(100, color=blue, lw=1.9, ls="--", label=r"legacy fixed-$\langle N\rangle=100$")
axN.set_xscale("log"); axN.set_yscale("log")
axN.set_xlabel(r"$z_s$"); axN.set_ylabel(r"explicit halos $\langle N\rangle$ per LOS")
axN.set_title("Cost / allocation")
axN.grid(color=grid_c, lw=0.7, which="both"); axN.set_axisbelow(True)
axN.legend(frameon=False, fontsize=8.8, loc="upper left")

# (2) convergence: JSD-to-reference vs kappa_thr
axJ = axes[1]
for z in zorder:
    r = results[z]
    ks = np.array(r["grid"]); js = np.array([r["js"][k] for k in r["grid"]])
    m = js > 0
    axJ.plot(ks[m], js[m], "o-", color=colz[z], lw=1.9, label=rf"$z_s={z:g}$")
    axJ.axvline(r["legacy_kthr"], color=colz[z], lw=1.3, ls=":")
axJ.set_xscale("log"); axJ.set_yscale("log")
axJ.set_xlabel(r"flat $\kappa_{\rm thr}$")
axJ.set_ylabel(r"JSD($\ln\mu$) to $\kappa_{\rm thr}\!\to\!0$ ref")
axJ.set_title("Convergence toward truth\n(dotted = legacy $\\langle N\\rangle{=}100$ threshold)")
axJ.grid(color=grid_c, lw=0.7, which="both"); axJ.set_axisbelow(True)
axJ.legend(frameon=False, fontsize=9.3)

# (3) wall time vs <N>
axT = axes[2]
for z in zorder:
    r = results[z]
    Ns = np.array([r["Nexp"][k] for k in r["grid"]])
    ts = np.array([r["times"][k] for k in r["grid"]])
    axT.plot(Ns, ts, "o-", color=colz[z], lw=1.9, label=rf"$z_s={z:g}$")
axT.set_xscale("log"); axT.set_yscale("log")
axT.set_xlabel(r"explicit halos $\langle N\rangle$")
axT.set_ylabel(rf"wall time / {NS:,} realizations [s]")
axT.set_title("Sampling cost")
axT.grid(color=grid_c, lw=0.7, which="both"); axT.set_axisbelow(True)
axT.legend(frameon=False, fontsize=9.3)

fig.suptitle(r"Threshold rule: flat $\kappa_{\rm thr}$ vs fixed-$\langle N\rangle$ "
             "(subhalo off, same seed)", y=1.02, fontsize=13)
fig.tight_layout()
out = repo / "plots" / "kappathr_decision.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
fig.savefig(repo / "plots" / "kappathr_decision.pdf", bbox_inches="tight")
print("wrote", out)

np.savez(repo / "data" / "results" / "kappathr_decision.npz",
         summary_json=np.array(json.dumps(
             {str(z): {"legacy_kthr": results[z]["legacy_kthr"],
                       "js": {f"{k:.1e}": results[z]["js"][k] for k in results[z]["grid"]},
                       "Nexp": {f"{k:.1e}": results[z]["Nexp"][k] for k in results[z]["grid"]},
                       "times": {f"{k:.1e}": results[z]["times"][k] for k in results[z]["grid"]}}
              for z in results})))
print(json.dumps({str(z): {"legacy_kthr": results[z]["legacy_kthr"],
                           "js": {f"{k:.1e}": round(results[z]["js"][k], 6) for k in results[z]["grid"]}}
                  for z in results}, indent=2))
