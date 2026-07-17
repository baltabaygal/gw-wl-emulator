"""
Proof figure: why subhalo_factor must be ~1e-5 even though the implied clump
kappa-threshold (~1.3e-9) looks absurdly small.

Three panels, all at the fiducial single-host configuration of the closure
analysis (host M = 1e13 Msun, z_l = 0.5, z_s = 1, kappa_thr_host = 1.276e-4):

A. kappa(d) of a single NFW clump vs ray-clump distance d. Shows kappa ~ m/d^2,
   so the SAME 1e7 Msun clump produces kappa = 2e-4 (> the HOST threshold) at
   d = 1 kpc and 1.3e-9 at 453 kpc. The clump threshold is evaluated at the
   host-center distance r, not at the clump's actual distance d, so it must
   price the worst case (clump lands on the ray).

B. One brute clump catalog (matched realization), ray at r = 300 kpc from the
   host center, production gate at subhalo_factor = 1e-4: every clump plotted
   at (d, m). The proxy gate keeps m >= m_res(r) -- a horizontal line blind to
   d. The truth boundary kappa(m, d) = kappa_thr,clump is the diagonal. Clumps
   in the wedge between them are dropped despite sitting next to the ray with
   individually large kappa. At f = 1e-5 the horizontal line falls below the
   1e7 hard floor and the wedge empties (gate == brute).

C. The observable: population-aggregated bias of the substructure kappa-variance
   excess vs subhalo_factor (playground/subhalo_factor_population_zs*.json).
   Only f = 1e-5 keeps |bias| < 1% for every source redshift.

Physics is imported from scripts/subhalo_factor_proxy_check.py (validated
against the C++ to <1%). Output: plots/subhalo_factor_proof.{png,pdf}.
"""
from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/Users/baltabay/Desktop/gw-wl-emulator")
sys.path.insert(0, str(ROOT / "scripts"))

import subhalo_factor_proxy_check as P  # noqa: E402

P.sigma_crit = functools.lru_cache(maxsize=None)(P.sigma_crit)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ----------------------------------------------------------------------------
# palette (validated: dataviz reference palette, light surface)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUES3 = ["#86b6ef", "#2a78d6", "#104281"]            # ordinal: m = 1e7, 1e8, 1e9
BLUES4 = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # ordinal: z_s = 0.5, 1, 2, 5
KEEP, DROP_BAD = "#2a78d6", "#e34948"

# ----------------------------------------------------------------------------
# fiducial configuration (matches the closure analysis)
HOST_M = 1.0e13
ZL, ZS = 0.5, 1.0
KTHR_HOST = 1.276e-4
M_FLOOR = 1.0e7
R_RAY = 300.0
F_SHOW = 1.0e-4          # the gate factor panel B dissects
FACTORS_A = [1.0e-4, 1.0e-5]


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


# ============================================================================
# Panel A data: kappa(d) per clump mass
d_grid = np.logspace(-1, 3.5, 300)
masses_a = [1.0e7, 1.0e8, 1.0e9]
kappa_curves = {
    m: np.array([P.kappa_nfw(m, ZL, ZS, d) for d in d_grid]) for m in masses_a
}
reach_1e7 = {f: P.reach_radius(1.0e7, ZL, ZS, f * KTHR_HOST) for f in FACTORS_A}

# ============================================================================
# Panel B data: one matched catalog under the f = 1e-4 proxy-r gate
rs_h, rhos_h, c_h, r200_h = P.nfw_params(HOST_M, ZL)
nt, _zf = P.n_tau(HOST_M, ZL)
fs = 0.3563 / nt**0.6 - 0.075
gamma = P.gamma_norm(fs)

real = P.sample_brute_realization(1, HOST_M, ZL, M_FLOOR, gamma, r200_h, c_h)
mass = np.asarray(real["mass"], dtype=float)
d_clump = np.sqrt((np.asarray(real["x"]) - R_RAY) ** 2 + np.asarray(real["y"]) ** 2)

kthr_sub = F_SHOW * KTHR_HOST
m_grid = np.logspace(np.log10(M_FLOOR), 12.0, 200)
reach_grid = np.array([P.reach_radius(m, ZL, ZS, kthr_sub) for m in m_grid])
reach_of_m = lambda mm: np.interp(np.log(mm), np.log(m_grid), reach_grid)  # noqa: E731

reach = reach_of_m(mass)
proxy_keep = reach >= R_RAY
truth_keep = reach >= d_clump
missed = (~proxy_keep) & truth_keep

kap = np.array([P.kappa_nfw(m, ZL, ZS, max(dd, 1e-9)) for m, dd in zip(mass, d_clump)])
worst = int(np.argmax(np.where(missed, kap, -np.inf)))
k2_missed_frac = kap[missed] @ kap[missed] / max(kap[truth_keep] @ kap[truth_keep], 1e-300)

# proxy floor at this ray and gate boundaries
m_res_proxy = float(np.exp(np.interp(R_RAY, reach_grid, np.log(m_grid))))
d_bound = np.logspace(np.log10(3.0), np.log10(1500.0), 120)
m_truth_bound = np.exp(np.interp(d_bound, reach_grid, np.log(m_grid)))

print(f"panel B: n={len(mass)}  proxy keeps {proxy_keep.sum()}  "
      f"missed {missed.sum()} (kappa^2 missed/truth = {100 * k2_missed_frac:.1f}%)")
print(f"  proxy floor m_res({R_RAY:.0f} kpc; f={F_SHOW:.0e}) = {m_res_proxy:.2e} Msun")
print(f"  worst missed clump: m={mass[worst]:.2e} Msun at d={d_clump[worst]:.1f} kpc, "
      f"kappa={kap[worst]:.2e}  ({kap[worst] / kthr_sub:.0f}x the clump threshold)")

# ============================================================================
# Panel C data: population bias vs factor per z_s
pop = {}
for zs_tag in ["0.5", "1", "2", "5"]:
    with open(ROOT / f"playground/subhalo_factor_population_zs{zs_tag}.json") as fh:
        rows = json.load(fh)["rows"]
    pop[zs_tag] = (
        np.array([r["subhalo_factor"] for r in rows]),
        np.array([r["bias_pct"] for r in rows]),
    )

# ============================================================================
# figure
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), facecolor=SURFACE)
fig.subplots_adjust(left=0.06, right=0.985, top=0.82, bottom=0.14, wspace=0.32)

# --- Panel A -----------------------------------------------------------------
ax = axes[0]
style_axis(ax)
for m, color in zip(masses_a, BLUES3):
    ax.loglog(d_grid, kappa_curves[m], color=color, lw=2)
    if m > 1.5e7:
        ax.annotate(f"$m=10^{{{int(np.log10(m))}}}\\,M_\\odot$",
                    xy=(3200, kappa_curves[m][-1]), xytext=(0, 6),
                    textcoords="offset points", ha="right",
                    color=SECONDARY, fontsize=8.5)
ax.annotate("$m=10^{7}\\,M_\\odot$", xy=(170, 4.5e-8), ha="left",
            color=SECONDARY, fontsize=8.5)

ax.axhline(KTHR_HOST, color=MUTED, lw=1, ls=(0, (5, 3)))
ax.annotate("host threshold $\\kappa_{\\rm thr,host}$", xy=(2800, KTHR_HOST),
            xytext=(0, 4), textcoords="offset points", ha="right",
            color=SECONDARY, fontsize=8)
thr_label_x = {1.0e-4: 0.13, 1.0e-5: 0.13}
for f, ls in zip(FACTORS_A, [(0, (5, 3)), (0, (2, 2))]):
    kt = f * KTHR_HOST
    ax.axhline(kt, color=MUTED, lw=1, ls=ls)
    ax.annotate(f"clump threshold, $f=10^{{{int(np.log10(f))}}}$",
                xy=(thr_label_x[f], kt), xytext=(0, -11),
                textcoords="offset points", color=SECONDARY, fontsize=8)
    r = reach_1e7[f]
    ax.plot([r], [kt], "o", ms=6, mfc=BLUES3[0], mec=SURFACE, mew=1.2, zorder=5)

k_at_1 = float(P.kappa_nfw(1e7, ZL, ZS, 1.0))
ax.plot([1.0], [k_at_1], "o", ms=6, mfc=BLUES3[0], mec=SURFACE, mew=1.2, zorder=5)
ax.annotate("same clump on the ray:\n$\\kappa=2\\times10^{-4}$ > host thr.",
            xy=(1.0, k_at_1), xytext=(9.0, 3e-3), color=INK, fontsize=8.5,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.annotate("reach at $f=10^{-5}$:\n453 kpc $\\approx$ host disk",
            xy=(reach_1e7[1e-5], 1e-5 * KTHR_HOST), xytext=(3.5, 2e-8),
            color=INK, fontsize=8.5,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.set_xlim(0.12, 3500)
ax.set_ylim(1e-10, 0.4)
ax.set_xlabel("distance from clump  $d$  [kpc]", color=SECONDARY, fontsize=9.5)
ax.set_ylabel("$\\kappa_{\\rm clump}(d)$", color=SECONDARY, fontsize=9.5)
ax.set_title("A — one clump: $\\kappa\\propto m/d^2$ spans 5 decades,\n"
             "so the threshold must price the worst-case $d$",
             color=INK, fontsize=10, loc="left", pad=10)

# --- Panel B -----------------------------------------------------------------
ax = axes[1]
style_axis(ax)
harmless = (~proxy_keep) & (~truth_keep)
ax.scatter(d_clump[harmless], mass[harmless], s=7, c=BASELINE, lw=0, alpha=0.55,
           label="dropped, negligible (both gates agree)")
ax.scatter(d_clump[proxy_keep], mass[proxy_keep], s=9, c=KEEP, lw=0, alpha=0.8,
           label="kept by production gate")
ax.scatter(d_clump[missed], mass[missed], s=16, c=DROP_BAD, lw=0,
           label="dropped by production gate, yet significant")
ax.plot(d_bound, m_truth_bound, color=INK, lw=1.4, ls=(0, (5, 3)))
ax.annotate("significance boundary  $\\kappa(m,d)=\\kappa_{\\rm thr,clump}$",
            xy=(4.5, 7.1e6), color=SECONDARY, fontsize=8.5, ha="left")
ax.axhline(m_res_proxy, color=DROP_BAD, lw=1.4)
ax.annotate("production floor — blind to $d$",
            xy=(0.5, m_res_proxy), xytext=(2, 5), textcoords="offset points",
            ha="left", color=DROP_BAD, fontsize=8.5)
ax.annotate(
    f"dropped clump at $d=0.9$ kpc:\n$\\kappa={kap[worst]:.0e}$ "
    f"($\\sim\\!2\\times10^4\\times$ the threshold)",
    xy=(d_clump[worst], mass[worst]), xytext=(1.1, 1.5e9),
    color=INK, fontsize=8.5,
    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.4, 2000)
ax.set_ylim(6e6, 3e11)
ax.set_xlabel("actual ray–clump distance  $d$  [kpc]", color=SECONDARY, fontsize=9.5)
ax.set_ylabel("clump mass  [$M_\\odot$]", color=SECONDARY, fontsize=9.5)
ax.set_title(f"B — matched catalog, ray at $r=300$ kpc, $f=10^{{-4}}$:\n"
             f"the $r$-keyed gate discards {100 * k2_missed_frac:.0f}% "
             "of this catalog's clump $\\kappa^2$",
             color=INK, fontsize=10, loc="left", pad=10)
leg = ax.legend(loc="upper left", fontsize=7.5, frameon=True, facecolor=SURFACE,
                edgecolor=GRID, borderpad=0.7, framealpha=0.95)
for t in leg.get_texts():
    t.set_color(SECONDARY)

# --- Panel C -----------------------------------------------------------------
ax = axes[2]
style_axis(ax)
ax.axhspan(-1.0, 1.0, color=GRID, alpha=0.6, lw=0)
ax.annotate("$\\pm1\\%$ tolerance", xy=(1.35e-5, 0.35), color=SECONDARY, fontsize=8.5)
ax.axhline(0.0, color=BASELINE, lw=1)
for (zs_tag, (fx, bias)), color in zip(pop.items(), BLUES4):
    ax.plot(fx, bias, color=color, lw=2, marker="o", ms=4.5,
            mec=SURFACE, mew=0.8, label=f"$z_s={zs_tag}$")
ax.axvline(1e-5, color=MUTED, lw=1, ls=(0, (2, 2)))
ax.annotate("production\ndefault", xy=(1e-5, -18.5), xytext=(6, 0),
            textcoords="offset points", ha="left", color=SECONDARY, fontsize=8.5)
leg_c = ax.legend(loc="center right", fontsize=8, frameon=True, facecolor=SURFACE,
                  edgecolor=GRID, borderpad=0.7, framealpha=0.95)
for t in leg_c.get_texts():
    t.set_color(SECONDARY)
ax.set_xscale("log")
ax.set_xlim(6e-6, 6e-3)
ax.set_ylim(-20, 2.5)
ax.set_xlabel("subhalo_factor", color=SECONDARY, fontsize=9.5)
ax.set_ylabel("bias of $\\sigma^2_{\\rm sub}$  [%]", color=SECONDARY, fontsize=9.5)
ax.set_title("C — the observable: population substructure\n"
             "$\\kappa$-variance bias vs brute force",
             color=INK, fontsize=10, loc="left", pad=10)

out_png = ROOT / "plots/subhalo_factor_proof.png"
fig.savefig(out_png, dpi=200, facecolor=SURFACE)
fig.savefig(out_png.with_suffix(".pdf"), facecolor=SURFACE)
print(f"wrote {out_png}")
