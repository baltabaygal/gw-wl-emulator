#!/usr/bin/env python
"""
Diagnostic: at FIXED psi_min, how does the substructure convergence perturbation respond
to a per-subhalo threshold kappa_thr,sub -- i.e. dropping every clump whose own
convergence AT THE RAY, kappa_i = kappa_NFW(m_i, d_i), falls below the cut?

This is deliberately NOT Fig 5. There the knob is psi_min, the resolved MASS floor, and
the kappa axis is only a relabeling of it. Here the clump population is frozen at the
production floor psi_min = m_floor/M -- every subhalo is always sampled -- and the cut
acts on each clump's actual convergence at the ray, which depends on impact parameter as
well as mass. The two do not commute: a massive clump far from the ray is dropped while a
light clump the ray nearly hits survives.

That makes kappa_thr,sub the SAME KIND OF OBJECT as the host counting threshold: the
r_thr table in cpp/subhalo.cpp is exactly "keep the object while its kappa at the ray
exceeds kappa_thr". So the two thresholds can be compared directly, which the psi_min
version could not do. It also removes the cusp ambiguity of the psi_min top axis -- a
clump's r=0 convergence is log-divergent and grid-defined, its convergence at the actual
separation is not.

Y-AXIS CHOICE (2026-07-27). The earlier version of this figure plotted the bare
sigma_kappa, area-weighted over the whole aperture. That is uninterpretable. Its absolute
size (~1e-3) is set as much by the aperture rmax_host = 8.2 r200 as by the substructure,
and it has no reference to be compared against, because with the host mass and position
fixed the smooth host carries no scatter at all. Dividing the two area-weighted scalars is
worse than useless -- sigma_kappa/<kappa_nosub> = 1.35 -- since the rms is dominated by
the inner r200 while the aperture mean is dominated by the outskirts.

The fix is to drop the aperture average and work at ONE fixed impact parameter y, where
the ratio

    sigma_kappa(y) / kappa_smooth(y)

is well posed: numerator and denominator live at the same y, so no weighting enters, and
the number reads directly as the fractional convergence perturbation substructure imprints
on a ray passing at y. kappa_smooth is the convergence of the SAME host with no
substructure at all, so this is the "compared to no subhalos" reference the old y-axis
lacked. The three components are the same ones as before -- total, subhalo sum, and the
carved-host response -- each divided by that one reference.

The fixed radius is Y_OVER_R200 below. At the default y = 0.5 r200 the fully-populated
perturbation is ~50%. It is a strong function of y (~9% at 0.1 r200, peaking near 70% at
0.7 r200 where the smooth profile has fallen but the clumps have not, collapsing past the
edge of the subhalo population), so the chosen radius must be stated in the caption.

Key finding (chat 2026-07-24, unchanged): the perturbation is essentially flat across the
range where the retained clump count collapses. At the host threshold kappa_thr = 1.28e-4
only ~8 clumps per sightline survive out of ~3.4e4, yet the perturbation is unchanged to
four digits. The subhalo scatter is carried entirely by rare close encounters. Note this
is an inside-r200 statement -- outside the subhalo population the same threshold already
removes ~40% of the perturbation, which the old area-weighted curve hid. The retained
counts are printed to the console rather than plotted.

Engine (exact Campbell quadrature, no MC): playground/analytic/sigma_vs_subkappathr.py.
Output: paper_prod/plots/figures/fig_subhalo_sigma_vs_subkappathr.{png,pdf}
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from paper_prod.plot_style import (apply_style, FIGURE_SIZES, SUBPLOTS_ADJUST,
                                   format_log_axis_decimal)
from plot_fig_subhalo_population import guard_broken_latex
from playground.analytic.sigma_vs_subkappathr import run_kthr

OUT_DIR = REPO / "paper_prod" / "plots" / "figures"

# ---- fixed host, fixed clump population, fixed ray; sweep only the convergence cut ----
HOST_MASS, Z_LENS, Z_SOURCE = 1.0e14, 0.5, 1.0
KAPPA_THR_HOST = 1.276e-4                  # host aperture threshold, for comparison
PRODUCTION_M_FLOOR = 1.0e7                 # psi_min = 1e7/1e14 = 1e-7, held fixed
Y_OVER_R200 = 0.5                          # the one impact parameter the figure is drawn at

KTHR = np.logspace(-6, 0.0, 56)
meta, rows = run_kthr(HOST_MASS, Z_LENS, Z_SOURCE, KAPPA_THR_HOST,
                      np.concatenate([[0.0], KTHR]),
                      m_floor=PRODUCTION_M_FLOOR, n_d=2400,
                      y_probe_r200=[Y_OVER_R200])

base_row, rows = rows[0], rows[1:]
kthr = np.array([r["kappa_thr_sub"] for r in rows])

# the no-substructure host at this ray -- the reference everything is divided by
k_smooth = float(meta["kappa_nosub_probe"][0])
y_kpc = float(meta["y_probe"][0])

sig_tot = np.array([r["probe"]["sigma_total"][0] for r in rows]) / k_smooth
sig_sub = np.array([r["probe"]["sigma_sub"][0] for r in rows]) / k_smooth
sig_host = np.array([r["probe"]["sigma_host"][0] for r in rows]) / k_smooth
n_ret = np.array([r["probe"]["n_retained"][0] for r in rows])
base = base_row["probe"]["sigma_total"][0] / k_smooth

# ---- figure --------------------------------------------------------------------
apply_style()
guard_broken_latex()
import matplotlib.pyplot as plt

C_TOT, C_HOST, C_SUB = 'k', 'C1', 'C0'
OUT_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=FIGURE_SIZES["single"])
fig.subplots_adjust(**SUBPLOTS_ADJUST["single"])

ax.plot(kthr, sig_tot, color=C_TOT, lw=1.5, label=r'$\sigma_{\kappa,\rm total}$')
ax.plot(kthr, sig_sub, color=C_SUB, lw=1.3, ls=':', label=r'$\sigma_{\kappa,\rm sub}$')
ax.plot(kthr, sig_host, color=C_HOST, lw=1.3, ls='--', label=r'$\sigma_{\kappa,\rm host}$')

# the threshold the HOST halos already use -- deep in the plateau
ax.axvline(KAPPA_THR_HOST, color='0.5', lw=0.8, ls=(0, (1, 1)))
ax.text(KAPPA_THR_HOST * 1.25, base * 0.66, r'host $\kappa_{\rm thr}$',
        fontsize=5.5, color='0.4', va='center', ha='left', rotation=90)

info = (rf'$M=10^{{{int(round(np.log10(HOST_MASS)))}}}\,M_\odot$' + '\n'
        + rf'$z_l={Z_LENS:g},\ z_s={Z_SOURCE:g}$' + '\n'
        + rf'$y={Y_OVER_R200:g}\,r_{{200}}$')
ax.text(0.03, 0.33, info, transform=ax.transAxes, ha='left', va='bottom',
        fontsize=6.0, linespacing=1.4)

ax.set_xscale('log')
ax.set_xlim(kthr.min(), kthr.max())
ax.set_ylim(0, max(sig_sub.max(), base) * 1.15)
ax.set_xlabel(r'$\kappa_{\rm thr,\,sub}$')
ax.set_ylabel(r'$\sigma_\kappa\,/\,\kappa_{\rm smooth}$')
format_log_axis_decimal(ax, axis='x')
ax.legend(fontsize=6.5, loc='upper right', handlelength=1.7, borderaxespad=0.6)

out_png = OUT_DIR / 'fig_subhalo_sigma_vs_subkappathr.png'
out_pdf = out_png.with_suffix('.pdf')
fig.savefig(out_png, dpi=300, facecolor='white')
fig.savefig(out_pdf, facecolor='white')
print(f"wrote {out_pdf.relative_to(REPO)}")
print(f"wrote {out_png.relative_to(REPO)}")

# ---- console summary ------------------------------------------------------------
print(f"\nhost r200={meta['r200_host']:.4g} kpc; ray at y={Y_OVER_R200:g} r200 "
      f"= {y_kpc:.4g} kpc")
print(f"no-substructure reference kappa_smooth(y) = {k_smooth:.4e}")
print(f"fully populated (no cut): sigma_tot/kappa_smooth={base:.4f}, "
      f"<N>={base_row['probe']['n_retained'][0]:.4g} clumps on this ray, "
      f"<kappa_sub>={base_row['probe']['kappa_sub'][0]:.4e}")
f = sig_tot / base
for frac in (0.99, 0.95, 0.90, 0.50):
    j = np.where(f <= frac)[0]
    if len(j):
        i = j[0]
        print(f"  perturbation -> {frac:.0%} of baseline at kappa_thr,sub={kthr[i]:.3g} "
              f"(<N_ret>={n_ret[i]:.3g})")
i_h = int(np.argmin(np.abs(kthr - KAPPA_THR_HOST)))
print(f"  at the HOST threshold {KAPPA_THR_HOST:.3e}: sigma_tot/kappa_smooth="
      f"{sig_tot[i_h]:.4f} ({f[i_h]:.4f} of baseline), <N_ret>={n_ret[i_h]:.3g} "
      f"(a {base_row['probe']['n_retained'][0] / n_ret[i_h]:.3g}x cut in clump count)")
print(f"  mean: <kappa_tot>/kappa_smooth "
      f"{base_row['probe']['kappa_total'][0] / k_smooth:.4f} -> "
      f"{rows[i_h]['probe']['kappa_total'][0] / k_smooth:.4f}")
