#!/usr/bin/env python
"""
Fig (subhalo subsection): convergence scatter sigma_kappa of a single lensing halo
under the simplified production model (subhalo_model = 4) as a function of the
resolved-clump mass floor psi_min = m_floor / M.

Companion to fig_subhalo_sigma_decomposition (which shows sigma_kappa vs. host-centric
impact parameter r at the fixed production floor). Here we instead sweep the floor:
everything with psi >= psi_min is drawn as an explicit clump (contributing to the
resolved sum Sum_i kappa_i), the smooth host is carved to M - Sum_i m_i (exact
per-realization mass conservation), and nothing fills in psi < psi_min -- i.e. the
"resolved-only, no kappa_u" scheme. The point of the figure: sigma_kappa is FLAT across
many decades of psi_min and the production floor (m_floor = 1e7 Msun) sits deep in that
plateau, so the result is insensitive to the exact floor.

Curves (all area-weighted over the ray impact parameter y across the aperture):
  sigma_kappa            total halo scatter (host + resolved clumps, carve-correlated)
  sigma_{kappa,host}     carved smooth host alone (first-order dkappa/dM carve response)
  sigma_{kappa,sub}      resolved clump population alone (Campbell 2nd moment)
Note sigma_total < sigma_sub because the mass-conserving carve makes host and clumps
ANTI-correlated: Var_tot = Var_host + Var_sub + 2 Cov, Cov < 0 (a heavier realized
clump mass carves the host down). Confirmed against production C++ at 25/25 r-points
(playground/subhalo_single_host_probe.cpp; CLAUDE.md 2026-07-23).

Top axis translates psi_min to the equivalent peak (r=0) convergence of a single clump
sitting exactly at the resolved floor, 2 kappa0(psi_min * M).

Physics: exact-quadrature Campbell moments (no MC), same validated machinery
(<~1.6% vs C++, docs/subhalo/subhalo_combining.md) reused from
playground/analytic/plot_sigma_vs_psimin.py.

Output: paper_prod/plots/figures/fig_subhalo_sigma_vs_psimin.{png,pdf}
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from paper_prod.plot_style import (apply_style, FIGURE_SIZES, SUBPLOTS_ADJUST,
                                   format_log_axis_decimal)
from plot_fig_subhalo_population import guard_broken_latex
from playground.analytic.plot_sigma_vs_psimin import run

OUT_DIR = REPO / "paper_prod" / "plots" / "figures"

# ---- representative host + sweep grid --------------------------------------
HOST_MASS, Z_LENS, Z_SOURCE = 1.0e14, 0.5, 1.0
KAPPA_THR_HOST = 1.276e-4
PRODUCTION_M_FLOOR = 1.0e7
PSI_PROD = PRODUCTION_M_FLOOR / HOST_MASS          # = 1e-7

# sweep just past psi_max = 1 so the collapse-to-zero is visible without dead space
PSI_MINS = np.logspace(-8, 0.2, 47)

meta, rows = run(HOST_MASS, Z_LENS, Z_SOURCE, KAPPA_THR_HOST, PSI_MINS)

psi_min = np.array([r["psi_min"] for r in rows])
sig_host = np.array([r["sigma_host"] for r in rows])
sig_sub = np.array([r["sigma_sub"] for r in rows])
sig_tot = np.array([r["sigma_total"] for r in rows])

# ---- figure ----------------------------------------------------------------
apply_style()
guard_broken_latex()
import matplotlib.pyplot as plt

# same palette as fig_subhalo_sigma_decomposition: total = black, host = C1, sub = C0
C_TOT, C_HOST, C_SUB = 'k', 'C1', 'C0'
OUT_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=FIGURE_SIZES["single"])
fig.subplots_adjust(**SUBPLOTS_ADJUST["single"])

ax.plot(psi_min, sig_tot, color=C_TOT, lw=1.5, label=r'$\sigma_{\kappa,\rm total}$')
ax.plot(psi_min, sig_sub, color=C_SUB, lw=1.3, ls=':', label=r'$\sigma_{\kappa,\rm sub}$')
ax.plot(psi_min, sig_host, color=C_HOST, lw=1.3, ls='--', label=r'$\sigma_{\kappa,\rm host}$')

ax.axvline(PSI_PROD, color='0.5', lw=0.8, ls=(0, (1, 1)))
ax.text(PSI_PROD * 1.7, sig_tot.max() * 0.62, r'$m_{\rm floor}=10^{7}\,M_\odot$',
        fontsize=5.5, color='0.4', va='center', ha='left', rotation=90)

# host configuration, top-right (empty corner once the curves have dropped)
_mexp = int(round(np.log10(HOST_MASS)))
info = (rf'$M=10^{{{_mexp}}}\,M_\odot$' + '\n'
        + rf'$z_l={Z_LENS:g},\ z_s={Z_SOURCE:g}$')
ax.text(0.97, 0.96, info, transform=ax.transAxes, ha='right', va='top',
        fontsize=6.5, linespacing=1.4)

ax.set_xscale('log')
ax.set_xlim(psi_min.min(), psi_min.max())
ax.set_ylim(0, sig_tot.max() * 1.18)
ax.set_xlabel(r'$\psi_{\rm min}$')
ax.set_ylabel(r'$\sigma_\kappa$')
format_log_axis_decimal(ax, axis='x')
ax.legend(fontsize=6.5, loc='lower left', handlelength=1.7, borderaxespad=0.5)

out_png = OUT_DIR / 'fig_subhalo_sigma_vs_psimin.png'
out_pdf = out_png.with_suffix('.pdf')
fig.savefig(out_png, dpi=300, facecolor='white')
fig.savefig(out_pdf, facecolor='white')
print(f"wrote {out_pdf.relative_to(REPO)}")
print(f"wrote {out_png.relative_to(REPO)}")

# console summary
i_prod = int(np.argmin(np.abs(psi_min - PSI_PROD)))
print(f"\nhost M={HOST_MASS:.0e}  z_l={Z_LENS}  z_s={Z_SOURCE}")
print(f"at production psi_min={PSI_PROD:.1e}: sig_tot={sig_tot[i_prod]:.4e} "
      f"(host={sig_host[i_prod]:.4e}, sub={sig_sub[i_prod]:.4e})")
print(f"plateau flatness sig_tot(1e-8)/sig_tot(1e-3) = "
      f"{np.interp(np.log10(1e-8), np.log10(psi_min), sig_tot) / np.interp(np.log10(1e-3), np.log10(psi_min), sig_tot):.4f}")
