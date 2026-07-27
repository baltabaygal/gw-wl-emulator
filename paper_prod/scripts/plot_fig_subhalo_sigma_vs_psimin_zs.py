#!/usr/bin/env python
"""
Multi-z_s companion to plot_fig_subhalo_sigma_vs_psimin.py.

Same figure (sigma_kappa decomposition vs. resolved floor psi_min for the
resolved-only, no-kappa_u scheme, subhalo_model=4), but repeated as one panel
per source redshift z_s to show how the plateau amplitude and the floor-
insensitivity evolve with z_s. The host (M, z_l) and the convergence threshold
that sets the aperture rmax are held fixed; only z_s changes panel to panel.

Curves per panel (area-weighted over ray impact parameter y):
  sigma_{kappa,total}  host + resolved clumps (carve-correlated)
  sigma_{kappa,sub}    resolved clump population alone (Campbell 2nd moment)
  sigma_{kappa,host}   carved smooth host alone (first-order carve response)

Output: paper_prod/plots/figures/fig_subhalo_sigma_vs_psimin_zs.{png,pdf}
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from paper_prod.plot_style import (apply_style, format_log_axis_decimal)
from plot_fig_subhalo_population import guard_broken_latex
from playground.analytic.plot_sigma_vs_psimin import run

OUT_DIR = REPO / "paper_prod" / "plots" / "figures"

# ---- fixed host + sweep grid; only z_s varies ------------------------------
HOST_MASS, Z_LENS = 1.0e14, 0.5
KAPPA_THR_HOST = 1.276e-4                 # aperture convergence threshold (fixed)
PRODUCTION_M_FLOOR = 1.0e7
PSI_PROD = PRODUCTION_M_FLOOR / HOST_MASS  # = 1e-7
PSI_MINS = np.logspace(-8, 0.2, 47)

# z_s must exceed the lens redshift z_l=0.5 (a source at the lens plane is not
# lensed and Sigma_crit diverges), so the lowest sampled source is 0.7.
Z_SOURCES = [0.7, 1.0, 2.0, 5.0, 10.0]

# same palette as the single-panel figure: total = black, host = C1, sub = C0
C_TOT, C_HOST, C_SUB = 'k', 'C1', 'C0'

# ---- compute all panels ----------------------------------------------------
results = []
for z_s in Z_SOURCES:
    meta, rows = run(HOST_MASS, Z_LENS, z_s, KAPPA_THR_HOST, PSI_MINS)
    results.append({
        "z_s": z_s,
        "psi_min": np.array([r["psi_min"] for r in rows]),
        "sig_host": np.array([r["sigma_host"] for r in rows]),
        "sig_sub": np.array([r["sigma_sub"] for r in rows]),
        "sig_tot": np.array([r["sigma_total"] for r in rows]),
    })

# ---- figure: 2 rows x 3 cols (5 panels + shared legend in the 6th) ---------
apply_style()
guard_broken_latex()
import matplotlib.pyplot as plt

OUT_DIR.mkdir(parents=True, exist_ok=True)
nrow, ncol = 2, 3
fig, axes = plt.subplots(nrow, ncol, figsize=(7.1, 4.4))
fig.subplots_adjust(left=0.09, right=0.98, bottom=0.11, top=0.95,
                    wspace=0.30, hspace=0.32)
axes_flat = axes.flatten()

handles_ref = None
for ax, res in zip(axes_flat, results):
    psi_min = res["psi_min"]
    ln_tot, = ax.plot(psi_min, res["sig_tot"], color=C_TOT, lw=1.5,
                      label=r'$\sigma_{\kappa,\rm total}$')
    ln_sub, = ax.plot(psi_min, res["sig_sub"], color=C_SUB, lw=1.3, ls=':',
                      label=r'$\sigma_{\kappa,\rm sub}$')
    ln_host, = ax.plot(psi_min, res["sig_host"], color=C_HOST, lw=1.3, ls='--',
                       label=r'$\sigma_{\kappa,\rm host}$')
    handles_ref = [ln_tot, ln_sub, ln_host]

    ax.axvline(PSI_PROD, color='0.5', lw=0.8, ls=(0, (1, 1)))

    ax.set_xscale('log')
    ax.set_xlim(psi_min.min(), psi_min.max())
    ymax = res["sig_tot"].max()
    ax.set_ylim(0, ymax * 1.18 if ymax > 0 else 1.0)
    format_log_axis_decimal(ax, axis='x')

    # per-panel z_s label
    ax.text(0.05, 0.94, rf'$z_s={res["z_s"]:g}$', transform=ax.transAxes,
            ha='left', va='top', fontsize=8)
    # use offset-style tick labels for the small y numbers
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 3))
    ax.yaxis.get_offset_text().set_fontsize(5.5)

# shared x / y labels
fig.supxlabel(r'$\psi_{\rm min}$', fontsize=9, y=0.02)
fig.supylabel(r'$\sigma_\kappa$', fontsize=9, x=0.01)

# 6th cell: host info + shared legend
ax_leg = axes_flat[-1]
ax_leg.axis('off')
_mexp = int(round(np.log10(HOST_MASS)))
info = (rf'$M=10^{{{_mexp}}}\,M_\odot$' + '\n'
        + rf'$z_l={Z_LENS:g}$' + '\n'
        + r'$m_{\rm floor}=10^{7}\,M_\odot$' + '\n'
        + r'(dotted grey line)')
ax_leg.text(0.5, 0.72, info, transform=ax_leg.transAxes, ha='center', va='center',
            fontsize=7.5, linespacing=1.5)
ax_leg.legend(handles=handles_ref, fontsize=8.5, loc='center',
              bbox_to_anchor=(0.5, 0.30), handlelength=1.8, frameon=False)

out_png = OUT_DIR / 'fig_subhalo_sigma_vs_psimin_zs.png'
out_pdf = out_png.with_suffix('.pdf')
fig.savefig(out_png, dpi=300, facecolor='white')
fig.savefig(out_pdf, facecolor='white')
print(f"wrote {out_pdf.relative_to(REPO)}")
print(f"wrote {out_png.relative_to(REPO)}")

# console summary
for res in results:
    i_prod = int(np.argmin(np.abs(res["psi_min"] - PSI_PROD)))
    print(f"z_s={res['z_s']:>4g}: sig_tot(prod)={res['sig_tot'][i_prod]:.4e}  "
          f"host={res['sig_host'][i_prod]:.4e}  sub={res['sig_sub'][i_prod]:.4e}")
