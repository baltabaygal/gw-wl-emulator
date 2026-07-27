#!/usr/bin/env python
"""
Companion to plot_fig_subhalo_sigma_vs_psimin.py (Fig 5), with the axes swapped:
here the convergence threshold kappa_thr is on the x-axis and the resolved-clump mass
floor psi_min is HELD FIXED at the production value psi_min = m_floor / M = 1e7 Msun / M.

kappa_thr is exactly the SAME kind of threshold used to count host / field halos: it is
the impact-parameter cut, rmax = radius where the host convergence drops to kappa_thr,
out to which the halo (here, its substructure) is tallied along a line of sight (for
field halos this is what fixes <N> ~ 1/kappa_thr). It need not equal the value used for
the host halos.

y-axis -- the halo-style quantity. A naive per-ray MEAN scatter <sigma_kappa>_aperture
FALLS as kappa_thr -> 0, but that is a pure aperture-normalization artifact: a smaller
kappa_thr enlarges rmax (up to ~95 r200 at 1e-6) so the average is taken over an ever-
larger, mostly-empty disk (sigma_mean ~ 1/rmax). The quantity that actually enters the
line-of-sight variance -- exactly as for the counted field halos -- is the aperture-
INTEGRATED variance, int Var(kappa|y) 2 pi y dy = pi rmax^2 <Var>_aperture. We plot its
square root normalized to the host's own r200 footprint,

    sigma_kappa^LOS = sqrt( int_0^rmax Var(kappa|y) 2 pi y dy / (pi r200^2) )
                    = (rmax / r200) * sigma_kappa^mean,

i.e. the effective convergence scatter this host+substructure contributes per r200-sized
beam. This is FLAT in kappa_thr over 1e-6..~5e-3 (threshold-insensitive, mirroring the
psi_min-insensitivity of Fig 5) and only rolls off once rmax < r200 and the aperture
starts clipping the substructure.

Curves (each the (rmax/r200)-scaled area-weighted RMS):
  sigma_{kappa,total}^LOS  host + resolved clumps (carve-correlated)
  sigma_{kappa,sub}^LOS    resolved clump population alone (Campbell 2nd moment)
  sigma_{kappa,host}^LOS   carved smooth host alone (first-order carve response)

Output: paper_prod/plots/figures/fig_subhalo_sigma_vs_kappathr.{png,pdf}
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
from playground.dgate.subhalo_factor_dgate_area_scan import host_rmax
from scripts.subhalo_gate.subhalo_factor_proxy_check import nfw_params

OUT_DIR = REPO / "paper_prod" / "plots" / "figures"

# ---- fixed host + fixed floor; sweep the counting threshold ----------------
HOST_MASS, Z_LENS, Z_SOURCE = 1.0e14, 0.5, 1.0
PRODUCTION_M_FLOOR = 1.0e7
PSI_MIN_FIXED = PRODUCTION_M_FLOOR / HOST_MASS      # = 1e-7, held fixed
KAPPA_THR_FID = 1.276e-4                            # fiducial counting threshold

_, _, _, R200 = nfw_params(HOST_MASS, Z_LENS)

# host peak convergence ~0.061 for this host; stay well below it
KAPPA_THRS = np.logspace(-6, np.log10(0.03), 55)

# aperture-integrated RMS normalized to the host r200 footprint:
#   sigma^LOS = (rmax/r200) * sigma^mean   (see module docstring)
sig_host, sig_sub, sig_tot = [], [], []
for kt in KAPPA_THRS:
    _, rows = run(HOST_MASS, Z_LENS, Z_SOURCE, kt, np.array([PSI_MIN_FIXED]))
    f = host_rmax(HOST_MASS, Z_LENS, Z_SOURCE, kt) / R200
    sig_host.append(f * rows[0]["sigma_host"])
    sig_sub.append(f * rows[0]["sigma_sub"])
    sig_tot.append(f * rows[0]["sigma_total"])
sig_host = np.array(sig_host)
sig_sub = np.array(sig_sub)
sig_tot = np.array(sig_tot)

# ---- figure ----------------------------------------------------------------
apply_style()
guard_broken_latex()
import matplotlib.pyplot as plt

# same palette as fig_subhalo_sigma_vs_psimin: total = black, host = C1, sub = C0
C_TOT, C_HOST, C_SUB = 'k', 'C1', 'C0'
OUT_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=FIGURE_SIZES["single"])
fig.subplots_adjust(**SUBPLOTS_ADJUST["single"])

ax.plot(KAPPA_THRS, sig_tot, color=C_TOT, lw=1.5, label=r'$\sigma_{\kappa,\rm total}^{\rm LOS}$')
ax.plot(KAPPA_THRS, sig_sub, color=C_SUB, lw=1.3, ls=':', label=r'$\sigma_{\kappa,\rm sub}^{\rm LOS}$')
ax.plot(KAPPA_THRS, sig_host, color=C_HOST, lw=1.3, ls='--', label=r'$\sigma_{\kappa,\rm host}^{\rm LOS}$')

# fiducial counting threshold
ax.axvline(KAPPA_THR_FID, color='0.5', lw=0.8, ls=(0, (1, 1)))
ax.text(KAPPA_THR_FID * 1.4, sig_tot.max() * 0.30,
        r'$\kappa_{\rm thr}^{\rm fid}$', fontsize=5.5, color='0.4',
        va='center', ha='left', rotation=90)

# host configuration + fixed floor, upper-left corner
_mexp = int(round(np.log10(HOST_MASS)))
info = (rf'$M=10^{{{_mexp}}}\,M_\odot$' + '\n'
        + rf'$z_l={Z_LENS:g},\ z_s={Z_SOURCE:g}$')
ax.text(0.03, 0.55, info, transform=ax.transAxes, ha='left', va='top',
        fontsize=6.5, linespacing=1.4)

ax.set_xscale('log')
ax.set_xlim(KAPPA_THRS.min(), KAPPA_THRS.max())
ax.set_ylim(0, sig_sub.max() * 1.18)
ax.set_xlabel(r'$\kappa_{\rm thr}$')
ax.set_ylabel(r'$\sigma_\kappa^{\rm LOS}$  (per $r_{200}$ beam)')
format_log_axis_decimal(ax, axis='x')
ax.legend(fontsize=6.5, loc='lower center', ncol=3, columnspacing=1.2,
          handlelength=1.7, borderaxespad=0.6)

out_png = OUT_DIR / 'fig_subhalo_sigma_vs_kappathr.png'
out_pdf = out_png.with_suffix('.pdf')
fig.savefig(out_png, dpi=300, facecolor='white')
fig.savefig(out_pdf, facecolor='white')
print(f"wrote {out_pdf.relative_to(REPO)}")
print(f"wrote {out_png.relative_to(REPO)}")

# console summary
i_fid = int(np.argmin(np.abs(KAPPA_THRS - KAPPA_THR_FID)))
print(f"\nhost M={HOST_MASS:.0e}  z_l={Z_LENS}  z_s={Z_SOURCE}  "
      f"psi_min={PSI_MIN_FIXED:.1e} (fixed)")
print(f"at fiducial kappa_thr={KAPPA_THR_FID:.3e}: sig_tot^LOS={sig_tot[i_fid]:.4e} "
      f"(host={sig_host[i_fid]:.4e}, sub={sig_sub[i_fid]:.4e})")
# plateau flatness over the counting-threshold range (before the rmax<r200 rolloff)
lo = np.interp(np.log10(1e-6), np.log10(KAPPA_THRS), sig_tot)
hi = np.interp(np.log10(3e-3), np.log10(KAPPA_THRS), sig_tot)
print(f"plateau flatness sig_tot^LOS(1e-6)/sig_tot^LOS(3e-3) = {lo/hi:.4f}")
