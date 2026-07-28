"""
Fig: the model-5 per-clump threshold across the whole source-redshift range.

The companion figure fig_subhalo_sigma_ratio_vs_subkappathr establishes the threshold at
z_s = 1. This one answers the obvious follow-up -- does the same threshold still throw
nothing away at other source redshifts? -- over z_s in [0.2, 10].

LEFT: the threshold response at every z_s. x is the production knob
mult = kappa_thr,sub / kappa_thr(z_s) (`subhalo_kappathr_factor`, default 0.1), y is the
fraction of the substructure convergence scatter that survives the cut. The curves are
ordered in z_s but the ordering is mild: the knee sits near mult ~ 10 at every redshift,
one to two decades above the default.

RIGHT: the two numbers that matter, read off at the default. The loss rises monotonically
with z_s but stays under a quarter of a percent of the SUBSTRUCTURE scatter everywhere;
since substructure carries ~10% of Var(kappa) at z_s = 1, the loss on the total is another
order of magnitude smaller. The rendered clump budget stays near 10^2 per sightline while
the full population it stands in for falls from 1.7e6 to 3.2e5, i.e. a 3e3 to 2.4e4
reduction across the range.

WHAT IS AND IS NOT PREDICTED HERE. These are pure substructure quantities the population
Campbell quadrature computes exactly, so no MC enters. The BOOST AMPLITUDE
sigma_ON/sigma_OFF is deliberately absent: it needs the host + smooth-field variance,
which the MC measures on a kappa <= 1 clipped estimator while the Campbell integral is
unclipped, and CLAUDE.md item 13 warns those differ by tens of percent in a way that
worsens with z_s. Plotting an analytic boost here would manufacture a z_s trend out of an
estimator mismatch. The measured boost per z_s comes from
scripts/convergence/subhalo_kappathr_zs_mc.py.

Host weights: production-engine dumps at z_s = 0.5/1/5, the validated Python replica
(playground/analytic/host_weights_py.py, gated to 0.02% on NhfNFW and the per-cell
weights) elsewhere. The three engine points are marked so the reader can see they sit on
the same trend as the replica ones.

Input:  playground/analytic/subkappathr_zs.json
Output: paper_prod/plots/figures/fig_subhalo_subkappathr_vs_zs.{png,pdf}
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "paper_prod" / "scripts"))

from paper_prod.plot_style import (apply_style, FIGURE_SIZES,  # noqa: E402
                                   format_log_axis_decimal)
from plot_fig_subhalo_population import guard_broken_latex  # noqa: E402

OUT_DIR = REPO / "paper_prod" / "plots" / "figures"
IN_JSON = REPO / "playground" / "analytic" / "subkappathr_zs.json"
MULT_DEFAULT = 0.1

res = json.loads(IN_JSON.read_text())["results"]
zs_all = np.array(sorted(float(k) for k in res))

apply_style()
guard_broken_latex()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib import cm  # noqa: E402

OUT_DIR.mkdir(parents=True, exist_ok=True)
W, H = FIGURE_SIZES["double"]
fig, (ax, bx) = plt.subplots(1, 2, figsize=(W, 2.95))
fig.subplots_adjust(left=0.085, right=0.915, bottom=0.175, top=0.945, wspace=0.42)

norm = mcolors.LogNorm(vmin=zs_all.min(), vmax=zs_all.max())
cmap = cm.viridis

# ---------------------------------------------------------------------------------
# left: threshold response at every z_s
# ---------------------------------------------------------------------------------
for zs in zs_all:
    r = res[str(zs)]
    m = np.array(r["mults"])
    s = np.array(r["sigma_sub_ratio"])
    good = m > 0
    ax.plot(m[good], s[good], color=cmap(norm(zs)), lw=1.1, alpha=0.95)

ax.axvline(MULT_DEFAULT, color='C3', lw=0.9, alpha=0.8)
ax.text(MULT_DEFAULT / 1.6, 0.06, r'production default', fontsize=6, color='C3',
        rotation=90, va='bottom', ha='right')
ax.axhline(1.0, color='0.45', lw=0.8, ls=(0, (1, 1.6)))

ax.set_xscale('log')
ax.set_xlim(1e-4, 1e4)
ax.set_ylim(0.0, 1.06)
ax.set_xlabel(r'$\kappa_{\rm thr,\,sub}/\kappa_{\rm thr}(z_s)$')
ax.set_ylabel(r'$\sigma_{\rm sub}(\kappa_{\rm thr,\,sub})\,/\,\sigma_{\rm sub}(0)$')
format_log_axis_decimal(ax, axis='x')

sm = cm.ScalarMappable(norm=norm, cmap=cmap)
cb = fig.colorbar(sm, ax=ax, pad=0.03, aspect=26)
cb.set_label(r'$z_s$', fontsize=8)
cb.ax.tick_params(labelsize=7)
cb.ax.minorticks_off()
cb.set_ticks([0.2, 0.5, 1, 2, 5, 10])
cb.set_ticklabels(['0.2', '0.5', '1', '2', '5', '10'])

# ---------------------------------------------------------------------------------
# right: the two numbers at the default, vs z_s
# ---------------------------------------------------------------------------------
def at_mult(r, mult, key):
    m = np.array(r["mults"])
    good = m > 0
    return float(np.interp(np.log(mult), np.log(m[good]), np.array(r[key])[good]))


loss_def = np.array([100 * (1 - at_mult(res[str(z)], MULT_DEFAULT, "sigma_sub_ratio"))
                     for z in zs_all])
loss_one = np.array([100 * (1 - at_mult(res[str(z)], 1.0, "sigma_sub_ratio"))
                     for z in zs_all])
clumps_def = np.array([np.exp(np.interp(
    np.log(MULT_DEFAULT),
    np.log(np.array(res[str(z)]["mults"])[1:]),
    np.log(np.array(res[str(z)]["clumps_per_ray"])[1:]))) for z in zs_all])
clumps_full = np.array([res[str(z)]["clumps_per_ray"][0] for z in zs_all])
is_engine = np.array([res[str(z)]["weight_source"] == "engine_probe" for z in zs_all])

bx.plot(zs_all, loss_def, '-', color='C0', lw=1.4, zorder=3,
        label=r'default, $0.1\,\kappa_{\rm thr}$')
bx.plot(zs_all[is_engine], loss_def[is_engine], 'o', ms=4.2, mfc='none',
        mec='C0', mew=1.0, zorder=4)
bx.plot(zs_all, loss_one, '--', color='C0', lw=1.0, alpha=0.65, zorder=3,
        label=r'$1.0\,\kappa_{\rm thr}$')
bx.set_xscale('log')
bx.set_yscale('log')          # the default curve spans 0.06-0.24%, the 1.0 curve 10x that
bx.set_xlim(0.18, 11.5)
bx.set_ylim(0.04, 4.0)
bx.set_xlabel(r'source redshift $z_s$')
bx.set_ylabel(r'$\sigma_{\kappa,\rm sub}$ lost  [\%]' if plt.rcParams['text.usetex']
              else r'$\sigma_{\kappa,\rm sub}$ lost  [%]', color='C0')
bx.tick_params(axis='y', colors='C0')
format_log_axis_decimal(bx, axis='x')
format_log_axis_decimal(bx, axis='y')
bx.legend(fontsize=6.2, loc='upper left', handlelength=1.8, borderaxespad=0.5,
          labelspacing=0.35)
bx.text(0.035, 0.62, 'open symbols: engine host weights,\nline: validated Python replica',
        transform=bx.transAxes, ha='left', va='top', fontsize=5.6, color='0.45',
        linespacing=1.4)

cx = bx.twinx()
cx.plot(zs_all, clumps_def, '-', color='C2', lw=1.4)
cx.plot(zs_all, clumps_full, ':', color='C2', lw=1.0, alpha=0.8)
cx.set_yscale('log')
cx.set_ylim(20, 2e8)
cx.set_ylabel(r'clumps per ray', color='C2')
cx.tick_params(axis='y', colors='C2')
cx.minorticks_off()
cx.set_yticks([1e2, 1e3, 1e4, 1e5, 1e6])
cx.annotate('all subhalos', xy=(10.6, clumps_full[-1] * 1.5), fontsize=6, color='C2',
            ha='right', va='bottom')
cx.annotate('rendered', xy=(10.6, clumps_def[-1] * 1.7), fontsize=6, color='C2',
            ha='right', va='bottom')

out_png = OUT_DIR / 'fig_subhalo_subkappathr_vs_zs.png'
fig.savefig(out_png, dpi=300, facecolor='white')
fig.savefig(out_png.with_suffix('.pdf'), facecolor='white')
print(f"wrote {out_png.with_suffix('.pdf').relative_to(REPO)}")
print(f"wrote {out_png.relative_to(REPO)}")

# ---------------------------------------------------------------------------------
# console summary
# ---------------------------------------------------------------------------------
print(f"\n{'z_s':>6} {'kappa_thr':>11} {'weights':>15} {'all clumps':>12} "
      f"{'rendered':>9} {'reduction':>10} {'loss@0.1':>9} {'loss@1.0':>9}")
for i, z in enumerate(zs_all):
    r = res[str(z)]
    print(f"{z:6.2f} {r['kappa_thr']:11.4e} {r['weight_source']:>15} "
          f"{clumps_full[i]:12.4g} {clumps_def[i]:9.4g} "
          f"{clumps_full[i] / clumps_def[i]:10.4g} {loss_def[i]:8.3f}% {loss_one[i]:8.3f}%")

# where is the knee? the mult at which 1% / 5% of sigma_sub is lost
print("\nknee location (mult at which the stated fraction of sigma_sub is lost):")
for z in zs_all:
    r = res[str(z)]
    m = np.array(r["mults"])[1:]
    s = np.array(r["sigma_sub_ratio"])[1:]
    row = []
    for frac in (0.01, 0.05, 0.5):
        j = np.where(s <= 1 - frac)[0]
        row.append(f"{frac:.0%}: {m[j[0]]:.3g}" if len(j) else f"{frac:.0%}: >{m[-1]:.0g}")
    print(f"  z_s={z:<6g} " + "   ".join(row))

print(f"\nmonotonic in z_s: loss {'YES' if np.all(np.diff(loss_def) > 0) else 'NO'}, "
      f"rendered {'YES' if np.all(np.diff(clumps_def) > 0) else 'NO'}")
print(f"loss at the default spans {loss_def.min():.3f}% (z_s={zs_all[0]:g}) to "
      f"{loss_def.max():.3f}% (z_s={zs_all[-1]:g})")
