"""
Fig: what substructure does to sigma_kappa, and what the per-clump threshold costs.

Replaces the first cut of this figure (2026-07-27, 16:08), which plotted the bare MC
ratio sigma_ON/sigma_OFF with a LOESS trend line through it. That was misleading in a
specific way: the wiggles the smoother traced -- a dip near 1e-5, a bump at 2e-4 -- are
sampling noise on a single 3e5-ray realization, not features of the model, and drawing a
curve through them invited the reader to interpret them. This version keeps the same MC
data and replaces the smoother with (i) per-point error bars, (ii) the independent
analytic prediction for the SHAPE, and (iii) the variance decomposition, all on ONE axis.

SINGLE PANEL, DECOMPOSED BOOST (user decision 2026-07-27). Earlier drafts of this figure
carried a second panel -- first the rendered clump budget, then the variance shares of
Var(kappa_sub). Both were dropped because the decomposition can be shown in the units the
reader already cares about, on the same y axis as the measurement. Since
    sigma_ON^2/sigma_OFF^2 = 1 + Var_sub,tot/Var_OFF
and Var_sub,tot = Var_clumps + Var_host + 2Cov exactly (Campbell identity), the boost
itself decomposes: plotting sqrt(1 + A*Var_clumps/Var_tot(0)) and
sqrt(1 + A*(Var_clumps+Var_host)/Var_tot(0)) against the full prediction shows each
channel's contribution as a vertical offset in sigma_ON/sigma_OFF. The cross term is then
the visible gap between the clumps+host curve and the total -- it pulls the boost DOWN,
which a shares plot states but does not show.

WHY THIS FIGURE AND NOT fig_subhalo_sigma_vs_subkappathr
    That one is a single host (M=1e14, z_l=0.5) at a single impact parameter y=0.5 r200,
    where the substructure perturbation is ~50% of kappa_smooth. Its magnitude is a
    strong function of y (~9% at 0.1 r200, ~70% at 0.7 r200), so the y-axis is set by a
    hand-picked radius, and it largely duplicates Fig. 4 (the host/subhalo decomposition
    at fixed threshold vs r, same single-host Campbell). This figure integrates over the
    whole host population on a sightline, so nothing is hand-picked, and the number it
    reports (+5% on sigma_kappa) is the one that propagates to the magnification PDF.

THE MEASUREMENT
    MC: sigma_kappa with model-5 substructure over sigma_kappa with substructure off,
    z_s = 1, 3e5 rays per threshold, common random numbers along the threshold grid
    (playground/sigma_on_off_vs_subkappathr_model5_crn.json).

    Curve: the population-weighted Campbell prediction. The analytic calculation gives
    the substructure variance Var_sub(kappa_thr,sub) summed over the host population
    (playground/analytic/subkappathr_dense.json), hence the shape

        sigma_ON/sigma_OFF = c * sqrt(1 + A * s(kappa_thr,sub)^2),
        s = sigma_sub(kappa_thr,sub)/sigma_sub(0),

    with ONE fitted scalar and no fitted shape: A = Var_sub(0)/Var_OFF, the substructure
    amplitude, which the analytic side cannot supply because it does not model the
    host/smooth-field variance. c is not fitted; it is a calibration, see below.

    THE NULL REGION IS THE INSTRUMENT. At kappa_thr,sub >~ 0.4 the retained clump
    population is empty (<1e-7 clumps per ray), so the ratio is exactly 1 by construction
    -- model 5 with nothing rendered has Msum = 0, an uncarved host, and therefore exactly
    the subhalo-off physics (cpp/lensing.cpp:975). Two things follow.

    (1) Common mode. Every ratio divides by ONE sigma_OFF run, so a fluctuation in that
        single reference shifts all 35 points together; nothing in the model can produce
        a genuine deficit in the null region, and a leak would push up, not down. The
        null-region mean therefore measures the offset directly: c = 0.9965, i.e.
        sigma_OFF happened to come out 0.35% high. Its uncertainty relative to 1 is
        eps_null*sqrt(1/n_null + 1) = 0.0034, NOT the standard error on the mean of the
        null points -- the shared sigma_OFF error does not average down with n_null.
        (Using the SEM alone reads 2.7 sigma instead of the correct 1.0 sigma; that
        mistake was made and corrected on 2026-07-27.) User decision the same day: divide
        the offset out, so the null region and the prediction both land on the 1.0
        reference line and the plateau reads straight off the axis. The shaded band is
        the residual +-0.0034 on that calibration. THE CAPTION MUST SAY SO.

    (2) Independent error. The scatter of the null points is a direct empirical
        measurement of the ratio's noise. It is larger where substructure IS on (rendered
        clumps add tail variance, which makes sigma harder to estimate at fixed N), so
        the plotted error is eps(k) = sqrt(eps_null^2 + eps_extra^2 * s(k)^2), both terms
        measured from the residuals. chi2/dof is printed as a check that this is honest.

    Amplitude precision. The plateau is good to ~half a percent, not three digits: the
    independent 1e5-ray run refits to +5.8% against +5.1% here. Quote one digit. Note the
    two runs SHARE seed 20260727, so their sigma_OFF is the same draw and their agreement
    on c is not independent confirmation -- only the plateau comparison is.

BOTTOM PANEL -- the cost
    Expected clumps rendered per sightline, same population weighting. This is the axis
    that justifies the model-5 default: over the range where the top panel is flat, the
    clump budget falls by four orders of magnitude.

Inputs (both cached, no C++ needed):
    playground/sigma_on_off_vs_subkappathr_model5_crn.json   MC, z_s=1
    playground/analytic/subkappathr_dense.json               analytic, z_s=0.5/1/5
Output: paper_prod/plots/figures/fig_subhalo_sigma_ratio_vs_subkappathr.{png,pdf}
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "paper_prod" / "scripts"))

from paper_prod.plot_style import (apply_style, FIGURE_SIZES,  # noqa: E402
                                   format_log_axis_decimal)
from plot_fig_subhalo_population import guard_broken_latex  # noqa: E402

OUT_DIR = REPO / "paper_prod" / "plots" / "figures"
MC_JSON = REPO / "playground" / "sigma_on_off_vs_subkappathr_model5_crn.json"
AN_JSON = REPO / "playground" / "analytic" / "subkappathr_dense.json"
COMP_JSON = REPO / "playground" / "analytic" / "subkappathr_components.json"

Z_SOURCE = 1.0
KTHR_FACTOR_DEFAULT = 0.1          # production default subhalo_kappathr_factor
S_NULL = 0.05                      # s below this -> substructure is off by construction
S_ON = 0.95                        # s above this -> substructure fully populated

# ---------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------
mc = json.loads(MC_JSON.read_text())
assert mc["model"] == 5 and mc["cosmo"]["z"] == Z_SOURCE, "MC json is not the z_s=1 model-5 run"
k_mc = np.array([r["kthr_sub"] for r in mc["rows"]])
ratio = np.array([r["ratio"] for r in mc["rows"]])
N_RAY = mc["nsamples"]

an_all = json.loads(AN_JSON.read_text())["results"]
an = an_all[str(Z_SOURCE)]
comp = json.loads(COMP_JSON.read_text())
assert comp["z_source"] == Z_SOURCE and comp["kthr_sub"] == an["kthr_sub"], \
    "component sweep is on a different z_s or threshold grid than the dense sweep"
KAPPA_THR_HOST = an["kappa_thr"]
k_an = np.array(an["kthr_sub"][1:])                  # drop the kappa_thr,sub = 0 entry
s_an = np.array(an["sigma_sub_ratio"][1:])
clumps_an = np.array(an["clumps_per_ray"][1:])
CLUMPS_FULL = an["clumps_per_ray"][0]                # kappa_thr,sub = 0, whole population

s_at = np.interp(np.log(k_mc), np.log(k_an), s_an)   # analytic shape at the MC thresholds
clumps_at = np.exp(np.interp(np.log(k_mc), np.log(k_an), np.log(clumps_an + 1e-300)))

# ---------------------------------------------------------------------------------
# calibration: null region fixes c, the substructure-on points fix A
# ---------------------------------------------------------------------------------
null = s_at < S_NULL
on = s_at > S_ON
eps_null = float(ratio[null].std(ddof=1))

# c = the common-mode offset. Every ratio divides by ONE sigma_OFF run, so a fluctuation
# in that single reference moves all 35 points together. The null region is known to be
# exactly 1 (nothing rendered -> Msum = 0 -> uncarved host -> subhalo-off physics), so its
# mean measures that offset directly.
c_raw = float(ratio[null].mean())

# Uncertainty of c RELATIVE TO 1. NOTE this is not the standard error on the mean of the
# null points: the comparison against 1 also carries the error of the single shared
# sigma_OFF, which does not average down with the number of null points. Getting this
# wrong makes a 1-sigma fluctuation look like a 2.7-sigma one.
c_err = float(eps_null * np.sqrt(1.0 / null.sum() + 1.0))

# 2026-07-27 (user decision): divide the offset out rather than carry it in the fit, so
# the null region and the prediction both land on the 1.0 reference line and the plateau
# reads straight off the axis. The caption must state the calibration.
ratio = ratio / c_raw
c = 1.0

A = float(least_squares(lambda p: c * np.sqrt(1 + p[0] * s_at[~null] ** 2) - ratio[~null],
                        [0.11]).x[0])
model = c * np.sqrt(1 + A * s_at ** 2)
resid = ratio - model
eps_on = float(resid[on].std(ddof=1))
eps_extra = float(np.sqrt(max(eps_on ** 2 - eps_null ** 2, 0.0)))
err = np.sqrt(eps_null ** 2 + (eps_extra * s_at) ** 2)
chi2_dof = float(np.sum((resid / err) ** 2) / (len(k_mc) - 2))

PLATEAU = float(np.sqrt(1 + A))
K_DEFAULT = KTHR_FACTOR_DEFAULT * KAPPA_THR_HOST

# ---------------------------------------------------------------------------------
# decompose the boost. Var_sub,tot = Var_clumps + Var_host + 2Cov exactly, and the boost
# is sqrt(1 + A * Var_x / Var_tot(0)), so each channel maps onto the same y axis as the
# measurement. Normalizing by Var_tot(0) (a fixed number) rather than Var_tot(k) also
# avoids the underflow that makes the SHARES meaningless past kappa_thr,sub ~ 3.
# ---------------------------------------------------------------------------------
V0 = comp["var_tot"][0]
k_cmp = np.array(comp["kthr_sub"][1:])
q_clm = np.array(comp["var_clumps"][1:]) / V0
q_hst = np.array(comp["var_host"][1:]) / V0
q_tot = np.array(comp["var_tot"][1:]) / V0

# consistency: the component sweep's total must be the same object the dense sweep's
# sigma_sub_ratio encodes, otherwise the curves below are not comparable to the fit
_s_from_comp = np.sqrt(q_tot)
assert np.allclose(_s_from_comp, s_an, rtol=2e-3), \
    "component var_tot disagrees with the dense sweep's sigma_sub_ratio"


def boost(q):
    """sigma_ON/sigma_OFF implied by a substructure variance q * Var_tot(0)."""
    return c * np.sqrt(1.0 + A * q)

# ---------------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------------
apply_style()
guard_broken_latex()
import matplotlib.pyplot as plt  # noqa: E402

C_MC, C_AN, C_REF = 'C0', '#123b5c', '0.45'
OUT_DIR.mkdir(parents=True, exist_ok=True)

W, _ = FIGURE_SIZES["single"]
fig, ax = plt.subplots(figsize=(W, 3.05))
fig.subplots_adjust(left=0.185, right=0.965, bottom=0.145, top=0.975)

XLIM = (k_mc.min() / 1.6, k_mc.max() * 1.6)

# --- the measurement ---------------------------------------------------------------
ax.axhline(1.0, color=C_REF, lw=0.8, ls=(0, (1, 1.6)), zorder=1)

k_fine = np.logspace(np.log10(XLIM[0]), np.log10(XLIM[1]), 400)


def on_fine(arr):
    """Interpolate an analytic-grid quantity onto the plotting grid, flat outside."""
    return np.interp(np.log(k_fine), np.log(k_cmp), arr, left=arr[0], right=arr[-1])


m_fine = boost(on_fine(q_tot))
b_clm = boost(on_fine(q_clm))
b_ch = boost(on_fine(q_clm + q_hst))

# the cross term is the gap between "clumps + host" and the truth: it pulls the boost DOWN
jd = int(np.argmin(np.abs(np.log(k_cmp / K_DEFAULT))))
b_clm_d, b_ch_d, b_tot_d = (boost(q_clm[jd]), boost(q_clm[jd] + q_hst[jd]),
                            boost(q_tot[jd]))

# The percentages live in the legend labels rather than in a text block: the plateau is
# where every curve is flat, so a separate annotation just fights the curves for space.
# The two dashed/dotted curves are CUMULATIVE partial sums, not separate contributions.
# Labelling the orange one "+ carved host" invited the reading that the host contributes
# 5.9% on its own and therefore exceeds the 5.1% total, which is nonsense. It is
# clumps+host with the covariance still omitted, and it lies ABOVE the total precisely
# because 2Cov < 0: the clump sum and the host response anticorrelate (the carve conserves
# mass, so a ray with more clump mass nearby sees a lighter host), and adding the two
# variances without the cross term overestimates the boost. The labels now say so.
ax.fill_between(k_fine, m_fine, b_ch, color='0.55', alpha=0.20, lw=0, zorder=2,
                label=r'cross term $2\,\mathrm{Cov}<0$')
ax.plot(k_fine, b_clm, color='C0', lw=1.1, ls=(0, (1, 1.3)), zorder=3,
        label=rf'$\sum_i\kappa_i$ only: $+{100 * (b_clm_d - 1):.1f}\%$')
ax.plot(k_fine, b_ch, color='C1', lw=1.1, ls=(0, (4, 1.6)), zorder=3,
        label=rf'$+$ host, no cross: $+{100 * (b_ch_d - 1):.1f}\%$')

# The +-c_err calibration band is deliberately NOT drawn now that the MC points are off
# the canvas. Two shaded regions of similar weight read as one confused object, and the
# cross-term shading is the physics -- the calibration band was only there to show that
# the points were consistent with the curve. Its size is stated in the caption instead:
# the amplitude carries +-c_err = +-0.34% from the null-region normalisation.
ax.plot(k_fine, m_fine, color=C_AN, lw=1.5, zorder=5,
        label=rf'$+$ cross $=$ total: $+{100 * (b_tot_d - 1):.1f}\%$')
# The MC points are NOT drawn (user decision 2026-07-27): the figure shows the model's
# variance budget, and 35 noisy points at +-0.3-0.6% obscured the 0.4% separation between
# the clumps-alone and clumps+host curves, which is the thing being shown. The MC has not
# stopped mattering -- the amplitude A is still fitted to it, and the fit quality is still
# computed and printed below (chi2/dof, the null-region calibration, the independent-run
# cross-check) so the provenance of the y axis is on the record even though the points are
# off the canvas. If the measurement needs to be SHOWN rather than used, re-enable this.
SHOW_MC = False
if SHOW_MC:
    ax.errorbar(k_mc, ratio, yerr=err, fmt='o', ms=2.9, color=C_MC, mfc=C_MC,
                ecolor=C_MC, elinewidth=0.7, capsize=1.3, alpha=0.9, zorder=6,
                label=rf'MC, $N={N_RAY // 1000}\times10^3$ rays')

# Headroom above the plateau so the legend can sit in the upper right, clear of every
# curve. That is what lets the default marker run the FULL height: earlier versions
# clipped it (ymin=0.52) only because the legend sat in the lower left, directly in its
# path, and the house style draws legends without a frame so an overlap is not survivable.
YLO, YHI = 0.9985, 1.0765
ax.axvline(K_DEFAULT, color='C3', lw=0.9, ls='-', alpha=0.75, zorder=1)
ax.text(K_DEFAULT / 1.55, YLO + 0.30 * (YHI - YLO), r'default', fontsize=6,
        color='C3', rotation=90, va='bottom', ha='right')

# Everything else that used to be annotated on the canvas -- z_s, the rendered clump
# count, the host kappa_thr marker, the provenance of the amplitude -- was removed on
# request (2026-07-27) and belongs in the caption now. The console summary below is
# unchanged, so none of those numbers are lost; they are just not drawn.
n_def = float(np.exp(np.interp(np.log(K_DEFAULT), np.log(k_an), np.log(clumps_an))))

ax.set_xscale('log')
ax.set_xlim(*XLIM)
ax.set_ylim(YLO, YHI)
ax.set_xlabel(r'per-subhalo convergence threshold $\kappa_{\rm thr,\,sub}$')
ax.set_ylabel(r'$\sigma_\kappa^{\rm sub}\,/\,\sigma_\kappa^{\rm no\ sub}$')
format_log_axis_decimal(ax, axis='x')
# legend in the empty band between the 1.0 reference line and the lowest MC point:
# the frame is off in the house style, so the box must not cross any line at all
handles, labels = ax.get_legend_handles_labels()
wanted = [rf'$\sum_i\kappa_i$ only: $+{100 * (b_clm_d - 1):.1f}\%$',
          rf'$+$ host, no cross: $+{100 * (b_ch_d - 1):.1f}\%$',
          r'cross term $2\,\mathrm{Cov}<0$',
          rf'$+$ cross $=$ total: $+{100 * (b_tot_d - 1):.1f}\%$']
if SHOW_MC:
    wanted.insert(0, rf'MC, $N={N_RAY // 1000}\times10^3$ rays')
order = [labels.index(l) for l in wanted]
ax.legend([handles[i] for i in order], [labels[i] for i in order],
          fontsize=5.5, loc='upper right', handlelength=1.5, borderaxespad=0.6,
          labelspacing=0.34, handletextpad=0.55)

out_png = OUT_DIR / 'fig_subhalo_sigma_ratio_vs_subkappathr.png'
out_pdf = out_png.with_suffix('.pdf')
fig.savefig(out_png, dpi=300, facecolor='white')
fig.savefig(out_pdf, facecolor='white')
print(f"wrote {out_pdf.relative_to(REPO)}")
print(f"wrote {out_png.relative_to(REPO)}")

# ---------------------------------------------------------------------------------
# console summary / verification
# ---------------------------------------------------------------------------------
print(f"\nz_s={Z_SOURCE:g}  host kappa_thr={KAPPA_THR_HOST:.4e}  N_ray={N_RAY}")
print(f"null region: {null.sum()} points at kappa_thr,sub >= {k_mc[null].min():.3g} "
      f"(analytic s < {S_NULL}, clumps/ray < {clumps_at[null].max():.1e})")
print(f"  common-mode offset c = {c_raw:.5f} +- {c_err:.5f}  "
      f"({abs(1 - c_raw) / c_err:.2f} sigma from 1) -> DIVIDED OUT of the plotted ratios")
print("    (the +-  includes the single shared sigma_OFF error, which does not average "
      "down; using SEM alone would read "
      f"{abs(1 - c_raw) / (eps_null / np.sqrt(null.sum())):.1f} sigma and be wrong)")
print(f"  eps_null = {eps_null:.4f}   eps_on = {eps_on:.4f}   "
      f"-> eps_extra = {eps_extra:.4f}")
print(f"fit: A = Var_sub/Var_off = {A:.4f}  ->  plateau = {PLATEAU:.4f} "
      f"(+{100 * (PLATEAU - 1):.2f}% on sigma_kappa, "
      f"{100 * A / (1 + A):.1f}% of Var(kappa))")
print(f"  chi2/dof = {chi2_dof:.2f} over {len(k_mc)} points "
      f"(1 fitted scalar A + 1 null-region calibration, shape predicted)")
print(f"  max |resid|/err = {np.max(np.abs(resid / err)):.2f} at "
      f"kappa_thr,sub={k_mc[np.argmax(np.abs(resid / err))]:.3g}")

# independent realization: the earlier 1e5-ray non-CRN run of the same sweep. Refitting
# it end to end is the honest check on the plateau amplitude, because A depends on c and
# c is pinned by only ~5 null points.
alt_path = REPO / "playground" / "sigma_on_off_vs_subkappathr_model5.json"
if alt_path.exists():
    alt = json.loads(alt_path.read_text())
    k_a = np.array([r["kthr_sub"] for r in alt["rows"]])
    r_a = np.array([r["ratio"] for r in alt["rows"]])
    s_a = np.interp(np.log(k_a), np.log(k_an), s_an)
    n_a = s_a < S_NULL
    if n_a.sum() >= 3:
        c_a = float(r_a[n_a].mean())
        A_a = float(least_squares(
            lambda p: c_a * np.sqrt(1 + p[0] * s_a[~n_a] ** 2) - r_a[~n_a], [0.11]).x[0])
        print(f"\ncross-check on the independent N={alt['nsamples']} run: "
              f"c={c_a:.5f}, plateau={np.sqrt(1 + A_a):.4f} "
              f"(+{100 * (np.sqrt(1 + A_a) - 1):.2f}%) vs +{100 * (PLATEAU - 1):.2f}% here")
        print("  -> quote the plateau to ~half a percent, not three digits: it inherits "
              "the calibration error on c")

print("\nthreshold ledger (analytic, population weighted):")
for mult in (0.01, KTHR_FACTOR_DEFAULT, 1.0, 10.0):
    kk = mult * KAPPA_THR_HOST
    ss = float(np.interp(np.log(kk), np.log(k_an), s_an))
    cc = float(np.exp(np.interp(np.log(kk), np.log(k_an), np.log(clumps_an))))
    tot = np.sqrt(1 + A * ss ** 2) / PLATEAU
    print(f"  f={mult:<5g} kappa_thr,sub={kk:.3e}  clumps/ray={cc:9.4g}  "
          f"({CLUMPS_FULL / cc:9.3g}x fewer)  sigma_sub loss={100 * (1 - ss):.3f}%  "
          f"total sigma loss={100 * (1 - tot):.4f}%")
