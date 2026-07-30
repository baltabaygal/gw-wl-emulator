"""Figures: the Hubble diagram (his Fig. 4) and the corner plot (his Fig. 5).

Deliberately NOT importing paper_prod/plot_style.py: these are diagnostic
figures for a pipeline that currently runs on a MOCK P(mu), and they must not
be mistakable for paper figures. If an HDR figure ever goes in the paper, it
gets rebuilt against the paper style (and the log-axis tick rule in CLAUDE.md).
No `corner` dependency -- the corner plot is hand-rolled from numpy histograms.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .background import Background  # noqa: E402

MOCK_BANNER = "MOCK P(mu) -- pipeline diagnostic, NOT a forecast"

# Display labels for the 1+6d parameter space (ml/params.py CONTEXT_KEYS order,
# minus z). Keys not listed fall back to the raw key.
LABELS = {
    "h": r"$h$",
    "Om": r"$\Omega_M$",
    "sigma8": r"$\sigma_8$",
    "Ob": r"$\Omega_b$",
    "ns": r"$n_s$",
    "zeq": r"$z_{\rm eq}$",
}


def label(k: str) -> str:
    return LABELS.get(k, k)


def _hpd_levels(H, probs=(0.68, 0.95)):
    """Contour levels enclosing the given posterior mass."""
    flat = np.sort(H.ravel())[::-1]
    csum = np.cumsum(flat)
    csum /= csum[-1]
    return [flat[np.searchsorted(csum, p)] for p in sorted(probs, reverse=True)]


def corner_plot(flat, keys, truth=None, path=None, title=None, banner=None):
    """68%/95% contours + 1-D marginals. flat: (n, npar)."""
    n = len(keys)
    fig, axes = plt.subplots(n, n, figsize=(2.4 * n, 2.4 * n))
    axes = np.atleast_2d(axes)
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                ax.hist(flat[:, i], bins=40, color="0.3", histtype="stepfilled",
                        alpha=0.7, density=True)
                if truth is not None and keys[i] in truth:
                    ax.axvline(truth[keys[i]], color="crimson", lw=1.2)
                ax.set_yticks([])
            else:
                H, xe, ye = np.histogram2d(flat[:, j], flat[:, i], bins=32)
                H = H.T
                lv = _hpd_levels(H)
                ax.contourf(
                    0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), H,
                    levels=lv + [H.max() + 1], colors=["0.75", "0.4"],
                )
                if truth is not None:
                    if keys[j] in truth:
                        ax.axvline(truth[keys[j]], color="crimson", lw=0.8)
                    if keys[i] in truth:
                        ax.axhline(truth[keys[i]], color="crimson", lw=0.8)
            if i == n - 1:
                ax.set_xlabel(label(keys[j]), fontsize=13)
                ax.tick_params(axis="x", labelrotation=45, labelsize=8)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(label(keys[i]), fontsize=13)
                ax.tick_params(axis="y", labelsize=8)
            elif j > 0:
                ax.set_yticklabels([])
            # keep tick counts low so 6x6 panels stay readable
            ax.locator_params(axis="x", nbins=4)
            if i != j:
                ax.locator_params(axis="y", nbins=4)
    if title:
        fig.suptitle(title, y=0.98)
    fig.text(0.5, 0.005, banner or MOCK_BANNER, ha="center", color="crimson",
             fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    if path:
        fig.savefig(path, dpi=140)
        plt.close(fig)
    return fig


def hubble_diagram(cat, theta_fid, provider=None, path=None, banner=None):
    """Catalogue points + true Dtilde(z) + a lensing band (his Fig. 4).

    The band is the 1-99 percentile of Dtilde/sqrt(mu) under the SOURCE-plane
    PDF -- i.e. lensing scatter only, no measurement error.
    """
    bg = Background(theta_fid)
    zg = np.linspace(max(1e-3, float(np.min(cat.z)) * 0.5),
                     float(np.max(cat.z)) * 1.05, 120)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(zg, bg.DL(zg) / 1e3, "k-", lw=1.6, label=r"$\tilde D_L(z)$, truth")

    if provider is not None:
        lo, hi = [], []
        for z in zg:
            g = provider(z, theta_fid).source_plane()
            r = np.exp(-0.5 * g.lnmu)
            dl = float(g.lnmu[1] - g.lnmu[0])
            cdf = np.cumsum(g.p * dl)
            cdf /= cdf[-1]
            # r decreases with lnmu, so percentiles swap
            lo.append(np.interp(0.99, cdf, r))
            hi.append(np.interp(0.01, cdf, r))
        DLg = bg.DL(zg)
        ax.fill_between(zg, DLg * np.array(lo) / 1e3, DLg * np.array(hi) / 1e3,
                        color="steelblue", alpha=0.25,
                        label="98% lensing band")

    ax.errorbar(cat.z, cat.dL_obs / 1e3, yerr=cat.sigma_dL / 1e3, fmt=".",
                ms=4, lw=0.6, color="crimson", alpha=0.75,
                label=f"{cat.arm} ({len(cat)} events)")
    ax.set_xlabel("$z$")
    ax.set_ylabel("$D_L$ [Gpc]")
    ax.legend(frameon=False, fontsize=9)
    fig.text(0.5, 0.005, banner or MOCK_BANNER, ha="center", color="crimson",
             fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    if path:
        fig.savefig(path, dpi=140)
        plt.close(fig)
    return fig


def sigma8_vs_N(scan, path=None, banner=None, title=None):
    """sigma_8 precision vs catalogue size, with realization scatter.

    `scan` is the dict returned by run_forecast.n_scan(): keys 'N', 'median',
    'lo', 'hi' (percentiles over catalogue realizations), all as fractional
    errors on sigma_8.

    The point of this figure is that N is an ASSUMPTION, not a measurement:
    Vaskonen asserts 300 ET bright sirens, De Leo+ decline to assert an absolute
    yield at all and parametrise by N. Plotting the curve puts the assumption on
    an axis instead of burying it in a single number.
    """
    N = np.asarray(scan["N"], float)
    med = 100 * np.asarray(scan["median"], float)
    lo = 100 * np.asarray(scan["lo"], float)
    hi = 100 * np.asarray(scan["hi"], float)

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.fill_between(N, lo, hi, color="steelblue", alpha=0.22,
                    label="16-84% over catalogue realizations")
    ax.plot(N, med, "o-", color="steelblue", lw=1.8, ms=5, label="median")

    # 1/sqrt(N) reference anchored at the smallest N
    ref = med[0] * np.sqrt(N[0] / N)
    ax.plot(N, ref, "k--", lw=1.0, alpha=0.7, label=r"$\propto N^{-1/2}$")

    # survey assumptions
    ax.axvspan(5, 60, color="darkorange", alpha=0.12, zorder=0)
    ax.text(17, hi.max() * 0.97, "De Leo+ 2026\nsub-catalogues\n($N=5-60$)",
            color="darkorange", fontsize=8, va="top", ha="center")
    ax.axvline(300, color="crimson", lw=1.2, ls=":")
    ax.text(330, hi.max() * 0.97, "Vaskonen 2026\nassumes $N=300$",
            color="crimson", fontsize=8, va="top")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$N$ bright sirens (ET, BNS, $z<2$)")
    ax.set_ylabel(r"$\sigma_8$ precision  $\sigma(\sigma_8)/\sigma_8$  [%]")
    # Plain decimal ticks: matplotlib's default log formatter renders these as
    # "6x10^0", which is unreadable for a percentage axis.
    for axis, ticks in ((ax.xaxis, [10, 30, 100, 300, 1000, 3000]),
                        (ax.yaxis, [2, 3, 5, 10, 20, 30, 50])):
        axis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
        axis.set_minor_locator(matplotlib.ticker.NullLocator())
        axis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.legend(frameon=False, fontsize=9)
    if title:
        ax.set_title(title, fontsize=11)
    fig.text(0.5, 0.005, banner or MOCK_BANNER, ha="center", color="crimson",
             fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    if path:
        fig.savefig(path, dpi=140)
        plt.close(fig)
    return fig


def pmu_panel(provider, theta, zs=(0.5, 1.0, 5.0), path=None, banner=None):
    """P(lnmu) in both planes -- the figure that makes a plane error visible."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for z in zs:
        gi = provider(z, theta)
        gs = gi.source_plane()
        axes[0].plot(gi.lnmu, gi.p, label=f"$z_s={z}$")
        axes[1].plot(gs.lnmu, gs.p, label=f"$z_s={z}$")
    for ax, ttl in zip(axes, ["image plane $P_I$", r"source plane $P_S=\mu^{-1}P_I$"]):
        ax.set_xlabel(r"$\ln\mu$")
        ax.set_ylabel(r"$dP/d\ln\mu$")
        ax.set_title(ttl)
        ax.legend(frameon=False, fontsize=9)
        ax.set_xlim(-0.8, 1.5)
    fig.text(0.5, 0.005, banner or MOCK_BANNER, ha="center", color="crimson",
             fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    if path:
        fig.savefig(path, dpi=140)
        plt.close(fig)
    return fig
