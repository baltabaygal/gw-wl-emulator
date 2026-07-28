#!/usr/bin/env python
"""
Magnification-PDF figures for Sec. II.B (Magnification PDF).

Produces two single-panel paper figures (Vaskonen-2026 Fig. 2 / Fig. 3 style):

  1. fig_magnification_pdf_zs.{png,pdf}
     dP/dmu vs mu for several source redshifts z_s, at the full production config.
     Shows the sharp low-mu edge and the heavy high-mu tail growing with z_s.

  2. fig_magnification_pdf_ingredients.{png,pdf}
     dP/dmu at a fixed z_s (default 5), for three cumulative model configs:
        (a) baseline      = Vaskonen 2026 (legacy iid clustering, no subhalos)
        (b) + subhalos    = baseline + subhalo_model 4 (carved host)
        (c) + clustering  = full production (correlated field + weak arm + subhalos)
     Shows what the two new ingredients of this paper do to the PDF.

--------------------------------------------------------------------------------
RUN IT YOURSELF -- see paper_prod/scripts/README_magnification_pdf.md for the
full recipe (env, realization counts, parallel shards, timing). Quick reference:

  PY=/Users/baltabay/miniforge3/envs/test/bin/python   # conda "test", py3.12

  # quick smoke test (fast, noisy):
  $PY paper_prod/scripts/plot_fig_magnification_pdf.py --nreal 50000 --tag test

  # one big smooth run (slow, single process):
  $PY paper_prod/scripts/plot_fig_magnification_pdf.py --nreal 3000000 --tag run1

  # or shard over seeds for speed, then combine (histograms just add):
  for s in 1 2 3 4 5 6 7 8; do \
    $PY paper_prod/scripts/plot_fig_magnification_pdf.py \
        --nreal 1000000 --seed $s --tag shard$s --no-plot & done; wait
  $PY paper_prod/scripts/plot_fig_magnification_pdf.py \
        --combine paper_prod/plots/data/magpdf_shard*.npz --tag combined

  # re-plot / restyle without re-running the MC:
  $PY paper_prod/scripts/plot_fig_magnification_pdf.py --replot --tag run1

Each MC run writes a small histogram cache (a few kB) to
paper_prod/plots/data/magpdf_<tag>.npz, so re-plotting is instant. The heavy
part is the Monte Carlo; smoothness is bought with --nreal (and/or shards).
--------------------------------------------------------------------------------
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))            # for paper_prod.plot_style
sys.path.insert(0, str(REPO / "build"))  # for gwlensing (cpython-3.12 .so)

# ----------------------------------------------------------------------------
# Fiducial cosmology (Planck 2018, the Vaskonen 2026 benchmark; CLAUDE.md defaults)
# ----------------------------------------------------------------------------
COSMO = dict(OmegaM=0.315, sigma8=0.811, h=0.674)

# Source redshifts for figure 1 (the z_s ramp).
ZS_LIST = [0.5, 1.0, 2.0, 5.0, 10.0]

# ----------------------------------------------------------------------------
# Model configurations. Flag semantics (cpp/python_bindings.cpp, cpp/lensing.h):
#   subhalo_model 4  -> every subhalo sampled to m_floor, host carved (REQUIRES
#                       subhalo_carve=True; throws otherwise).
#   bias_model 1     -> correlated 1D density field (vs 0 = legacy iid cell bias).
#   bias_window 1    -> spherical top-hat on |k| (REQUIRES bias_model=1).
#   bias_weak True   -> conditional sub-threshold weak arm (REQUIRES bias_model=1).
#   bias_Rperp 20000 -> R_s = 20 Mpc (comoving kpc), the settled paper value.
#   fil_bias True    -> filament PBS bias b_F (vs halo bias) for the count modulation.
#   kappa_anchor 1   -> ROBUST flux anchor <kappa>=0 over rays with kappa<=1 only,
#                       avoiding the monster-ray batch-mean shift (lensing.h:128).
# NOTE: model 4 + carve requires the post-2026-07-23 Mac build of gwlensing.
# ----------------------------------------------------------------------------
_COMMON = dict(filaments=True, ell=True, Nhalos=100, Mmin=1e7,
               NM=100, Nz=100, kappa_anchor=1, kappa_anchor_cut=1.0)

CONFIG_FULL = dict(_COMMON, bias=True,
                   subhalo=True, subhalo_model=4, subhalo_carve=True,
                   bias_model=1, bias_window=1, bias_Rperp=20000.0,
                   bias_weak=True, fil_bias=True)

CONFIG_BASE = dict(_COMMON, bias=True,
                   subhalo=False,
                   bias_model=0, bias_window=0, bias_Rperp=8441.0,
                   bias_weak=False, fil_bias=False)

CONFIG_SUBH = dict(CONFIG_BASE, subhalo=True, subhalo_model=4, subhalo_carve=True)

# ----------------------------------------------------------------------------
# Fine histogram grid used for the on-disk cache. We store binned counts (tiny)
# rather than raw samples, so re-plotting/re-binning is free and shards just add.
# The stored range is wide enough to hold everything we might plot; the display
# range is a sub-window rebinned coarser at plot time.
#
# LOG-SPACED, WIDE (changed 2026-07-28). The previous grid was 2000 LINEAR bins
# over mu in [0.40, 5.00], which cannot support either feature discussed in
# Sec. II.B / App. edge_tail:
#   - the mu^-2 tail is fitted above mu >= 8 (docs/edge_tail_flux_note.md Sec. 2),
#     i.e. entirely outside the old upper edge;
#   - the low-mu edge is soft, with a sparse kappa>1 population extending to
#     lnmu ~ -10; showing that it is soft needs room below the edge, which the
#     old lower edge of 0.40 (lnmu = -0.92) did not leave at high z_s.
# The MC is the expensive step and the cache stores only counts, so the range
# must be right BEFORE the production run -- widening it afterwards means
# re-running, not re-plotting. Resolution is dlnmu = 2.1e-3, matching the old
# grid's effective resolution at mu ~ 1.
# ----------------------------------------------------------------------------
STORE_LO, STORE_HI, STORE_NBINS = 0.05, 200.0, 4000
STORE_EDGES = np.geomspace(STORE_LO, STORE_HI, STORE_NBINS + 1)

DATA_DIR = REPO / "paper_prod" / "plots" / "data"
OUT_DIR = REPO / "paper_prod" / "plots" / "figures"


# ============================================================================
# Monte-Carlo generation
# ============================================================================
def run_series(series, nreal, seed):
    """Run the MC for each (tag, zs, config) and return {tag: (counts, ntot)}.

    counts: histogram over STORE_EDGES of mu = exp(lnmu) (finite rays only).
    ntot:   number of finite rays (the normalization for dP/dmu).
    nlo:    finite rays below STORE_LO (underflow).
    nhi:    finite rays above STORE_HI (overflow).

    nlo/nhi are stored so that quantiles -- in particular the 0.1% low-mu edge
    marker -- can be computed exactly from the cache. Without them the CDF built
    from `counts` alone silently omits the sparse kappa>1 population that extends
    far below STORE_LO, which is exactly the population that makes the edge soft.
    """
    import gwlensing as gw

    out = {}
    for tag, zs, cfg in series:
        print(f"[mc] {tag}: z_s={zs}  Nreal={nreal:,}  seed={seed}", flush=True)
        lnmu = np.asarray(gw.sample_lnmu(z=zs, Nreal=int(nreal), seed=int(seed),
                                         **COSMO, **cfg), dtype=float)
        lnmu = lnmu[np.isfinite(lnmu)]
        mu = np.exp(lnmu)
        counts, _ = np.histogram(mu, bins=STORE_EDGES)
        nlo = int(np.count_nonzero(mu < STORE_LO))
        nhi = int(np.count_nonzero(mu > STORE_HI))
        out[tag] = (counts.astype(np.int64), int(mu.size), nlo, nhi)
        # report a couple of body-safe diagnostics (clipped, per standing rules)
        print(f"      finite rays={mu.size:,}  frac(mu>1.5)={np.mean(mu>1.5):.3e}"
              f"  median(mu)={np.median(mu):.4f}"
              f"  underflow={nlo}  overflow={nhi}", flush=True)
    return out


def build_series(zs_ing):
    """Return the deduplicated list of (tag, zs, config) to simulate."""
    series, seen = [], set()

    def add(tag, zs, cfg):
        if tag not in seen:
            series.append((tag, zs, cfg))
            seen.add(tag)

    for zs in ZS_LIST:
        add(f"full_z{zs:g}", zs, CONFIG_FULL)
    add(f"base_z{zs_ing:g}", zs_ing, CONFIG_BASE)
    add(f"subh_z{zs_ing:g}", zs_ing, CONFIG_SUBH)
    add(f"full_z{zs_ing:g}", zs_ing, CONFIG_FULL)  # reused if zs_ing in ZS_LIST
    return series


def save_cache(path, data, meta):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    kw = {"edges": STORE_EDGES, "_meta": np.array(json.dumps(meta))}
    for tag, (counts, ntot, nlo, nhi) in data.items():
        kw[f"counts__{tag}"] = counts
        kw[f"ntot__{tag}"] = np.array(ntot, dtype=np.int64)
        kw[f"nlo__{tag}"] = np.array(nlo, dtype=np.int64)
        kw[f"nhi__{tag}"] = np.array(nhi, dtype=np.int64)
    np.savez_compressed(path, **kw)
    print(f"[cache] wrote {path}", flush=True)


def load_cache(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["_meta"]))
    data = {}
    for k in z.files:
        if k.startswith("counts__"):
            tag = k[len("counts__"):]
            data[tag] = (z[k], int(z[f"ntot__{tag}"]),
                         int(z[f"nlo__{tag}"]) if f"nlo__{tag}" in z.files else 0,
                         int(z[f"nhi__{tag}"]) if f"nhi__{tag}" in z.files else 0)
    return data, meta


def combine_caches(paths):
    """Sum histogram counts (and ntot) across shard caches with matching edges."""
    total, meta = {}, None
    for p in paths:
        data, meta = load_cache(p)
        edges = np.load(p, allow_pickle=False)["edges"]
        if edges.shape != STORE_EDGES.shape or not np.allclose(edges, STORE_EDGES):
            raise SystemExit(
                f"[combine] {p} was written with a different histogram grid "
                f"({edges.size - 1} bins over [{edges[0]:g}, {edges[-1]:g}]) than the "
                f"current STORE_EDGES ({STORE_NBINS} bins over "
                f"[{STORE_LO:g}, {STORE_HI:g}]). Summing them would be wrong; "
                f"re-run the shard or check out the matching script revision.")
        for tag, (counts, ntot, nlo, nhi) in data.items():
            if tag not in total:
                total[tag] = [np.zeros_like(counts), 0, 0, 0]
            total[tag][0] = total[tag][0] + counts
            total[tag][1] += ntot
            total[tag][2] += nlo
            total[tag][3] += nhi
    return {t: tuple(v) for t, v in total.items()}, meta


# ============================================================================
# Plotting
# ============================================================================
def _rebin(counts, edges, factor):
    n = (len(counts) // factor) * factor
    c = counts[:n].reshape(-1, factor).sum(axis=1)
    e = edges[: n + 1 : factor]
    return c, e


def _dpdmu(counts, edges, ntot, mu_lo, mu_hi, factor):
    counts, edges = _rebin(counts, edges, factor)
    # geometric centers: STORE_EDGES is log-spaced, so the arithmetic midpoint
    # would bias each bin outward on a log axis.
    centers = np.sqrt(edges[:-1] * edges[1:])
    width = np.diff(edges)
    dpdmu = counts / (ntot * width)
    m = (centers >= mu_lo) & (centers <= mu_hi) & (counts > 0)
    return centers[m], dpdmu[m]


def guard_broken_latex():
    """apply_style enables usetex when a `latex` binary exists; fall back to
    mathtext if the TeX tree is unusable (e.g. sandbox installs).

    Same guard as plot_fig_subhalo_population.py -- keep them in sync.
    """
    import shutil
    import subprocess
    import matplotlib as mpl
    if not mpl.rcParams.get("text.usetex"):
        return
    ok = False
    if shutil.which("kpsewhich"):
        # type1ec.sty (cm-super) is matplotlib's hard usetex requirement
        r = subprocess.run(["kpsewhich", "type1ec.sty"], capture_output=True)
        ok = r.returncode == 0
    if not ok:
        mpl.rcParams["text.usetex"] = False
        mpl.rcParams["mathtext.fontset"] = "cm"


def _edge_quantile(counts, edges, ntot, nlo, q):
    """The q-quantile of mu, computed exactly from the stored histogram.

    `nlo` (rays below STORE_LO) is included in the cumulative count, so the
    sparse kappa>1 population below the stored range is not silently dropped.
    Returns None if the quantile falls in the underflow, i.e. if the marker
    would be off the stored grid and cannot be placed honestly.
    """
    target = q * ntot
    if nlo >= target:
        return None
    cum = nlo + np.cumsum(counts)
    j = int(np.searchsorted(cum, target, side="left"))
    if j >= len(counts):
        return None
    # linear interpolation in lnmu within the crossing bin
    below = cum[j - 1] if j > 0 else nlo
    frac = (target - below) / max(counts[j], 1)
    lo, hi = np.log(edges[j]), np.log(edges[j + 1])
    return float(np.exp(lo + frac * (hi - lo)))


def _maybe_smooth(y, window):
    if window and window >= 3 and window % 2 == 1 and len(y) > window:
        try:
            from scipy.signal import savgol_filter
            ly = savgol_filter(np.log10(y), window, 3)
            return 10.0 ** ly
        except Exception:
            pass
    return y


def plot_figures(data, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from paper_prod.plot_style import apply_style, FIGURE_SIZES

    apply_style()
    guard_broken_latex()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    edges = STORE_EDGES
    factor = args.rebin
    mu_lo, mu_hi = args.mu_range

    # ---- Figure 1: dP/dmu vs mu for several z_s (full production config) -----
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single"])
    cmap = plt.get_cmap("plasma")
    colors = [cmap(t) for t in np.linspace(0.05, 0.85, len(ZS_LIST))]
    for zs, col in zip(ZS_LIST, colors):
        tag = f"full_z{zs:g}"
        if tag not in data:
            continue
        counts, ntot, nlo, nhi = data[tag]
        x, y = _dpdmu(counts, edges, ntot, mu_lo, mu_hi, factor)
        y = _maybe_smooth(y, args.smooth)
        ax.plot(x, y, color=col, lw=1.1, label=fr"${zs:g}$")
        # low-mu edge marker: the args.edge_q quantile, drawn as a rug tick at
        # the bottom of the axes. This is the MEASURED edge of the simulated
        # distribution, not the analytic empty-beam value (which lies well below
        # it -- see App. edge_tail).
        if args.edge_q > 0:
            mu_edge = _edge_quantile(counts, edges, ntot, nlo, args.edge_q)
            if mu_edge is not None and mu_lo <= mu_edge <= mu_hi:
                ax.plot([mu_edge], [args.ylim[0]], marker="|", color=col,
                        ms=6, mew=1.2, clip_on=False, zorder=5)
    ax.set_yscale("log")
    ax.set_xlim(mu_lo, mu_hi)
    ax.set_ylim(*args.ylim)
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$\mathrm{d}P/\mathrm{d}\mu$")
    # legend upper-LEFT: the tail inset below occupies the upper-right corner
    ax.legend(title=r"$z_s$", frameon=False, fontsize=7, title_fontsize=7,
              handlelength=1.0, labelspacing=0.25, loc="upper left")

    # ---- inset: the high-mu tail on log-log axes, with a mu^-2 slope guide ---
    # The body panel above is linear in mu and stops at mu_hi, so the power-law
    # regime discussed in Sec. II.B is invisible there. The guide line shows the
    # SLOPE only; it is not a fit, and the absolute normalization of the far
    # tail is not certified (standing rule -- state this in the caption).
    if args.inset:
        from matplotlib.ticker import LogLocator
        axi = ax.inset_axes(args.inset_box)
        ti_lo, ti_hi = args.tail_range
        anchor_mu, anchor_y = max(ti_lo * 1.5, 3.0), 0.0
        for zs, col in zip(ZS_LIST, colors):
            tag = f"full_z{zs:g}"
            if tag not in data:
                continue
            counts, ntot, nlo, nhi = data[tag]
            xt, yt = _dpdmu(counts, edges, ntot, ti_lo, ti_hi, args.tail_rebin)
            if xt.size:
                axi.plot(xt, _maybe_smooth(yt, args.smooth), color=col, lw=0.9)
                anchor_y = max(anchor_y, float(np.interp(anchor_mu, xt, yt)))
        # Slope guide, anchored a factor --guide-norm above the highest curve at
        # anchor_mu so it never floats off the panel. It shows the SLOPE only:
        # it is not a fit, and the absolute normalization of the far tail is not
        # certified (standing rule -- say so in the caption).
        if anchor_y > 0:
            gx = np.geomspace(anchor_mu, ti_hi, 32)
            gy = args.guide_norm * anchor_y * (gx / anchor_mu) ** (-2.0)
            axi.plot(gx, gy, color="0.3", lw=0.8, ls="--", zorder=1)
            axi.text(gx[8], gy[8] * 1.6, r"$\propto\mu^{-2}$", fontsize=6,
                     color="0.3", ha="left", va="bottom")
        axi.set_xscale("log")
        axi.set_yscale("log")
        axi.set_xlim(ti_lo, ti_hi)
        axi.yaxis.set_major_locator(LogLocator(numticks=4))
        axi.yaxis.set_minor_locator(LogLocator(subs=(), numticks=4))
        axi.tick_params(labelsize=5.5, pad=1.0, length=2.0)
        axi.set_xlabel(r"$\mu$", fontsize=6, labelpad=-0.5)

    fig.subplots_adjust(left=0.17, right=0.96, bottom=0.16, top=0.95)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig_magnification_pdf_zs.{ext}", dpi=300)
    plt.close(fig)
    print(f"[fig] wrote {OUT_DIR / 'fig_magnification_pdf_zs.{pdf,png}'}", flush=True)

    # ---- Figure 2: dP/dmu at fixed z_s, ingredients + ratio to baseline ------
    # The subhalo/clustering effects are ~10-20% in localized regions, invisible
    # on a bare log overlay, so the lower panel shows the ratio to the baseline.
    zi = args.zs_ingredients
    base_tag = f"base_z{zi:g}"
    ing = [
        (f"base_z{zi:g}", "baseline (Vaskonen 2026)", "0.35", "-"),
        (f"subh_z{zi:g}", r"$+$ subhalos", "#1f77b4", "--"),
        (f"full_z{zi:g}", r"$+$ clustering (full)", "#d62728", "-"),
    ]

    def _grid(tag):
        counts, ntot = data[tag][0], data[tag][1]
        c, e = _rebin(counts, edges, factor)
        cen = 0.5 * (e[:-1] + e[1:])
        return cen, c, c / (ntot * np.diff(e))

    fig, (axt, axr) = plt.subplots(
        2, 1, sharex=True, figsize=(FIGURE_SIZES["single"][0], 3.5),
        gridspec_kw={"height_ratios": [2.0, 1.0]})

    cen, cbase, ybase = _grid(base_tag)
    inwin = (cen >= mu_lo) & (cen <= mu_hi)
    for tag, lab, col, ls in ing:
        if tag not in data:
            continue
        _, c, y = _grid(tag)
        m = inwin & (c > 0)
        axt.plot(cen[m], _maybe_smooth(y[m], args.smooth), color=col, ls=ls,
                 lw=1.1, label=lab)
        if tag != base_tag:
            # ratio only where the baseline bin has enough counts to be meaningful
            mr = inwin & (cbase >= 50) & (c > 0)
            axr.plot(cen[mr], (y / ybase)[mr], color=col, ls=ls, lw=1.1)
    axt.set_yscale("log")
    axt.set_xlim(mu_lo, mu_hi)
    axt.set_ylim(*args.ylim)
    axt.set_ylabel(r"$\mathrm{d}P/\mathrm{d}\mu$")
    axt.text(0.95, 0.92, fr"$z_s = {zi:g}$", transform=axt.transAxes,
             ha="right", va="top", fontsize=8)
    axt.legend(frameon=False, fontsize=7, handlelength=1.4, labelspacing=0.25)
    axr.axhline(1.0, color="0.35", lw=0.7, ls=":")
    axr.set_ylim(0.7, 2.0)
    axr.set_xlabel(r"$\mu$")
    axr.set_ylabel(r"ratio")
    fig.subplots_adjust(left=0.17, right=0.96, bottom=0.12, top=0.97, hspace=0.08)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig_magnification_pdf_ingredients.{ext}", dpi=300)
    plt.close(fig)
    print(f"[fig] wrote {OUT_DIR / 'fig_magnification_pdf_ingredients.{pdf,png}'}",
          flush=True)


# ============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nreal", type=int, default=500_000,
                    help="realizations per series (per z_s / per config). "
                         "Use >=2e6, or shard over seeds, for smooth tails.")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", type=str, default="run1",
                    help="cache label -> paper_prod/plots/data/magpdf_<tag>.npz")
    ap.add_argument("--zs-ingredients", type=float, default=5.0,
                    help="fixed z_s for the ingredient-comparison figure")
    ap.add_argument("--replot", action="store_true",
                    help="skip MC, load the <tag> cache and just (re)plot")
    ap.add_argument("--combine", nargs="+", metavar="NPZ",
                    help="sum these shard caches (glob ok), save as <tag>, plot")
    ap.add_argument("--no-plot", action="store_true",
                    help="run the MC and write the cache, but do not plot")
    # display knobs
    ap.add_argument("--mu-range", type=float, nargs=2, default=[0.6, 1.8])
    ap.add_argument("--edge-q", type=float, default=0.001,
                    help="quantile marking the low-mu edge on Fig. 1 "
                         "(0 disables the markers)")
    ap.add_argument("--inset", dest="inset", action="store_true", default=True,
                    help="draw the log-log high-mu tail inset on Fig. 1")
    ap.add_argument("--no-inset", dest="inset", action="store_false")
    ap.add_argument("--tail-range", type=float, nargs=2, default=[1.5, 100.0],
                    help="mu window shown in the tail inset")
    ap.add_argument("--tail-rebin", type=int, default=40,
                    help="coarser rebin for the inset (tail bins are sparse)")
    ap.add_argument("--guide-norm", type=float, default=1.0,
                    help="normalization of the mu^-2 slope guide in the inset; "
                         "purely cosmetic, set it so the guide sits beside the "
                         "curves without overlapping them")
    ap.add_argument("--inset-box", type=float, nargs=4,
                    default=[0.50, 0.42, 0.46, 0.50],
                    help="inset axes box (x0 y0 w h) in axes fraction")
    ap.add_argument("--ylim", type=float, nargs=2, default=[1e-2, 3e1])
    ap.add_argument("--rebin", type=int, default=8,
                    help="merge this many stored fine bins per display bin "
                         "(2000 stored / 8 ~= 250 display bins)")
    ap.add_argument("--smooth", type=int, default=0,
                    help="odd Savitzky-Golay window on log10(dP/dmu) for display "
                         "(0 = off; prefer more --nreal over smoothing)")
    args = ap.parse_args()

    cache_path = DATA_DIR / f"magpdf_{args.tag}.npz"

    if args.combine:
        paths = []
        for pat in args.combine:
            paths.extend(sorted(glob.glob(pat)))
        if not paths:
            sys.exit(f"--combine matched no files: {args.combine}")
        print(f"[combine] {len(paths)} shard(s)", flush=True)
        data, meta = combine_caches(paths)
        save_cache(cache_path, data, {"combined_from": paths})
    elif args.replot:
        if not cache_path.exists():
            sys.exit(f"--replot: cache not found: {cache_path}")
        data, meta = load_cache(cache_path)
    else:
        series = build_series(args.zs_ingredients)
        data = run_series(series, args.nreal, args.seed)
        save_cache(cache_path, data,
                   {"cosmo": COSMO, "nreal": args.nreal, "seed": args.seed,
                    "zs_list": ZS_LIST, "zs_ingredients": args.zs_ingredients})

    if not args.no_plot:
        plot_figures(data, args)


if __name__ == "__main__":
    main()
