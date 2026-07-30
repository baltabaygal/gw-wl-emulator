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

  # PRODUCTION (2026-07-28): 4e6 per series, 8 seed shards of 5e5, then combine.
  for s in 1 2 3 4 5 6 7 8; do \
    $PY paper_prod/scripts/plot_fig_magnification_pdf.py \
        --nreal 500000 --seed $s --tag shard$s --no-plot & done; wait
  $PY paper_prod/scripts/plot_fig_magnification_pdf.py \
        --combine paper_prod/plots/data/magpdf_shard*.npz --tag combined

  # re-plot / restyle without re-running the MC:
  $PY paper_prod/scripts/plot_fig_magnification_pdf.py --replot --tag run1

Each MC run writes a small histogram cache (a few kB) to
paper_prod/plots/data/magpdf_<tag>.npz, so re-plotting is instant. The heavy
part is the Monte Carlo; smoothness is bought with --nreal (and/or shards).

WARNING -- the pre-2026-07-28 caches in that directory are NOT reusable. They
carry 2000 LINEAR bins over mu in [0.4, 5] (no tail window, no under/overflow
counts), and were run with subhalo_model=4 without subhalo_virial, before the
halobias and subhalo-profile fixes. --combine refuses to mix grids; --replot on
an old cache works but silently cannot show the tail figure.
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
# Cosmologies (Planck 2018 fiducial vs high-structure stress corner)
# ----------------------------------------------------------------------------
COSMO_FID = dict(OmegaM=0.315, sigma8=0.811, h=0.674)
COSMO_HIGH = dict(OmegaM=0.38, sigma8=1.00, h=0.72)
COSMO = COSMO_FID

# Source redshifts for figure 1 (the z_s ramp).
# 3 and 7 added 2026-07-28 so the compensated tail figure has five fully
# populated curves rather than three: at 480k realizations the number of rays
# above mu = 8 runs 1 / 7 / 69 / 154 / 423 / 637 / 981 for
# z_s = 0.5 / 1 / 2 / 3 / 5 / 7 / 10, so everything from z_s = 2 up carries a
# usable tail while z_s <= 1 does not (a physics limit, not a depth one:
# S(mu>5) = 2e-6 at z_s = 0.5).
ZS_LIST = [0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]

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
               NM=100, Nz=100)

# The "full" arm IS the paper's production physics -- import it rather than
# restating it, so this figure cannot drift from ml/params.py. Before
# 2026-07-28 this file hard-coded subhalo_model=4 and omitted
# subhalo_kappathr_factor and subhalo_virial, i.e. it plotted a different
# subhalo population from the one the draft describes (model 4 and 5 agree in
# physics, but subhalo_virial does change the population).
from ml.params import PRODUCTION_CONFIG, PRODUCTION_CONFIG_HASH  # noqa: E402

CONFIG_FULL = dict(_COMMON, bias=True, **PRODUCTION_CONFIG)

# Baseline = Vaskonen 2026: legacy iid clustering layer, no subhalos. Keeps the
# production anchor so the two arms differ only in the ingredients under study.
CONFIG_BASE = dict(_COMMON, bias=True,
                   subhalo=False,
                   bias_model=0, bias_window=0, bias_Rperp=8441.0,
                   bias_weak=False, fil_bias=False,
                   kappa_anchor=PRODUCTION_CONFIG["kappa_anchor"],
                   kappa_anchor_cut=PRODUCTION_CONFIG["kappa_anchor_cut"])

CONFIG_SUBH = dict(CONFIG_BASE,
                   **{k: v for k, v in PRODUCTION_CONFIG.items()
                      if k.startswith("subhalo") or k == "m_floor"})

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


def build_series(zs_ing, zs_list=None, ingredients=True):
    """Return the deduplicated list of (tag, zs, config) to simulate.

    `zs_list` overrides ZS_LIST and `ingredients=False` drops the base/subh arms,
    so a TOP-UP run can add one missing source redshift to an existing shard set
    and be `--combine`d into it. combine_caches() unions tags and SUMS per tag,
    so a top-up shard must contain ONLY the new tags -- rerunning the ingredient
    arms here would double their statistics on combine while leaving every other
    tag alone, which no downstream check would catch.
    """
    series, seen = [], set()

    def add(tag, zs, cfg):
        if tag not in seen:
            series.append((tag, zs, cfg))
            seen.add(tag)

    for zs in (ZS_LIST if zs_list is None else zs_list):
        add(f"full_z{zs:g}", zs, CONFIG_FULL)
    if ingredients:
        add(f"base_z{zs_ing:g}", zs_ing, CONFIG_BASE)
        add(f"subh_z{zs_ing:g}", zs_ing, CONFIG_SUBH)
        add(f"full_z{zs_ing:g}", zs_ing, CONFIG_FULL)  # reused if zs_ing in ZS_LIST
    return series


def save_cache(path, data, meta, edges=None):
    # `edges` must be the grid the counts were binned on -- when re-saving a
    # combination of OLD-grid shards, writing the current STORE_EDGES would
    # mislabel every bin.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    edges = STORE_EDGES if edges is None else np.asarray(edges, dtype=float)
    kw = {"edges": edges, "_meta": np.array(json.dumps(meta))}
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
    # Return the cache's OWN edges. Plotting must use these, not the module-level
    # STORE_EDGES: a cache written before the 2026-07-28 log-grid change stores
    # 2000 linear bins over [0.40, 5.00], and re-plotting it against the current
    # STORE_EDGES would silently place every count at the wrong mu.
    return data, meta, z["edges"]


def combine_caches(paths):
    """Sum histogram counts (and ntot) across shard caches with matching edges."""
    total, meta, ref_edges, ref_path = {}, None, None, None
    for p in paths:
        data, meta, edges = load_cache(p)
        # Shards may only be summed if they share a histogram grid. They are NOT
        # required to match the current STORE_EDGES -- an old linear-grid set can
        # still be combined and replotted among itself; it just cannot show the
        # tail inset, whose window lies outside that grid.
        if ref_edges is None:
            ref_edges, ref_path = edges, p
        elif edges.shape != ref_edges.shape or not np.allclose(edges, ref_edges):
            raise SystemExit(
                f"[combine] histogram grids differ: {ref_path} has "
                f"{ref_edges.size - 1} bins over [{ref_edges[0]:g}, {ref_edges[-1]:g}], "
                f"{p} has {edges.size - 1} bins over [{edges[0]:g}, {edges[-1]:g}]. "
                f"Summing them would be wrong; re-run the odd shard out.")
        for tag, (counts, ntot, nlo, nhi) in data.items():
            if tag not in total:
                total[tag] = [np.zeros_like(counts), 0, 0, 0]
            total[tag][0] = total[tag][0] + counts
            total[tag][1] += ntot
            total[tag][2] += nlo
            total[tag][3] += nhi
    return {t: tuple(v) for t, v in total.items()}, meta, ref_edges


# ============================================================================
# Plotting
# ============================================================================
def _rebin(counts, edges, factor):
    n = (len(counts) // factor) * factor
    c = counts[:n].reshape(-1, factor).sum(axis=1)
    e = edges[: n + 1 : factor]
    return c, e


def _auto_rebin(counts, edges, target_bins_per_sigma=9.0, lo=1, hi=60):
    """Display rebin factor matched to this curve's own width.

    sigma(lnmu) spans 0.009 (z_s=0.2) to 0.28 (z_s=10), a factor of 30, while
    the stored grid is uniform in lnmu. A single factor therefore either shreds
    the wide curves into noise or collapses the narrow ones onto a handful of
    bins -- at rebin=8 the z_s=0.2 body occupies about four display bins and
    renders as a spike that appears truncated. Choosing the factor per curve
    keeps roughly `target_bins_per_sigma` display bins across one sigma.
    """
    c = counts.astype(float)
    if c.sum() <= 0:
        return lo
    ln = np.log(np.sqrt(edges[:-1] * edges[1:]))
    q = c / c.sum()
    mean = (q * ln).sum()
    sd = np.sqrt(max((q * (ln - mean) ** 2).sum(), 0.0))
    dln = np.diff(np.log(edges)).mean()
    if not np.isfinite(sd) or sd <= 0 or dln <= 0:
        return lo
    return int(np.clip(round(sd / (target_bins_per_sigma * dln)), lo, hi))


def _rebin_wings(counts, edges, target):
    """Uniform bins in the core, merged bins in the wings.

    A single bin width cannot carry a distribution whose density spans four
    decades across the panel: fine enough to resolve the peak leaves the wings
    with a handful of counts per bin, and dropping those bins truncates the
    curve partway down the y-axis instead of following it to the bottom. Here
    the walk merges input bins until each output bin holds `target` counts, so
    the peak keeps the input resolution while the wings widen and the curve
    continues as far as there is any data at all.

    Also returns a per-bin PLOT CENTER that is the counts-weighted mean of the
    underlying fine-bin centers, not the geometric mean of the merged bin's
    outer edges. For a near-uniformly-populated bin these agree; but a wing
    bin that swallows a long near-empty desert plus a denser sliver at one
    end (e.g. the sparse soft low-mu edge merging into the sharp body edge --
    docs/edge_tail_flux_note.md) has its edge-geometric-center sitting far
    from where its counts actually are, which draws a misleading straight
    line out into the desert. The weighted center fixes the x-position using
    the same real data, without fabricating anything.
    """
    nz = np.nonzero(counts)[0]
    if nz.size == 0:
        return counts, edges, np.sqrt(edges[:-1] * edges[1:])
    i0, i1 = int(nz[0]), int(nz[-1]) + 1
    bounds, acc, start = [i0], 0, i0
    for i in range(i0, i1):
        acc += counts[i]
        if acc >= target:
            bounds.append(i + 1)
            acc, start = 0, i + 1
    if bounds[-1] != i1:                 # trailing remainder
        if len(bounds) > 1:
            bounds[-1] = i1
        else:
            bounds.append(i1)
    b = np.array(bounds)
    merged_counts = np.add.reduceat(counts, b[:-1])
    fine_centers = np.sqrt(edges[:-1] * edges[1:])
    edge_centers = np.sqrt(edges[b][:-1] * edges[b][1:])
    weighted = edge_centers.copy()
    for k in range(len(b) - 1):
        lo, hi = b[k], b[k + 1]
        if hi - lo > 1 and merged_counts[k] > 0:
            weighted[k] = np.average(fine_centers[lo:hi], weights=counts[lo:hi])
    return merged_counts, edges[b], weighted


def _rebin_piecewise(counts, edges, f_lo, f_hi, pivot):
    """Merge `f_lo` fine bins below `pivot` and `f_hi` above it.

    On a log grid a mu^-2 tail loses counts per bin as fast as mu^-1, so one
    rebin factor cannot serve both the sharp body peak near mu ~ 1 and the tail:
    fine enough for the peak leaves the tail as single-count hash. Coarsening
    only above the pivot keeps the peak resolved and the tail smooth.
    """
    bounds, i, n = [0], 0, len(counts)
    while i < n:
        step = f_lo if edges[i] < pivot else f_hi
        i = min(i + max(int(step), 1), n)
        bounds.append(i)
    b = np.array(bounds)
    c = np.add.reduceat(counts, b[:-1])
    return c, edges[b]


def _dpdmu(counts, edges, ntot, mu_lo, mu_hi, factor,
           f_hi=None, pivot=None, min_counts=1, wings=False):
    if f_hi is not None and pivot is not None:
        counts, edges = _rebin_piecewise(counts, edges, factor, f_hi, pivot)
        # geometric centers: STORE_EDGES is log-spaced, so the arithmetic
        # midpoint would bias each bin outward on a log axis.
        centers = np.sqrt(edges[:-1] * edges[1:])
    elif wings:
        counts, edges = _rebin(counts, edges, factor)
        counts, edges, centers = _rebin_wings(counts, edges, min_counts)
    else:
        counts, edges = _rebin(counts, edges, factor)
        centers = np.sqrt(edges[:-1] * edges[1:])
    width = np.diff(edges)
    dpdmu = counts / (ntot * width)
    # min_counts drops bins too sparse to be a density estimate; without it the
    # far tail degenerates into a jagged single-count floor that reads as signal.
    m = (centers >= mu_lo) & (centers <= mu_hi) & (counts >= max(min_counts, 1))
    return centers[m], dpdmu[m]


def _to_source_plane(x, y, counts, edges, ntot):
    """Convert an image-plane density to the source plane.

    dP_S/dmu = mu^-1 dP_I/dmu / <mu^-1>_I, since a random direction on the sky
    over-weights high mu by exactly the factor mu relative to a random source
    (dOmega_I = mu dOmega_S). This is the same conversion lensing.cpp::Plnmuf
    applies at :1396 (`Plnmu[j][1] *= exp(-Plnmu[j][0])`); `sample_lnmu`, which
    feeds this script, does NOT apply it and returns image-plane rays.

    ⚠ The normalization is the delicate part. <mu^-1> is dominated by the
    SMALLEST mu, i.e. by exactly the sparse kappa > 1 population that makes the
    raw sample mean unusable (it evaluates to 1.08 at z_s = 1 on a handful of
    rays). Here it is summed over the STORED grid only, so rays below
    STORE_LO = 0.05 are excluded. That keeps the curve stable but means the
    normalization is support-restricted, not the exact sum rule -- fine for
    looking at the shape, not a flux measurement.
    """
    cen = np.sqrt(edges[:-1] * edges[1:])
    inv_mean = float(np.sum(counts / cen) / ntot)
    return x, (y / x) / inv_mean


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


_GUIDE_STYLE = {-2.0: ("0.3", (4, 2)), -3.0: ("0.55", (1.2, 1.6))}


def _slope_guide(axes, curves, mu0, mu1, above, slopes=(-2.0,), fontsize=7):
    """Power-law guides sharing one anchor: `above` x the highest curve at mu0.

    A common anchor is the point -- the guides then differ only in slope, so the
    eye compares exponents rather than normalizations. mu^-2 is the image-plane
    fold-caustic law, mu^-3 the source-plane one; this simulator samples random
    sky directions, so mu^-2 is the expected behaviour and mu^-3 is drawn only
    as the contrast. Neither is a fit, and the absolute normalization of the far
    tail is not certified.
    """
    anchor = 0.0
    for x, y in curves:
        if x.size and x[0] <= mu0 <= x[-1]:
            anchor = max(anchor, float(np.interp(mu0, x, y)))
    if anchor <= 0:
        return
    gx = np.geomspace(mu0, mu1, 64)
    # Stagger the labels along the lines: two guides from a common anchor stay
    # close together near it, so labels placed at the same x would overlap.
    fracs = (0.80,) if len(slopes) == 1 else np.linspace(0.86, 0.52, len(slopes))
    for s, f in zip(slopes, fracs):
        col, dashes = _GUIDE_STYLE.get(float(s), ("0.55", (1.2, 1.6)))
        gy = above * anchor * (gx / mu0) ** float(s)
        axes.plot(gx, gy, color=col, lw=0.8, ls="--", dashes=dashes, zorder=1)
        k = int(f * (len(gx) - 1))
        axes.text(gx[k], gy[k] * 1.45, rf"$\propto\mu^{{{int(s)}}}$",
                  fontsize=fontsize, color=col, ha="center", va="bottom")


def _coarse(counts, edges, new_edges):
    """Sum fine bins into `new_edges`. Returns (counts, edges) on the new grid."""
    idx = np.clip(np.searchsorted(edges, new_edges, side="left"), 0, len(counts))
    c = np.array([counts[idx[k]:idx[k + 1]].sum() for k in range(len(idx) - 1)],
                 dtype=np.int64)
    return c, edges[idx]


def _compensated_tail(counts, edges, ntot, mu_lo, mu_hi, nbins, min_counts=8):
    """mu^2 dP/dmu on a coarse log grid, with Poisson errors.

    Flat  <=>  dP/dmu ~ mu^-2  (image plane, what this simulator samples)
    ~mu^-1 <=> dP/dmu ~ mu^-3  (source plane)

    Compensating is what makes the two distinguishable by eye: as plain curves
    on log-log they are near-parallel lines, and over the one decade the tail
    actually populates the difference is easy to mistake for normalization.
    """
    new_edges = np.geomspace(mu_lo, mu_hi, nbins + 1)
    c, e = _coarse(counts, edges, new_edges)
    if len(e) < 2:
        return (np.array([]),) * 3
    cen = np.sqrt(e[:-1] * e[1:])
    y = cen ** 2 * c / (ntot * np.diff(e))
    m = c >= max(min_counts, 1)
    # Poisson error on the bin count propagates straight through
    return cen[m], y[m], y[m] / np.sqrt(c[m])


def _maybe_smooth(y, window):
    if window and window >= 3 and window % 2 == 1 and len(y) > window:
        try:
            from scipy.signal import savgol_filter
            ly = savgol_filter(np.log10(y), window, 3)
            return 10.0 ** ly
        except Exception:
            pass
    return y


def plot_figures(data, args, edges=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from paper_prod.plot_style import apply_style, FIGURE_SIZES

    from paper_prod.plot_style import format_log_axis_decimal
    apply_style()
    guard_broken_latex()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Always plot against the grid the counts were BINNED on.
    edges = STORE_EDGES if edges is None else np.asarray(edges, dtype=float)
    factor = args.rebin
    if args.inset and args.logx:
        print("[info] log-x main panel shows the tail directly; inset disabled "
              "(--inset --no-logx to get the linear-mu panel with an inset).",
              flush=True)
        args.inset = False
    if args.logx and args.mu_range[1] > edges[-1]:
        print(f"[warn] requested mu up to {args.mu_range[1]:g} but this cache "
              f"stops at {edges[-1]:g}; the panel will end there.", flush=True)
        args.mu_range = [args.mu_range[0], float(edges[-1])]
    if args.inset and edges[-1] < args.tail_range[1]:
        print(f"[warn] this cache stops at mu = {edges[-1]:g}, below the tail "
              f"window {args.tail_range[1]:g}; the mu^-2 inset needs a cache "
              f"re-run on the current (log, wide) grid. Disabling the inset.",
              flush=True)
        args.inset = False
    mu_lo, mu_hi = args.mu_range

    # ---- Figure 1: dP/dmu vs mu for several z_s (full production config) -----
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single"])
    cmap = plt.get_cmap("plasma")
    # widened from (0.05, 0.85) when ZS_LIST grew to 8 entries: the low-z_s
    # curves were all landing in the same purple.
    colors = [cmap(t) for t in np.linspace(0.0, 0.92, len(ZS_LIST))]
    main_curves = []
    body_zs = [z for z in ZS_LIST if not args.body_zs or z in args.body_zs]
    for zs, col in zip(ZS_LIST, colors):
        tag = f"full_z{zs:g}"
        if tag not in data or zs not in body_zs:
            continue
        counts, ntot, nlo, nhi = data[tag]
        f = _auto_rebin(counts, edges) if factor <= 0 else factor
        x, y = _dpdmu(counts, edges, ntot, mu_lo, mu_hi, f,
                      f_hi=args.rebin_hi if args.logx else None,
                      pivot=args.rebin_pivot if args.logx else None,
                      min_counts=args.min_counts, wings=not args.logx)
        if args.plane == "source":
            x, y = _to_source_plane(x, y, counts, edges, ntot)
        y = _maybe_smooth(y, args.smooth)
        main_curves.append((x, y))
        ds = "steps-mid" if getattr(args, "as_hist", False) else "default"
        ax.plot(x, y, color=col, lw=1.1, label=fr"${zs:g}$", drawstyle=ds)
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
    if args.logx:
        # Log-log main panel: the mu^-2 regime is a straight line here, so the
        # tail is shown directly and the inset is redundant (auto-disabled
        # below). The body peak near mu ~ 1 is compressed relative to the
        # linear-mu view of Ref. [Vaskonen] -- that is the trade.
        ax.set_xscale("log")
        from paper_prod.plot_style import format_log_axis_decimal
        format_log_axis_decimal(ax, axis="x")
        _slope_guide(ax, main_curves, args.guide_mu, mu_hi, args.guide_norm,
                     slopes=args.guide_slopes)
    ax.set_xlim(mu_lo, mu_hi)
    ax.set_ylim(*args.ylim)
    # paper log-tick convention: 0.1 / 1 / 10 as decimals, other decades 10^n
    format_log_axis_decimal(ax, axis="y")
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$\mathrm{d}P_S/\mathrm{d}\mu$" if args.plane == "source"
                  else r"$\mathrm{d}P_I/\mathrm{d}\mu$")
    # With a log-x panel the curves run to the lower right, so the legend goes
    # lower-left; on the linear-mu panel the inset occupies the upper right.
    ax.legend(title=r"$z_s$", frameon=False, fontsize=7, title_fontsize=7,
              handlelength=1.0, labelspacing=0.25,
              loc="lower left" if args.logx else "upper left")

    # ---- inset: the high-mu tail on log-log axes, with a mu^-2 slope guide ---
    # The body panel above is linear in mu and stops at mu_hi, so the power-law
    # regime discussed in Sec. II.B is invisible there. The guide line shows the
    # SLOPE only; it is not a fit, and the absolute normalization of the far
    # tail is not certified (standing rule -- state this in the caption).
    if args.inset:
        from matplotlib.ticker import LogLocator, NullFormatter
        from paper_prod.plot_style import format_log_axis_decimal
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
            _slope_guide(axi, [(gx, np.full_like(gx, anchor_y))], anchor_mu,
                         ti_hi, args.guide_norm, slopes=args.guide_slopes,
                         fontsize=6)
        axi.set_xscale("log")
        axi.set_yscale("log")
        axi.set_xlim(ti_lo, ti_hi)
        axi.yaxis.set_major_locator(LogLocator(numticks=4))
        axi.yaxis.set_minor_locator(LogLocator(subs=(), numticks=4))
        # Over a narrow tail window matplotlib labels the MINOR decades too
        # (2x10^0, 3x10^0, ...), which collide at inset font size. Label major
        # decades only, in the paper's 0.1/1/10 convention.
        axi.xaxis.set_major_locator(LogLocator(numticks=5))
        axi.xaxis.set_minor_locator(LogLocator(subs=(2.0, 5.0), numticks=12))
        axi.xaxis.set_minor_formatter(NullFormatter())
        format_log_axis_decimal(axi, axis="x")
        axi.tick_params(labelsize=5.5, pad=1.0, length=2.0)
        axi.tick_params(which="minor", length=1.2)
        axi.set_xlabel(r"$\mu$", fontsize=6, labelpad=-0.5)

    fig.subplots_adjust(left=0.17, right=0.96, bottom=0.16, top=0.95)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig_magnification_pdf_zs{args.out_suffix}.{ext}", dpi=300)
    plt.close(fig)
    print(f"[fig] wrote {OUT_DIR}/fig_magnification_pdf_zs{args.out_suffix}.{{pdf,png}}", flush=True)

    # ---- Figure 3: compensated tail, mu^2 dP/dmu (appendix) -----------------
    # Separate figure because the body and the tail cannot share an axis: the
    # 1-99% mass spans 0.07 decades at z_s=0.5 and 0.61 at z_s=10, against the
    # ~2 decades the tail needs. One panel per job.
    tail_zs = [z for z in args.tail_zs if f"full_z{z:g}" in data]
    if tail_zs and edges[-1] >= args.tail_fit_mu:
        figt, axt3 = plt.subplots(figsize=FIGURE_SIZES["single"])
        # Same colour per z_s as the body figure, so a reader comparing the two
        # panels can follow a redshift across them.
        tcolors = [colors[ZS_LIST.index(z)] if z in ZS_LIST else cmap(0.5)
                   for z in tail_zs]
        lo, hi = args.tail_mu
        hi = min(hi, float(edges[-1]))
        ref = 0.0
        for zs, col in zip(tail_zs, tcolors):
            counts, ntot, nlo, nhi = data[f"full_z{zs:g}"]
            x, y, ey = _compensated_tail(counts, edges, ntot, lo, hi,
                                         args.tail_nbins, args.min_counts)
            if not x.size:
                print(f"[warn] tail panel: z_s={zs:g} has no bin with "
                      f">={args.min_counts} rays over mu in [{lo:g},{hi:g}]",
                      flush=True)
                continue
            axt3.errorbar(x, y, yerr=ey, color=col, lw=1.0, marker="o", ms=2.5,
                          capsize=1.5, elinewidth=0.7, label=fr"${zs:g}$")
            # Each curve gets its OWN flat reference at its own asymptotic
            # level. A single shared line would sit on the topmost curve and
            # read as a fit to it; the claim here is that EVERY curve is flat.
            asym = x >= args.tail_fit_mu
            if asym.any():
                lvl = float(np.median(y[asym]))
                gxi = np.geomspace(args.tail_fit_mu, hi, 8)
                axt3.plot(gxi, np.full_like(gxi, lvl), color=col, lw=0.7,
                          ls="--", dashes=(3, 2), alpha=0.55, zorder=1)
                ref = max(ref, lvl)
        if ref > 0:
            gx = np.geomspace(args.tail_fit_mu, hi, 32)
            # NB: \td is a preamble macro of the paper, not known to matplotlib
            # mathtext -- always spell derivatives as \mathrm{d} in figure text.
            # Labels sit INSIDE the axes: right-aligned on the last gridpoint,
            # so the text grows leftward and cannot run off the frame however
            # long it is (centering it clipped the exponent).
            axt3.text(gx[-1], ref * 1.55,
                      r"flat $\Leftrightarrow\ \mathrm{d}P_I/\mathrm{d}\mu"
                      r"\propto\mu^{-2}$",
                      fontsize=6.5, color="0.3", ha="right", va="bottom")
            if args.tail_mu3:
                # Source-plane contrast. Off by default: in the compensated
                # variable the data being FLAT already says mu^-2, and a steeply
                # falling reference line the data never approaches adds nothing
                # but ink and vertical range.
                gy3 = ref * (gx / args.tail_fit_mu) ** (-1.0)
                axt3.plot(gx, gy3, color="0.55", lw=0.8, ls=":", zorder=1)
                k3 = int(0.45 * (len(gx) - 1))
                axt3.text(gx[k3] * 1.15, gy3[k3], r"$\propto\mu^{-3}$",
                          fontsize=6.5, color="0.55", ha="left", va="center")
        axt3.axvline(args.tail_fit_mu, color="0.75", lw=0.6, ls="-", zorder=0)
        axt3.set_xscale("log")
        axt3.set_yscale("log")
        axt3.set_xlim(lo, hi)
        # headroom for the legend above and the mu^-3 guide below
        y0, y1 = axt3.get_ylim()
        axt3.set_ylim(y0 * 0.75, y1 * 3.5)
        format_log_axis_decimal(axt3, axis="x")
        axt3.set_xlabel(r"$\mu$")
        axt3.set_ylabel(r"$\mu^2\,\mathrm{d}P_I/\mathrm{d}\mu$")
        axt3.legend(title=r"$z_s$", frameon=False, fontsize=7, title_fontsize=7,
                    handlelength=1.2, labelspacing=0.25, loc="upper left",
                    ncol=3, columnspacing=1.0, borderaxespad=0.4)
        figt.subplots_adjust(left=0.17, right=0.96, bottom=0.16, top=0.95)
        for ext in ("pdf", "png"):
            figt.savefig(OUT_DIR / f"fig_magnification_pdf_tail{args.out_suffix}.{ext}", dpi=300)
        plt.close(figt)
        print(f"[fig] wrote {OUT_DIR}/fig_magnification_pdf_tail{args.out_suffix}.{{pdf,png}}",
              flush=True)
    elif tail_zs:
        print(f"[warn] tail figure skipped: cache ends at mu = {edges[-1]:g}, "
              f"below the asymptotic threshold mu = {args.tail_fit_mu:g}. "
              f"Re-run the MC on the current (log, wide) grid.", flush=True)

    # ---- Figure 2: dP/dmu at fixed z_s, ingredients + ratio to baseline ------
    # The subhalo/clustering effects are ~10-20% in localized regions, invisible
    # on a bare log overlay, so the lower panel shows the ratio to the baseline.
    zi = args.zs_ingredients
    base_tag = f"base_z{zi:g}"
    # A cache written with --no-ingredients (or a top-up shard set) carries only
    # the full_z* arms. Skip the ingredient figure instead of raising KeyError
    # halfway through, after the body and tail figures have already been written.
    if base_tag not in data:
        print(f"[warn] ingredient figure skipped: no '{base_tag}' in this cache "
              f"(written with --no-ingredients?). Body and tail figures are done.",
              flush=True)
        return
    ing = [
        (f"base_z{zi:g}", "baseline (Vaskonen 2026)", "0.35", "-"),
        (f"subh_z{zi:g}", r"$+$ subhalos", "#1f77b4", "--"),
        (f"full_z{zi:g}", r"$+$ clustering (full)", "#d62728", "-"),
    ]

    def _grid(tag):
        counts, ntot = data[tag][0], data[tag][1]
        # all three ingredient arms share one z_s, so they must share one rebin
        # factor for the ratio panel to be meaningful -- take it from the
        # baseline arm rather than per curve.
        f = _auto_rebin(data[base_tag][0], edges) if factor <= 0 else factor
        c, e = _rebin(counts, edges, f)
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
    axt.set_ylabel(r"$\mathrm{d}P_I/\mathrm{d}\mu$")
    axt.text(0.95, 0.92, fr"$z_s = {zi:g}$", transform=axt.transAxes,
             ha="right", va="top", fontsize=8)
    axt.legend(frameon=False, fontsize=7, handlelength=1.4, labelspacing=0.25)
    axr.axhline(1.0, color="0.35", lw=0.7, ls=":")
    axr.set_ylim(0.7, 2.0)
    axr.set_xlabel(r"$\mu$")
    axr.set_ylabel(r"ratio")
    fig.subplots_adjust(left=0.17, right=0.96, bottom=0.12, top=0.97, hspace=0.08)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig_magnification_pdf_ingredients{args.out_suffix}.{ext}", dpi=300)
    plt.close(fig)
    print(f"[fig] wrote {OUT_DIR}/fig_magnification_pdf_ingredients{args.out_suffix}.{{pdf,png}}",
          flush=True)


# ============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nreal", type=int, default=4_000_000,
                    help="realizations per series (per z_s / per config). "
                         "4e6 is the production value (2026-07-28): it puts the "
                         "worst compensated-tail bin at 5-7%% for z_s >= 5 and "
                         "11-17%% at z_s = 2-3. Shard over seeds and --combine "
                         "rather than running this in one process.")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", type=str, default="run1",
                    help="cache label -> paper_prod/plots/data/magpdf_<tag>.npz")
    ap.add_argument("--body-zs", type=float, nargs="*",
                    default=[0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
                    help="subset of ZS_LIST to DRAW on the body figure. All of "
                         "ZS_LIST is still simulated -- 3 and 7 are omitted "
                         "here to keep the body legible but are used by the "
                         "compensated tail figure. Empty = draw all")
    ap.add_argument("--zs-ingredients", type=float, default=5.0,
                    help="fixed z_s for the ingredient-comparison figure")
    ap.add_argument("--plane", choices=("image", "source"), default="image",
                    help="which plane the BODY figure shows. image (default) is "
                         "what sample_lnmu returns and what the paper uses. "
                         "source reweights by 1/mu -- for comparison with "
                         "Vaskonen 2026 Fig. 2 and with ACE, which are source "
                         "plane. Use --out-suffix so it cannot overwrite the "
                         "paper figure.")
    ap.add_argument("--zs-list", type=float, nargs="+", default=None,
                    help="override ZS_LIST for THIS run. Use for a top-up shard "
                         "that adds one source redshift to an existing set, "
                         "then --combine old and new shards together.")
    ap.add_argument("--no-ingredients", action="store_true",
                    help="skip the base/subh ingredient arms. REQUIRED on a "
                         "top-up run: --combine SUMS matching tags, so rerunning "
                         "the ingredient arms would silently double their counts.")
    ap.add_argument("--replot", action="store_true",
                    help="skip MC, load the <tag> cache and just (re)plot")
    ap.add_argument("--combine", nargs="+", metavar="NPZ",
                    help="sum these shard caches (glob ok), save as <tag>, plot")
    ap.add_argument("--no-plot", action="store_true",
                    help="run the MC and write the cache, but do not plot")
    # display knobs
    ap.add_argument("--mu-range", type=float, nargs=2, default=[0.5, 2.5],
                    help="displayed mu window on Fig. 1 (the BODY figure). "
                         "Wider than Vaskonen 2026 Fig. 2 ([0.6,1.8]) because "
                         "the lower ylim exposes more of both wings")
    ap.add_argument("--logx", dest="logx", action="store_true", default=False,
                    help="log-log main panel. NOT the default and not "
                         "recommended: the 1-99%% mass spans 0.07 decades at "
                         "z_s=0.5 vs the 2.5 decades a tail panel needs, so "
                         "the low-z_s bodies degenerate into vertical spikes. "
                         "The tail lives in Fig. 3 instead.")
    ap.add_argument("--no-logx", dest="logx", action="store_false",
                    help="linear-mu body panel in the style of Vaskonen 2026 "
                         "Fig. 2 (default)")
    ap.add_argument("--guide-mu", type=float, default=8.0,
                    help="mu at which the slope guides are anchored on the "
                         "log-log panel (8 = the tail-fit threshold)")
    ap.add_argument("--guide-slopes", type=float, nargs="+", default=[-2.0],
                    help="power-law guides to draw, sharing one anchor. "
                         "-2 = image-plane fold-caustic law (what this "
                         "simulator samples); add -3 for the source-plane "
                         "contrast, e.g. --guide-slopes -2 -3")
    ap.add_argument("--edge-q", type=float, default=0.0,
                    help="if > 0, mark this quantile of each curve as a tick on "
                         "the lower axis of Fig. 1 (the measured low-mu edge). "
                         "Off by default: the curves already terminate at the "
                         "edge, so the ticks duplicated it. 0.001 restores them")
    ap.add_argument("--inset", dest="inset", action="store_true", default=False,
                    help="draw the log-log high-mu tail inset on Fig. 1; only "
                         "meaningful with --no-logx")
    ap.add_argument("--no-inset", dest="inset", action="store_false")
    ap.add_argument("--tail-range", type=float, nargs=2, default=[1.5, 100.0],
                    help="mu window shown in the tail inset")
    ap.add_argument("--tail-rebin", type=int, default=40,
                    help="coarser rebin for the inset (tail bins are sparse)")
    ap.add_argument("--guide-norm", type=float, default=3.0,
                    help="how far ABOVE the highest tail curve to place the "
                         "mu^-2 slope guide (multiplicative); purely cosmetic")
    ap.add_argument("--inset-box", type=float, nargs=4,
                    default=[0.52, 0.46, 0.44, 0.48],
                    help="inset axes box (x0 y0 w h) in axes fraction")
    ap.add_argument("--ylim", type=float, nargs=2, default=[2e-2, 110.0],
                    help="dP/dmu range on the body figure (the z_s=0.2 peak "
                         "reaches ~50-100, hence the high upper limit)")
    # ---- Fig. 3: compensated tail panel (appendix) --------------------------
    ap.add_argument("--tail-zs", type=float, nargs="+",
                    default=[2.0, 3.0, 5.0, 7.0, 10.0],
                    help="source redshifts on the compensated tail figure. "
                         "z_s <= 1 is omitted by default because the tail is "
                         "unsampled there (S(mu>8) ~ 1e-6 at z_s=0.5, i.e. ~1 "
                         "ray in 480k -- not a depth problem, a physics one)")
    ap.add_argument("--tail-nbins", type=int, default=7,
                    help="log bins across the compensated tail window")
    ap.add_argument("--tail-mu", type=float, nargs=2, default=[2.0, 100.0],
                    help="mu window of the compensated tail figure")
    ap.add_argument("--tail-mu3", action="store_true",
                    help="also draw the source-plane mu^-3 contrast on the "
                         "compensated tail figure (off by default)")
    ap.add_argument("--tail-fit-mu", type=float, default=20.0,
                    help="mu above which the mu^-2 asymptote is taken to hold. "
                         "Marked on the figure and used for each curve's flat "
                         "reference level. 20, not the 8 of "
                         "docs/edge_tail_flux_note.md: measured on real data "
                         "the survival exponent is still ~2.25 at mu>8 and only "
                         "reaches 2.0 above mu~20-30, so a reference taken from "
                         "mu>8 sits above the actual plateau")
    ap.add_argument("--rebin-hi", type=int, default=60,
                    help="rebin factor ABOVE --rebin-pivot on the log-log panel "
                         "(the tail needs coarser bins than the body)")
    ap.add_argument("--rebin-pivot", type=float, default=1.6,
                    help="mu at which --rebin gives way to --rebin-hi")
    ap.add_argument("--min-counts", type=int, default=8,
                    help="drop display bins with fewer raw counts than this; "
                         "stops the far tail degenerating into a single-count "
                         "floor that reads as signal")
    ap.add_argument("--rebin", type=int, default=0,
                    help="merge this many stored fine bins per display bin. "
                         "0 (default) = choose per curve from its own "
                         "sigma(lnmu), which is what keeps the narrow low-z_s "
                         "bodies resolved and the wide ones smooth")
    ap.add_argument("--high-structure", action="store_true",
                    help="use high-structure cosmology (h=0.72, Om=0.38, s8=1.0) "
                         "where the mu^-2 tail is populated and clean across redshifts")
    ap.add_argument("--out-suffix", type=str, default="",
                    help="appended to every output figure stem, e.g. "
                         "'_highstruct'. The paper uses the FIDUCIAL cosmology "
                         "for fig_magnification_pdf_zs and the HIGH-STRUCTURE "
                         "one for fig_magnification_pdf_tail_highstruct, so the "
                         "two must not overwrite each other -- pass this "
                         "whenever --high-structure is set")
    ap.add_argument("--as-hist", action="store_true",
                    help="plot curves as step histograms rather than smooth lines")
    ap.add_argument("--smooth", type=int, default=0,
                    help="odd Savitzky-Golay window on log10(dP/dmu) for display "
                         "(0 = off; prefer more --nreal over smoothing)")
    args = ap.parse_args()

    global COSMO
    if args.high_structure:
        COSMO = COSMO_HIGH

    cache_path = DATA_DIR / f"magpdf_{args.tag}.npz"

    if args.combine:
        paths = []
        for pat in args.combine:
            paths.extend(sorted(glob.glob(pat)))
        if not paths:
            sys.exit(f"--combine matched no files: {args.combine}")
        print(f"[combine] {len(paths)} shard(s)", flush=True)
        data, meta, edges = combine_caches(paths)
        save_cache(cache_path, data, {"combined_from": paths}, edges)
    elif args.replot:
        if not cache_path.exists():
            sys.exit(f"--replot: cache not found: {cache_path}")
        data, meta, edges = load_cache(cache_path)
    else:
        zs_list = args.zs_list if args.zs_list else ZS_LIST
        series = build_series(args.zs_ingredients, zs_list=zs_list,
                              ingredients=not args.no_ingredients)
        data = run_series(series, args.nreal, args.seed)
        edges = STORE_EDGES
        save_cache(cache_path, data,
                   {"cosmo": COSMO, "nreal": args.nreal, "seed": args.seed,
                    "zs_list": zs_list, "zs_ingredients": args.zs_ingredients,
                    "ingredients": not args.no_ingredients,
                    # what physics this cache actually holds -- see the class of
                    # drift documented in paper_writer.md sec 7
                    "physics_config_hash": PRODUCTION_CONFIG_HASH,
                    "physics_config": PRODUCTION_CONFIG},
                   edges)

    if not args.no_plot:
        plot_figures(data, args, edges)


if __name__ == "__main__":
    main()
