#!/usr/bin/env python
"""Fig. flow_transformation: the generative path of the conditional NSF.

Compact single-column figure showing how the flow carries the standardized
Gaussian base density through its K stacked rational-quadratic spline
transforms onto the magnification PDF dP/dmu at one (z_s, Theta).

Supersedes `scripts/figures/plot_flow_layer_transformation.py` (2026-07-28).
Three things changed; read them before comparing to the old 8-panel version.

1. LAYER ORDER.  zuko's `Flow.transform` is a ComposedTransform that maps
   DATA -> LATENT by applying transforms[0] first and transforms[-1] last.  The
   generative (latent -> data) path is therefore the inverses applied in
   REVERSE order: transforms[-1].inv, ..., transforms[0].inv.  The old script
   walked them forward, so its intermediate densities were not points on the
   flow's own path and its final panel did not reproduce `model.log_prob` --
   which is visible in the old figure as a peak/offset mismatch against the
   dashed reference.  This script does not assume the convention: it walks BOTH
   orders, scores each final density against `model.log_prob`, asserts one of
   them matches, and prints both residuals (`--verbose` for the full table).

2. HONEST REFERENCE CURVE.  `ConditionalNSF.log_prob` is the bare flow: no
   mu^-2 tail, no empty-beam edge, no flux calibration.  The dashed curve is
   labelled as the flow density, not as the composite/hybrid model.  It is a
   CLOSURE CHECK on the layer walk (final layer must land on it), not an
   accuracy claim about the emulator.

3. MODEL-AGNOSTIC.  Context dimension is read off the checkpoint's
   `context_mean` buffer, so the same script runs on the legacy 1+3d
   checkpoint (context = z_s, h, Om, sigma8) and on a retrained 1+6d model
   (context = ml.params.CONTEXT_KEYS).  The context vector is built from
   ml.params.FIDUCIAL unless overridden.

NOTE ON WORDING: with `features=1` zuko's NSF is autoregressive (a context-
conditioned monotonic spline per transform), NOT a coupling flow -- there is no
second half of the vector to condition on.  Say "spline transforms", not
"coupling layers", in the caption.

Usage (needs the `test` conda env: torch + zuko live there, not in system py3.13)

    PY=/Users/baltabay/miniforge3/envs/test/bin/python
    $PY paper_prod/scripts/plot_fig_flow_transformation.py
    $PY paper_prod/scripts/plot_fig_flow_transformation.py --zs 5.0 --verbose
    $PY paper_prod/scripts/plot_fig_flow_transformation.py --replot   # restyle from cache

Writes:
    paper_prod/plots/figures/fig_flow_transformation.{pdf,png}
    paper_prod/plots/data/flow_transformation_<tag>.npz   (histogram cache)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from paper_prod.plot_style import apply_style, FIGURE_SIZES  # noqa: E402

FIG_DIR = ROOT / "paper_prod" / "plots" / "figures"
CACHE_DIR = ROOT / "paper_prod" / "plots" / "data"
DEFAULT_MODEL = ROOT / "data" / "models" / "conditional_nsf_backend_current.pt"


def guard_broken_latex():
    """apply_style enables usetex when a `latex` binary exists; fall back to
    mathtext if the TeX tree is unusable (e.g. sandbox installs).
    Same guard as plot_fig_subhalo_population.py."""
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


# --------------------------------------------------------------------------- #
# context construction
# --------------------------------------------------------------------------- #
def build_context(zs: float, context_dim: int, overrides: dict) -> tuple[np.ndarray, str]:
    """Context row for the checkpoint's dimensionality, from ml.params.FIDUCIAL.

    context_dim == 4 -> legacy 1+3d layout (z, h, Om, sigma8)
    context_dim == 7 -> ml.params.CONTEXT_KEYS (z, h, Om, sigma8, Ob, ns, zeq/1000)
    """
    from ml.params import FIDUCIAL

    p = dict(FIDUCIAL)
    p.update({k: v for k, v in overrides.items() if v is not None})

    if context_dim == 4:
        ctx = np.array([[zs, p["h"], p["Om"], p["sigma8"]]], dtype=np.float64)
        label = "1+3d"
    elif context_dim == 7:
        from ml.params import theta_to_context

        ctx = theta_to_context(zs, p["h"], p["Om"], p["sigma8"],
                               p["Ob"], p["ns"], p["zeq"]).reshape(1, -1)
        label = "1+6d"
    else:
        raise SystemExit(
            f"context_dim={context_dim} is neither the legacy 4 nor the 6d 7; "
            "add the layout here before using this checkpoint."
        )
    return ctx, label


# --------------------------------------------------------------------------- #
# the generative walk
# --------------------------------------------------------------------------- #
def layer_densities(model, ctx_np, mu_edges, n_samples, seed, verbose=False):
    """Histogram dP/dmu after each step of the latent -> data path.

    Returns (hists, ref, order_used, residuals) where hists has shape
    (K+1, nbins): row 0 is the base density mapped into mu, row k is the
    density after k inverse transforms, row K is the flow's own density.
    """
    import torch

    ctx = torch.tensor(ctx_np, dtype=torch.float32)
    ctx_norm = (ctx - model.context_mean) / model.context_std

    torch.manual_seed(seed)
    flow_dist = model.flow(ctx_norm)
    transforms = list(flow_dist.transform.transforms)
    K = len(transforms)

    z_base = flow_dist.base.sample((n_samples,))

    centers = 0.5 * (mu_edges[:-1] + mu_edges[1:])

    # reference: the flow's own density, evaluated analytically.
    # dP/dmu = (dP/dlnmu) / mu.
    mu_ref = centers
    with torch.no_grad():
        logp_lnmu = model.log_prob(np.log(mu_ref), ctx_np)
    ref = np.asarray(np.exp(logp_lnmu), dtype=np.float64) / mu_ref

    def walk(order):
        """Apply .inv along `order`; return per-step mu-space histograms."""
        rows = []
        lnmu = (z_base * model.lnmu_std + model.lnmu_mean).squeeze(-1).detach().numpy()
        rows.append(np.histogram(np.exp(lnmu).ravel(), bins=mu_edges, density=True)[0])
        curr = z_base
        for t in order:
            curr = t.inv(curr)
            lnmu = (curr * model.lnmu_std + model.lnmu_mean).squeeze(-1).detach().numpy()
            rows.append(np.histogram(np.exp(lnmu).ravel(), bins=mu_edges, density=True)[0])
        return np.array(rows)

    # Do not trust a remembered zuko convention -- test both and let the
    # closure against log_prob decide which one is the generative path.
    cand = {
        "reversed": walk(list(reversed(transforms))),
        "forward": walk(transforms),
    }
    # score on the body only, where the histogram has samples to speak with
    body = ref > 0.02 * ref.max()

    def resid(h):
        return float(np.max(np.abs(h[-1][body] - ref[body])) / ref.max())

    residuals = {k: resid(v) for k, v in cand.items()}
    order_used = min(residuals, key=residuals.get)

    if verbose:
        print("  closure of the final layer against model.log_prob "
              "(max |diff| / peak, body only):")
        for k, v in sorted(residuals.items(), key=lambda kv: kv[1]):
            print(f"    {k:>9s}: {v:.4f}")

    tol = 0.03
    if residuals[order_used] > tol:
        raise SystemExit(
            f"neither transform order closes against model.log_prob "
            f"(best = {order_used} at {residuals[order_used]:.3f} > {tol}). "
            "The layer walk is wrong -- do not ship this figure."
        )
    if residuals[order_used] > 0.4 * min(v for k, v in residuals.items() if k != order_used):
        print("  WARNING: the two orders score similarly; the closure test is "
              "not discriminating. Check the flow depth and the base sample.")

    return cand[order_used], ref, order_used, residuals


# --------------------------------------------------------------------------- #
# cosmetic smoothing (plot-time only -- never touches the cache or the
# closure check, which must run on the raw histogram)
# --------------------------------------------------------------------------- #
def gaussian_smooth_rows(hists, sigma_bins):
    """Light Gaussian-kernel smoothing of each histogram row.

    At finite sample count the intermediate RQS transforms histogram noisier
    than the base/final layers: they have sharp local derivatives at spline
    knots, so a fixed sample count lands unevenly bin to bin even at N~1e6.
    This is display-only -- a KDE-equivalent smoothing of the same estimator,
    same role as `scipy.ndimage.gaussian_filter1d` in the superseded script.
    Implemented with a plain np.convolve kernel so `--replot` does not need
    scipy installed on top of the `test` env's torch/zuko.
    """
    if sigma_bins <= 0:
        return hists
    radius = max(1, int(round(4 * sigma_bins)))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel /= kernel.sum()
    out = np.empty_like(hists)
    for i, row in enumerate(hists):
        # edge-reflect padding: rows are histograms with the total density
        # sitting away from either edge over most of the plotted range
        padded = np.pad(row, radius, mode="reflect")
        out[i] = np.convolve(padded, kernel, mode="valid")
    return out


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def make_figure(hists, ref, mu_centers, zs, ctx_label, out_stem, xlim, ylim):
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors as mcolors
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    apply_style()
    guard_broken_latex()
    plt.rcParams["axes.formatter.use_mathtext"] = True  # cmr10 wants this

    K = hists.shape[0] - 1  # number of transforms
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single"])

    cmap = plt.get_cmap("viridis")  # cm.get_cmap was removed in mpl 3.9
    norm = mcolors.Normalize(vmin=0, vmax=K)

    # intermediate steps: thin, so the eye follows the ramp not any one curve
    for k in range(K + 1):
        c = cmap(norm(k))
        if k == 0:
            ax.plot(mu_centers, hists[k], color=c, lw=1.4, zorder=3)
        elif k == K:
            ax.plot(mu_centers, hists[k], color=c, lw=1.6, zorder=4)
        else:
            ax.plot(mu_centers, hists[k], color=c, lw=0.8, alpha=0.85, zorder=2)

    # closure check: the final layer must sit on the flow's analytic density
    ax.plot(mu_centers, ref, color="0.15", ls=(0, (3.2, 2.0)), lw=1.0, zorder=5,
            label=r"flow density")

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"magnification $\mu$")
    ax.set_ylabel(r"$\mathrm{d}P/\mathrm{d}\mu$")

    ax.legend(loc="upper right", fontsize=7, frameon=False, handlelength=1.6)

    # Slim colorbar carries the layer index; no in-axes annotations, they
    # collide with the stack of intermediate curves in the peak region.
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="4.5%", pad=0.06)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_ticks(list(range(0, K + 1, max(1, K // 3))))
    cb.set_label(r"spline transform $k$ \ ($k=0$: base)"
                 if plt.rcParams.get("text.usetex") else
                 r"spline transform $k$  ($k=0$: base)",
                 fontsize=6.5, labelpad=2)
    cb.ax.tick_params(labelsize=6.5, length=2)

    fig.subplots_adjust(left=0.155, right=0.855, bottom=0.175, top=0.965)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = FIG_DIR / f"{out_stem}.{ext}"
        fig.savefig(path, dpi=300 if ext == "png" else None)
        print(f"  wrote {path.relative_to(ROOT)}")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--zs", type=float, default=5.0)
    ap.add_argument("--h", type=float, default=None)
    ap.add_argument("--Om", type=float, default=None)
    ap.add_argument("--sigma8", type=float, default=None)
    ap.add_argument("--nsamples", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nbins", type=int, default=260)
    ap.add_argument("--murange", type=float, nargs=2, default=(0.30, 2.60),
                    help="histogram range in mu")
    ap.add_argument("--xlim", type=float, nargs=2, default=None,
                    help="plot xlim (default: murange trimmed to the support)")
    ap.add_argument("--ylim", type=float, nargs=2, default=None)
    ap.add_argument("--out", default="fig_flow_transformation")
    ap.add_argument("--tag", default=None, help="cache tag (default: zs<z>)")
    ap.add_argument("--replot", action="store_true",
                    help="rebuild the figure from the cached npz, no torch needed")
    ap.add_argument("--smooth-sigma", type=float, default=1.5,
                    help="Gaussian smoothing kernel width in bins, display-only "
                         "(the cache and the closure check always use the raw "
                         "histogram; 0 disables)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    tag = args.tag or f"zs{args.zs:g}"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"flow_transformation_{tag}.npz"

    if args.replot:
        if not cache.exists():
            raise SystemExit(f"no cache at {cache}; run without --replot first")
        d = np.load(cache, allow_pickle=False)
        hists, ref, centers = d["hists"], d["ref"], d["mu_centers"]
        zs, ctx_label = float(d["zs"]), str(d["ctx_label"])
        print(f"replotting from {cache.relative_to(ROOT)} "
              f"(order used at generation: {str(d['order_used'])})")
    else:
        import torch  # noqa: F401  (fail early with a clear message)
        from ml.phase3_common import load_nsf_model

        # read the context dimension off the checkpoint rather than assuming it
        sd = torch.load(args.model, map_location="cpu", weights_only=False)
        state = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
        context_dim = int(np.asarray(state["context_mean"]).shape[-1])

        model = load_nsf_model(args.model) if context_dim == 4 else None
        if model is None:
            from ml.nsf_model import ConditionalNSF
            model = ConditionalNSF(input_dim=1, context_dim=context_dim)
            model.load_checkpoint(args.model)
        model.eval()

        ctx_np, ctx_label = build_context(
            args.zs, context_dim,
            dict(h=args.h, Om=args.Om, sigma8=args.sigma8),
        )
        print(f"model  : {Path(args.model).name}  (context_dim={context_dim}, {ctx_label})")
        print(f"context: {np.array2string(ctx_np[0], precision=4)}")
        print(f"samples: {args.nsamples:,}")

        mu_edges = np.linspace(args.murange[0], args.murange[1], args.nbins + 1)
        centers = 0.5 * (mu_edges[:-1] + mu_edges[1:])

        hists, ref, order_used, residuals = layer_densities(
            model, ctx_np, mu_edges, args.nsamples, args.seed, verbose=args.verbose
        )
        print(f"generative order: transforms {order_used} "
              f"(closure {residuals[order_used]:.4f} vs "
              f"{residuals['forward' if order_used == 'reversed' else 'reversed']:.4f})")

        np.savez_compressed(
            cache, hists=hists, ref=ref, mu_centers=centers,
            zs=args.zs, ctx=ctx_np, ctx_label=ctx_label,
            order_used=order_used, nsamples=args.nsamples,
            model=Path(args.model).name,
        )
        print(f"  cached {cache.relative_to(ROOT)}")
        zs = args.zs

    # display-only smoothing; cache above and the closure check inside
    # layer_densities() already ran on the raw (unsmoothed) histogram
    hists_plot = gaussian_smooth_rows(hists, args.smooth_sigma)
    if args.smooth_sigma > 0:
        print(f"  smoothing: Gaussian sigma = {args.smooth_sigma:g} bins "
              f"(display only; raw histogram is cached and was used for closure)")

    # default limits: trim to where there is density, round the top up a little
    peak = float(max(hists_plot.max(), ref.max()))
    if args.xlim is None:
        sig = np.where(hists_plot.max(axis=0) > 0.004 * peak)[0]
        lo = float(centers[max(sig[0] - 2, 0)])
        hi = float(centers[min(sig[-1] + 2, len(centers) - 1)])
        xlim = (lo, hi)
    else:
        xlim = tuple(args.xlim)
    ylim = tuple(args.ylim) if args.ylim else (0.0, 1.08 * peak)

    make_figure(hists_plot, ref, centers, zs, ctx_label, args.out, xlim, ylim)


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
