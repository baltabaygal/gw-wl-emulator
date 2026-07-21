#!/usr/bin/env python3
"""
Production plot: convergence variance partitioning vs host resolution threshold
under the correlated bias field (bias_model=1, production Rperp=8441 kpc) WITH
the conditional weak arm (bias_weak) — the bias-era counterpart of
plot_sigma_partition_vs_kthr.py.

All three curves are MEASURED from the production MC via the kappa_weak per-ray
diagnostic (weak = background arm draw, strong = kappa - kappa_weak), on the
shared kappa_tot <= 1 core mask (weak-lensing validity, kappa_anchor_cut
convention), pooled over seed ensembles; on the same masked rays
Var_w + Var_s + 2 Cov = Var_tot holds exactly, so the flatness of sigma_total
in kappa_thr is the conservation statement for the certified core.

Why clipped, not raw: the formal (raw) clustering variance of the exponentiated
linear-bias field is dominated by >5 sigma field excursions (P ~ 1e-7; the
6-sigma production table clamp is all that keeps it finite) and raw Var
estimates are monster-ray seed junk — standing rule. The law-level conservation
of the RAW variance is established analytically by
playground/sigma_partition_bias_vs_kthr.cpp (shot part to 1.4e-9, total
including clustering + cross terms to 4.7e-4 with common-mode field draws);
its curves are stored in the npz (sigW/sigS/cov/sig_tot _analytic) but are NOT
plotted: a raw-law curve over core-clipped measurements would mix estimators.

Inputs (regenerate before use), for the default ENSEMBLE = "crn":
  playground/partition_components_zs1_seed<100..131>_crn.npz
      <- sweep_sigma_partition_components.py ... main 1   (32 parallel seeds,
         common random numbers along the kappa_thr grid)
  playground/partition_components_zs1_deepcore_seed<100..131>_crn.npz
      <- sweep_sigma_partition_components.py ... deepcore 1  (the three
         in-domain deep points at 4x rays)
  playground/partition_components_zs1_deep_seed<0..7>.npz
      <- mode=deep; ONLY the sub-floor points (kappa_thr < kappa_min) are taken,
         as npz documentation of the floor semantics — they are not plotted
  playground/sigma_partition_bias_vs_kthr_zs1.txt
      <- build/sigma_partition_bias_vs_kthr (analytic probe)
ENSEMBLE = "legacy" reproduces the original 8-seedbase independent-seed figure
from partition_components_zs1_{seed,deep_seed}<0..7>.npz.
Consolidates into data/sigma_partition_vs_kthr_bias_z1[_crn].npz (citable source),
writes paper_prod/plots/figures/sigma_partition_vs_kthr_bias.{png,pdf} and a
metadata JSON.
"""
from pathlib import Path
import glob
import json
import datetime
import sys
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper_prod.plot_style import (  # noqa: E402
    apply_style,
    FIGURE_SIZES,
    SUBPLOTS_ADJUST,
    format_log_axis_decimal,
)

SIZES = apply_style()
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"]
mpl.rcParams["mathtext.fontset"] = "cm"

PG = ROOT / "playground"
_DATA = {"legacy": "sigma_partition_vs_kthr_bias_z1.npz",
         "crn": "sigma_partition_vs_kthr_bias_z1_crn.npz"}
OUT_DIR = ROOT / "paper_prod" / "plots" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MD_DIR = ROOT / "paper_prod" / "metadata"
MD_DIR.mkdir(parents=True, exist_ok=True)

ESTIMATOR = "clip1"  # shared-mask core estimator plotted (kappa_tot <= 1)
KMIN = 1.278e-7      # absolute background floor 1e-3 * kappa_thr(N=100, zs=1)

# Which sweep ensemble to consolidate. "crn" = the 32-seedbase common-random-
# numbers run (one seed per seedbase for the whole kappa_thr grid, so the
# estimation errors are correlated ALONG the sweep) plus the deepcore
# regeneration of the three in-domain deep points at 4x rays. Each point stays
# individually unbiased; CRN only suppresses the point-to-point scatter, which
# is the thing this figure is reading off. "legacy" = the original 8-seedbase
# independent-seed run (seed0+i), kept so the published figure stays
# reproducible. See sweep_sigma_partition_components.py.
ENSEMBLE = "crn"
DATA = ROOT / "data" / _DATA[ENSEMBLE]


def git_commit_short():
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT)).decode().strip()
    except Exception:
        return "unknown"


def _load_group(pattern, kthr_max=None):
    """Stack one mode's seed files -> (kthr, {est: (nseed, nkt, 7)}, nsamples_pt).

    kthr_max, if given, keeps only points strictly below it — used to take the
    sub-floor documentation points from mode=deep without duplicating the
    in-domain ones that mode=deepcore regenerates at higher statistics.
    """
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    per = [np.load(f) for f in files]
    kthr = per[0]["kthr"]
    ns_pt = (per[0]["nsamples_pt"] if "nsamples_pt" in per[0].files
             else np.full(kthr.size, int(per[0]["nsamples"])))
    keep = np.ones(kthr.size, bool) if kthr_max is None else kthr < kthr_max
    if not keep.any():
        return None
    return (kthr[keep],
            {est: np.stack([p[est] for p in per])[:, keep, :]
             for est in ("raw", "clip1", "q999")},
            ns_pt[keep])


def consolidate():
    """Rebuild the npz from the sweep outputs (if they exist).

    Groups may carry different seed counts (the CRN main/deepcore runs use 32
    seedbases, the sub-floor deep points are inherited from the original
    8-seedbase run), so the per-seed cube is NaN-padded to the widest group and
    all reductions are nan-aware. Seed counts per point are recorded in
    nseeds_pt; `nseeds` is the count backing the plotted domain.
    """
    suffix = "_crn" if ENSEMBLE == "crn" else ""
    main = _load_group(str(PG / f"partition_components_zs1_seed*{suffix}.npz"))
    if ENSEMBLE == "crn":
        deepcore = _load_group(str(PG / "partition_components_zs1_deepcore_seed*_crn.npz"))
        # sub-floor points only: npz documentation, not plotted, not regenerated
        deep = _load_group(str(PG / "partition_components_zs1_deep_seed*.npz"), kthr_max=KMIN)
        groups = [g for g in (main, deepcore, deep) if g is not None]
    else:
        deep = _load_group(str(PG / "partition_components_zs1_deep_seed*.npz"))
        groups = [g for g in (main, deep) if g is not None]
    probe = PG / "sigma_partition_bias_vs_kthr_zs1.txt"
    if main is None or not probe.exists():
        return
    kthr = np.concatenate([g[0] for g in groups])
    order = np.argsort(kthr)
    kthr = kthr[order]
    ns_pt = np.concatenate([g[2] for g in groups])[order]
    nseed_max = max(g[1]["raw"].shape[0] for g in groups)
    nseeds_pt = np.concatenate([
        np.full(g[0].size, g[1]["raw"].shape[0]) for g in groups])[order]
    save = {"kthr": kthr, "zs": 1.0, "rperp": 8441.0, "ensemble": ENSEMBLE,
            "nsamples_pt": ns_pt, "nseeds_pt": nseeds_pt,
            "nseeds": int(nseeds_pt[kthr >= KMIN].min())}
    for est in ("raw", "clip1", "q999"):
        # (nseed, nkt, 7): var_w var_s cov var_t mean_w mean_s n
        blocks = []
        for g in groups:
            a = g[1][est]
            if a.shape[0] < nseed_max:
                pad = np.full((nseed_max - a.shape[0],) + a.shape[1:], np.nan)
                a = np.concatenate([a, pad], axis=0)
            blocks.append(a)
        arr = np.concatenate(blocks, axis=1)[:, order, :]
        n_eff = np.sum(~np.isnan(arr[:, :, 3]), axis=0)
        save[f"{est}_per_seed"] = arr
        save[f"var_w_{est}"] = np.nanmean(arr[:, :, 0], axis=0)
        save[f"var_s_{est}"] = np.nanmean(arr[:, :, 1], axis=0)
        save[f"cov_ws_{est}"] = np.nanmean(arr[:, :, 2], axis=0)
        save[f"var_t_{est}"] = np.nanmean(arr[:, :, 3], axis=0)
        save[f"var_t_{est}_sem"] = np.nanstd(arr[:, :, 3], axis=0, ddof=1) / np.sqrt(n_eff)
    (kt_a, sWsh, sWco, sWt, sSsh, sSco, sSt, cov_a, sT_a) = np.loadtxt(probe, unpack=True)
    save.update(kthr_analytic=kt_a,
                sigW_shot_analytic=sWsh, sigW_corr_analytic=sWco, sigW_analytic=sWt,
                sigS_shot_analytic=sSsh, sigS_corr_analytic=sSco, sigS_analytic=sSt,
                cov_ws_analytic=cov_a, sig_tot_analytic=sT_a)
    # <N>(kappa_thr) mapping for the <N>-axis companion figure (same lattice)
    nexp_file = PG / "nexp_vs_kthr_zs1.txt"
    if nexp_file.exists():
        kt_n, nexp = np.loadtxt(nexp_file, unpack=True)
        nmap = {round(4.0 * np.log10(k)): n for k, n in zip(kt_n, nexp)}
        save["nexp"] = np.array([nmap.get(round(4.0 * np.log10(k)), np.nan) for k in kthr])
        save["nexp_analytic"] = np.array([nmap.get(round(4.0 * np.log10(k)), np.nan) for k in kt_a])
    np.savez(DATA, **save)
    print(f"Consolidated {DATA.relative_to(ROOT)} ({save['nseeds']} seeds, {kthr.size} kthr points)")


def main():
    consolidate()
    d = np.load(DATA)
    kthr = d["kthr"]
    sig_w = np.sqrt(d[f"var_w_{ESTIMATOR}"])
    sig_s = np.sqrt(d[f"var_s_{ESTIMATOR}"])
    sig_t = np.sqrt(d[f"var_t_{ESTIMATOR}"])

    # weak-arm decomposition: shot = the analytic Campbell sigma_W (exact — it is
    # what production injects, and it is clip-insensitive: a ~0.027-sigma Gaussian
    # loses nothing to the kappa<=1 mask); corr = the field-driven part, derived
    # on the shared core mask as sqrt(Var_w - sigma_shot^2). Grids are both on the
    # quarter-decade lattice, so the subtraction is point-exact (no interpolation).
    shot_map = {round(4.0 * np.log10(k)): s
                for k, s in zip(d["kthr_analytic"], d["sigW_shot_analytic"])}
    sig_w_shot = np.array([shot_map[round(4.0 * np.log10(k))] for k in kthr])
    sig_w_corr = np.sqrt(np.maximum(sig_w**2 - sig_w_shot**2, 0.0))

    core = kthr >= KMIN
    plateau = float(sig_t[core].mean())
    flat = (sig_t[core].max() - sig_t[core].min()) / plateau
    deep_ratio = float(sig_t.max() / plateau)

    # Plot the model domain only: kappa_thr >= kappa_min. Below the floor the
    # sweep leaves the model's fixed halo budget — the weak arm is empty by
    # construction and the explicit arm picks up the sub-floor band no arm ever
    # covered, so the total genuinely grows (strong == total there; sub-floor
    # points are kept in the npz as documentation of the floor semantics).
    kthr_p, w_p, wsh_p, wco_p, s_p, t_p = (a[core] for a in
        (kthr, sig_w, sig_w_shot, sig_w_corr, sig_s, sig_t))

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single"])
    fig.subplots_adjust(**SUBPLOTS_ADJUST["single"])

    ax.axhline(plateau, color="#94a3b8", ls=":", lw=0.8, zorder=1)
    w_line, = ax.plot(kthr_p, w_p, lw=2.0, color="#0f4c81", zorder=6)
    wsh_line, = ax.plot(kthr_p, wsh_p, lw=1.0, ls=(0, (1, 1.2)), color="#0f4c81",
                        alpha=0.85, zorder=3)
    wco_line, = ax.plot(kthr_p, wco_p, lw=1.0, ls=(0, (4, 1.5, 1, 1.5)),
                        color="#5b8fbe", zorder=3)
    s_line, = ax.plot(kthr_p, s_p, lw=1.5, color="#d97706", zorder=4)
    t_line, = ax.plot(kthr_p, t_p, ls="--", lw=1.2, color="#94a3b8", zorder=7)

    ax.set_xscale("log")
    ax.set_xlabel(r"$\kappa_{\rm threshold}$")
    ax.set_ylabel(r"$\sigma_{\kappa}$")
    ax.set_ylim(-0.05 * plateau, 1.23 * plateau)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.grid(False)
    ax.legend([w_line, wsh_line, wco_line, s_line, t_line],
              [r"weak", r"weak (shot)", r"weak (corr.)", r"strong", r"total"],
              loc="center left", fontsize=6, handlelength=2.0, borderaxespad=0.8,
              labelspacing=0.35)

    try:
        format_log_axis_decimal(ax, axis='x')
    except Exception:
        pass

    out_png = OUT_DIR / "sigma_partition_vs_kthr_bias.png"
    out_pdf = out_png.with_suffix('.pdf')
    fig.savefig(out_png, dpi=300, facecolor="white")
    fig.savefig(out_pdf, facecolor="white")
    plt.close(fig)

    meta = {
        "script": str(Path(__file__).relative_to(ROOT)),
        "generated": datetime.datetime.utcnow().isoformat() + "Z",
        "git_commit": git_commit_short(),
        "source_data": str(DATA.relative_to(ROOT)),
        "estimator": ESTIMATOR,
        "ensemble": ENSEMBLE,
        "nseeds_plotted_domain": int(d["nseeds"]),
        "plateau_sigma": plateau,
        "flatness_relrange_sigma_total_core": float(flat),
        "below_floor_max_sigma_ratio": deep_ratio,
        "notes": "zs=1, halo-only + correlated bias field (bias_model=1, Rperp=8441 kpc) "
                 "with the conditional weak arm (bias_weak). weak/strong/total measured from "
                 "sample_lensing_raw_ml via the kappa_weak per-ray split, kappa_tot<=1 core "
                 "mask, %d seeds, 32k-120k rays per threshold (adaptive below 1e-5); "
                 "the plotted ensemble uses common random numbers along the kappa_thr "
                 "grid (one seed per seedbase for the whole sweep), which leaves every "
                 "point individually unbiased but correlates the errors along the sweep "
                 "and so suppresses point-to-point scatter (pilot: rms second-difference "
                 "roughness 2.71%% -> 1.36%% at fixed cost); the plateau LEVEL is "
                 "therefore backed by the seedbase count, not by nseeds*npoints. "
                 "Var_w+Var_s+2Cov=Var_tot exact on the shared mask. weak(shot) = analytic "
                 "Campbell sigma_W (floor-consistent, = production injection, clip-"
                 "insensitive); weak(corr.) = sqrt(Var_w - shot^2) on the core mask. "
                 "The FIGURE plots the model domain kappa_thr >= kappa_min = 1.28e-7 "
                 "(absolute floor 1e-3*kappa_thr(N=100)) where conservation holds; the "
                 "npz additionally holds sub-floor points (down to 1e-8) documenting the "
                 "floor semantics: there the weak arm is empty by construction, strong == "
                 "total exactly, and the total genuinely grows (law-level 1.145x plateau "
                 "in Var at 1e-8) because the explicit arm re-includes the sub-floor band "
                 "no arm ever covered — the band's clustering mean grows ~ln(1/kappa) "
                 "(untruncated NFW), which is why the model is defined with an absolute "
                 "floor; below kt~3e-8 the rmaxfNFW r<=1e6 kpc bisection ceiling also "
                 "truncates the largest cells. Raw-law "
                 "conservation (incl. clustering + cross terms) established analytically by "
                 "playground/sigma_partition_bias_vs_kthr.cpp (shot to 4.4e-9, total to "
                 "4.7e-4 for kt>=kappa_min); raw MC Var is tail-junk (formal clustering "
                 "variance doubles between 5 and 6 sigma field excursions - clamp scan in "
                 "the probe)." % int(d["nseeds"]),
        "output_png": str(out_png.relative_to(ROOT)),
        "output_pdf": str(out_pdf.relative_to(ROOT)),
    }
    (MD_DIR / "sigma_partition_vs_kthr_bias.metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"plateau sigma_total (kt>=kappa_min) = {plateau:.6f}   "
          f"flatness (max-min)/mean = {flat:.4f}")
    print(f"below-floor max sigma_total = {deep_ratio:.3f} x plateau")
    print(f"Wrote {out_png} and {out_pdf} and metadata")


if __name__ == '__main__':
    main()
