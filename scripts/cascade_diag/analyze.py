#!/usr/bin/env python
"""Analyze the cascade-residual diagnostic (subhalo ingredient).

Reads the npz shards written by run_all.py and produces, per z_s and split into
three regions (low-mu EDGE / BODY / high-mu TAIL, assigned by the base-arm CDF):

  (1) dynamic range of dlogP = logP_B - logP_A  vs  base logP_A
  (2) MC noise floor of logP_A and dlogP from the 8-seed block, vs signal
  (3) smoothness in theta: leave-one-out interpolation error of the base curve
      vs the residual curve along each of the Om/s8/h axes
  (4) edge motion: how far the empty-beam wall shifts A->B, and where logP is
      undefined in one arm but not the other.

Writes report.md, tables.md and plots under data/results/cascade_residual/.
Run in any python with numpy+matplotlib (no C++ needed for this step).
"""
import os, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATADIR = ("/private/tmp/claude-501/-Users-baltabay-Desktop-gw-wl-emulator/"
           "ac612839-196b-4778-8b02-eab1b572a0e1/scratchpad/cascade_data")
OUTDIR = "data/results/cascade_residual"
PLOTS = os.path.join(OUTDIR, "plots")
os.makedirs(PLOTS, exist_ok=True)

CMIN = 8          # min counts in a bin for logP to be "defined"
NBINS = 300
TAIL_CLIP_PCT = 99.9   # grid upper edge (robust; far tail is junk)

idx = json.load(open(os.path.join(DATADIR, "index.json")))
ZS_LIST = idx["zs_list"]
AXES = idx["axes"]


def load(tag):
    d = np.load(os.path.join(DATADIR, tag + ".npz"))
    return d["lnmu"].astype(np.float64)


def hist_logP(x, edges, cmin=CMIN):
    """Return (logP, counts, density) on bin centers; logP=NaN where counts<cmin."""
    c, _ = np.histogram(x, bins=edges)
    w = np.diff(edges)
    dens = c / (x.size * w)
    lp = np.where(c >= cmin, np.log(np.where(dens > 0, dens, 1.0)), np.nan)
    return lp, c, dens


def region_masks(cdf):
    """cdf on bin centers (from base arm A). edge<0.01, tail>0.99, else body."""
    edge = cdf < 0.01
    tail = cdf > 0.99
    body = (~edge) & (~tail)
    return {"edge": edge, "body": body, "tail": tail}


def span(v):
    v = v[np.isfinite(v)]
    if v.size < 3:
        return np.nan, np.nan
    return v.max() - v.min(), np.subtract(*np.percentile(v, [95, 5]))


# ---------------------------------------------------------------------------
report = []
tables = []
report.append("# Cascade-residual diagnostic — subhalo ingredient\n")
report.append("Paired arms: **A = halo-only**, **B = halo+sub** (subhalo_model=3, "
              "subhalo_factor=1e-2). Residual `dlogP = logP_B - logP_A` on a shared "
              "lnμ grid per z_s. Regions assigned by the base-arm (A) CDF: "
              "EDGE (CDF<1%), BODY (1–99%), TAIL (CDF>99%). "
              f"logP defined where bin counts ≥{CMIN}. N={idx['n_grid']:,} rays/config.\n")
report.append("> **CRN note:** at a fixed seed the two arms are *not* paired per-ray "
              "(subhalos reshuffle the RNG stream; measured per-ray corr≈0), so dlogP "
              "carries the MC noise of both estimates. The 8-seed block quantifies that "
              "floor directly.\n")

summary = {}

for zs in ZS_LIST:
    # ---- noise block: 8 seeds, arms A/B at fiducial -----------------------
    seedsA = [load(f"noise_zs{zs}_seed{s}_sub0") for s in range(1, idx["nseed"] + 1)]
    seedsB = [load(f"noise_zs{zs}_seed{s}_sub1") for s in range(1, idx["nseed"] + 1)]
    pool = np.concatenate(seedsA)
    lo = np.percentile(pool, 0.02)
    hi = np.percentile(pool, TAIL_CLIP_PCT)
    edges = np.linspace(lo, hi, NBINS + 1)
    cen = 0.5 * (edges[:-1] + edges[1:])

    # reference base logP + CDF from pooled A
    lpA_ref, cA_ref, densA = hist_logP(pool, edges, cmin=CMIN * len(seedsA))
    cdfA = np.cumsum(cA_ref) / cA_ref.sum()
    reg = region_masks(cdfA)

    # per-seed logP for noise floor
    lpA = np.array([hist_logP(x, edges)[0] for x in seedsA])
    lpB = np.array([hist_logP(x, edges)[0] for x in seedsB])
    dlp = lpB - lpA                         # (nseed, nbin)
    dlp_mean = np.nanmean(dlp, axis=0)
    lpA_mean = np.nanmean(lpA, axis=0)
    # per-bin noise (std across seeds)
    with np.errstate(all="ignore"):
        noise_base = np.nanstd(lpA, axis=0)
        noise_res = np.nanstd(dlp, axis=0)

    # ---- per-region dynamic range + noise --------------------------------
    rows = []
    for rname in ("edge", "body", "tail"):
        m = reg[rname]
        # bins where residual is defined in the noise-mean sense
        dm = dlp_mean.copy(); dm[~m] = np.nan
        bm = lpA_mean.copy(); bm[~m] = np.nan
        dr_res_full, dr_res_rob = span(dm)
        dr_base_full, dr_base_rob = span(bm)
        nb = np.nanmedian(noise_base[m]) if np.isfinite(noise_base[m]).any() else np.nan
        nr = np.nanmedian(noise_res[m]) if np.isfinite(noise_res[m]).any() else np.nan
        rows.append(dict(region=rname, dr_base_rob=dr_base_rob, dr_res_rob=dr_res_rob,
                         dr_base_full=dr_base_full, dr_res_full=dr_res_full,
                         noise_base=nb, noise_res=nr,
                         ndef=int(np.isfinite(dm).sum()),
                         snr_res=(dr_res_rob / nr) if nr else np.nan))
    summary[zs] = dict(rows=rows, edges=edges, cen=cen, reg=reg,
                       lpA_ref=lpA_ref, lpA_mean=lpA_mean, dlp_mean=dlp_mean,
                       noise_base=noise_base, noise_res=noise_res, cdfA=cdfA)

    # ---- edge motion ------------------------------------------------------
    poolB = np.concatenate(seedsB)
    qA = np.percentile(pool, 0.5); qB = np.percentile(poolB, 0.5)
    mnA, mnB = pool.min(), poolB.min()
    summary[zs]["edge_motion"] = dict(q05A=qA, q05B=qB, dq=qB - qA, mnA=mnA, mnB=mnB, dmn=mnB - mnA)

# ---------------------------------------------------------------------------
# Smoothness / interpolation along theta axes (grid block, single seed)
# ---------------------------------------------------------------------------
def interp_loo_err(curves, vals):
    """curves: (npt, nbin) values along an axis; leave-one-out linear interp of
    interior points; return per-bin median abs error over interior points."""
    npt = curves.shape[0]
    errs = np.full(curves.shape[1], np.nan)
    acc = np.zeros((0, curves.shape[1]))
    for i in range(1, npt - 1):
        lo, hi = i - 1, i + 1
        frac = (vals[i] - vals[lo]) / (vals[hi] - vals[lo])
        pred = curves[lo] + frac * (curves[hi] - curves[lo])
        acc = np.vstack([acc, np.abs(pred - curves[i])])
    return np.nanmedian(acc, axis=0)

smooth = {}
for zs in ZS_LIST:
    S = summary[zs]
    edges, reg = S["edges"], S["reg"]
    smooth[zs] = {}
    for axis, vals in AXES.items():
        vals = np.array(vals)
        lpA_ax, dlp_ax = [], []
        for i in range(len(vals)):
            a = load(f"grid_{axis}_{i}_zs{zs}_sub0")
            b = load(f"grid_{axis}_{i}_zs{zs}_sub1")
            la = hist_logP(a, edges)[0]
            lb = hist_logP(b, edges)[0]
            lpA_ax.append(la); dlp_ax.append(lb - la)
        lpA_ax = np.array(lpA_ax); dlp_ax = np.array(dlp_ax)
        eb = interp_loo_err(lpA_ax, vals)
        er = interp_loo_err(dlp_ax, vals)
        # dynamic range along the axis (per bin, then region median) for normalization
        dr_base = np.nanmax(lpA_ax, 0) - np.nanmin(lpA_ax, 0)
        dr_res = np.nanmax(dlp_ax, 0) - np.nanmin(dlp_ax, 0)
        rec = {}
        for rname in ("edge", "body", "tail"):
            m = reg[rname]
            rec[rname] = dict(
                err_base=np.nanmedian(eb[m]), err_res=np.nanmedian(er[m]),
                drax_base=np.nanmedian(dr_base[m]), drax_res=np.nanmedian(dr_res[m]),
                # normalized: interp error relative to how much the curve moves along axis
                rel_base=np.nanmedian(eb[m]) / np.nanmedian(dr_base[m]) if np.nanmedian(dr_base[m]) else np.nan,
                rel_res=np.nanmedian(er[m]) / np.nanmedian(dr_res[m]) if np.nanmedian(dr_res[m]) else np.nan,
            )
        smooth[zs][axis] = rec

# ---------------------------------------------------------------------------
# TABLES
# ---------------------------------------------------------------------------
tables.append("# Cascade-residual diagnostic — numbers\n")
tables.append("## (1)+(2) Dynamic range & MC noise floor (fiducial, 8-seed block)\n")
tables.append("Robust dynamic range = 5–95 pctile spread of logP over defined bins. "
              "Noise = median per-bin std across seeds. SNR_res = dr_res_rob / noise_res.\n")
tables.append("| z_s | region | dr_base(rob) | dr_res(rob) | res/base | noise_base | noise_res | SNR_res | n_def |")
tables.append("|----|----|----|----|----|----|----|----|----|")
for zs in ZS_LIST:
    for r in summary[zs]["rows"]:
        ratio = r["dr_res_rob"] / r["dr_base_rob"] if r["dr_base_rob"] else np.nan
        tables.append("| %.1f | %s | %.3f | %.3f | %.2f | %.4f | %.4f | %.1f | %d |" % (
            zs, r["region"], r["dr_base_rob"], r["dr_res_rob"], ratio,
            r["noise_base"], r["noise_res"], r["snr_res"], r["ndef"]))

tables.append("\n## (3) Smoothness in theta — leave-one-out interp error (grid block)\n")
tables.append("err = median |interp − truth| of the curve across a 7-pt axis; "
              "rel = err / (axis dynamic range). Lower rel ⇒ needs fewer θ samples.\n")
tables.append("| z_s | axis | region | err_base | err_res | rel_base | rel_res | rel_res/rel_base |")
tables.append("|----|----|----|----|----|----|----|----|")
for zs in ZS_LIST:
    for axis in AXES:
        for r in ("edge", "body", "tail"):
            d = smooth[zs][axis][r]
            rr = d["rel_res"] / d["rel_base"] if d["rel_base"] else np.nan
            tables.append("| %.1f | %s | %s | %.4f | %.4f | %.3f | %.3f | %.2f |" % (
                zs, axis, r, d["err_base"], d["err_res"], d["rel_base"], d["rel_res"], rr))

tables.append("\n## (4) Edge motion (empty-beam wall, fiducial pooled)\n")
tables.append("| z_s | q0.5%_A | q0.5%_B | Δq0.5% | min_A | min_B | Δmin |")
tables.append("|----|----|----|----|----|----|----|")
for zs in ZS_LIST:
    e = summary[zs]["edge_motion"]
    tables.append("| %.1f | %.4f | %.4f | %+.4f | %.4f | %.4f | %+.4f |" % (
        zs, e["q05A"], e["q05B"], e["dq"], e["mnA"], e["mnB"], e["dmn"]))

open(os.path.join(OUTDIR, "tables.md"), "w").write("\n".join(tables) + "\n")

# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------
colors = {"edge": "#d62728", "body": "#1f77b4", "tail": "#2ca02c"}

fig, axz = plt.subplots(len(ZS_LIST), 2, figsize=(13, 4 * len(ZS_LIST)))
for j, zs in enumerate(ZS_LIST):
    S = summary[zs]; cen = S["cen"]; reg = S["reg"]
    ax = axz[j, 0]
    ax.plot(cen, S["lpA_mean"], label="logP_A (halo)", color="k", lw=1.3)
    lpB = S["lpA_mean"] + S["dlp_mean"]
    ax.plot(cen, lpB, label="logP_B (halo+sub)", color="C1", lw=1.0, alpha=0.8)
    for rname, m in reg.items():
        ax.fill_between(cen, ax.get_ylim()[0], ax.get_ylim()[1], where=m,
                        color=colors[rname], alpha=0.06)
    ax.set_title(f"z_s={zs}: base log P"); ax.set_xlabel("lnμ"); ax.set_ylabel("log P")
    ax.legend(fontsize=8)

    ax = axz[j, 1]
    ax.plot(cen, S["dlp_mean"], color="purple", lw=1.2, label="ΔlogP (mean)")
    ax.fill_between(cen, S["dlp_mean"] - S["noise_res"], S["dlp_mean"] + S["noise_res"],
                    color="purple", alpha=0.2, label="±seed noise")
    ax.axhline(0, color="gray", lw=0.6)
    for rname, m in reg.items():
        yl = ax.get_ylim()
        ax.fill_between(cen, yl[0], yl[1], where=m, color=colors[rname], alpha=0.06)
    ax.set_title(f"z_s={zs}: residual ΔlogP_sub"); ax.set_xlabel("lnμ"); ax.set_ylabel("ΔlogP")
    ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "residual_curves.png"), dpi=120)
plt.close(fig)

# summary bars: dynamic-range ratio + interp rel ratio, per region/zs
fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
regs = ["edge", "body", "tail"]
x = np.arange(len(regs)); w = 0.25
for k, zs in enumerate(ZS_LIST):
    dr_ratio = []
    for rname in regs:
        r = next(rr for rr in summary[zs]["rows"] if rr["region"] == rname)
        dr_ratio.append(r["dr_res_rob"] / r["dr_base_rob"] if r["dr_base_rob"] else np.nan)
    axs[0].bar(x + (k - 1) * w, dr_ratio, w, label=f"z_s={zs}")
axs[0].axhline(1, color="k", lw=0.8, ls="--")
axs[0].set_xticks(x); axs[0].set_xticklabels(regs)
axs[0].set_ylabel("dr_res / dr_base"); axs[0].set_title("Residual dynamic range vs base (<1 = smaller)")
axs[0].legend(fontsize=8)

for k, zs in enumerate(ZS_LIST):
    rel_ratio = []
    for rname in regs:
        # average the axis-relative ratio across axes
        vv = [smooth[zs][axis][rname]["rel_res"] / smooth[zs][axis][rname]["rel_base"]
              if smooth[zs][axis][rname]["rel_base"] else np.nan for axis in AXES]
        rel_ratio.append(np.nanmean(vv))
    axs[1].bar(x + (k - 1) * w, rel_ratio, w, label=f"z_s={zs}")
axs[1].axhline(1, color="k", lw=0.8, ls="--")
axs[1].set_xticks(x); axs[1].set_xticklabels(regs)
axs[1].set_ylabel("rel_res / rel_base (avg over axes)")
axs[1].set_title("Residual θ-interp error vs base (<1 = smoother)")
axs[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "summary_bars.png"), dpi=120)
plt.close(fig)

# edge zoom
fig, axz = plt.subplots(1, len(ZS_LIST), figsize=(5 * len(ZS_LIST), 4))
for j, zs in enumerate(ZS_LIST):
    S = summary[zs]; cen = S["cen"]
    m = S["reg"]["edge"] | (S["cdfA"] < 0.05)
    ax = axz[j]
    ax.plot(cen[m], S["lpA_mean"][m], "k.-", ms=3, label="A halo")
    lpB = (S["lpA_mean"] + S["dlp_mean"])
    ax.plot(cen[m], lpB[m], "C1.-", ms=3, label="B halo+sub")
    e = S["edge_motion"]
    ax.axvline(e["q05A"], color="k", ls=":", lw=0.8)
    ax.axvline(e["q05B"], color="C1", ls=":", lw=0.8)
    ax.set_title(f"z_s={zs} edge (Δq0.5%={e['dq']:+.4f})"); ax.set_xlabel("lnμ"); ax.set_ylabel("logP")
    ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "edge_zoom.png"), dpi=120)
plt.close(fig)

print("analysis done ->", OUTDIR)
print("\n".join(tables[:40]))
