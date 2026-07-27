#!/usr/bin/env python
"""#4 step-1: recompute the σ shift + composition in a 2nd-MOMENT (clipped-Var)
width instead of the IQR proxy that flipped sign at z_s=5. Cheap, existing shards.

The caveat: IQR σ=(q84-q16)/2 disagreed in SIGN with the production clipped-Var σ
for clustering at z_s=5 — exactly where the 37% non-additivity lived. Here we
recompute Δσ_bias and the composition NA using std(lnμ) on a monster-excluded
core, at several clip levels, to see whether the interaction survives the metric.

std_kappa: κ_tot≤1 core is the project's certified clip, but these shards carry
only lnμ. We bracket it with variance on lnμ trimmed to [qLO, qHI] percentiles,
same clip across all four arms (base/sub/bias/full). Also carries the IQR σ for
side-by-side. Env: any numpy. Writes report_metric.md, tables_metric.md, plot.
"""
import os, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATADIR = ("/private/tmp/claude-501/-Users-baltabay-Desktop-gw-wl-emulator/"
           "ac612839-196b-4778-8b02-eab1b572a0e1/scratchpad/cascade_data")
OUTDIR = "data/results/cascade_residual"
PLOTS = os.path.join(OUTDIR, "plots")

idx = json.load(open(os.path.join(DATADIR, "index.json")))
ZS_LIST = idx["zs_list"]
SUF = {"base": "sub0", "sub": "sub1", "bias": "bias", "full": "full"}
# clip levels (lower%, upper%): light -> aggressive monster exclusion
CLIPS = {"iqr": None, "clipVar_q999": (0.1, 99.9),
         "clipVar_q995": (0.1, 99.5), "clipVar_q99": (0.5, 99.0)}


def load(tag):
    return np.load(os.path.join(DATADIR, tag + ".npz"))["lnmu"].astype(np.float64)


def sigma(x, clip):
    if clip is None:                      # IQR proxy (for reference)
        q16, q84 = np.percentile(x, [15.865, 84.135])
        return 0.5 * (q84 - q16)
    lo, hi = np.percentile(x, clip)
    return x[(x >= lo) & (x <= hi)].std()


def noise_tag(zs, s, arm):
    return f"noise_zs{zs}_seed{s}_{SUF[arm]}"


# fiducial 8-seed block -> Δσ_bias + composition NA per clip metric
rows = []
for zs in ZS_LIST:
    S = {arm: [load(noise_tag(zs, s, arm)) for s in range(1, idx["nseed"] + 1)]
         for arm in SUF}
    for mname, clip in CLIPS.items():
        col = {arm: np.array([sigma(x, clip) for x in S[arm]]) for arm in SUF}
        d_sub = col["sub"] - col["base"]
        d_bias = col["bias"] - col["base"]
        d_full = col["full"] - col["base"]
        NA = col["full"] - (col["base"] + d_sub + d_bias)
        rows.append(dict(
            zs=zs, metric=mname, base=col["base"].mean(),
            d_sub=d_sub.mean(), d_bias=d_bias.mean(), d_full=d_full.mean(),
            add=(d_sub + d_bias).mean(),
            snr_bias=abs(d_bias.mean()) / d_bias.std(ddof=1),
            NA=NA.mean(), NA_noise=NA.std(ddof=1),
            snr_NA=abs(NA.mean()) / NA.std(ddof=1),
            fracNA=abs(NA.mean()) / abs(d_full.mean()) if d_full.mean() else np.nan,
        ))

T = ["# σ metric recompute — clipped-Var vs IQR (#4 step 1)\n",
     "Δσ shift + composition non-additivity (NA) at fiducial, 8-seed block, for the "
     "IQR proxy vs 2nd-moment std on lnμ trimmed to [lo,hi]%. Same clip across all "
     "four arms. Question: does the z_s=5 interaction survive a variance-based σ?\n",
     "| z_s | metric | σ_base | Δσ_sub | Δσ_bias | SNR_bias | NA | fracNA | SNR_NA |",
     "|----|----|----|----|----|----|----|----|----|"]
for r in rows:
    T.append("| %.1f | %s | %.4f | %+.4f | %+.4f | %.1f | %+.5f | %.2f | %.1f |" % (
        r["zs"], r["metric"], r["base"], r["d_sub"], r["d_bias"], r["snr_bias"],
        r["NA"], r["fracNA"], r["snr_NA"]))
open(os.path.join(OUTDIR, "tables_metric.md"), "w").write("\n".join(T) + "\n")

# plot: Δσ_bias sign and fracNA vs metric, per z_s
fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
metrics = list(CLIPS.keys())
x = np.arange(len(metrics)); w = 0.25
for k, zs in enumerate(ZS_LIST):
    db = [next(r for r in rows if r["zs"] == zs and r["metric"] == m)["d_bias"] for m in metrics]
    axs[0].bar(x + (k - 1) * w, db, w, label=f"z_s={zs}")
axs[0].axhline(0, color="k", lw=0.8); axs[0].set_xticks(x)
axs[0].set_xticklabels(metrics, rotation=20, fontsize=8)
axs[0].set_ylabel("Δσ_bias"); axs[0].set_title("Clustering σ shift: sign vs metric")
axs[0].legend(fontsize=8)
for k, zs in enumerate(ZS_LIST):
    fn = [next(r for r in rows if r["zs"] == zs and r["metric"] == m)["fracNA"] for m in metrics]
    axs[1].bar(x + (k - 1) * w, fn, w, label=f"z_s={zs}")
axs[1].set_xticks(x); axs[1].set_xticklabels(metrics, rotation=20, fontsize=8)
axs[1].set_ylabel("|NA|/|Δσ_full|"); axs[1].set_title("σ non-additivity vs metric")
axs[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "sigma_metric_recompute.png"), dpi=120)
plt.close(fig)

print("\n".join(T))
