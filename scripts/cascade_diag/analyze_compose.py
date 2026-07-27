#!/usr/bin/env python
"""#2 (clustering generalization) + #3 (composition / order-dependence).

Four arms per config, all at matched θ/seed:
  base = halo            (existing sub0)
  sub  = halo+sub        (existing sub1)
  bias = halo+bias       (new; production clustering: bias_model=1, top-hat, 20 Mpc, weak)
  full = halo+sub+bias   (new)

#2: does the parametric-shift picture (σ↑, edge↦out, tail) generalize to clustering?
    -> ΔS_bias = S(bias) - S(base), same detectability/fittability table as #1.

#3: do the corrections COMPOSE, or is host-clump/field covariance order-dependent?
    additive prediction : S_base + ΔS_sub + ΔS_bias
    measured full       : S(full)
    non-additivity      : NA = S(full) - [S_base + ΔS_sub + ΔS_bias]
                        = [S(full)-S(sub)] - [S(bias)-S(base)]   (bias-on-sub  vs  bias-on-base)
    Report NA vs the seed noise of S(full) and vs the size of the shifts being added.

Pure numpy. Writes report_compose.md, tables_compose.md, plots. Env: any numpy.
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

idx = json.load(open(os.path.join(DATADIR, "index.json")))
ZS_LIST = idx["zs_list"]
AXES = idx["axes"]
SCALARS = ["mean", "sigma", "skew", "edge", "tail", "flux"]
# arm -> filename suffix. base/sub reuse the original sub0/sub1 tags.
SUF = {"base": "sub0", "sub": "sub1", "bias": "bias", "full": "full"}


def load(tag):
    return np.load(os.path.join(DATADIR, tag + ".npz"))["lnmu"].astype(np.float64)


def params(x):
    q = np.percentile(x, [0.5, 15.865, 25, 50, 75, 84.135, 99, 99.5])
    q05, q16, q25, q50, q75, q84, q99, q995 = q
    sigma = 0.5 * (q84 - q16)
    skew = (q84 + q16 - 2 * q50) / (q84 - q16) if (q84 - q16) else np.nan
    core = x[(x >= q05) & (x <= q995)]
    mean = core.mean(); flux = np.log(np.mean(np.exp(-core)))
    xs = np.sort(x); n = xs.size
    surv = 1.0 - (np.arange(n) + 0.5) / n
    m = (xs >= q75) & (xs <= q99) & (surv > 0)
    b = np.polyfit(xs[m], np.log(surv[m]), 1)[0] if m.sum() > 20 else np.nan
    return dict(mean=mean, sigma=sigma, skew=skew, edge=q05, tail=b, flux=flux)


def grid_tag(axis, i, zs, arm):
    return f"grid_{axis}_{i}_zs{zs}_{SUF[arm]}"


def noise_tag(zs, s, arm):
    return f"noise_zs{zs}_seed{s}_{SUF[arm]}"


# ---------------------------------------------------------------------------
# Noise block: 4 arms x 8 seeds at fiducial -> shifts + non-additivity + noise
# ---------------------------------------------------------------------------
noise = {}
for zs in ZS_LIST:
    P = {arm: [params(load(noise_tag(zs, s, arm))) for s in range(1, idx["nseed"] + 1)]
         for arm in SUF}
    rec = {}
    for s in SCALARS:
        col = {arm: np.array([d[s] for d in P[arm]]) for arm in SUF}
        d_sub = col["sub"] - col["base"]
        d_bias = col["bias"] - col["base"]
        d_full = col["full"] - col["base"]
        NA = col["full"] - (col["base"] + d_sub + d_bias)   # per-seed non-additivity
        rec[s] = dict(
            base=col["base"].mean(),
            d_sub=d_sub.mean(), d_bias=d_bias.mean(), d_full=d_full.mean(),
            add=(d_sub + d_bias).mean(),
            NA=NA.mean(), NA_noise=NA.std(ddof=1),
            full_noise=col["full"].std(ddof=1),
            d_bias_noise=d_bias.std(ddof=1),
            snr_bias=abs(d_bias.mean()) / d_bias.std(ddof=1) if d_bias.std(ddof=1) else np.inf,
            snr_NA=abs(NA.mean()) / NA.std(ddof=1) if NA.std(ddof=1) else np.inf,
        )
    noise[zs] = rec

# ---------------------------------------------------------------------------
# Grid block: ΔS_bias(θ) fittability + NA(θ) across axes
# ---------------------------------------------------------------------------
def loo_err(vals, y):
    e = []
    for i in range(1, len(vals) - 1):
        f = (vals[i] - vals[i - 1]) / (vals[i + 1] - vals[i - 1])
        e.append(abs((y[i - 1] + f * (y[i + 1] - y[i - 1])) - y[i]))
    return np.median(e)


smooth = {}
for zs in ZS_LIST:
    smooth[zs] = {}
    for axis, vals in AXES.items():
        vals = np.array(vals)
        cols = {arm: [params(load(grid_tag(axis, i, zs, arm))) for i in range(len(vals))]
                for arm in SUF}
        rec = {}
        for s in SCALARS:
            base = np.array([d[s] for d in cols["base"]])
            sub = np.array([d[s] for d in cols["sub"]])
            bias = np.array([d[s] for d in cols["bias"]])
            full = np.array([d[s] for d in cols["full"]])
            d_bias = bias - base
            NA = full - (base + (sub - base) + (bias - base))
            rec[s] = dict(d_bias=d_bias, NA=NA,
                          err_bias=loo_err(vals, d_bias),
                          rng_bias=d_bias.max() - d_bias.min())
        smooth[zs][axis] = rec

# ---------------------------------------------------------------------------
# TABLES
# ---------------------------------------------------------------------------
T = []
T.append("# Clustering generalization (#2) + composition/order test (#3)\n")
T.append("Clustering arm = production config (bias_model=1, spherical top-hat, R_s=20 Mpc, "
         "weak arm on). Four arms per config at matched θ/seed. base/sub reuse the #1 shards.\n")

T.append("## #2 — does the σ↑/edge/tail picture generalize to clustering? (fiducial 8-seed)\n")
T.append("| z_s | scalar | S_base | ΔS_bias | noise | SNR_bias | fit err_bias(θ) | verdict |")
T.append("|----|----|----|----|----|----|----|----|")
for zs in ZS_LIST:
    for s in SCALARS:
        r = noise[zs][s]
        err = np.median([smooth[zs][ax][s]["err_bias"] for ax in AXES])
        v = "drop (≈invariant)" if r["snr_bias"] < 3 else (
            "**at-MC-floor**" if err <= 2 * r["d_bias_noise"] else "needs-more-θ")
        T.append("| %.1f | %s | %+.4f | %+.4f | %.2e | %.1f | %.2e | %s |" % (
            zs, s, r["base"], r["d_bias"], r["d_bias_noise"], r["snr_bias"], err, v))

T.append("\n## #3 — composition / order-dependence (fiducial 8-seed)\n")
T.append("additive = ΔS_sub+ΔS_bias, measured = ΔS_full. NA = measured − additive "
         "(= [bias-on-sub] − [bias-on-base]). |NA|/|full| = fractional non-additivity of "
         "the shift; SNR_NA = |NA|/noise(NA) (>3 ⇒ order-dependence is REAL, not MC scatter).\n")
T.append("| z_s | scalar | ΔS_sub | ΔS_bias | additive | ΔS_full(meas) | NA | |NA|/|full| | SNR_NA |")
T.append("|----|----|----|----|----|----|----|----|----|")
for zs in ZS_LIST:
    for s in SCALARS:
        r = noise[zs][s]
        frac = abs(r["NA"]) / abs(r["d_full"]) if r["d_full"] else np.nan
        T.append("| %.1f | %s | %+.4f | %+.4f | %+.4f | %+.4f | %+.4f | %.2f | %.1f |" % (
            zs, s, r["d_sub"], r["d_bias"], r["add"], r["d_full"], r["NA"], frac, r["snr_NA"]))

open(os.path.join(OUTDIR, "tables_compose.md"), "w").write("\n".join(T) + "\n")

# ---------------------------------------------------------------------------
# PLOT: composition bars for sigma (the workhorse) across z_s
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(1, len(ZS_LIST), figsize=(5 * len(ZS_LIST), 4.2))
for j, zs in enumerate(ZS_LIST):
    ax = axs[j]
    labels = ["Δσ_sub", "Δσ_bias", "additive\n(sum)", "Δσ_full\n(measured)"]
    r = noise[zs]["sigma"]
    vals = [r["d_sub"], r["d_bias"], r["add"], r["d_full"]]
    errs = [0, r["d_bias_noise"], 0, r["full_noise"]]
    cols = ["#1f77b4", "#ff7f0e", "#7f7f7f", "#2ca02c"]
    ax.bar(range(4), vals, yerr=errs, color=cols, capsize=3)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(f"z_s={zs}: σ composition (NA={r['NA']:+.4f}, {abs(r['NA'])/abs(r['d_full'])*100:.0f}%)")
    ax.set_ylabel("Δσ")
fig.suptitle("Do σ-shifts compose?  additive(sum) vs measured full  [gap = non-additivity]")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "composition_sigma.png"), dpi=120)
plt.close(fig)

print("compose analysis done ->", OUTDIR)
print("\n".join(T))
