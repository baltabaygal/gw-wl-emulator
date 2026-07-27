#!/usr/bin/env python
"""Parametric-shift diagnostic (follow-up #1) — REUSES the existing cascade shards.

For each arm (A=halo, B=halo+sub) and config, fit the structured/parametric
representation the production model uses, then difference A->B to get the
ingredient shift as a function of theta:

  mean    = robust mean lnμ (trimmed)
  sigma   = robust width, (q84.135 - q15.865)/2   (= σ for a Gaussian body)
  skew    = quantile (Bowley) skew, ((q84+q16-2q50)/(q84-q16))
  edge    = empty-beam wall, q0.5% of lnμ
  tail    = upper-shoulder decay slope: fit ln(1-CDF) vs lnμ over [q75,q99]
            (moderate, well-populated tail — NOT the uncertified far tail)
  flux    = ln <1/μ>_trim  (mean exp(-lnμ) over the [q0.5,q99.5] core)

Then, per scalar, re-ask the three diagnostic questions:
  (1) detectability: |ΔS| vs its seed-to-seed noise (8-seed block)   -> SNR_Δ
  (2) θ-smoothness : leave-one-out interp error of ΔS(θ) and S_A(θ)
      along each Om/σ8/h axis, normalized by the axis spread          -> rel
  (3) resolvable?  : is the leave-one-out error ABOVE the seed noise
      (genuine θ-curvature) rather than drowned in it?

Pure numpy; no C++ needed. Writes report_param.md, tables_param.md, plots.
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


def load(tag):
    return np.load(os.path.join(DATADIR, tag + ".npz"))["lnmu"].astype(np.float64)


def params(x):
    """Extract the structured scalars from an lnμ sample array."""
    q = np.percentile(x, [0.5, 15.865, 25, 50, 75, 84.135, 99, 99.5])
    q05, q16, q25, q50, q75, q84, q99, q995 = q
    sigma = 0.5 * (q84 - q16)
    skew = (q84 + q16 - 2 * q50) / (q84 - q16) if (q84 - q16) else np.nan
    edge = q05
    # trimmed mean/flux over the [q0.5,q99.5] core (monster-ray safe)
    core = x[(x >= q05) & (x <= q995)]
    mean = core.mean()
    flux = np.log(np.mean(np.exp(-core)))
    # upper-shoulder slope: ln survival vs lnμ over [q75,q99]
    xs = np.sort(x)
    n = xs.size
    surv = 1.0 - (np.arange(n) + 0.5) / n
    m = (xs >= q75) & (xs <= q99) & (surv > 0)
    if m.sum() > 20:
        b = np.polyfit(xs[m], np.log(surv[m]), 1)[0]
    else:
        b = np.nan
    return dict(mean=mean, sigma=sigma, skew=skew, edge=edge, tail=b, flux=flux)


# ---------------------------------------------------------------------------
# Noise block: 8 seeds, arms A/B at fiducial -> per-scalar seed noise for A & ΔS
# ---------------------------------------------------------------------------
noise = {}   # noise[zs] = dict(scalar -> (Amean, Anoise, dmean, dnoise, snr))
for zs in ZS_LIST:
    A = [params(load(f"noise_zs{zs}_seed{s}_sub0")) for s in range(1, idx["nseed"] + 1)]
    B = [params(load(f"noise_zs{zs}_seed{s}_sub1")) for s in range(1, idx["nseed"] + 1)]
    rec = {}
    for s in SCALARS:
        a = np.array([d[s] for d in A]); b = np.array([d[s] for d in B])
        d = b - a
        rec[s] = dict(Amean=a.mean(), Anoise=a.std(ddof=1),
                      dmean=d.mean(), dnoise=d.std(ddof=1),
                      snr=abs(d.mean()) / d.std(ddof=1) if d.std(ddof=1) else np.inf)
    noise[zs] = rec


# ---------------------------------------------------------------------------
# Grid block: ΔS(θ) along each axis + leave-one-out interpolation error
# ---------------------------------------------------------------------------
def loo_err(vals, y):
    """leave-one-out linear interp of interior points; return median |err|."""
    errs = []
    for i in range(1, len(vals) - 1):
        lo, hi = i - 1, i + 1
        frac = (vals[i] - vals[lo]) / (vals[hi] - vals[lo])
        pred = y[lo] + frac * (y[hi] - y[lo])
        errs.append(abs(pred - y[i]))
    return np.median(errs)


smooth = {}  # smooth[zs][axis][scalar] = dict(...)
for zs in ZS_LIST:
    smooth[zs] = {}
    for axis, vals in AXES.items():
        vals = np.array(vals)
        A = [params(load(f"grid_{axis}_{i}_zs{zs}_sub0")) for i in range(len(vals))]
        B = [params(load(f"grid_{axis}_{i}_zs{zs}_sub1")) for i in range(len(vals))]
        rec = {}
        for s in SCALARS:
            ya = np.array([d[s] for d in A])
            yd = np.array([d[s] for d in B]) - ya
            ea, ed = loo_err(vals, ya), loo_err(vals, yd)
            ra = ya.max() - ya.min(); rd = yd.max() - yd.min()
            rec[s] = dict(sA=ya, dS=yd, err_base=ea, err_res=ed,
                          range_base=ra, range_res=rd,
                          rel_base=ea / ra if ra else np.nan,
                          rel_res=ed / rd if rd else np.nan)
        smooth[zs][axis] = rec


# ---------------------------------------------------------------------------
# TABLES
# ---------------------------------------------------------------------------
T = []
T.append("# Parametric-shift diagnostic — numbers (reuses cascade shards)\n")
T.append("## (1) Detectability of the ingredient shift ΔS (fiducial, 8-seed block)\n")
T.append("S_A = base value, ΔS = B−A, seed-noise from 8 seeds. SNR_Δ = |ΔS| / noise(ΔS). "
         "SNR≫1 ⇒ the shift is well-determined from one 200k-ray run.\n")
T.append("| z_s | scalar | S_A | noise(S_A) | ΔS | noise(ΔS) | SNR_Δ |")
T.append("|----|----|----|----|----|----|----|")
for zs in ZS_LIST:
    for s in SCALARS:
        r = noise[zs][s]
        T.append("| %.1f | %s | %+.4f | %.2e | %+.4f | %.2e | %.1f |" % (
            zs, s, r["Amean"], r["Anoise"], r["dmean"], r["dnoise"], r["snr"]))

T.append("\n## (2)+(3) Fittability of ΔS(θ) — compact summary (median over Om/σ8/h axes)\n")
T.append("The right question is NOT 'is there curvature above noise' but 'can we predict "
         "ΔS(θ) to an error small vs the shift itself'. err_res = leave-one-out interp "
         "error of ΔS(θ) (7-pt axis); floor = seed noise. frac = err_res / |ΔS_fid| = "
         "interp error as a fraction of the effect being modeled. VERDICT: 'drop' if the "
         "shift is < 3·noise (invariant); 'at-MC-floor' if err_res ≤ 2·noise (7 pts "
         "already interpolate to the sim's own noise floor — the WIN); 'needs-more-θ' "
         "otherwise.\n")
T.append("| z_s | scalar | SNR_Δ | |ΔS_fid| | med err_res | floor | med frac | verdict |")
T.append("|----|----|----|----|----|----|----|----|")
for zs in ZS_LIST:
    for s in SCALARS:
        sn = noise[zs][s]["dnoise"]
        dfid = abs(noise[zs][s]["dmean"])
        snr = noise[zs][s]["snr"]
        errs = np.array([smooth[zs][axis][s]["err_res"] for axis in AXES])
        med_err = np.median(errs)
        frac = med_err / dfid if dfid else np.inf
        if snr < 3:
            verdict = "drop (≈invariant)"
        elif med_err <= 2 * sn:
            verdict = "**at-MC-floor**"
        else:
            verdict = "needs-more-θ"
        T.append("| %.1f | %s | %.1f | %.4f | %.2e | %.2e | %.2f | %s |" % (
            zs, s, snr, dfid, med_err, sn, frac, verdict))

T.append("\n## (2b) Full per-axis leave-one-out detail\n")
T.append("| z_s | axis | scalar | rel_base | rel_res | err_res | floor(ΔS) |")
T.append("|----|----|----|----|----|----|----|")
for zs in ZS_LIST:
    for axis in AXES:
        for s in SCALARS:
            d = smooth[zs][axis][s]
            sn = noise[zs][s]["dnoise"]
            T.append("| %.1f | %s | %s | %.3f | %.3f | %.2e | %.2e |" % (
                zs, axis, s, d["rel_base"], d["rel_res"], d["err_res"], sn))
open(os.path.join(OUTDIR, "tables_param.md"), "w").write("\n".join(T) + "\n")

# ---------------------------------------------------------------------------
# PLOTS: ΔS(θ) along each axis, per scalar, one figure per z_s
# ---------------------------------------------------------------------------
for zs in ZS_LIST:
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for k, s in enumerate(SCALARS):
        ax = axs.flat[k]
        for axis, vals in AXES.items():
            vals = np.array(vals)
            d = smooth[zs][axis][s]
            v0 = idx["fid"][{"Om": "Om", "s8": "s8", "h": "h"}[axis]]
            ax.plot((vals - v0) / (vals.max() - vals.min()), d["dS"], "o-",
                    ms=4, label=axis)
        sn = noise[zs][s]["dnoise"]
        dm = noise[zs][s]["dmean"]
        ax.axhspan(dm - sn, dm + sn, color="gray", alpha=0.2)
        ax.set_title(f"Δ{s}  (SNR_Δ={noise[zs][s]['snr']:.0f})")
        ax.set_xlabel("normalized θ offset"); ax.set_ylabel(f"Δ{s} (B−A)")
        if k == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f"Parametric shifts ΔS(θ), z_s={zs}  (gray band = seed noise)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, f"param_shifts_zs{zs}.png"), dpi=110)
    plt.close(fig)

print("param analysis done ->", OUTDIR)
print("\n".join(T))
