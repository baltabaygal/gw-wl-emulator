"""L1 / downstream-impact check: how much does the tail error actually matter?

For each (z, cosmology), compare simulator truth vs three emulators (production cutoff,
simple bound-fixed Normal flow, body+GPD-tail splice) on:
  - rare-event RATES P(mu>t)        (what tail-sensitive science needs; no dilution)
  - lensing dispersion sigma_mu     (linear in mu -> tail-weighted)
  - magnitude scatter sigma_dm, dm=2.5log10(mu)  (log compresses tail -> diluted)
  - PDF L1 distance  int|p_model-p_sim| dmu
  - mock-catalog log-likelihood bias per event (binned science likelihood -> inference impact)

Truth from fresh simulator samples. Model stats by integrating the model density over the
science window. Self-contained; prints a table.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
from pathlib import Path
import numpy as np
import torch

from ml.autoresearch import train_ar, splice_tail as SP, validate_ar as V

AR = Path(__file__).resolve().parent
REPO = AR.parents[1]
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

CENTRAL = (0.67, 0.30, 0.85); STRESS = (0.72, 0.38, 1.00)
POINTS = [("central", CENTRAL, 2.0), ("central", CENTRAL, 5.0), ("central", CENTRAL, 8.0),
          ("stress", STRESS, 2.0), ("stress", STRESS, 3.5), ("stress", STRESS, 5.0), ("stress", STRESS, 8.0)]
NSIM = 400_000
LO, HI = np.log(0.5), np.log(12.18)         # science window in lnmu (mu ~ [0.61, 12.18])
GRID = np.linspace(LO, HI, 3000)
MUG = np.exp(GRID)
BIN_EDGES = np.linspace(-0.5, 2.5, 101)      # science histogram bins (matches load_bin_edges)
THRS = (2.0, 3.0, 5.0)


def stats_from_density(p_lnmu):
    """p_lnmu on GRID (dP/dlnmu); restrict to window, normalize, return stats dict."""
    p = np.clip(p_lnmu, 0, None)
    Z = _trapz(p, GRID)
    if Z <= 0:
        return None
    p = p / Z
    mu = MUG
    mean_mu = _trapz(p * mu, GRID)
    var_mu = _trapz(p * (mu - mean_mu) ** 2, GRID)
    dm = 2.5 * np.log10(mu)
    mean_dm = _trapz(p * dm, GRID); var_dm = _trapz(p * (dm - mean_dm) ** 2, GRID)
    surv = {t: float(_trapz(p[GRID > np.log(t)], GRID[GRID > np.log(t)])) for t in THRS}
    return dict(sig_mu=np.sqrt(max(var_mu, 0)), sig_dm=np.sqrt(max(var_dm, 0)), surv=surv, p=p)


def stats_from_samples(lnmu):
    s = lnmu[(lnmu >= LO) & (lnmu <= HI)]; mu = np.exp(s)
    dm = 2.5 * np.log10(mu)
    surv = {t: float(np.mean(np.exp(lnmu) > t)) for t in THRS}   # unconditional rate
    return dict(sig_mu=mu.std(), sig_dm=dm.std(), surv=surv)


def model_p(fn, z, theta):
    return np.exp(np.asarray(fn(GRID, z, theta), float))


def main():
    sys.path.insert(0, str(REPO / "build")); import gwlensing as gw
    prod = V.prod_fn()
    body_fn, _ = SP.load_body_fn()
    simple = body_fn
    splice = SP.make_spliced_log_prob_fn(body_fn, mu_u=2.0, alpha=3.4, anchor="survival")
    models = {"production": prod, "simple": simple, "splice": splice}

    hdr = f"{'point':14s} {'model':>10s} | {'sig_mu rel':>10s} {'sig_dm rel':>10s} | " \
          f"{'P(>3) rel':>10s} {'P(>5) rel':>10s} | {'L1':>6s} {'dlogL/evt':>9s}"
    print(hdr); print("-" * len(hdr))
    for cname, theta, z in POINTS:
        res = gw.sample_lnmu_ml_with_diagnostics(float(z), *map(float, theta), NSIM, 100, False)
        lnmu = np.asarray(res["lnmu"], float); lnmu = lnmu[np.isfinite(lnmu)]
        st = stats_from_samples(lnmu)
        # sim PDF (normalized over window) for L1 + truth histogram for logL
        ph, _ = np.histogram(np.clip(lnmu, BIN_EDGES[0] + 1e-9, BIN_EDGES[-1] - 1e-9), bins=BIN_EDGES)
        p_sim_bins = ph / ph.sum()
        sim_dens = stats_from_density(np.interp(GRID, 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:]),
                                                ph / (ph.sum() * np.diff(BIN_EDGES)[0]), left=0, right=0))
        cat = ph  # mock catalog counts = simulator histogram (N=NSIM)
        logL_truth = np.sum(cat * np.log(p_sim_bins + 1e-12))
        for mname, fn in models.items():
            sm = stats_from_density(model_p(fn, z, theta))
            if sm is None:
                print(f"{cname+'_z'+str(z):14s} {mname:>10s} | model collapsed"); continue
            # model binned probs for logL
            centers = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
            pm = np.exp(np.asarray(fn(centers, z, theta), float)); pm = pm / pm.sum()
            logL_model = np.sum(cat * np.log(pm + 1e-12))
            dlogL = (logL_model - logL_truth) / cat.sum()
            L1 = float(_trapz(np.abs(sm["p"] - sim_dens["p"]), GRID)) if sim_dens else float("nan")
            rel = lambda a, b: (a - b) / b if b else float("nan")
            print(f"{cname+'_z'+str(z):14s} {mname:>10s} | "
                  f"{rel(sm['sig_mu'], st['sig_mu']):+10.3f} {rel(sm['sig_dm'], st['sig_dm']):+10.3f} | "
                  f"{rel(sm['surv'][3.0], st['surv'][3.0]):+10.3f} {rel(sm['surv'][5.0], st['surv'][5.0]):+10.3f} | "
                  f"{L1:6.3f} {dlogL:+9.4f}")
        print()


if __name__ == "__main__":
    main()
