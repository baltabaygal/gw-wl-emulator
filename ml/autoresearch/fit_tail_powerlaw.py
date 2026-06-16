"""Fit the simulator magnification tail to a power law dP/dmu ~ mu^-alpha.

Uses cached simulator samples (no new simulator calls). Two estimators:
  - log-log histogram regression (visual slope, R^2)
  - Hill MLE: alpha = 1 + n / sum ln(mu_i/mu_min)  (rigorous for tails)
Also a KS goodness-of-fit of the fitted power law vs the empirical tail.
Goal: see if a single power law describes the tail well and whether the slope
is universal across z (amplitude varying) -> a clean benchmark for the emulator.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AR = Path(__file__).resolve().parent
CACHE = AR / "cache" / "groundtruth.npz"
MU_MIN = 3.0      # fit mu > MU_MIN
MU_MAX = 12.0     # cache body window tops out near mu=12.18 (lnmu=2.5)
POINTS = ["stress_z2.0", "stress_z3.5", "stress_z5.0", "stress_z8.0",
          "central_z5.0", "central_z8.0"]


def hill(mu, mu_min):
    x = mu[mu >= mu_min]
    n = x.size
    if n < 50:
        return np.nan, np.nan, n
    alpha = 1.0 + n / np.sum(np.log(x / mu_min))
    return alpha, (alpha - 1.0) / np.sqrt(n), n


def ks_powerlaw(mu, mu_min, alpha):
    """KS distance between empirical tail CDF and power-law CDF (Clauset et al.)."""
    x = np.sort(mu[mu >= mu_min]); n = x.size
    if n < 50:
        return np.nan
    cdf_emp = np.arange(1, n + 1) / n
    cdf_pl = 1.0 - (x / mu_min) ** (1.0 - alpha)   # CDF of power law, exponent alpha
    return float(np.max(np.abs(cdf_emp - cdf_pl)))


def loglog_slope(mu, mu_min, mu_max, nb=14):
    x = mu[(mu >= mu_min) & (mu <= mu_max)]
    edges = np.geomspace(mu_min, mu_max, nb + 1)
    c, _ = np.histogram(x, bins=edges)
    w = 0.5 * (edges[:-1] + edges[1:]); bw = np.diff(edges)
    dens = c / (mu.size * bw)
    m = c >= 20
    if m.sum() < 3:
        return np.nan, np.nan, (w, dens, m)
    X = np.log10(w[m]); Y = np.log10(dens[m])
    A = np.vstack([X, np.ones_like(X)]).T
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    yhat = A @ coef
    ss_res = np.sum((Y - yhat) ** 2); ss_tot = np.sum((Y - Y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return -coef[0], r2, (w, dens, m)


def main():
    d = np.load(CACHE, allow_pickle=True)
    tags = [str(t) for t in d["tags"]]
    print(f"{'point':14s} | {'alpha_hist':>10} {'R2':>6} | {'alpha_Hill':>10} {'+-':>6} {'n_tail':>7} | {'KS':>6}")
    print("-" * 78)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), squeeze=False)
    for k, pt in enumerate(POINTS):
        if pt not in tags:
            continue
        body = d[f"body_{pt}"]
        mu = np.exp(np.asarray(body, float))
        a_h, r2, (w, dens, m) = loglog_slope(mu, MU_MIN, MU_MAX)
        a_hill, a_err, ntail = hill(mu, MU_MIN)
        ks = ks_powerlaw(mu, MU_MIN, a_hill)
        print(f"{pt:14s} | {a_h:10.3f} {r2:6.3f} | {a_hill:10.3f} {a_err:6.3f} {ntail:7d} | {ks:6.3f}")
        ax = axes[k // 3][k % 3]
        ax.scatter(np.log10(w[m]), np.log10(dens[m]), s=18, color="#1f77b4", label="simulator")
        xx = np.log10(w[m])
        # overlay Hill power law normalized to the data at the first bin
        if np.isfinite(a_hill):
            yint = np.log10(dens[m][0]) + a_hill * (xx[0])
            ax.plot(xx, -a_hill * xx + yint, "g-", lw=2, label=f"Hill α={a_hill:.2f}")
        if np.isfinite(a_h):
            ax.plot(xx, -a_h * xx + (np.log10(dens[m][0]) + a_h * xx[0]), "r--", lw=1.5,
                    label=f"hist α={a_h:.2f}")
        ax.set_title(pt); ax.set_xlabel(r"$\log_{10}\mu$"); ax.set_ylabel(r"$\log_{10} dP/d\mu$")
        ax.legend(fontsize=8); ax.grid(True, ls=":", alpha=0.4)
    fig.suptitle(r"Simulator tail power-law fit ($\mu>%.0f$): $dP/d\mu\propto\mu^{-\alpha}$" % MU_MIN)
    fig.tight_layout()
    out = AR / "tail_powerlaw_fit.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
