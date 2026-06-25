"""Measure the simulator's magnification tail slope from a LARGE sample (full tail,
no mu<12 truncation), in parallel. Answers: is dP/dmu a clean power law mu^-alpha,
what is alpha, and does it vary with z?
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np

AR = Path(__file__).resolve().parent
REPO = AR.parents[1]
THETA = (0.72, 0.38, 1.00)
NPER = 1_000_000          # per worker
NWORK = 10                # parallel workers -> NPER*NWORK total
ZS = [2.0, 5.0, 8.0]


def _worker(args):
    z, seed = args
    sys.path.insert(0, str(REPO / "build"))
    import gwlensing as gw
    r = gw.sample_lnmu_ml_with_diagnostics(float(z), *map(float, THETA), int(NPER), int(seed), False)
    x = np.asarray(r["lnmu"], float)
    return x[np.isfinite(x)]


def hill(mu, mu_min):
    x = mu[mu >= mu_min]; n = x.size
    a = 1.0 + n / np.sum(np.log(x / mu_min))   # density index: dP/dmu ~ mu^-a
    return a, (a - 1) / np.sqrt(n), n


def main():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), squeeze=False)
    print(f"{'z':>4} | {'Ntot':>9} {'maxmu':>9} | {'Hill a (mu>5)':>14} {'(mu>10)':>9} {'(mu>30)':>9} | loglog a[5-300]")
    for j, z in enumerate(ZS):
        t = time.time()
        with Pool(NWORK) as p:
            parts = p.map(_worker, [(z, 100 + i) for i in range(NWORK)])
        lnmu = np.concatenate(parts); mu = np.exp(lnmu); N = mu.size
        a5, e5, n5 = hill(mu, 5); a10, _, _ = hill(mu, 10); a30, _, _ = hill(mu, 30)
        # log-log regression of dP/dmu over a wide window
        edges = np.geomspace(5, 300, 25); c, _ = np.histogram(mu, bins=edges)
        ctr = np.sqrt(edges[:-1] * edges[1:]); bw = np.diff(edges); dens = c / (N * bw); m = c >= 30
        sl = np.polyfit(np.log10(ctr[m]), np.log10(dens[m]), 1)[0]
        print(f"{z:>4} | {N:>9} {mu.max():>9.0f} | {a5:>8.3f}±{e5:.3f} {a10:>9.3f} {a30:>9.3f} | {-sl:.3f}  ({time.time()-t:.0f}s)")
        ax = axes[0][j]
        ax.scatter(ctr[m], dens[m], s=20, color="#1f77b4", label="simulator (10M)")
        xx = ctr[m]
        ax.plot(xx, dens[m][0] * (xx / xx[0]) ** (sl), "g-", lw=2, label=f"fit α={-sl:.2f}")
        ax.plot(xx, dens[m][0] * (xx / xx[0]) ** (-3.0), "k--", lw=1, alpha=.6, label="α=3 ref")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(f"z={z}")
        ax.set_xlabel(r"$\mu$"); ax.set_ylabel(r"$dP/d\mu$"); ax.legend(fontsize=8); ax.grid(True, which="both", ls=":", alpha=.4)
    fig.suptitle(r"Simulator tail, full range (10M samples) — measured power-law slope")
    fig.tight_layout(); out = AR / "measured_tail.png"; fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
