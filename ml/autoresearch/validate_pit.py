"""PIT (probability integral transform) calibration test for the production model.

If the model density is correct, U = F_model(lnmu) evaluated at simulator samples is
Uniform(0,1). We draw FRESH samples (new seeds -> clean held-out set) at the 15
(cosmology, z) panels, compute the PIT per panel, and report the KS distance and
where the deviation lives (PIT histogram). Writes validate_pit.png.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

AR = Path(__file__).resolve().parent; REPO = AR.parents[1]
COSMOS = [("prior center", (0.67, 0.30, 0.85)),
          ("high structure", (0.72, 0.38, 1.00)),
          ("low structure", (0.60, 0.22, 0.70))]
ZS = [1.0, 2.0, 3.5, 5.0, 8.0]
NSIM = 200_000


def _worker(args):
    z, th, seed = args
    sys.path.insert(0, str(REPO / "build"))
    import gwlensing as gw
    r = gw.sample_lnmu_ml_with_diagnostics(float(z), *map(float, th), int(NSIM), int(seed), False)
    x = np.asarray(r["lnmu"], float)
    return x[np.isfinite(x)]


def main():
    from ml.autoresearch import smooth_model
    fn = smooth_model.load_smooth_fn()
    jobs = [(z, th, 777_000 + 13 * j) for j, (z, th) in
            enumerate((z, th) for _, th in COSMOS for z in ZS)]
    with Pool(10) as p:
        samples = p.map(_worker, jobs)

    g = np.linspace(-3.0, np.log(400), 6000)
    fig, ax = plt.subplots(3, 5, figsize=(19, 10), sharex=True, sharey=True)
    print(f"{'panel':26s} {'N':>7} {'KS':>8} {'KS*sqrt(N)':>10}   (KS*sqrt(N) < 1.36 ~ 5% level)")
    k = 0
    for r, (cname, th) in enumerate(COSMOS):
        for c, z in enumerate(ZS):
            x = samples[k]; k += 1
            lp = np.asarray(fn(g, z, th), np.float64)
            pg = np.exp(lp)
            cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pg[1:] + pg[:-1]) * np.diff(g))])
            cdf /= cdf[-1]
            u = np.interp(x, g, cdf)
            us = np.sort(u); n = us.size
            ks = float(np.max(np.abs(us - (np.arange(1, n + 1) - 0.5) / n)))
            print(f"{cname:14s} z={z:<4}   {n:7d} {ks:8.4f} {ks*np.sqrt(n):10.1f}")
            a = ax[r][c]
            a.hist(u, bins=50, range=(0, 1), density=True, color="#d62728", alpha=0.75)
            a.axhline(1.0, color="k", ls="--", lw=1)
            a.set_ylim(0, 2.0)
            a.text(0.03, 1.82, f"KS={ks:.3f}", fontsize=9)
            if r == 0: a.set_title(f"z = {z}")
            if c == 0: a.set_ylabel(f"{cname}\nPIT density")
            if r == 2: a.set_xlabel("PIT value")
    fig.suptitle("PIT calibration: histogram of F_model(lnmu) at fresh simulator samples "
                 "(flat = perfectly calibrated)", fontsize=13)
    fig.tight_layout()
    out = AR / "validate_pit.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
