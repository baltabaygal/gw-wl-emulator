"""Parameter-space validation of the production smooth model (exp18).

Grid: 3 cosmologies (prior center / high-structure stress / low-structure corner)
x 5 redshifts. Each panel: fresh simulator histogram (400k, image-plane dP/dmu)
vs the new flow_smooth+blend production model, old flow_ar dashed for contrast.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

AR = Path(__file__).resolve().parent; REPO = AR.parents[1]

COSMOS = [
    ("prior center",          (0.67, 0.30, 0.85)),
    ("high structure",        (0.72, 0.38, 1.00)),   # stress corner (harness)
    ("low structure",         (0.60, 0.22, 0.70)),   # held-out corner (not in harness)
]
ZS = [1.0, 2.0, 3.5, 5.0, 8.0]
NSIM = 400_000
EDGES = np.geomspace(0.3, 150.0, 75)
CACHE = AR / "cache" / "param_space_ref.npz"


def _worker(args):
    (z, theta, seed) = args
    sys.path.insert(0, str(REPO / "build"))
    import gwlensing as gw
    r = gw.sample_lnmu_ml_with_diagnostics(float(z), *map(float, theta), int(NSIM), int(seed), False)
    x = np.asarray(r["lnmu"], float); x = x[np.isfinite(x)]
    c, _ = np.histogram(np.exp(x), bins=EDGES)
    return c, x.size


def load_ref():
    """Cached 15-point simulator reference; build once (~3 min, 10 workers)."""
    if CACHE.exists():
        d = np.load(CACHE)
        return list(zip(d["counts"], d["sizes"]))
    jobs = [(z, th, 7000 + 17 * j) for j, (z, th) in
            enumerate((z, th) for _, th in COSMOS for z in ZS)]
    with Pool(10) as p:
        hists = p.map(_worker, jobs)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, counts=np.array([h[0] for h in hists]),
             sizes=np.array([h[1] for h in hists]), edges=EDGES)
    return hists


def panel_shape_mae(fn, hists, mu_lo=1.2, mu_hi=9.0, min_cnt=25):
    """Extended score: mean |log10 model - log10 sim| of dP/dlnmu per panel,
    over bins with enough counts in [mu_lo, mu_hi]. Covers low z + low structure
    (the frozen harness doesn't)."""
    ctr = np.sqrt(EDGES[:-1] * EDGES[1:]); lw = np.diff(np.log(EDGES))
    out = {}
    k = 0
    for cname, th in COSMOS:
        for z in ZS:
            cnt, N = hists[k]; k += 1
            dpd = cnt / (N * lw)                      # dP/dlnmu
            m = (cnt >= min_cnt) & (ctr >= mu_lo) & (ctr <= mu_hi)
            lp = np.asarray(fn(np.log(ctr[m]), z, th), np.float64)
            out[(cname, z)] = float(np.mean(np.abs(lp / np.log(10) - np.log10(dpd[m]))))
    return out


def main():
    from ml.autoresearch import smooth_model
    new_fn = smooth_model.load_smooth_fn()
    old_body, _ = smooth_model.load_body_fn(AR / "models" / "flow_ar.pt")

    hists = load_ref()
    mae = panel_shape_mae(new_fn, hists)
    print("extended shapeMAE per panel (mu 1.2-9, incl. low z / low structure):")
    for (cname, z), v in mae.items():
        print(f"  {cname:15s} z={z:<4} {v:.3f}")
    print(f"  MEAN = {np.mean(list(mae.values())):.4f}")

    ctr = np.sqrt(EDGES[:-1] * EDGES[1:]); bw = np.diff(EDGES)
    lg = np.linspace(np.log(0.4), np.log(300), 1200); mug = np.exp(lg)

    fig, ax = plt.subplots(len(COSMOS), len(ZS), figsize=(21, 11), sharex=True, sharey=True)
    k = 0
    for r, (cname, th) in enumerate(COSMOS):
        for c_i, z in enumerate(ZS):
            a = ax[r][c_i]
            cnt, N = hists[k]; k += 1
            m = cnt >= 15
            a.scatter(ctr[m], cnt[m] / (N * bw[m]), s=10, color="#1f77b4", zorder=3,
                      label="simulator 400k" if (r == 0 and c_i == 0) else None)
            a.plot(mug, np.exp(np.asarray(old_body(lg, z, th), float)) / mug,
                   color="#7f7f7f", lw=0.9, ls="--", alpha=0.8,
                   label="old flow_ar" if (r == 0 and c_i == 0) else None)
            a.plot(mug, np.exp(np.asarray(new_fn(lg, z, th), float)) / mug,
                   color="#d62728", lw=1.6,
                   label="NEW production" if (r == 0 and c_i == 0) else None)
            a.set_xscale("log"); a.set_yscale("log")
            a.set_xlim(0.4, 300); a.set_ylim(1e-7, 6)
            a.grid(True, which="both", ls=":", alpha=.3)
            if r == 0:
                a.set_title(f"z = {z}", fontsize=12)
            if c_i == 0:
                a.set_ylabel(f"{cname}\n(h,Ωm,σ8)={th}\n" + r"$dP/d\mu$", fontsize=9)
            if r == len(COSMOS) - 1:
                a.set_xlabel(r"$\mu$")
    ax[0][0].legend(fontsize=9, loc="lower left")
    fig.suptitle("NEW production model (flow_smooth + tanh blend) across parameter space — "
                 "image-plane magnification PDF vs fresh simulator", fontsize=13)
    fig.tight_layout()
    out = AR / "param_space_check.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
