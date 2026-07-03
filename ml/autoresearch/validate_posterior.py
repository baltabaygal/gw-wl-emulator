"""End-to-end posterior recovery test (the decisive science check).

Mock GW catalog lensed by the SIMULATOR at a fiducial cosmology; inference of
(h, Om) with the PRODUCTION MODEL as the lensing likelihood (s8 fixed to truth).
Model-induced parameter bias is reported in units of the posterior width.

Control run: the same inference on a catalog lensed by the MODEL itself
(any residual bias there is pipeline-, not model-mismatch-, induced).

Setup: N_EV events, z ~ uniform [0.3, 4], 7% log-normal distance noise,
   ln DL_obs = ln DL(z; h,Om) - lnmu/2 + eps.
Likelihood: L_i(theta) = int p_model(u | z_i, theta) N(lnDLobs_i - lnDL(z_i;theta)
   + u/2; 0, sig) du   on a (h, Om) grid. Lensing PDFs cached per (z-bin, theta).
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

AR = Path(__file__).resolve().parent; REPO = AR.parents[1]
TRUTH = dict(h=0.674, Om=0.315, s8=0.811)
N_EV = 1000
SIG = 0.07                       # 7% distance noise (in ln DL)
ZBINS = np.linspace(0.3, 4.0, 17)          # 16 bins
HGRID = np.linspace(0.60, 0.76, 33)
OMGRID = np.linspace(0.20, 0.40, 33)
SEED = 424242

CH = 299792.458  # km/s


def lnDL(z, h, om):   # flat LCDM, no radiation; DL in Mpc (const offsets cancel)
    zg = np.linspace(0, z, 256)
    E = np.sqrt(om * (1 + zg) ** 3 + (1 - om))
    dc = np.trapezoid(1.0 / E, zg) * CH / (100.0 * h)
    return np.log((1 + z) * dc)


_ZG = np.linspace(1e-4, 4.2, 2048)
def lnDL_vec(z, h, om):
    """ln DL at exact per-event z (vectorized, shared cumulative grid)."""
    E = np.sqrt(om * (1 + _ZG) ** 3 + (1 - om))
    dc = np.concatenate([[0.0], np.cumsum(0.5 * (1/E[1:] + 1/E[:-1]) * np.diff(_ZG))])
    dcz = np.interp(z, _ZG, dc) * CH / (100.0 * h)
    return np.log((1 + z) * dcz)


def _sim_worker(args):
    z, th, n, seed = args
    sys.path.insert(0, str(REPO / "build"))
    import gwlensing as gw
    r = gw.sample_lnmu_ml_with_diagnostics(float(z), *map(float, th), int(n), int(seed), False)
    x = np.asarray(r["lnmu"], float)
    return x[np.isfinite(x)]


def make_catalog(rng, lens_source, fn=None):
    """Events (z_i, lnDL_obs_i) lensed by simulator or by the model itself."""
    z_ev = rng.uniform(0.3, 4.0, N_EV)
    zbin = np.clip(np.digitize(z_ev, ZBINS) - 1, 0, len(ZBINS) - 2)
    zc = 0.5 * (ZBINS[:-1] + ZBINS[1:])
    th = (TRUTH["h"], TRUTH["Om"], TRUTH["s8"])
    lnmu_ev = np.empty(N_EV)
    if lens_source == "sim":
        jobs = [(zc[b], th, 120_000, SEED + 31 * b) for b in range(len(zc))]
        with Pool(8) as p:
            pools = p.map(_sim_worker, jobs)
        for b in range(len(zc)):
            m = zbin == b
            lnmu_ev[m] = rng.choice(pools[b], size=int(m.sum()), replace=False)
    else:                                   # lensed by the model itself (control)
        g = np.linspace(-2.5, np.log(300), 4000)
        for b in range(len(zc)):
            m = zbin == b
            if not m.any(): continue
            p = np.exp(np.asarray(fn(g, zc[b], th), np.float64))
            cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(g))])
            cdf /= cdf[-1]
            lnmu_ev[m] = np.interp(rng.uniform(0, 1, int(m.sum())), cdf, g)
    lnDL_true = lnDL_vec(z_ev, TRUTH["h"], TRUTH["Om"])
    lnDL_obs = lnDL_true - 0.5 * lnmu_ev + SIG * rng.standard_normal(N_EV)
    return z_ev, zbin, lnDL_obs


def run_inference(fn, z_ev, zbin, lnDL_obs, tag):
    zc = 0.5 * (ZBINS[:-1] + ZBINS[1:])
    ug = np.linspace(-1.5, np.log(60), 700); du = ug[1] - ug[0]
    t0 = time.time()
    # cache model lensing pdf per (zbin, Om-ish dependence): p depends on theta=(h,om,s8)
    logL = np.zeros((len(HGRID), len(OMGRID)))
    for io, om in enumerate(OMGRID):
        # lensing pdf depends on (h, om): evaluate per h too (h-dependence is mild but keep honest)
        for ih, h in enumerate(HGRID):
            th = (h, om, TRUTH["s8"])
            lnDL_ev = lnDL_vec(z_ev, h, om)     # exact per-event distances
            ll = 0.0
            for b in range(len(zc)):
                m = zbin == b
                if not m.any(): continue
                pu = np.exp(np.asarray(fn(ug, zc[b], th), np.float64))
                pu /= np.sum(pu) * du
                # residual r_i = lnDLobs - lnDL(z_i;theta); L_i = sum_u p(u) N(r + u/2; 0, sig) du
                r = lnDL_obs[m] - lnDL_ev[m]
                arg = (r[:, None] + 0.5 * ug[None, :]) / SIG
                Li = np.sum(pu[None, :] * np.exp(-0.5 * arg ** 2), axis=1) * du / (SIG * np.sqrt(2 * np.pi))
                ll += float(np.sum(np.log(np.maximum(Li, 1e-300))))
            logL[ih, io] = ll
        print(f"[{tag}] Om={om:.3f} done ({time.time()-t0:.0f}s)", flush=True)
    return logL


def summarize(logL, tag):
    P = np.exp(logL - logL.max())
    P /= P.sum()
    ph = P.sum(1); pom = P.sum(0)
    mh = np.sum(ph * HGRID); sh = np.sqrt(np.sum(ph * (HGRID - mh) ** 2))
    mo = np.sum(pom * OMGRID); so = np.sqrt(np.sum(pom * (OMGRID - mo) ** 2))
    bh = (mh - TRUTH["h"]) / sh; bo = (mo - TRUTH["Om"]) / so
    print(f"[{tag}] h  = {mh:.4f} +- {sh:.4f}  (truth {TRUTH['h']})  bias = {bh:+.2f} sigma")
    print(f"[{tag}] Om = {mo:.4f} +- {so:.4f}  (truth {TRUTH['Om']}) bias = {bo:+.2f} sigma")
    return P, (mh, sh, mo, so)


M_REAL = 16   # independent mock catalogs; mean bias = model systematics


def run_many(fn):
    """Cache the model lensing PDF per (z-bin, theta) once; evaluate M_REAL
    independent sim-lensed and model-lensed catalogs against it."""
    zc = 0.5 * (ZBINS[:-1] + ZBINS[1:])
    ug = np.linspace(-1.5, np.log(60), 700); du = ug[1] - ug[0]
    th_true = (TRUTH["h"], TRUTH["Om"], TRUTH["s8"])

    # simulator lnmu pools (shared across realizations, independent draws)
    jobs = [(zc[b], th_true, 250_000, SEED + 31 * b) for b in range(len(zc))]
    with Pool(8) as p:
        pools = p.map(_sim_worker, jobs)
    # model inverse-CDFs at truth for the control catalogs
    g = np.linspace(-2.5, np.log(300), 4000)
    icdf = []
    for b in range(len(zc)):
        pg = np.exp(np.asarray(fn(g, zc[b], th_true), np.float64))
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pg[1:] + pg[:-1]) * np.diff(g))])
        icdf.append((cdf / cdf[-1], g))

    # build M_REAL catalogs of each kind
    cats = {"sim": [], "model": []}
    for r_i in range(M_REAL):
        rng = np.random.default_rng(SEED + 1000 * r_i)
        z_ev = rng.uniform(0.3, 4.0, N_EV)
        zbin = np.clip(np.digitize(z_ev, ZBINS) - 1, 0, len(ZBINS) - 2)
        noise = SIG * rng.standard_normal(N_EV)
        lnDL_true = lnDL_vec(z_ev, TRUTH["h"], TRUTH["Om"])
        for kind in ("sim", "model"):
            lnmu = np.empty(N_EV)
            for b in range(len(zc)):
                m = zbin == b
                if not m.any(): continue
                if kind == "sim":
                    lnmu[m] = rng.choice(pools[b], size=int(m.sum()), replace=False)
                else:
                    cdf, gg = icdf[b]
                    lnmu[m] = np.interp(rng.uniform(0, 1, int(m.sum())), cdf, gg)
            cats[kind].append((z_ev, zbin, lnDL_true - 0.5 * lnmu + noise))

    # grid inference, PDF computed once per (theta, z-bin), reused by all catalogs
    logL = {k: np.zeros((M_REAL, len(HGRID), len(OMGRID))) for k in cats}
    t0 = time.time()
    for io, om in enumerate(OMGRID):
        for ih, h in enumerate(HGRID):
            th = (h, om, TRUTH["s8"])
            pu_b = [None] * len(zc)
            for b in range(len(zc)):
                pu = np.exp(np.asarray(fn(ug, zc[b], th), np.float64))
                pu_b[b] = pu / (np.sum(pu) * du)
            for kind in cats:
                for r_i, (z_ev, zbin, obs) in enumerate(cats[kind]):
                    lnDL_ev = lnDL_vec(z_ev, h, om)
                    ll = 0.0
                    for b in range(len(zc)):
                        m = zbin == b
                        if not m.any(): continue
                        r = obs[m] - lnDL_ev[m]
                        arg = (r[:, None] + 0.5 * ug[None, :]) / SIG
                        Li = np.sum(pu_b[b][None, :] * np.exp(-0.5 * arg ** 2), axis=1) * du
                        ll += float(np.sum(np.log(np.maximum(Li, 1e-300))))
                    logL[kind][r_i, ih, io] = ll
        print(f"Om={om:.3f} done ({time.time()-t0:.0f}s)", flush=True)
    return logL


def main():
    from ml.autoresearch import smooth_model
    fn = smooth_model.load_smooth_fn()
    logL = run_many(fn)
    print(f"\n=== {M_REAL} independent catalogs, {N_EV} events each ===")
    for kind in ("sim", "model"):
        bh, bo = [], []
        for r_i in range(M_REAL):
            P = np.exp(logL[kind][r_i] - logL[kind][r_i].max()); P /= P.sum()
            ph = P.sum(1); pom = P.sum(0)
            mh = np.sum(ph * HGRID); sh = np.sqrt(np.sum(ph * (HGRID - mh) ** 2))
            mo = np.sum(pom * OMGRID); so = np.sqrt(np.sum(pom * (OMGRID - mo) ** 2))
            bh.append((mh - TRUTH["h"]) / sh); bo.append((mo - TRUTH["Om"]) / so)
        bh, bo = np.array(bh), np.array(bo)
        tag = "sim-lensed (REAL TEST)" if kind == "sim" else "model-lensed (control)"
        print(f"[{tag}] mean h bias = {bh.mean():+.2f} +- {bh.std()/np.sqrt(M_REAL):.2f} sigma "
              f"(scatter {bh.std():.2f})")
        print(f"[{tag}] mean Om bias = {bo.mean():+.2f} +- {bo.std()/np.sqrt(M_REAL):.2f} sigma "
              f"(scatter {bo.std():.2f})")
    np.savez(AR / "cache" / "posterior_recovery.npz", **{k: v for k, v in logL.items()})
    print("wrote cache/posterior_recovery.npz")



if __name__ == "__main__":
    main()
