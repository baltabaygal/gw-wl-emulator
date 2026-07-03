"""exp23: conditional POT tail amplitude (see RESEARCH_tails.md round 2).

Fit A_u(ctx) = P(mu > u | z, h, Om, s8) by ridge-penalized Poisson regression on the
per-config exceedance counts of datasets_logz_1k (487 cfg x ~10k) + lowz_aug (150 cfg
x 40k):  counts_i ~ Poisson(N_i * exp(f(ctx_i))),  f linear in the edge-fit features.
Two thresholds are fit (u=3 amplitude anchor, u=8 for the transition slope k(ctx)).

Validation: predicted survivals vs the fresh 400k reference at 15 (cosmo, z) points.
Writes cache/tail_amp_fit.json for smooth_model.py.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

from ml.data import load_dataset


def tail_features(z, h, om, s8):
    """Feature map v2 for the amplitude fit (richer than the edge fit: the tail
    amplitude tracks the halo abundance, which is strongly om/s8-dependent)."""
    lz = np.log1p(z)
    return np.stack([np.ones_like(lz), lz, lz**2, lz**3,
                     om, h, s8, om*lz, h*lz, s8*lz,
                     s8**2, om*s8, s8**2*lz, om*s8*lz], axis=-1)

AR = Path(__file__).resolve().parent
OUT = AR / "cache" / "tail_amp_fit.json"
THRESHOLDS = (2.0, 3.0, 8.0)
RIDGE = 1e-2


def config_counts():
    """(ctx (M,4), N (M,), counts (M,len(THR))) pooled from both datasets."""
    ctxs, Ns, cnts = [], [], []
    ds = load_dataset("datasets_logz_1k")["train"]
    lnmu, vc = ds["lnmu"], ds["valid_counts"].astype(int)
    for i in range(len(vc)):
        x = lnmu[i, :vc[i]]
        ctxs.append([ds["z"][i], ds["h"][i], ds["OmegaM"][i], ds["sigma8"][i]])
        Ns.append(x.size)
        cnts.append([int(np.sum(x > np.log(u))) for u in THRESHOLDS])
    d = np.load(AR / "cache" / "lowz_aug.npz")
    X, Y = d["X"], d["Y"]
    # group by config (rows of identical context)
    _, idx, inv = np.unique(X, axis=0, return_index=True, return_inverse=True)
    for g in range(idx.size):
        m = inv == g
        ctxs.append(X[idx[g]].tolist())
        Ns.append(int(m.sum()))
        cnts.append([int(np.sum(Y[m] > np.log(u))) for u in THRESHOLDS])
    extra = AR / "cache" / "tail_counts_extra.npz"   # exp24 counting runs
    if extra.exists():
        e = np.load(extra)
        assert tuple(e["thresholds"]) == THRESHOLDS, "threshold mismatch in extra counts"
        ctxs.extend(e["ctx"].tolist()); Ns.extend(e["N"].tolist())
        cnts.extend(e["counts"].tolist())
        print(f"including {len(e['N'])} counting-run configs (exp24)")
    return np.array(ctxs), np.array(Ns, float), np.array(cnts, float)


def fit_poisson(F, N, c, ridge=RIDGE):
    """max sum[c*f - N*exp(f)] - ridge*|w|^2, f = F@w. Returns w."""
    def nll(w):
        f = F @ w
        lam = N * np.exp(np.clip(f, -30, 5))
        return float(np.sum(lam - c * f) + ridge * np.sum(w[1:] ** 2))
    def grad(w):
        f = F @ w
        lam = N * np.exp(np.clip(f, -30, 5))
        g = F.T @ (lam - c)
        g[1:] += 2 * ridge * w[1:]
        return g
    w0 = np.zeros(F.shape[1]); w0[0] = np.log(max(c.sum(), 1.0) / N.sum())
    r = minimize(nll, w0, jac=grad, method="L-BFGS-B", options=dict(maxiter=2000))
    assert r.success, r.message
    return r.x


def main():
    ctx, N, cnts = config_counts()
    F = tail_features(ctx[:, 0], ctx[:, 1], ctx[:, 2], ctx[:, 3])
    print(f"configs: {len(N)}   total exceedances: u=3: {int(cnts[:,0].sum())}, u=8: {int(cnts[:,1].sum())}")
    ws = {}
    for j, u in enumerate(THRESHOLDS):
        w = fit_poisson(F, N, cnts[:, j])
        # in-sample check: deviance-ish, pooled by z bin
        pred = N * np.exp(F @ w)
        for zlo, zhi in [(0, 1.5), (1.5, 3), (3, 10)]:
            m = (ctx[:, 0] >= zlo) & (ctx[:, 0] < zhi)
            print(f"  u={u}: z[{zlo},{zhi}) counts obs={int(cnts[m,j].sum()):6d} pred={pred[m].sum():8.1f}")
        ws[str(u)] = w.tolist()

    # out-of-sample validation vs the fresh 400k reference panels
    ref = np.load(AR / "cache" / "param_space_ref.npz")
    edges = ref["edges"]; counts = ref["counts"]; sizes = ref["sizes"]
    COSMOS = [(0.67, 0.30, 0.85), (0.72, 0.38, 1.00), (0.60, 0.22, 0.70)]
    ZS = [1.0, 2.0, 3.5, 5.0, 8.0]
    print("\nvalidation vs fresh sims (S = survival fraction):")
    print(f"{'panel':24s} {'S2 sim':>9} {'S2 fit':>9} {'ratio':>6}  {'S3 sim':>9} {'S3 fit':>9} {'ratio':>6}  {'S8 sim':>9} {'S8 fit':>9} {'ratio':>6}")
    k = 0
    for th in COSMOS:
        for z in ZS:
            c, Np = counts[k], int(sizes[k]); k += 1
            feats = tail_features(np.array([z]), *[np.array([t]) for t in th])[0]
            line = f"th={th} z={z:<4}"
            for j, u in enumerate(THRESHOLDS):
                s_sim = c[np.sqrt(edges[:-1]*edges[1:]) > u].sum() / Np
                s_fit = float(np.exp(feats @ np.array(ws[str(u)])))
                line += f" {s_sim:9.2e} {s_fit:9.2e} {s_fit/max(s_sim,1e-12):6.2f}  "
            print(line)
    OUT.write_text(json.dumps(dict(thresholds=list(THRESHOLDS), coef={k2: v for k2, v in ws.items()}), indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
