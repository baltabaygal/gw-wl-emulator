"""Validate the C++ BiasField1D (cpp/lensing.cpp, bias_model=1) against the
notebook reference (playground/bias_field/field1d_prototype.ipynb), pre-build.

Two layers:
  1. [default] pure-numpy VERBATIM replica of BiasField1D::build (same grids,
     same trapezoids, same table interpolation, same mode/quadrature branch,
     same Cholesky) vs the notebook's exact P1D integral + discrete-mode
     covariance. Catches algorithm/derivation errors before the Mac build.
  2. [--module] after `make build`: drive gwlensing.sample_lensing_raw_ml with
     bias_model=1 and check <lambda>=1 bookkeeping + Var(kappa) response vs
     bias_model=0 arms (smoke-level; the PDF scan is the real acceptance).

Run: python3 playground/bias_field/validate_field_covariance.py
Seed namespace: none needed for layer 1 (deterministic); layer 2 uses 950_000_000.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "convergence"))
from bias_field_prototype import Cosmo, PI

PAD = 1.05          # hard-coded in the C++ (L = 1.05 chi(zs))


# ---------------------------------------------------------------- Bessel J1
try:
    from scipy.special import j1 as _j1
except ImportError:  # Abramowitz-Stegun 9.4 polynomial (|err| < 4e-8), as in the notebook
    def _j1(x):
        x = np.asarray(x, float); ax = np.abs(x)
        small = ax < 8.0
        y = np.where(small, x * x, 0.0)
        p1 = x * (72362614232.0 + y * (-7895059235.0 + y * (242396853.1
             + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))))
        q1 = 144725228442.0 + y * (2300535178.0 + y * (18583304.74
             + y * (99447.43394 + y * (376.9991397 + y))))
        z = np.where(~small, 8.0 / np.where(ax == 0, 1.0, ax), 0.0)
        y2 = z * z
        p2 = 1.0 + y2 * (0.183105e-2 + y2 * (-0.3516396496e-4
             + y2 * (0.2457520174e-5 + y2 * (-0.240337019e-6))))
        q2 = 0.04687499995 + y2 * (-0.2002690873e-3 + y2 * (0.8449199096e-5
             + y2 * (-0.88228987e-6 + y2 * 0.105787412e-6)))
        xx = ax - 2.356194491
        big = np.sqrt(0.636619772 / np.where(ax == 0, 1.0, ax)) * (
            np.cos(xx) * p2 - z * np.sin(xx) * q2)
        return np.where(small, p1 / q1, np.sign(x) * big)


def Wdisk(x):
    x = np.asarray(x, float)
    return np.where(x < 1e-6, 1.0, 2.0 * _j1(np.where(x < 1e-6, 1.0, x))
                    / np.where(x < 1e-6, 1.0, x))


def shells(C, zs):
    """Shell edges exactly as BiasField1D::build: jz = 1.. while zlist[jz] < zs."""
    lo, hi = [], []
    for jz in range(1, C.Nz):
        if C.zlist[jz] >= zs:
            break
        lo.append(float(C.dc(C.zlist[jz - 1])))
        hi.append(float(C.dc(C.zlist[jz])))
    lo, hi = np.array(lo), np.array(hi)
    return 0.5 * (lo + hi), hi - lo


# ------------------------------------------------- verbatim C++ replica
def cpp_field(C, zs, Rperp):
    """Replicates BiasField1D::build number-for-number (numpy-vectorized)."""
    c, Lh = shells(C, zs)
    n = len(c)
    L = PAD * float(C.dc(zs))
    kmin = 2.0 * PI / L
    Nmax = max(4, int(np.floor(L / Rperp)))
    kmax = 2.0 * PI * Nmax / L

    # P1D table: nkperp=2048 log grid [1e-9, 60/max(R,10)], trapezoid in ln kperp;
    # nktab=600 log-log table on [0.5*kmin, kmax]
    nkperp, nktab = 2048, 600
    Rw = max(Rperp, 10.0)
    kperp = np.exp(np.linspace(np.log(1e-9), np.log(60.0 / Rw), nkperp))
    dlnkp = np.log(kperp[1] / kperp[0])
    W2 = Wdisk(kperp * Rperp) ** 2
    lktab = np.linspace(np.log(0.5 * kmin), np.log(kmax), nktab)
    Ptab = np.empty(nktab)
    for t0 in range(0, nktab, 64):
        t1 = min(t0 + 64, nktab)
        kk = np.sqrt(np.exp(lktab[None, t0:t1]) ** 2 + kperp[:, None] ** 2)
        f = kperp[:, None] ** 2 * C.Pk(kk) * W2[:, None]
        Ptab[t0:t1] = np.trapezoid(f, dx=dlnkp, axis=0) / (2.0 * PI)
    lPtab = np.log(np.maximum(Ptab, 1e-300))
    P1D = lambda k: np.exp(np.interp(np.log(np.clip(k, np.exp(lktab[0]),
                                                    np.exp(lktab[-1]))),
                                     lktab, lPtab))

    # EXACT mode sum, always (matches the C++ after the 2026-07-16 quadrature
    # rejection; the C++ uses a trig recursion — same numbers, different order)
    Cov = np.zeros((n, n))
    for n0 in range(0, Nmax, 200_000):
        n1 = min(n0 + 200_000, Nmax)
        kq = 2.0 * PI * np.arange(n0 + 1, n1 + 1) / L
        wq = 2.0 * P1D(kq) / L
        snc = np.sinc(np.outer(Lh / 2.0, kq) / PI)      # sin(x)/x, x = k Lh/2
        rw = np.sqrt(wq)[None, :]
        Ac = rw * np.cos(np.outer(c, kq)) * snc
        As = rw * np.sin(np.outer(c, kq)) * snc
        Cov += Ac @ Ac.T + As @ As.T
    sig2 = np.diag(Cov).copy()
    jit = 1e-12 * sig2.mean()
    chol = np.linalg.cholesky(Cov + jit * np.eye(n))
    return dict(Cov=Cov, sig2=sig2, chol=chol, n=n, L=L, Nmax=Nmax,
                c=c, Lh=Lh)


# ------------------------------------------------- notebook exact reference
def notebook_cov(C, zs, Rperp, nkperp=4096, nkpar=1600):
    """Exact discrete-mode covariance built from the notebook's P1D class
    (independent integral: different grids, exact per-mode evaluation)."""
    c, Lh = shells(C, zs)
    L = PAD * float(C.dc(zs))
    Nmax = max(4, int(np.floor(L / Rperp)))
    kp_max = 60.0 / max(Rperp, 10.0)
    kperp = np.exp(np.linspace(np.log(1e-9), np.log(kp_max), nkperp))
    dlnkp = np.log(kperp[1] / kperp[0])
    W2 = Wdisk(kperp * Rperp) ** 2
    ktab = np.exp(np.linspace(np.log(1e-9), np.log(kp_max), nkpar))
    Ptab = np.empty(nkpar)
    for t0 in range(0, nkpar, 128):
        t1 = min(t0 + 128, nkpar)
        kk = np.sqrt(ktab[None, t0:t1] ** 2 + kperp[:, None] ** 2)
        Ptab[t0:t1] = np.trapezoid(kperp[:, None] ** 2 * C.Pk(kk) * W2[:, None],
                                   dx=dlnkp, axis=0) / (2.0 * PI)
    lk, lP = np.log(ktab), np.log(np.maximum(Ptab, 1e-300))
    P1D = lambda k: np.exp(np.interp(np.log(np.clip(k, ktab[0], None)), lk, lP))

    n = len(c)
    Cov = np.zeros((n, n))
    for n0 in range(0, Nmax, 200_000):                   # chunk the mode sum
        n1 = min(n0 + 200_000, Nmax)
        kq = 2.0 * PI * np.arange(n0 + 1, n1 + 1) / L
        wq = 2.0 * P1D(kq) / L
        snc = np.sinc(np.outer(Lh / 2.0, kq) / PI)
        rw = np.sqrt(wq)[None, :]
        Ac = rw * np.cos(np.outer(c, kq)) * snc
        As = rw * np.sin(np.outer(c, kq)) * snc
        Cov += Ac @ Ac.T + As @ As.T
    return Cov


def compare(zs, Rperp, C):
    f = cpp_field(C, zs, Rperp)
    ref = notebook_cov(C, zs, Rperp)
    scale = np.max(np.diag(ref))
    dev_cov = np.max(np.abs(f["Cov"] - ref)) / scale
    dev_sig = np.max(np.abs(np.sqrt(f["sig2"]) / np.sqrt(np.diag(ref)) - 1.0))
    # Cholesky reproduces Cov
    dev_ch = np.max(np.abs(f["chol"] @ f["chol"].T - f["Cov"])) / scale
    # <lambda> = 1 bookkeeping at b*sig = 0.5 (MC, 200k draws; at b*sig >> 1 the
    # SAMPLE mean is heavy-tail-limited — notebook cell 19 — so gate in the
    # regime where the estimator converges)
    rng = np.random.default_rng(12345)
    g = rng.standard_normal((200_000, f["n"]))
    d = g @ f["chol"].T
    i = int(np.argmax(f["sig2"]))
    bDg = 0.5 / np.sqrt(f["sig2"][i])
    lam = np.exp(bDg * d[:, i] - 0.5 * bDg ** 2 * f["sig2"][i])
    print(f"zs={zs:4g} Rperp={Rperp:9.4g} kpc  Nmax={f['Nmax']:>8d} "
          f"n={f['n']:3d}  |dCov|/diag={dev_cov:.2e}  dsig={dev_sig:.2e}  "
          f"|chol@cholT-Cov|={dev_ch:.2e}  <lam>={lam.mean():.4f}"
          f"  sig_max={np.sqrt(f['sig2'].max()):.4f}")
    return dev_cov, dev_sig, abs(lam.mean() - 1.0)


def module_smoke():
    """Layer 2 (after `make build` in the test env): drive the actual module."""
    sys.path.insert(0, str(REPO / "build"))
    import gwlensing as gw
    ok = True

    cfgd = gw.get_simulator_config()
    ok &= cfgd["bias_model"] == 0 and abs(cfgd["bias_Rperp"] - 8441.0) < 1e-12
    print(f"config dict: bias_model={cfgd['bias_model']} "
          f"bias_Rperp={cfgd['bias_Rperp']}  [{'ok' if ok else 'BAD'}]")

    zs, N, seed = 1.0, 30_000, 950_000_001
    base = dict(z=zs, h=0.674, OmegaM=0.315, sigma8=0.811, nsamples=N, seed=seed)
    kap = {}
    for name, kw in (
        ("legacy",   dict()),
        ("field3M",  dict(bias_model=1, bias_Rperp=3000.0)),
        ("field3M_2", dict(bias_model=1, bias_Rperp=3000.0)),   # determinism
        ("field800M", dict(bias_model=1, bias_Rperp=8.441e5)),  # ~homogeneous
        ("nobias",   dict(bias=False)),
    ):
        r = gw.sample_lensing_raw_ml(**base, **kw)
        kap[name] = np.asarray(r["kappa"])
    det = np.array_equal(kap["field3M"], kap["field3M_2"])
    ok &= det
    clip = lambda k: k[np.abs(k - np.median(k)) < 10 * np.std(k)]
    v = {k: float(np.var(clip(v_))) for k, v_ in kap.items()}
    print(f"determinism (same seed, bias_model=1): {det}")
    print("clipped Var(kappa): " + "  ".join(f"{k}={v[k]:.3e}" for k in
          ("legacy", "field3M", "field800M", "nobias")))
    # huge-Rperp field ~ no clustering; small-Rperp adds variance above nobias
    ok &= v["field800M"] < 1.10 * v["nobias"] + 1e-12
    ok &= v["field3M"] > v["field800M"]
    # mean bookkeeping: <kappa> raw should be within MC error of the legacy arm
    m_leg, m_f = float(np.mean(kap["legacy"])), float(np.mean(kap["field3M"]))
    se = float(np.std(kap["legacy"])) / np.sqrt(N)
    print(f"<kappa>: legacy={m_leg:.5f} field={m_f:.5f} (se~{se:.5f})")
    ok &= abs(m_f - m_leg) < 8 * se
    print("MODULE SMOKE " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


def main():
    if "--module" in sys.argv:
        module_smoke()
        return
    ok = True
    for zs in (1.0, 5.0):
        C = Cosmo(Nz=100)
        rho = C.rhoM0
        for M in (1e5, 1e11, 1e14, 1e17, 1e20):
            RL = (3.0 * M / (4.0 * PI * rho)) ** (1.0 / 3.0)
            dev_cov, dev_sig, dlam = compare(zs, RL, C)
            # acceptance: table-interpolation-level agreement + exact bookkeeping
            ok &= dev_cov < 1e-3 and dev_sig < 1e-3 and dlam < 5e-3
    print("PASS" if ok else "FAIL: covariance deviates beyond table-interp error")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
