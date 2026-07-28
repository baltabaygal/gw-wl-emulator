"""
Pure-Python replica of playground/host_weight_probe.cpp.

WHY. The kappa_thr,sub population sweep needs the per-sightline host weight
dNh(z_l, M) for every source redshift studied. Those weights come from the production
engine via `host_weight_probe`, but that is a compiled arm64 binary, so extending the
sweep to a new z_s normally means going back to the Mac. This module reproduces the
same integrand in Python so the z_s scan can be driven anywhere, and -- crucially --
CHECKS ITSELF against the cached engine dumps before its output is used.

What is reproduced (cpp/lensing.cpp::NhfNFW, line 138):

    dNh(z_l, M) = c * pi * ((1+z_l) * rmax)^2 / H(z_l) * dndlnM * dlnM * dz

on the engine's own 100 x 100 grid (z 0.01..10.01 log, M 1e7..1e17 log), with
    rmax   = rmaxfNFW, the radius where the halo's own kappa falls to kappa_thr
    dndlnM = ellipsoidal first-crossing HMF (p=0.3, q=0.8) on the smooth-k window
    kappa_thr from the legacy fixed-<N>=100 rule, i.e. solve NhfNFW(z_s, k) = 100

Components are taken from the two existing engine-matched Python modules rather than
rewritten: `analytic/sgl.py` (HMF, growth, sigma_M, H(z)) configured with
window="smoothk" + transfer="eh98" to match cpp/cosmology.cpp, and
`scripts/subhalo_gate/subhalo_factor_proxy_check.py` (NFW params, Sigma_crit, Fg) which
supplies the same `host_rmax` the Campbell quadrature already uses -- so the aperture
convention is shared by construction, not by coincidence.

GATE. `validate()` compares, at every z_s with a cached dump:
    kappa_thr, the NhfNFW total, and the per-cell dNh over the heaviest cells.
Nothing downstream should use these weights unless that passes. Run directly to see it:

    python playground/analytic/host_weights_py.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analytic"))

import sgl  # noqa: E402
from scripts.subhalo_gate.subhalo_factor_proxy_check import (  # noqa: E402
    fg_kappa, nfw_params, sigma_crit,
)

# engine grid + constants (cpp/cosmology.cpp defaults; CLAUDE.md "Cosmology constants")
NZ, NM = 100, 100
ZMIN, ZMAX = 0.01, 10.01
MMIN, MMAX = 1e7, 1e17
CLIGHT_KPC_GYR = 306.535  # only used via the c/H(z) = 306.535/Hz(z) kpc identity below
N_HOST_TARGET = 100.0     # legacy fixed-<N> threshold rule

ZLIST = np.logspace(np.log10(ZMIN), np.log10(ZMAX), NZ)
MLIST = np.logspace(np.log10(MMIN), np.log10(MMAX), NM)

_COS = None


def cosmo():
    """sgl cosmology configured to match cpp/cosmology.cpp (smooth-k window, EH98)."""
    global _COS
    if _COS is None:
        _COS = sgl.Cosmology(window="smoothk", transfer="eh98")
    return _COS


def _rmax_vec(kappa0, rs, kappa_thr, iters=120):
    """Vectorized rmaxfNFW: solve 2*kappa0*Fg(r/rs) = kappa_thr for r [kpc].

    Same bisection-in-log-r as playground/dgate/subhalo_factor_dgate_area_scan.py::
    host_rmax (which the Campbell quadrature already uses, so the aperture convention is
    shared), just run on the whole grid at once. Fg is monotone decreasing, so bisection
    is unconditionally safe; cells whose central kappa never reaches kappa_thr get 0.
    """
    lo = np.full_like(kappa0, 1.0e-6)
    hi = np.full_like(kappa0, 1.0e7)
    reachable = 2.0 * kappa0 * fg_kappa(lo / rs) > kappa_thr
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        above = 2.0 * kappa0 * fg_kappa(np.maximum(mid / rs, 1e-12)) > kappa_thr
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
    return np.where(reachable, np.sqrt(lo * hi), 0.0)


_DC_CACHE: dict[float, float] = {}


def _dl_a(z: float) -> float:
    """(1+z)*dc(z)/(1+z)^2 -- the angular-diameter combination sigma_crit needs."""
    if z not in _DC_CACHE:
        from scripts.subhalo_gate.subhalo_factor_proxy_check import dl
        _DC_CACHE[z] = dl(z)
    return _DC_CACHE[z] / (1.0 + z) ** 2


def _sigma_crit_vec(zs: float, zl: np.ndarray) -> np.ndarray:
    """sigma_crit for an array of lens redshifts (same algebra as proxy_check)."""
    ds_a = _dl_a(zs)
    dl_a = np.array([_dl_a(float(z)) for z in zl])
    dls_a = ds_a - dl_a * (1.0 + zl) / (1.0 + zs)
    return 2.08871e16 * ds_a / (4.0 * np.pi * dl_a * dls_a)


def weight_grid(zs: float, kappa_thr: float):
    """Return (zl, M, dNh, rmax) flat arrays over the engine grid, as the probe prints.

    Vectorized over the full 99 x 99 grid: the scalar version needed a bisection per
    cell and was far too slow to sit inside the kappa_thr root-find.
    """
    cos = cosmo()
    jz = np.arange(1, NZ)
    zl1 = ZLIST[jz]
    sel = zl1 < zs
    zl1, jz = zl1[sel], jz[sel]
    dz1 = ZLIST[jz] - ZLIST[jz - 1]

    M1 = MLIST[1:]
    dlnM1 = np.log(MLIST[1:]) - np.log(MLIST[:-1])

    ZL = zl1[:, None] * np.ones_like(M1)[None, :]
    MM = np.ones_like(zl1)[:, None] * M1[None, :]
    DZ = dz1[:, None] * np.ones_like(M1)[None, :]
    DLNM = np.ones_like(zl1)[:, None] * dlnM1[None, :]

    # NFW scale radius / central convergence, engine units (kpc, Msun)
    rs, rhos, _, _ = nfw_params(MM, ZL)
    kappa0 = rs * rhos / _sigma_crit_vec(zs, zl1)[:, None]
    rmax = _rmax_vec(kappa0, rs, kappa_thr)

    # comoving c/H(z) in kpc  (engine: c/H(z) = 306.535/Hz(z) kpc comoving)
    c_over_H_kpc = (sgl.CKMS / np.array([cos.Hz(float(z)) for z in zl1]) * 1.0e3)[:, None]
    dndlnM_kpc3 = np.array([cos.dndlnM(M1, float(z)) for z in zl1]) * 1.0e-9

    dNh = c_over_H_kpc * np.pi * ((1.0 + ZL) * rmax) ** 2 * dndlnM_kpc3 * DLNM * DZ
    return ZL.ravel(), MM.ravel(), dNh.ravel(), rmax.ravel()


def n_host(zs: float, kappa_thr: float) -> float:
    return float(weight_grid(zs, kappa_thr)[2].sum())


def kappa_thr_fixed_N(zs: float, target: float = N_HOST_TARGET) -> float:
    """Legacy rule: the kappa_thr at which the expected host count equals `target`.

    Returns kappa_thr itself (not its log).
    """
    lk = brentq(lambda x: n_host(zs, np.exp(x)) - target,
                np.log(1e-8), np.log(1e-1), xtol=1e-7, rtol=1e-10)
    return float(np.exp(lk))


def load_engine_dump(path: Path):
    head, rows = {}, []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            p = line[1:].split()
            if len(p) >= 2 and p[0] in ("zs", "kappathr", "NhfNFW", "total"):
                head[p[0]] = float(p[1])
        elif line.startswith("W "):
            rows.append([float(v) for v in line.split()[1:]])
    a = np.array(rows)
    return head, a[:, 0], a[:, 1], a[:, 2], a[:, 3]


def validate(z_sources=(0.5, 1.0, 5.0), verbose=True) -> bool:
    """Gate the Python weights against every cached engine dump. True if all pass."""
    ok_all = True
    for zs in z_sources:
        p = ROOT / "tmp" / f"host_weights_zs{zs}.txt"
        if not p.exists():
            continue
        head, zl_e, M_e, w_e, rm_e = load_engine_dump(p)
        k_e = head["kappathr"]

        k_p = kappa_thr_fixed_N(zs)
        # compare the WEIGHTS at the engine's own kappa_thr, so a small threshold
        # difference does not contaminate the per-cell comparison
        _, _, w_p, rm_p = weight_grid(zs, k_e)
        tot_p = w_p.sum()

        heavy = np.argsort(w_e)[-500:]
        wr = w_p[heavy] / w_e[heavy]
        rr = rm_p[heavy] / rm_e[heavy]
        ok = (abs(k_p / k_e - 1) < 0.05 and abs(tot_p / head["NhfNFW"] - 1) < 0.05
              and abs(np.median(wr) - 1) < 0.05)
        ok_all &= ok
        if verbose:
            print(f"z_s={zs}: kappa_thr py/engine = {k_p / k_e:.5f}   "
                  f"NhfNFW py/engine = {tot_p / head['NhfNFW']:.5f}")
            print(f"         per-cell dNh ratio (500 heaviest): median {np.median(wr):.5f} "
                  f"[{wr.min():.4f}, {wr.max():.4f}]")
            print(f"         rmax ratio: median {np.median(rr):.5f} "
                  f"[{rr.min():.4f}, {rr.max():.4f}]   -> {'PASS' if ok else 'FAIL'}")
    return bool(ok_all)


if __name__ == "__main__":
    print("HMF checkpoint (tmp/engine_checkpoints.txt, engine values in brackets):")
    for z, M, ref in ((0.010000, 8.902151e11, 1.951187e-03),
                      (0.497984, 9.111628e12, 2.064092e-04),
                      (1.000667, 9.326033e13, 4.811894e-06)):
        got = cosmo().dndlnM(M, z)
        print(f"  z={z:<9.6f} M={M:.4e}  dndlnM {got:.6e}  [{ref:.6e}]  "
              f"ratio {got / ref:.5f}")
    print()
    print("Host-weight gate vs the cached engine dumps:")
    ok = validate()
    print()
    print("GATE PASSED" if ok else "GATE FAILED -- do not use these weights")
    sys.exit(0 if ok else 1)
