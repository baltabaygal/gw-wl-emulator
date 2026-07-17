"""
Component-level validation of the analytic deficit model at FIXED ray distance y.

For a few y values, draw many clump realizations (same machinery as the playground
scans) and compare, per gate:
    E[S]      vs analytic mu(y)
    Var(S)    vs analytic C(y)          (Poisson/Campbell variance)
    E[f_res]  vs analytic frac(y)
This isolates kernel/normalization errors from the heavy-tailed ray sampling.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path("/Users/baltabay/Desktop/gw-wl-emulator")
sys.path.insert(0, str(ROOT))

from scripts.subhalo_factor_proxy_check import (  # noqa: E402
    ALPHA, BETA, OMEGA, PSI_MAX,
    build_reach_interpolator, fg_kappa, gamma_norm, n_tau, nfw_params,
    sample_brute_realization, sigma_crit,
)
from playground.subhalo_factor_dgate_area_scan import host_rmax  # noqa: E402
from playground.subhalo_factor_analytic_deficit import (  # noqa: E402
    distance_kernel, projected_profile,
)

HOST = 1.0e13
ZL, ZS = 0.5, 1.0
KTHR_HOST = 1.276e-4
M_FLOOR = 1.0e7
FACTOR = 1.0e-3
Y_TEST = [50.0, 200.0, 400.0, 800.0]
NREAL = 4000

rs_h, rhos_h, c_h, r200_h = nfw_params(HOST, ZL)
nt, _ = n_tau(HOST, ZL)
fs = 0.3563 / nt**0.6 - 0.075
gamma = gamma_norm(fs)
sigmac = sigma_crit(ZS, ZL)
psi_lo = M_FLOOR / HOST
nbar = gamma / ALPHA * (PSI_MAX**ALPHA - psi_lo**ALPHA)

reach_interp = build_reach_interpolator(
    m_min=M_FLOOR, m_max=PSI_MAX * HOST, zl=ZL, zs=ZS, kappa_thr=FACTOR * KTHR_HOST,
)

# ---------- analytic pieces ----------
n_mass, n_d = 80, 240
lpsi = np.linspace(np.log(psi_lo), np.log(PSI_MAX), n_mass)
psi = np.exp(lpsi)
w_m = psi**ALPHA * np.exp(-BETA * psi**OMEGA)
w_m /= np.trapezoid(w_m, lpsi)
mass = psi * HOST
d_grid = np.logspace(-3, np.log10(host_rmax(HOST, ZL, ZS, KTHR_HOST) + r200_h), n_d)
k1 = np.empty((n_mass, n_d))
for i, m in enumerate(mass):
    rs_i, rhos_i, _, _ = nfw_params(float(m), ZL)
    k1[i] = 2.0 * (rs_i * rhos_i / sigmac) * fg_kappa(np.maximum(d_grid / rs_i, 1.0e-12))
k2 = k1**2
sigma_interp, _ = projected_profile(c_h, r200_h)
y_arr = np.array(Y_TEST)
f_dy = distance_kernel(y_arr, d_grid, sigma_interp, n_theta=512)
wt_d = np.gradient(np.log(d_grid))
fd = f_dy * (d_grid * wt_d)[None, :]
reach_m = reach_interp(mass)

print(f"host={HOST:.0e} zl={ZL} zs={ZS} factor={FACTOR:g}  nbar={nbar:.1f} fs={fs:.4f}")
print(f"kernel normalization check: int f(d|y) dd = {fd.sum(axis=1)}")

mean_psi = np.trapezoid(w_m * psi, lpsi)
print(f"analytic resolved mass frac (brute) = {nbar * mean_psi:.4f}  (cf fs={fs:.4f})")

ana = {}
for gate in ["brute", "proxy", "truth"]:
    if gate == "brute":
        keep_md = np.ones((n_mass, n_d))
        keep_my = np.ones((n_mass, len(y_arr)))
    elif gate == "proxy":
        keep_md = np.ones((n_mass, n_d))
        keep_my = (y_arr[None, :] <= reach_m[:, None]).astype(float)
    else:
        keep_md = (d_grid[None, :] <= reach_m[:, None]).astype(float)
        keep_my = np.ones((n_mass, len(y_arr)))
    J1 = np.einsum("yd,md->my", fd, k1 * keep_md) * keep_my
    J2 = np.einsum("yd,md->my", fd, k2 * keep_md) * keep_my
    Pk = np.einsum("yd,md->my", fd, np.broadcast_to(keep_md, (n_mass, n_d)).astype(float)) * keep_my
    mu = nbar * np.trapezoid(w_m[:, None] * J1, lpsi, axis=0)
    C = nbar * np.trapezoid(w_m[:, None] * J2, lpsi, axis=0)
    fr = nbar * np.trapezoid((w_m * psi)[:, None] * Pk, lpsi, axis=0)
    ana[gate] = (mu, C, fr)

# ---------- MC pieces ----------
mc = {g: {"S": [], "fr": []} for g in ["brute", "proxy", "truth"]}
rng_master = np.random.default_rng(7)
for it in range(NREAL):
    seed = int(rng_master.integers(0, 2**31))
    real = sample_brute_realization(seed, HOST, ZL, M_FLOOR, gamma, r200_h, c_h)
    m = np.asarray(real["mass"], dtype=float)
    if len(m) == 0:
        for g in mc:
            mc[g]["S"].append(np.zeros(len(y_arr)))
            mc[g]["fr"].append(np.zeros(len(y_arr)))
        continue
    x = np.asarray(real["x"], dtype=float)
    yy = np.asarray(real["y"], dtype=float)
    reach = reach_interp(m)
    rs_c = np.empty_like(m)
    k0_c = np.empty_like(m)
    for i, mi in enumerate(m):
        rs_i, rhos_i, _, _ = nfw_params(float(mi), ZL)
        rs_c[i] = rs_i
        k0_c[i] = rs_i * rhos_i / sigmac
    d = np.sqrt((x[None, :] - y_arr[:, None]) ** 2 + yy[None, :] ** 2)   # (ny, nclump)
    kap = 2.0 * k0_c[None, :] * fg_kappa(np.maximum(d / rs_c[None, :], 1.0e-12))
    keep_p = reach[None, :] >= y_arr[:, None]
    keep_t = reach[None, :] >= d
    mc["brute"]["S"].append(kap.sum(axis=1))
    mc["proxy"]["S"].append((kap * keep_p).sum(axis=1))
    mc["truth"]["S"].append((kap * keep_t).sum(axis=1))
    mc["brute"]["fr"].append(np.full(len(y_arr), m.sum() / HOST))
    mc["proxy"]["fr"].append((m[None, :] * keep_p).sum(axis=1) / HOST)
    mc["truth"]["fr"].append((m[None, :] * keep_t).sum(axis=1) / HOST)

print(f"\nNREAL={NREAL}, factor={FACTOR:g}")
for g in ["brute", "proxy", "truth"]:
    S = np.array(mc[g]["S"])
    fr = np.array(mc[g]["fr"])
    mu_a, C_a, fr_a = ana[g]
    print(f"\n== {g} ==")
    print(f"{'y':>6} {'E[S] mc':>12} {'mu ana':>12} {'ratio':>7} | "
          f"{'Var(S) mc':>12} {'C ana':>12} {'ratio':>7} | "
          f"{'E[fr] mc':>10} {'fr ana':>10}")
    for j, y in enumerate(Y_TEST):
        Es, Vs = S[:, j].mean(), S[:, j].var(ddof=1)
        sem = S[:, j].std(ddof=1) / np.sqrt(NREAL)
        print(f"{y:6.0f} {Es:12.4e} {mu_a[j]:12.4e} {mu_a[j]/max(Es,1e-300):7.3f} | "
              f"{Vs:12.4e} {C_a[j]:12.4e} {C_a[j]/max(Vs,1e-300):7.3f} | "
              f"{fr[:, j].mean():10.4f} {fr_a[j]:10.4f}")
