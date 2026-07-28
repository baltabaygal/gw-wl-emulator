#!/usr/bin/env python
"""
Fig 4 (replaces the obsolete fig:subhalo-factor / eps_sub partition figure).

Supervisor request (2026-07-23): show the total convergence scatter sigma_kappa of a
single lensing halo under the simplified production model (subhalo_model = 4):

    kappa_halo(r) = kappa_NFW[M - sum_i m_i](r) + sum_i kappa_i(r),

i.e. every subhalo sampled individually down to the absolute floor psi_min = m_floor/M,
the smooth host carved to M - sum_i m_i (exact per-realization mass conservation), and NO
unresolved term (no M_u, no kappa_u/Wsub, no dynamic floor, no subhalo_factor). The figure
plots, versus host-centric impact parameter r, the mean convergence and its scatter
sigma_kappa for (i) the total halo, (ii) the carved smooth host alone, and (iii) the
subhalo population alone.

Method: the discrete subhalos form a marked Poisson process (SHMF intensity x anti-biased
radial profile), so the mean and variance of sum_i kappa_i and sum_i m_i at a given ray are
the exact Campbell moments -- estimated here by a single-clump Monte-Carlo pool of the SAME
NFW kernel, concentration relation (Dutton-Maccio 2014) and radial profile B(x) used by the
C++ (cpp/subhalo.cpp). This is exact for the Poisson clump field (no per-realization brute
loop needed) and reproduces the draft's Eq.(kappau_moments)-style integrals. The carved-host
scatter and the host<->subhalo anti-correlation follow from the same pool via a first-order
(dkappa_host/dM) response to the realized carved mass sum_i m_i.

Representative host: M = 1e13 Msun, z_l = 0.5, z_s = 1 (bound fraction f_s from the JvdB14
dynamical-age fit is taken at its cluster-scale value f_s = 0.14 for this host; the shape of
the decomposition is insensitive to the exact f_s). Final paper numbers should be
cross-checked against a C++ single-host driver once model 4 is built on the Mac.

Output: paper_prod/plots/figures/fig_subhalo_sigma_decomposition_mean.{png,pdf}
        paper_prod/plots/figures/fig_subhalo_sigma_decomposition_scatter.{png,pdf}
        (single-panel figures -- one for the mean profile, one for the scatter;
        previously a single two-panel figure.)
"""
import sys
from pathlib import Path

import numpy as np
from scipy import integrate

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from paper_prod.plot_style import (apply_style, FIGURE_SIZES, SUBPLOTS_ADJUST,
                                   format_log_axis_decimal)
from plot_fig_subhalo_population import guard_broken_latex

OUT_DIR = REPO / "paper_prod" / "plots" / "figures"

rng = np.random.default_rng(20260723)

# ---- cosmology (CLAUDE.md internal constants: kpc, Msun) -------------------
Om, OL, h = 0.315, 0.685, 0.674
RHOC0 = 277.394 * h * h                      # Msun/kpc^3
CKMS = 2.998e5                               # km/s
G_KPC = 4.30091e-6                           # kpc (km/s)^2 / Msun
C2_4piG = CKMS**2 / (4 * np.pi * G_KPC)      # Msun/kpc


def Ez(z):
    return np.sqrt(Om * (1 + z)**3 + OL)


def OmegaMz(z):
    return Om * (1 + z)**3 / Ez(z)**2


def Dc(z):                                    # comoving distance [kpc]
    return (CKMS / (100 * h)) * 1e3 * integrate.quad(lambda zz: 1 / Ez(zz), 0, z)[0]


ZL, ZS = 0.5, 1.0
DcL, DcS = Dc(ZL), Dc(ZS)
DAL, DAS = DcL / (1 + ZL), DcS / (1 + ZS)
DALS = (DcS - DcL) / (1 + ZS)
SIGMA_C = C2_4piG * DAS / (DAL * DALS)        # Msun/kpc^2 (physical)
rhoc_zl = RHOC0 * Ez(ZL)**2


# ---- concentration: Dutton & Maccio 2014 (cons14, active in cpp) -----------
def cons14(M, z):
    # c200 for masses in Msun; DM14 quote in h^-1 Msun with pivot 1e12 h^-1 Msun
    a = 0.520 + (0.905 - 0.520) * np.exp(-0.617 * z**1.21)
    b = -0.101 + 0.026 * z
    return 10.0**(a + b * np.log10(M * h / 1e12))


def nfw_mu(y):
    return np.log(1.0 + y) - y / (1.0 + y)


def nfw_params(M, z=ZL):
    c = cons14(M, z)
    r200 = (3 * M / (4 * np.pi * 200 * (RHOC0 * Ez(z)**2)))**(1 / 3.)
    rs = r200 / c
    rhos = (200 / 3.) * (RHOC0 * Ez(z)**2) * c**3 / nfw_mu(c)
    return rs, rhos, r200, c


def Fg(x):                                     # NFW convergence shape f(x); kappa = 2 kappa0 f(x)
    x = np.asarray(x, float)
    out = np.empty_like(x)
    lo, hi = x < 1 - 1e-6, x > 1 + 1e-6
    mid = ~(lo | hi)
    xl, xh = x[lo], x[hi]
    out[lo] = (1 - 2 / np.sqrt(1 - xl**2) * np.arctanh(np.sqrt((1 - xl) / (1 + xl)))) / (xl**2 - 1)
    out[hi] = (1 - 2 / np.sqrt(xh**2 - 1) * np.arctan(np.sqrt((xh - 1) / (xh + 1)))) / (xh**2 - 1)
    out[mid] = 1 / 3.
    return out


def kappa_nfw_M(M, R, z=ZL):                   # convergence of NFW halo mass M at 2D sep R [kpc]
    rs, rhos, _, _ = nfw_params(M, z)
    kappa0 = rs * rhos / SIGMA_C
    return 2 * kappa0 * Fg(np.maximum(R / rs, 1e-6))


def kappa_nfw_arr(m, R):                        # vectorized over clump masses m (each its own c,rs)
    c = cons14(m, ZL)
    r200 = (3 * m / (4 * np.pi * 200 * rhoc_zl))**(1 / 3.)
    rs = r200 / c
    rhos = (200 / 3.) * rhoc_zl * c**3 / nfw_mu(c)
    kappa0 = rs * rhos / SIGMA_C
    return 2 * kappa0 * Fg(np.maximum(R / rs, 1e-6))


# ---- host + SHMF -----------------------------------------------------------
M_HOST = 1.0e13
RS_H, RHOS_H, R200, C_H = nfw_params(M_HOST, ZL)

ALPHA, BETA, OMEGA = -0.82, 50.0, 4.0
PSI_MAX = 1.0
PSI_MIN = 1.0e7 / M_HOST                        # = m_floor / M  (supervisor's floor)
F_B = 0.14                                       # JvdB14 bound fraction for this host (representative)


def dN_dpsi(psi):
    return psi**(ALPHA - 1.0) * np.exp(-BETA * psi**OMEGA)


mass_int = integrate.quad(lambda p: p * dN_dpsi(p), PSI_MIN, PSI_MAX)[0]
GAMMA = F_B / mass_int
N_ALL = integrate.quad(lambda p: GAMMA * dN_dpsi(p), PSI_MIN, PSI_MAX)[0]
print(f"M={M_HOST:.0e} R200={R200:.0f} kpc c_host={C_H:.2f} f_b={F_B} "
      f"psi_min={PSI_MIN:.1e} N_all={N_ALL:.3e} Sigma_c={SIGMA_C:.3e}")

# psi inverse-CDF sampler over the full band
_pg = np.logspace(np.log10(PSI_MIN), np.log10(PSI_MAX), 6000)
_w = dN_dpsi(_pg)
_cdf = np.concatenate([[0], np.cumsum(0.5 * (_w[1:] + _w[:-1]) * np.diff(_pg))])
_cdf /= _cdf[-1]


def sample_psi(n):
    return np.interp(rng.random(n), _cdf, _pg)


# ---- anti-biased radial profile B(x), x = r3d/r200 (matches cpp/subhalo.cpp) ----
BIAS_X0_RVIR, BIAS_EXP = 0.86, 2.5


def eta_vir_to_200(c200, z):
    d = OmegaMz(z) - 1.0
    Dvir = 18.0 * np.pi**2 + 82.0 * d - 39.0 * d * d
    target = 200.0 * c200**3 / nfw_mu(c200) / Dvir
    lo, hi = c200, 3.0 * c200
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if mid**3 / nfw_mu(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi) / c200


X0 = BIAS_X0_RVIR * eta_vir_to_200(C_H, ZL)     # transition scale in r200 units


def Bx(x):
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    m = x > 0
    out[m] = 1.0 / np.sqrt((x[m] / X0)**(-BIAS_EXP) + 1.0)
    return out


# inverse-CDF of dN/dx = 4 pi r^2 rho_NFW(x) B(x) ~ x/(1+c_host x)^2 B(x) on x in (0,1]
# (matches cpp/subhalo.cpp; corrected 2026-07-28 from x^2/(1+c x)^2 -- see the
# RADIAL SAMPLING CONVENTION block at the top of cpp/subhalo.cpp)
_xs = np.linspace(1e-4, 1.0, 6000)
_wx = _xs / (1 + C_H * _xs)**2 * Bx(_xs)
_cx = np.concatenate([[0], np.cumsum(0.5 * (_wx[1:] + _wx[:-1]) * np.diff(_xs))])
_cx /= _cx[-1]


def make_position_pool(n):
    # draw the 3D clump positions ONCE; reuse across all r (common random numbers) so the
    # Campbell moments are smooth in r rather than independently noisy per grid point.
    x = np.interp(rng.random(n), _cx, _xs)
    r3d = x * R200
    cth = rng.uniform(-1, 1, n)
    az = rng.uniform(0, 2 * np.pi, n)
    R2 = r3d * np.sqrt(1 - cth**2)
    return R2 * np.cos(az), R2 * np.sin(az)      # in-plane offsets of each clump


def sep_from_pool(px, py, r_ray):
    dx = r_ray - px
    return np.sqrt(dx * dx + py * py)


# ---- Campbell moments on an r-grid (single-clump MC pool) ------------------
# For a marked Poisson process with intensity N_all and single-clump law:
#   <sum kappa_i>   = N_all <kappa_c>
#   Var(sum kappa_i)= N_all <kappa_c^2>                          (Poisson sum variance)
#   Var(sum m_i)    = N_all <m^2>
#   Cov(sum m_i, sum kappa_i) = N_all <m kappa_c>
# The carved host kappa_h(r) = kappa_NFW(M - sum m_i, r); to first order in the mass
# fluctuation delta = sum m_i - <sum m_i>:
#   <kappa_h>   ~ kappa_NFW(M - <sum m_i>, r) = kappa_NFW((1-f_b)M, r)
#   sigma_h(r)  ~ |dkappa_NFW/dM|_{(1-f_b)M} * sqrt(Var(sum m_i))
#   Cov(kappa_h, sum kappa_i) ~ -(dkappa_NFW/dM) * Cov(sum m_i, sum kappa_i)   (carve anti-corr.)
def smooth(y, w=5):
    # running mean (odd window) for display of the heavy-tailed second-moment estimators;
    # sigma_sub is dominated by rare close (untruncated-NFW) clump encounters, so the raw
    # per-r Campbell estimator is noisy even with common random numbers.
    k = w // 2
    yp = np.pad(y, k, mode='edge')
    return np.convolve(yp, np.ones(w) / w, mode='valid')


RGRID = np.linspace(0.03 * R200, 1.0 * R200, 60)
NS = 4000000

mu_sub = np.zeros_like(RGRID)     # <sum kappa_i>
var_sub = np.zeros_like(RGRID)    # Var(sum kappa_i)
cov_mk = np.zeros_like(RGRID)     # Cov(sum m_i, sum kappa_i)
psi_pool = sample_psi(NS)
m_pool = psi_pool * M_HOST
var_M = N_ALL * (m_pool**2).mean()             # Var(sum m_i), r-independent
sig_M = np.sqrt(var_M)
pool_px, pool_py = make_position_pool(NS)       # common random positions, reused across r
for i, r in enumerate(RGRID):
    d = sep_from_pool(pool_px, pool_py, r)
    kc = kappa_nfw_arr(m_pool, d)
    mu_sub[i] = N_ALL * kc.mean()
    var_sub[i] = N_ALL * (kc**2).mean()
    cov_mk[i] = N_ALL * (kc * m_pool).mean()

# carved-host mean and dkappa/dM (numerical) at each r
Mred = (1.0 - F_B) * M_HOST
mu_host = kappa_nfw_M(Mred, RGRID)
dM = 1e-3 * Mred
dkdM = (kappa_nfw_M(Mred + dM, RGRID) - kappa_nfw_M(Mred - dM, RGRID)) / (2 * dM)

# smooth the heavy-tailed second moments for display (means left raw; they are smooth)
var_sub_s = smooth(var_sub)
cov_mk_s = smooth(cov_mk)

sig_host = np.abs(dkdM) * sig_M
var_host = sig_host**2
cov_hk = -dkdM * cov_mk_s                         # Cov(kappa_host, sum kappa_i)
mu_tot = mu_host + mu_sub
var_tot = var_host + var_sub_s + 2 * cov_hk
sig_tot = np.sqrt(np.maximum(var_tot, 0.0))
sig_sub = np.sqrt(np.maximum(var_sub_s, 0.0))

xr = RGRID / R200

# ---- figure ----------------------------------------------------------------
apply_style()
guard_broken_latex()
import matplotlib.pyplot as plt

# same hue set as the other subhalo-subsection figures: C0 = "this work"
# headline curve, C1 = the secondary component; total is plain black.
C_TOT, C_HOST, C_SUB = 'k', 'C1', 'C0'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save(fig, tag):
    out_png = OUT_DIR / f'fig_subhalo_sigma_decomposition_{tag}.png'
    out_pdf = out_png.with_suffix('.pdf')
    fig.savefig(out_png, dpi=300, facecolor='white')
    fig.savefig(out_pdf, facecolor='white')
    print(f"wrote {out_pdf.relative_to(REPO)}")
    print(f"wrote {out_png.relative_to(REPO)}")


# ---- mean convergence profile -- the mean is exactly additive
# (<kappa_halo> = <kappa_host> + <kappa_sub>), so plain component names are fine.
fig_m, ax_m = plt.subplots(figsize=FIGURE_SIZES["single"])
fig_m.subplots_adjust(**SUBPLOTS_ADJUST["single"])
ax_m.plot(xr, mu_tot, color=C_TOT, lw=1.5, label=r'$\kappa_{\rm halo}$')
ax_m.plot(xr, mu_host, color=C_HOST, lw=1.3, ls='--', label='host')
ax_m.plot(xr, mu_sub, color=C_SUB, lw=1.3, ls=':', label='subhalos')
ax_m.set_yscale('log')
ax_m.set_xlabel(r'$r/r_{200}$')
ax_m.set_ylabel(r'$\langle\kappa\rangle$')
format_log_axis_decimal(ax_m, axis='y')
ax_m.legend(fontsize=6.5, loc='upper right')
ax_m.set_xlim(0, 1)
save(fig_m, 'mean')

# ---- convergence scatter sigma_kappa -- NOT additive (sigma_tot != sigma_host
# + sigma_sub; the host<->subhalo carve anti-correlation contributes a cross
# term). Label each curve as its own sigma so "host"/"subhalos" read as the
# standalone scatter of that term, not as summands of the total.
fig_s, ax_s = plt.subplots(figsize=FIGURE_SIZES["single"])
fig_s.subplots_adjust(**SUBPLOTS_ADJUST["single"])
ax_s.plot(xr, sig_tot, color=C_TOT, lw=1.5, label=r'$\sigma_\kappa$')
ax_s.plot(xr, sig_host, color=C_HOST, lw=1.3, ls='--', label=r'$\sigma_{\kappa,\rm host}$')
ax_s.plot(xr, sig_sub, color=C_SUB, lw=1.3, ls=':', label=r'$\sigma_{\kappa,\rm sub}$')
ax_s.set_yscale('log')
ax_s.set_xlabel(r'$r/r_{200}$')
ax_s.set_ylabel(r'$\sigma_\kappa$')
format_log_axis_decimal(ax_s, axis='y')
ax_s.legend(fontsize=6.5, loc='upper right')
ax_s.set_xlim(0, 1)
save(fig_s, 'scatter')

# console summary at a few radii
print("\n r/r200   <k>_tot   <k>_host  <k>_sub   sig_tot   sig_host  sig_sub   sub/tot(var)")
for xi in (0.1, 0.3, 0.6, 0.9):
    i = np.argmin(np.abs(xr - xi))
    frac = var_sub_s[i] / var_tot[i]
    print(f"  {xr[i]:.2f}   {mu_tot[i]:.4e} {mu_host[i]:.4e} {mu_sub[i]:.4e}  "
          f"{sig_tot[i]:.3e} {sig_host[i]:.3e} {sig_sub[i]:.3e}   {frac:.3f}")
