#!/usr/bin/env python3
"""Radial subhalo bias B(x) in two unit systems, to compare r_vir vs r_200.

Version 1 (r_vir): the Green+21 / Bolshoi calibration in its NATIVE units
(x = r/r_vir); B normalized to unity at x = 1 (= r_vir).
Version 2 (r_200): the SAME physical profile expressed in the code's units
(x = r/r_200 = eta * (r/r_vir)); the r_vir-calibrated transition 0.54 lands at
0.54*eta, the profile reaches "unbiased" (B=1) at x = eta (= r_vir), and the code
truncates the population at x = 1 (= r_200). eta = r_vir/r_200 = c_vir/c_200 from
the Bryan-Norman virial overdensity, evaluated for the calibration host.
"""
import numpy as np
from math import pi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- eta = r_vir/r200 (Bryan-Norman + NFW), same as cpp/subhalo.cpp::etaVirTo200
Om0, OL = 0.315, 0.685
def OmegaMz(z): a = Om0*(1+z)**3 + OL; return Om0*(1+z)**3/a
def mfun(y):    return np.log(1+y) - y/(1+y)
def eta_vir_to_200(c200, z):
    d = OmegaMz(z) - 1.0
    Dvir = 18*pi*pi + 82*d - 39*d*d
    t = 200.0*c200**3/mfun(c200)/Dvir
    lo, hi = c200, 3*c200
    for _ in range(80):
        m = 0.5*(lo+hi)
        if m**3/mfun(m) < t: lo = m
        else: hi = m
    return 0.5*(lo+hi)/c200
def c200_DM14(M_h, z):                 # Dutton-Maccio 2014, M in h^-1 Msun
    a = 0.520 + (0.905-0.520)*np.exp(-0.617*z**1.21); b = -0.101 + 0.026*z
    return 10**(a + b*np.log10(M_h/1e12))

# calibration host: Green+21 Fig 7, M0 = 10^14.2 h^-1 Msun, z ~ 0
Z_CAL, M_CAL_h = 0.0, 10**14.2
C200_CAL = c200_DM14(M_CAL_h, Z_CAL)
ETA = eta_vir_to_200(C200_CAL, Z_CAL)

# ---- digitized Bolshoi points (r_vir native), from plot_fig_subhalo_population.py
BOL_LX = np.array([-1.5,-1.4,-1.3,-1.25,-1.15,-1.1,-1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.0])
BOL_LB = np.array([-1.52,-1.50,-1.33,-1.28,-1.30,-1.25,-1.10,-0.97,-0.85,-0.73,-0.58,-0.48,-0.38,-0.25,-0.15,-0.06,0.0])
x_bol_vir = 10**BOL_LX;  B_bol = 10**BOL_LB

def B_trans(xv, x0, a):                # normalized to 1 at x=1 (r_vir)
    raw = 1/np.sqrt((xv/x0)**(-a) + 1); return raw/(1/np.sqrt((1/x0)**(-a) + 1))
X0_OLD, A_OLD = 0.54, 2.5              # previous fit (to Green model curve)
X0_FREE, A_FREE = 0.675, 2.73         # refit to Bolshoi pts, free exponent
X0_5HALF, A_5HALF = 0.86, 2.5         # refit to Bolshoi pts, exponent fixed at 5/2
def B_adopt_vir(xv):  return B_trans(xv, X0_FREE, A_FREE)
SP08_A, SP08_X2, SP08_C = 0.678, 0.81, 16.11
def B_springel_vir(xv):
    def raw(y):
        ne = np.exp(-2.0/SP08_A*((y/SP08_X2)**SP08_A - 1.0))
        rn = 1.0/(SP08_C*y*(1.0+SP08_C*y)**2); return ne/rn
    return raw(xv)/raw(1.0)
def B_han_vir(xv): return xv**1.3      # normalized to 1 at x=1

xv = np.logspace(-1.6, 0.02, 300)

def make_panel(ax, unit):
    """unit = 'vir' or '200'. In r200, x = eta*x_vir (physical, not renormalized)."""
    s = ETA if unit == "200" else 1.0
    ax.plot(np.log10(xv*s), np.log10(B_trans(xv, X0_OLD, A_OLD)), color='0.6',
            lw=1.3, ls='--', label=r'previous ($0.54,\,5/2$)')
    ax.plot(np.log10(xv*s), np.log10(B_trans(xv, X0_5HALF, A_5HALF)), 'C2-', lw=1.5,
            label=r'refit, $5/2$ fixed ($x_0{=}0.86$)')
    ax.plot(np.log10(xv*s), np.log10(B_trans(xv, X0_FREE, A_FREE)), 'C0-', lw=2,
            label=r'refit, free ($x_0{=}0.68,\,p{=}2.73$)')
    ax.plot(BOL_LX + np.log10(s), BOL_LB, 'ks', ms=6, label='Bolshoi (Klypin+11)')
    ax.plot(np.log10(xv*s), np.log10(B_springel_vir(xv)), 'C4-.', lw=1.5,
            label='Springel+08 (Aq-A-1)')
    ax.plot(np.log10(xv*s), np.log10(B_han_vir(xv)), 'C3:', lw=1.5,
            label=r'Han+16 ($x^{1.3}$)')
    leg_loc = 'lower right'
    if unit == "vir":
        ax.set_xlabel(r'$\log(x = r/r_{\rm vir})$')
        ax.set_title(r'Native calibration units ($r_{\rm vir}$)', fontsize=10)
    else:
        ax.axvline(0.0, color='0.5', lw=1.2, ls='-')             # x = 1 = r200 (code truncation)
        ax.axvline(np.log10(ETA), color='0.5', lw=1.0, ls='--')  # x = eta = r_vir
        ax.text(0.0, 0.02, r'$r_{200}$', color='0.4', fontsize=8, ha='center', va='bottom')
        ax.text(np.log10(ETA), 0.02, r'$r_{\rm vir}$', color='0.4', fontsize=8, ha='center', va='bottom')
        ax.set_xlabel(r'$\log(x = r/r_{200})$')
        ax.set_title(r'Code units ($r_{200}$), $\eta=r_{\rm vir}/r_{200}=%.2f$' % ETA, fontsize=10)
        leg_loc = 'upper left'
    ax.set_ylabel(r'$\log B(x)=\log(n_{\rm sub}/n_{\rm host})$')
    ax.set_ylim(-2.0, 0.18); ax.set_xlim(-1.65, 0.22)
    ax.grid(alpha=0.3); ax.legend(fontsize=7.5, loc=leg_loc)

OUT = "/sessions/gracious-nice-lovelace/mnt/gw-wl-emulator/paper_prod/plots/figures"
import os; os.makedirs(OUT, exist_ok=True)
for unit, tag in [("vir","rvir"), ("200","r200")]:
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    make_panel(ax, unit)
    fig.tight_layout()
    for ext in ("png","pdf"):
        fig.savefig(f"{OUT}/fig_radial_bias_{tag}.{ext}", dpi=150)
    plt.close(fig)
    print(f"wrote fig_radial_bias_{tag}")
print(f"calibration host: M=10^14.2 h^-1, z={Z_CAL}: c200={C200_CAL:.2f}, eta={ETA:.3f}")
print(f"transition 0.54 r_vir -> {0.54*ETA:.3f} r200 ; B at r200 boundary = {B_adopt_vir(1.0/ETA):.3f}")
