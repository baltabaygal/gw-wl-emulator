"""Cross-check Fig 4 (subhalo sigma decomposition) against the production C++ sampler.

Oracle: playground/subhalo_single_host_probe.cpp, which calls the production
Subhalo::precompute + addClumps (model 4) + the production carve on ONE grid host.

Two things must agree or the figure is not a picture of the code:
  1. the SHMF normalization  -- the engine sets gamma so that JvdB14's f_s is the bound
     mass fraction over the RESOLVED band [psi_res = 1e-4, 1] (cpp/subhalo.cpp: gden is
     an incomplete Gamma from psi_res), NOT over [psi_min, 1].
  2. the radial profile      -- dN/dx ~ x B(x)/(1+cx)^2 with x0 = 0.86*eta in r_200 units.

Run:
  ./playground/subhalo_single_host_probe 4000 1e13 0.5 1.0 1e7 20260728 tmp/probe_fig4.csv 0
  /Users/baltabay/miniforge3/envs/test/bin/python paper_prod/scripts/check_fig4_vs_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import integrate
from scipy.special import gammaincc, gamma as gamma_fn

ROOT = Path("/Users/baltabay/Desktop/gw-wl-emulator")
sys.path.insert(0, str(ROOT))

ALPHA, BETA, OMEGA = -0.82, 50.0, 4.0
PSI_RES = 1.0e-4


def dN_dpsi(p):
    return p ** (ALPHA - 1.0) * np.exp(-BETA * p ** OMEGA)


def gamma_engine(fs):
    """cpp/subhalo.cpp: gam = omega beta^s / gden * fs, gden from psi_res."""
    s = (1.0 + ALPHA) / OMEGA
    gden = gamma_fn(s) * (gammaincc(s, BETA * PSI_RES ** OMEGA)
                          - gammaincc(s, BETA * 1.0 ** OMEGA))
    return OMEGA * BETA ** s / gden * fs


def gamma_figure(fb, psi_min):
    """Fig 4 as written: f_b spread over [psi_min, 1]."""
    return fb / integrate.quad(lambda p: p * dN_dpsi(p), psi_min, 1.0)[0]


def n_all(g, psi_min):
    return integrate.quad(lambda p: g * dN_dpsi(p), psi_min, 1.0)[0]


def read_probe(path):
    head = {}
    for tok in path.read_text().splitlines()[0].lstrip("#").split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            head[k] = float(v)
    return head, np.genfromtxt(path, delimiter=",", names=True, skip_header=1)


if __name__ == "__main__":
    head, d = read_probe(ROOT / "tmp/probe_fig4.csv")
    M, fs, psi_min_e = head["M"], head["fs_jvdb14"], head["psi_min"]
    print(f"engine grid host: M={M:.5e}  f_s(JvdB14)={fs:.6f}  psi_min={psi_min_e:.5e}")
    print(f"engine gnorm reported = {head['gnorm']:.6f}")

    g_e = gamma_engine(fs)
    print(f"\n  gamma engine convention  (f_s over [1e-4, 1]) = {g_e:.6f}  "
          f"-> N_all = {n_all(g_e, psi_min_e):8.1f}")
    print(f"  probe <N_c> (measured)                        = "
          f"{'':16s}{d['mean_Nc'].mean():8.1f}")

    # OLD Fig 4 convention (pre-2026-07-28): F_B spread over [psi_min, 1]
    g_old = gamma_figure(0.14, 1.0e-6)
    print(f"\n  gamma OLD Fig 4 convention (F_B over [1e-6, 1]) = {g_old:.6f}  "
          f"-> N_all = {n_all(g_old, 1.0e-6):8.1f}")

    # the SAME convention evaluated at the ENGINE's host, which is the like-for-like test
    g_old_e = gamma_figure(fs, psi_min_e)
    r_norm = n_all(g_e, psi_min_e) / n_all(g_old_e, psi_min_e)
    print(f"\n  like-for-like at the engine host (removes the 1e13 vs {M:.3g} offset):")
    print(f"    engine band [1e-4,1] : N_all = {n_all(g_e, psi_min_e):8.1f}   <-- correct")
    print(f"    old band [psi_min,1] : N_all = {n_all(g_old_e, psi_min_e):8.1f}")
    print(f"    ==> the old normalization band under-populated the host by {r_norm:.4f}x,")
    print(f"        i.e. sigma_sub low by {np.sqrt(r_norm):.4f}x (sigma_sub ~ sqrt(N_all))")

    # radial SHAPE check: sigma_sub(r)/sigma_sub(r_ref) is normalization-independent, so
    # it isolates the radial profile from the SHMF normalization.
    x, ss = d["x_r200"], d["sig_sub"]
    ref = np.argmin(np.abs(x - 0.5))
    print(f"\n  engine sigma_sub(r) shape, normalized at r/r200={x[ref]:.2f} "
          f"(profile check, normalization-free):")
    for i in range(0, len(x), 4):
        print(f"    r/r200={x[i]:5.2f}  sigma_sub/sigma_sub(ref) = {ss[i]/ss[ref]:7.4f}")
    print("\n  Compare against the 'sig_sub' column printed by "
          "plot_fig_subhalo_sigma_decomposition.py.")
