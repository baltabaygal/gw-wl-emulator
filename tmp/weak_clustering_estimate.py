"""Magnitude estimate: clustering of the SUB-THRESHOLD (weak) halo layer.

Legacy + current bias_model=1 treat the weak background as unclustered Poisson
(sigmakappaW = Campbell) drawn independently of the environment. If the weak
counts ride the same field delta_1D, the leading new term is the modulation of
the weak MEAN kappa per shell:
    dkappa_W = sum_i B_i deltabar_i,   B_i = sum_M b(M,z_i) Dg(z_i) m_iM,
    m_iM = int_{sub-threshold annuli} nbar kappa   (kappa^1 Campbell moment),
linearized in b*delta (good for the estimate; exact form is lognormal).
Also reported: the same linearized modulation of the EXPLICIT mean (A_i =
sum_M barN kbar bDg), their cross term (same field => fully correlated), and
sigma_W for scale.

Run: python3 tmp/weak_clustering_estimate.py
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "convergence"))
sys.path.insert(0, str(REPO / "playground" / "bias_field"))
from bias_field_prototype import Cosmo, PI, CLIGHT
from validate_field_covariance import cpp_field, shells

RL = lambda M, C: (3.0 * M / (4.0 * PI * C.rhoM0)) ** (1.0 / 3.0)


def weak_mean_bias_per_shell(C, zs, kappathr, eps_floor=0.001):
    """B[jz] = sum_M bDg * m_cell and m[jz] = sum_M m_cell for sub-threshold
    annuli (r from rmax outward, log-annulus measure, same loop shape as the
    port's sigmakappaW; kappa^1 instead of kappa^2)."""
    rmax, kappa0 = C.rmax_grid(zs, kappathr)
    zl = C.zlist
    dz = np.concatenate([[0.0], np.diff(zl)])
    mask = np.broadcast_to(zl[:, None] < zs, (C.Nz, C.NM)).copy()
    mask[0, :] = False
    mask[:, 0] = False
    r = np.where(rmax > 0.0, rmax, 1.0e-6)
    pref = (CLIGHT * 2.0 * PI * (1 + zl[:, None]) ** 2
            / C.Hz(zl)[:, None] * C.HMF0 * 0.01 * C.dlogM * dz[:, None])
    m_cell = np.zeros((C.Nz, C.NM))
    alive = mask.copy()
    Edlnr = np.exp(0.01)
    while alive.any():
        k = 2.0 * kappa0 * C.Fg0(r / C.rs)
        add = np.where(alive, pref * r ** 2 * k, 0.0)
        m_cell += add
        r = np.where(alive, r * Edlnr, r)
        alive &= k > eps_floor * kappathr
    bDg = C.biaslist * C.Dg(zl)[:, None]
    return np.sum(bDg * m_cell, axis=1), np.sum(m_cell, axis=1)


def explicit_mean_bias_per_shell(C, zs, kappathr):
    T = C.cell_tables(zs, kappathr)
    w = T["barN"] * T["kbar"] * T["bDg"]
    m = T["barN"] * T["kbar"]
    A = np.bincount(T["jz"], weights=w, minlength=C.Nz)
    Ame = np.bincount(T["jz"], weights=m, minlength=C.Nz)
    return A, Ame


def main():
    for zs in (1.0, 5.0):
        C = Cosmo(Nz=100)
        kt = C.find_kappathr(zs, 100)
        sigW = C.sigmakappaW(zs, kt)
        Bfull, mW = weak_mean_bias_per_shell(C, zs, kt)
        Afull, mE = explicit_mean_bias_per_shell(C, zs, kt)
        print(f"\n== zs={zs:g}  kappa_thr={kt:.3e}  sigma_W={sigW:.4f}  "
              f"<kappa_W>={mW.sum():.4f}  <kappa_expl>={mE.sum():.4f}")
        for M in (1e11, 1e14, 1e15):
            f = cpp_field(C, zs, RL(M, C))
            n = f["n"]                       # shells are jz=1..n
            Cov = f["Cov"]
            B = Bfull[1:n + 1]
            A = Afull[1:n + 1]
            sW = np.sqrt(B @ Cov @ B)
            sE = np.sqrt(A @ Cov @ A)
            sT = np.sqrt((A + B) @ Cov @ (A + B))
            print(f"  Rperp=R_L(1e{int(np.log10(M)):d})={RL(M, C)/1e3:7.2f} Mpc: "
                  f"weak-clust std={sW:.4f} ({sW/sigW:5.1%} of sigma_W)  "
                  f"expl-clust std={sE:.4f}  combined={sT:.4f}  "
                  f"[quad extra over expl-only: "
                  f"{np.sqrt(max(sT**2 - sE**2, 0.0)):.4f}]")


if __name__ == "__main__":
    main()
