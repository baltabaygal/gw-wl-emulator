"""Clustering-variance-weighted signal scale R_L, per bias_window (2026-07-20).

Reproduces the weighting behind the 2026-07-16 bias_Rperp = R_L(1e14) decision
and asks whether generalizing the window moves it.

Three weights over the (jz, jM) cells of the production grid:

  pop     w = (barN * kbar * b * Dg)^2
          — the 2026-07-16 weight as recorded (docs/claude_md_archive.md:
          "w = (barN kappabar btilde)^2, cell_tables port"). Reproduction
          target: q50/q90/q99 = 8.3/14/19 Mpc at z_s = 0.2, 6.1/11/17 at
          z_s = 1, 3.3/7.6/12 at z_s = 5, 2.8/7.1/12 at z_s = 10.

  field   w = pop * sigma_i^2(window)
          — the correlated-field-consistent refinement: within a shell every
          cell rides the SAME field, so a cell's clustering variance factorizes
          into the population part and the shell variance sigma_i^2 from
          BiasField1D. This is where, and the ONLY way, the window enters.

  mb_map  w = (barN * kbar * sigma_old)^2, sigma_old = Dg b sigma(M_b)
          — the weight used by mb_validity_map.py for the M_b/M validity
          question. sigma(M_b) is the LEGACY iid layer's per-cell amplitude and
          carries an M-dependence the field model does not have; shown only to
          make clear it answers a different question (it is NOT the scale calc).

Reported: weighted quantiles of R_L(M) = (3M / 4 pi rhoM0)^(1/3), the scale the
window radius is meant to represent.

Run: python3 scripts/convergence/bias_window_weighted_scale.py
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "convergence"))
sys.path.insert(0, str(REPO / "playground" / "bias_field"))

from bias_field_prototype import Cosmo, PI                    # noqa: E402
from validate_field_covariance import cpp_field, WINDOW_NAME  # noqa: E402

ZS_LIST = [0.2, 0.5, 1.0, 5.0, 10.0]
QS = [0.50, 0.90, 0.99]
NHALOS = 100
KPC2MPC = 1e-3
RPERP = 8441.0            # the window radius the shell variances are built at


def wquant(x, w, qs):
    o = np.argsort(x)
    x, w = x[o], w[o]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return np.full(len(qs), np.nan)
    return np.interp(qs, cw / cw[-1], x)


def main():
    C = Cosmo(Nz=100)
    rho = C.rhoM0
    print("Clustering-variance-weighted R_L quantiles [Mpc comoving]\n"
          f"(fixed-<N>={NHALOS} threshold, Nz=NM=100, subhalo off; shell "
          f"variances built at R_perp = {RPERP:.0f} kpc)\n")
    print(f"{'z_s':>5s} {'weight':>16s} " + " ".join(f"{'q'+str(int(q*100)):>7s}" for q in QS))
    for zs in ZS_LIST:
        kthr = C.find_kappathr(zs, NHALOS)
        T = C.cell_tables(zs, kthr)
        M = C.Mlist[T["jM"]]
        RL = (3.0 * M / (4.0 * PI * rho)) ** (1.0 / 3.0) * KPC2MPC
        base = (T["barN"] * T["kbar"] * T["bDg"]) ** 2
        print(f"{zs:5g} {'pop (2026-07-16)':>16s} "
              + " ".join(f"{v:7.2f}" for v in wquant(RL, base, QS)))
        # field weight: per-shell sigma_i^2 from the production covariance
        for window in (0, 1, 2):
            f = cpp_field(C, zs, RPERP, window)
            # cell shell index: cells at jz sit in shell jz-1 (BiasField1D::shell)
            sh = T["jz"] - 1
            ok = (sh >= 0) & (sh < f["n"])
            s2 = np.zeros(len(sh))
            s2[ok] = f["sig2"][sh[ok]]
            print(f"{'':5s} {'field/' + WINDOW_NAME[window]:>16s} "
                  + " ".join(f"{v:7.2f}" for v in wquant(RL[ok], (base * s2)[ok], QS)))
        w_mb = (T["barN"] * T["kbar"] * T["sigma_old"]) ** 2
        print(f"{'':5s} {'mb_map':>16s} "
              + " ".join(f"{v:7.2f}" for v in wquant(RL, w_mb, QS)))
        print()


if __name__ == "__main__":
    main()
