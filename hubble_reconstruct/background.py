"""Flat FRW background for the 1+6d space: H(z), D_C, D_L, dV_c/dz.

Numpy-only, vectorized, spline-free (linear interp on a dense grid; the
resolution gate in tests/run_gates.py checks convergence). Units: Mpc, km/s.

Radiation follows the engine convention (CLAUDE.md): Omega_R = Om/(1+zeq), so
freeing zeq = freeing radiation. Flat: OL = 1 - Om - Or. Only (h, Om, zeq)
enter the background; (sigma8, Ob, ns) act through P(mu) models only.
"""
from __future__ import annotations

import numpy as np

C_KMS = 299792.458  # km/s


class Background:
    def __init__(self, theta: dict, zmax: float = 12.0, ngrid: int = 4096):
        h = float(theta["h"])
        Om = float(theta["Om"])
        zeq = float(theta.get("zeq", 3402.0))
        self.h, self.Om, self.zeq = h, Om, zeq
        self.H0 = 100.0 * h                      # km/s/Mpc
        self.Or = Om / (1.0 + zeq)
        self.OL = 1.0 - Om - self.Or
        self.dH = C_KMS / self.H0                # Hubble distance, Mpc

        # QUADRATIC grid spacing: z = zmax * u^2. A uniform grid puts its first
        # node at zmax/ngrid ~ 3e-3, so D_L below that is a linear extrapolation
        # of the first trapezoid and carries a ~5e-4 relative error -- which the
        # background gate caught. Quadratic spacing makes the first cell ~1e-6
        # while keeping high-z resolution adequate for a smooth integrand.
        u = np.linspace(0.0, 1.0, ngrid)
        z = zmax * u**2
        E = self._E(z)
        integ = 1.0 / E
        chi = np.concatenate(
            [[0.0], np.cumsum(0.5 * (integ[1:] + integ[:-1]) * np.diff(z))]
        ) * self.dH                              # comoving distance, Mpc
        self._zgrid, self._chigrid, self._Egrid = z, chi, E

    def _E(self, z):
        zp1 = 1.0 + np.asarray(z, dtype=float)
        return np.sqrt(self.Or * zp1**4 + self.Om * zp1**3 + self.OL)

    def H(self, z):
        """H(z) in km/s/Mpc."""
        return self.H0 * self._E(z)

    def DC(self, z):
        """Comoving distance, Mpc."""
        return np.interp(z, self._zgrid, self._chigrid)

    def DL(self, z):
        """Luminosity distance, Mpc (flat)."""
        z = np.asarray(z, dtype=float)
        return (1.0 + z) * self.DC(z)

    def dVc_dz(self, z):
        """Comoving volume element per dz, up to a constant (flat):
        dV_c/dz \\propto D_C(z)^2 / E(z)."""
        return self.DC(z) ** 2 / self._E(z)
