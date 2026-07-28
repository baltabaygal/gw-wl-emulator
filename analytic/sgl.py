#!/usr/bin/env python
"""Semi-analytic sGL magnification PDF -- halos-only, spherical, unbiased Poisson.

Chain (no Monte Carlo, no fitting, no simulation input anywhere):

    dn/dlnM  ->  R(xi)  ->  Lambda(k)  ->  P(xi)  ->  dP/dmu

Physics is the validated chain of ``scripts/comparisons/analytic_pdf.py`` /
``analytic_chain_spec.md`` (every spec checkpoint is re-verified by
``checkpoints.py`` in this folder). What is NEW here relative to that script is
the last two links -- the Levy-Khintchine exponent ``Lambda_of_k`` and the
Fourier inversion ``P_of_xi`` -- which the spec describes (its Sec. 7) but the
reference script never implemented.

Model scope: spherical NFW lenses with the exact Wright-Brainerd projection,
Sheth-Tormen mass function, Poisson lens counts, NO linear bias, NO filaments,
NO ellipticity, NO subhalos.  That is deliberately the scope of the C++ engine
run with ``filaments=False, bias=False, ell=False, subhalo=False``.

Two conventions are selectable so the chain can be run either standalone or
matched to the C++ engine:

  ``window="tophat"``  real-space top-hat sigma(R) normalised to sigma8
                       (the spec / Vaskonen paper Sec. 2 convention)
  ``window="smoothk"`` the engine's smooth-k filter Ws(x)=1/(1+(0.43x)^6),
                       with sigma8 imposed *through that same filter* at
                       R = 8/h Mpc -- i.e. what ``cpp/cosmology.cpp::sigmaC``
                       actually does.  Reproduces the engine's sigma(M).

  ``transfer="nowiggle"``  Eisenstein-Hu 1998 no-wiggle (spec convention)
  ``transfer="eh98"``      full EH98 with the acoustic oscillations, which is
                           what ``cpp/cosmology.cpp::TM`` uses.

Units: Msun, Mpc (NOT Mpc/h), km/s/Mpc.
"""
from __future__ import annotations

import numpy as np
from numpy import log, exp, sqrt, sin, cos, pi
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from scipy.special import gamma as Gamma

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

# ============================== 0. constants ================================
# Planck-2018 benchmark == the C++ engine defaults.
DEFAULTS = dict(h=0.674, Om=0.315, Ob=0.0493, s8=0.811, ns=0.965,
                TCMB=2.7255, dc=1.686, zeq=3402.0)

CKMS = 299792.458            # km/s
RHOC0_UNIT = 2.77536627e11   # Msun/Mpc^3 / h^2
GNEWT = 4.30091e-9           # Mpc (km/s)^2 / Msun


class Cosmology:
    """Background + linear-theory sector.  Everything downstream reads this."""

    def __init__(self, h=None, Om=None, Ob=None, s8=None, ns=None,
                 TCMB=None, dc=None, zeq=None,
                 window="tophat", transfer="nowiggle"):
        d = DEFAULTS
        self.h = d["h"] if h is None else h
        self.Om = d["Om"] if Om is None else Om
        self.Ob = d["Ob"] if Ob is None else Ob
        self.s8 = d["s8"] if s8 is None else s8
        self.ns = d["ns"] if ns is None else ns
        self.TCMB = d["TCMB"] if TCMB is None else TCMB
        self.dc = d["dc"] if dc is None else dc
        self.zeq = d["zeq"] if zeq is None else zeq
        self.OL = 1.0 - self.Om
        self.H0 = 100.0 * self.h
        self.rhoc0 = RHOC0_UNIT * self.h**2
        self.rhom = self.Om * self.rhoc0        # comoving matter density
        if window not in ("tophat", "smoothk"):
            raise ValueError("window must be 'tophat' or 'smoothk'")
        if transfer not in ("nowiggle", "eh98"):
            raise ValueError("transfer must be 'nowiggle' or 'eh98'")
        self.window = window
        self.transfer = transfer
        self._build_growth()
        self._build_power()
        self._build_sigma()

    # ---------------------------- background --------------------------------
    def E(self, z):
        return sqrt(self.Om * (1 + z)**3 + self.OL)

    def Hz(self, z):
        return self.H0 * self.E(z)

    def rhocrit(self, z):                       # physical
        return self.rhoc0 * self.E(z)**2

    def chi(self, z):                           # comoving distance [Mpc]
        return quad(lambda zz: CKMS / self.H0 / self.E(zz), 0, z)[0]

    def DA(self, z):
        return self.chi(z) / (1 + z)

    def DA_ls(self, zl, zs):                    # flat universe
        return (self.chi(zs) - self.chi(zl)) / (1 + zs)

    def Sigma_cr(self, zl, zs):                 # Msun/Mpc^2, physical D_A's
        return (CKMS**2 * self.DA(zs)
                / (4 * pi * GNEWT * self.DA(zl) * self.DA_ls(zl, zs)))

    # ------------------------------ growth ----------------------------------
    def _build_growth(self):
        def Ea(a):
            return sqrt(self.Om * a**-3 + self.OL)

        def Dun(a):
            return Ea(a) * quad(lambda ap: 1.0 / (ap * Ea(ap))**3, 0, a)[0]

        self._Dun = Dun
        self._D0 = Dun(1.0)

    def D(self, z):
        a = 1.0 / (1 + z)
        return self._Dun(a) / self._D0

    # ------------------------ transfer function -----------------------------
    def _T_nowiggle(self, k):
        """Eisenstein & Hu 1998 no-wiggle (spec Sec. 1)."""
        wm, wb = self.Om * self.h**2, self.Ob * self.h**2
        th = self.TCMB / 2.7
        s = 44.5 * log(9.83 / wm) / sqrt(1 + 10 * wb**0.75)
        aG = (1 - 0.328 * log(431 * wm) * (wb / wm)
              + 0.38 * log(22.3 * wm) * (wb / wm)**2)
        Geff = self.Om * self.h * (aG + (1 - aG) / (1 + (0.43 * k * s)**4))
        q = k * th * th / (Geff * self.h)
        L = log(2 * np.e + 1.8 * q)
        C = 14.2 + 731.0 / (1 + 62.5 * q)
        return L / (L + C * q * q)

    def _T_eh98(self, k):
        """Full Eisenstein & Hu 1998 CDM+baryon transfer function.

        Transcribed to match ``cpp/cosmology.cpp::TM`` term for term (that
        routine is itself EH98 with k in Mpc^-1 and the sound horizon in Mpc).
        """
        h, Om, Ob = self.h, self.Om, self.Ob
        Oc = Om - Ob
        wm, wb = Om * h**2, Ob * h**2
        T0 = self.TCMB
        # z_eq is a free input in the engine (default 3402) and it sets
        # OmegaR = OmegaM/(1+zeq); with radiation included H(zeq)^2 =
        # 2 OmegaM H0^2 (1+zeq)^3, so k_eq = a_eq H(a_eq)/c reduces to
        zeq = self.zeq
        keq = (self.H0 / CKMS) * sqrt(2.0 * Om) * sqrt(1.0 + zeq)   # Mpc^-1
        ksilk = (0.0016 * wb**0.52 * wm**0.73
                 * (1 + (10.4 * wm)**-0.95))
        a1 = (46.9 * wm)**0.67 * (1 + (32.1 * wm)**-0.532)
        a2 = (12.0 * wm)**0.424 * (1 + (45.0 * wm)**-0.582)
        b1 = 0.944 / (1 + (458 * wm)**-0.708)
        b2 = (0.395 * wm)**-0.026
        ac = a1**(-Ob / Om) * a2**(-(Ob / Om)**3)
        bc = 1.0 / (1 + b1 * ((Oc / Om)**b2 - 1))
        s = 44.5 * log(9.83 / wm) / sqrt(1 + 10 * wb**0.75)     # Mpc
        b3 = 0.313 * wm**-0.419 * (1 + 0.607 * wm**0.674)
        b4 = 0.238 * wm**0.223
        zd = (1291 * wm**0.251 / (1 + 0.659 * wm**0.828) * (1 + b3 * wb**b4))
        Rd = 31.5 * wb * (T0 / 2.7)**-4 / (zd / 1000.0)

        def g2(y):
            return y * (-6 * sqrt(1 + y)
                        + (2.0 + 3.0 * y) * log((sqrt(1 + y) + 1)
                                                / (sqrt(1 + y) - 1)))

        ab = 2.07 * keq * s * (1 + Rd)**-0.75 * g2((1 + zeq) / (1 + zd))
        bb = (0.5 + Ob / Om
              + (3.0 - 2.0 * Ob / Om) * sqrt(1 + (17.2 * wm)**2))
        bnode = 8.41 * wm**0.435

        q = k / (13.41 * keq)
        fk = 1.0 / (1 + (k * s / 5.4)**4)

        def To1(kk, a_c, b_c):
            qq = kk / (13.41 * keq)
            C1 = 14.2 / a_c + 386.0 / (1 + 69.9 * qq**1.08)
            num = log(np.e + 1.8 * b_c * qq)
            return num / (num + C1 * qq * qq)

        TC = fk * To1(k, 1.0, bc) + (1 - fk) * To1(k, ac, bc)
        s3 = s / (1 + (bnode / (k * s))**3)**(1.0 / 3.0)
        x = k * s3
        jo = np.where(x < 1e-8, 1.0 - x * x / 6.0, sin(x) / np.where(x == 0, 1, x))
        TB = ((To1(k, 1.0, 1.0) / (1 + (k * s / 5.2)**2)
               + ab / (1 + (bb / (k * s))**3) * exp(-(k / ksilk)**1.4)) * jo)
        return Ob / Om * TB + Oc / Om * TC

    def T(self, k):
        return (self._T_nowiggle(k) if self.transfer == "nowiggle"
                else self._T_eh98(k))

    # --------------------------- sigma(M) -----------------------------------
    def _W(self, x):                             # real-space top hat
        return 3 * (sin(x) - x * cos(x)) / x**3

    def _Ws(self, x):                            # engine smooth-k filter
        return 1.0 / (1.0 + (0.43 * x)**6)

    def _build_power(self):
        self._kg = np.logspace(-5, 3, 4000)
        self._Pshape = self._kg**self.ns * self.T(self._kg)**2

    def _sigmaR_shape(self, R, window=None):
        w = self.window if window is None else window
        x = self._kg * R
        Wk = self._W(x) if w == "tophat" else self._Ws(x)
        return sqrt(_trapz(self._kg**2 * self._Pshape * Wk**2, self._kg)
                    / (2 * pi * pi))

    def _build_sigma(self):
        # normalise sigma8 THROUGH the selected window at R = 8/h Mpc, which is
        # exactly what the engine does with Ws (hence its top-hat sigma8 = 0.779)
        self._Anorm = self.s8 / self._sigmaR_shape(8.0 / self.h)
        Mg = np.logspace(4, 17, 500)
        lnsig = np.array([log(self.sigmaR(self.RofM(M))) for M in Mg])
        self._spl = CubicSpline(log(Mg), lnsig)

    def sigmaR(self, R):
        return self._Anorm * self._sigmaR_shape(R)

    def sigma_tophat8(self):
        """sigma8 measured with a real-space top hat, whatever the filter used
        for the normalisation.  Engine value: 0.7786."""
        return self._Anorm * self._sigmaR_shape(8.0 / self.h, window="tophat")

    def RofM(self, M):
        return (3 * M / (4 * pi * self.rhom))**(1.0 / 3.0)

    def sigmaM(self, M, z=0.0):
        return exp(self._spl(log(M))) * self.D(z)

    def dlnsig_dlnM(self, M):
        return self._spl(log(M), 1)

    # ----------------------- Sheth-Tormen mass function ---------------------
    P_ST, Q_ST = 0.3, 0.8
    A_ST = 1.0 / (1 + 2**-0.3 * Gamma(0.5 - 0.3) / sqrt(pi))

    def dndlnM(self, M, z):
        """Comoving number density per lnM [Mpc^-3].  nu = delta_c^2/sigma^2."""
        nu = (self.dc / self.sigmaM(M, z))**2
        qn = self.Q_ST * nu
        mult = (self.A_ST * (1 + qn**-self.P_ST) * sqrt(qn / (2 * pi))
                * exp(-qn / 2))
        return (self.rhom / M) * mult * (-2 * self.dlnsig_dlnM(M))

    # ------------------------------ NFW -------------------------------------
    @staticmethod
    def conc(M, z, h):
        """Dutton & Maccio 2014, 200c NFW (== the engine's ``cons14``)."""
        a = 0.520 + (0.905 - 0.520) * exp(-0.617 * z**1.21)
        b = -0.101 + 0.026 * z
        return 10.0**(a + b * np.log10(M * h / 1e12))

    def nfw_params(self, M, zl, zs):
        C = self.conc(M, zl, self.h)
        r200 = (3 * M / (4 * pi * 200 * self.rhocrit(zl)))**(1.0 / 3.0)
        rs = r200 / C
        fC = log(1 + C) - C / (1 + C)
        rhos = (200.0 / 3.0) * self.rhocrit(zl) * C**3 / fC
        ks = rhos * rs / self.Sigma_cr(zl, zs)
        return C, rs, ks, fC


# ===================== Wright & Brainerd projected NFW =======================
def _F(x):
    x = np.asarray(x, float)
    out = np.empty_like(x)
    lo, hi = x < 1 - 1e-6, x > 1 + 1e-6
    mid = ~(lo | hi)
    xl = x[lo]
    out[lo] = (1 / (xl * xl - 1)) * (1 - (2 / sqrt(1 - xl * xl))
                                     * np.arctanh(sqrt((1 - xl) / (1 + xl))))
    xh = x[hi]
    out[hi] = (1 / (xh * xh - 1)) * (1 - (2 / sqrt(xh * xh - 1))
                                     * np.arctan(sqrt((xh - 1) / (xh + 1))))
    xm = x[mid]
    out[mid] = 1.0 / 3.0 - 2.0 * (xm - 1) / 5.0
    return out


def _hfun(x):
    x = np.asarray(x, float)
    out = np.empty_like(x)
    lo, hi = x < 1 - 1e-6, x > 1 + 1e-6
    mid = ~(lo | hi)
    xl = x[lo]
    out[lo] = log(xl / 2) + np.arccosh(1 / xl) / sqrt(1 - xl * xl)
    xh = x[hi]
    out[hi] = log(xh / 2) + np.arccos(1 / xh) / sqrt(xh * xh - 1)
    xm = x[mid]
    out[mid] = log(xm / 2) + 1 - (xm - 1) / 3.0
    xs = x < 1e-3
    out[xs] = (x[xs]**2 / 2) * (log(2 / x[xs]) - 0.5)
    return out


def kappa_gamma(x, ks):
    k = 2 * ks * _F(x)
    kbar = 4 * ks * _hfun(x) / x**2
    return k, kbar - k


# ======================= jump-measure grid and binning =======================
# XI reaches 30: the near-caustic exponential floor of the cluster cross-section
# lives at xi ~ 1-30 and the spec's original cap of 4 truncated it.
XI = np.logspace(-7, np.log10(30.0), 560)
_lnXI = log(XI)
_XIE = np.empty(XI.size + 1)
_XIE[1:-1] = sqrt(XI[1:] * XI[:-1])
_XIE[0] = XI[0]**2 / _XIE[1]
_XIE[-1] = XI[-1]**2 / _XIE[-2]
_XIW = np.diff(_XIE)

_XG = np.logspace(-6, 3.5, 900)


def _refined_xgrid(ks):
    """Base log grid plus adaptive refinement around every zero of detA(x).

    Without this the thin annulus at the tangential critical curve of a
    cluster-scale lens is unresolved and dsigma/dxi is under-counted by up to
    ~4x for xi >~ 0.5 (spec Sec. 9 warning; the erratum found 2026-07-24).
    """
    k, g = kappa_gamma(_XG, ks)
    detA = (1 - k)**2 - g * g
    xs = [_XG]
    sgn = np.sign(detA)
    for i in np.nonzero(sgn[1:] * sgn[:-1] < 0)[0]:
        a, b = _XG[i], _XG[i + 1]
        for _ in range(70):                       # bisect detA = 0
            m = 0.5 * (a + b)
            km, gm = kappa_gamma(np.array([m]), ks)
            if np.sign((1 - km[0])**2 - gm[0]**2) == sgn[i]:
                a = m
            else:
                b = m
        xc = 0.5 * (a + b)
        eps = np.logspace(-11, -1.2, 22)
        xs.append(xc * (1.0 + eps))
        xs.append(xc * (1.0 - eps))
    x = np.unique(np.concatenate(xs))
    return x[x > 0]


def dsigma_bins(cos, M, zl, zs):
    """Deposit-binned single-lens cross-section dsigma/dxi on XI [Mpc^2].

    Every x-interval deposits its exact annulus area pi rs^2 (x2^2 - x1^2)
    uniformly over the xi range it spans -- branch-safe and area-exact, unlike
    the |dxi/dlnx| Jacobian recipe, which merges branches at the caustic.
    """
    C, rs, ks, fC = cos.nfw_params(M, zl, zs)
    x = _refined_xgrid(ks)
    k, g = kappa_gamma(x, ks)
    detA = (1 - k)**2 - g * g
    good = detA > 1e-300
    xi = np.where(good, -log(np.where(good, detA, 1.0)), np.nan)
    xa, xb = x[:-1], x[1:]
    fa, fb = xi[:-1], xi[1:]
    ok = np.isfinite(fa) & np.isfinite(fb)
    xa, xb, fa, fb = xa[ok], xb[ok], fa[ok], fb[ok]
    lo, hi = np.minimum(fa, fb), np.maximum(fa, fb)
    dsig = pi * rs * rs * (xb * xb - xa * xa)
    keep = (hi > _XIE[0]) & (lo < _XIE[-1]) & (dsig > 0)
    if not np.any(keep):
        return np.zeros_like(XI)
    lo, hi, dsig = lo[keep], hi[keep], dsig[keep]
    den = np.maximum(hi - lo, 1e-300)
    frac = np.clip((_XIE[:, None] - lo[None, :]) / den[None, :], 0.0, 1.0)
    return np.diff(frac @ dsig) / _XIW


def campbell_moments(cos, zs, kappa_min, Mmin=1e7, Mmax=1e16, Nz=40, NM=48):
    """Campbell (compound-Poisson) moments of the ADDITIVE fields kappa and
    gamma, integrated over the same lens population as ``R_of_xi``.

    For a compensated Poisson sum, Var(kappa) = int kappa^2 dR exactly and
    <|sum gamma_i|^2> = int gamma^2 dR (cross terms vanish for random lens
    position angles).  These involve NO scalar reduction and no Fourier
    inversion -- they test abundance x profile x geometry x counts alone, so
    they separate an error in R from an error in the kappa -> mu composition.

    Each halo is integrated out to the radius where its own convergence falls
    to ``kappa_min``, matching the engine's coverage: explicit lenses above
    kappa_thr plus the Gaussian background down to eps_floor * kappa_thr.
    """
    zgrid = np.linspace(1e-3, zs - 1e-3, Nz)
    dz = zgrid[1] - zgrid[0]
    Mgrid = np.logspace(np.log10(Mmin), np.log10(Mmax), NM)
    dlnM = log(Mgrid[1]) - log(Mgrid[0])
    acc = dict(kappa=0.0, kappa2=0.0, gamma2=0.0, xi=0.0, xi2=0.0, N=0.0)
    for z in zgrid:
        wz = (1 + z)**2 * CKMS / cos.Hz(z) * dz
        for M in Mgrid:
            C, rs, ks, fC = cos.nfw_params(M, z, zs)
            k, g = kappa_gamma(_XG, ks)
            sel = k > kappa_min                    # engine's radial coverage
            if not np.any(sel):
                continue
            x, kk, gg = _XG[sel], k[sel], g[sel]
            detA = (1 - kk)**2 - gg * gg
            xi = np.where(detA > 0, -log(np.where(detA > 0, detA, 1.0)), 0.0)
            pref = wz * cos.dndlnM(M, z) * dlnM * 2 * pi * rs * rs
            acc["N"] += pref * _trapz(x, x)
            acc["kappa"] += pref * _trapz(x * kk, x)
            acc["kappa2"] += pref * _trapz(x * kk * kk, x)
            acc["gamma2"] += pref * _trapz(x * gg * gg, x)
            acc["xi"] += pref * _trapz(x * xi, x)
            acc["xi2"] += pref * _trapz(x * xi * xi, x)
    return acc


def R_of_xi(cos, zs, Mmin=1e7, Mmax=1e16, Nz=40, NM=48):
    """Line-of-sight-integrated jump measure R(xi; zs) [per lnmu, per l.o.s.].

    R = int dz (1+z)^2 c/H(z) int dlnM dn/dlnM dsigma/dxi   (spec Sec. 5)
    """
    zgrid = np.linspace(1e-3, zs - 1e-3, Nz)
    dz = zgrid[1] - zgrid[0]
    Mgrid = np.logspace(np.log10(Mmin), np.log10(Mmax), NM)
    dlnM = log(Mgrid[1]) - log(Mgrid[0])
    R = np.zeros_like(XI)
    for z in zgrid:
        wz = (1 + z)**2 * CKMS / cos.Hz(z) * dz
        for M in Mgrid:
            R += wz * cos.dndlnM(M, z) * dlnM * dsigma_bins(cos, M, z, zs)
    return R


# ==================== Levy-Khintchine exponent and inversion =================
def tilt_source(R):
    """Source-plane Esscher tilt R_s(xi) = e^{-xi} R(xi) (spec Sec. 6)."""
    return exp(-XI) * R


def _fine_grid(R, nfine=6000):
    """Log-refine R onto a denser xi grid for the oscillatory k-integral.

    R is smooth in log-log, so linear interpolation of ln R vs ln xi is
    lossless here; the refinement only buys resolution of e^{i k xi}.
    """
    xif = np.logspace(log(XI[0]) / log(10), log(XI[-1]) / log(10), nfine)
    pos = R > 0
    lnR = np.interp(log(xif), log(XI[pos]), log(R[pos]),
                    left=-np.inf, right=-np.inf)
    return xif, np.where(np.isfinite(lnR), exp(lnR), 0.0)


def Lambda_of_k(k, R, nfine=6000, chunk=4000):
    """Compensated Levy exponent  Lambda(k) = int dxi R(xi)(e^{ikxi}-1-ikxi).

    Pass R already tilted (``tilt_source``) for source-plane statistics.
    The -ikxi subtraction is the ensemble-mean subtraction: <xi> = 0.
    """
    k = np.atleast_1d(np.asarray(k, float))
    xif, Rf = _fine_grid(R, nfine)
    out = np.empty(k.size, complex)
    for i0 in range(0, k.size, chunk):
        kk = k[i0:i0 + chunk][:, None]
        ph = kk * xif[None, :]
        integ = Rf[None, :] * (np.exp(1j * ph) - 1.0 - 1j * ph)
        out[i0:i0 + chunk] = _trapz(integ, xif, axis=-1)
    return out


def _kmax_auto(R, nfine=6000):
    """Spec Sec. 7 recipe: estimate the linear decay of Re Lambda and cut where
    e^{Re Lambda} ~ e^{-35}."""
    kp = np.array([50.0, 200.0])
    L = Lambda_of_k(kp, R, nfine)
    slope = (L[0].real - L[1].real) / (kp[1] - kp[0])   # >0
    if not np.isfinite(slope) or slope <= 0:
        return 2.0e4
    return float(np.clip(35.0 / slope, 2.0e3, 4.0e4))


def P_of_xi(R, xi_out=None, kmax=None, nk=60000, nfine=6000):
    """Invert to the pdf of xi = ln mu:  P(xi) = (1/pi) Re int_0^inf dk
    e^{-i k xi + Lambda(k)}.  Pass a tilted R for the source-plane PDF."""
    if xi_out is None:
        xi_out = np.linspace(-0.9, 1.6, 420)
    xi_out = np.asarray(xi_out, float)
    if kmax is None:
        kmax = _kmax_auto(R, nfine)
    kg = np.linspace(0.0, kmax, nk)
    Lam = Lambda_of_k(kg, R, nfine)
    Phi = np.exp(Lam)
    P = np.empty(xi_out.size)
    for i, xv in enumerate(xi_out):
        P[i] = _trapz((Phi * np.exp(-1j * kg * xv)).real, kg) / pi
    return xi_out, P


def dP_dmu(R, mu=None, **kw):
    """Source-plane magnification pdf.  dP/dmu = P(ln mu)/mu."""
    if mu is None:
        mu = np.linspace(0.5, 4.0, 400)
    xi = log(np.asarray(mu, float))
    _, P = P_of_xi(R, xi_out=xi, **kw)
    return mu, P / mu


# ============================== moments =====================================
def Ltilde(R, t):
    """Real-argument exponent  int R (e^{-t xi} - 1 + t xi) dxi (exact
    moments; pass a tilted R for source-plane)."""
    return _trapz(R * (exp(-t * XI) - 1 + t * XI), XI)


def sigma_DL_over_DL(R):
    """Exact fractional distance scatter, D_L ~ mu^{-1/2}:
    (sigma/D)^2 = exp[Ltilde(1) - 2 Ltilde(1/2)] - 1."""
    return sqrt(exp(Ltilde(R, 1.0) - 2 * Ltilde(R, 0.5)) - 1.0)


def moments(R):
    """Cumulants of xi for the compensated process: var, skew, kurtosis."""
    m2 = _trapz(R * XI**2, XI)
    m3 = _trapz(R * XI**3, XI)
    m4 = _trapz(R * XI**4, XI)
    return dict(var=m2, skew=m3 / m2**1.5, kurt=m4 / m2**2)


def slope_R(R):
    return -np.gradient(log(R), _lnXI)
