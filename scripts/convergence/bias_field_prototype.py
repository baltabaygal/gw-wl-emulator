"""Prototype: grid-decoupled bias (environment) layer — LOS-correlated field design.

Context (CLAUDE.md #13, docs/nz_bias_convergence_note.md): the BIAS layer draws an
independent lognormal count modulation per (jz,jM) grid cell with amplitude
sigma_b = Dg(zl)*sigma(M_b)*b(zl,sigma_M), where M_b is the grid cell's tube-segment
mass (lensing.cpp:123). The smoothing volume IS the numerical grid cell, so Nz, NM
and kappa_thr silently reparametrize the clustering physics: f(kappa>1) grows
unsaturated with Nz (no continuum limit) and sigma_b blows up for kappa_thr >~ 3e-3.

Proposed design prototyped here (NO free R_bg scale): the environment is a 1D
Gaussian field dbar(chi) = transverse-average of the linear density field along the
LOS. Its statistics are fully derived from the code's own linear P(k):
  - per-cell amplitude  = sigma_cyl(R_i, L_i): exact z=0 variance of the linear field
    averaged over the cell's OWN counting cylinder (comoving radius R=(1+zl)*rmax,
    length L=Delta_chi) — the window is the counting geometry, not a free knob;
  - cross-cell correlation = pencil-beam projected P_1D(k_par) segment covariance
    (all jM at one jz share the field; different jz correlated via P_1D);
  - count modulation lambda = exp(g - Var(g)/2), g = b(M,z)*Dg(z)*dbar_seg,
    same mean-one lognormal mapping as the current code.
As Nz -> inf, thin segments saturate at the disc-average variance and become
perfectly correlated with their neighbours => continuum limit exists. As
kappa_thr -> large, R -> 0 keeps sigma_cyl finite (pencil limit converges for CDM)
=> no blow-up.

Stages (run with the test env python; port-check and mc-reference need build/):
  port-check : validate the Python port of the C++ tables against gwlensing helpers
               (get_kappa_threshold / get_expected_halo_count / get_sigma_background)
  analytic   : (b) per-cell sigma_new vs sigma_old at the default grid;
               (c) sigma_b vs kappa_thr sweep (blow-up check); both + vs-Nz curves
  mc         : bias+halo-layer MC (kappa only, eps=0, no filaments), old arm
               (iid per-cell lognormal, replicating lensing.cpp) vs new arm
               (shared correlated field); f(kappa>1), clipped Var vs Nz
  report     : figures + report.md (overlays the measured full-model growth from
               data/results/vark_nz/vark_vs_nz.npz)

  /Users/baltabay/miniforge3/envs/test/bin/python \
      scripts/convergence/bias_field_prototype.py all

Outputs: data/results/bias_field_prototype/{port_validation.md, analytic.npz,
mc/*.npz, report.md}, plots/bias_field_prototype.png.

Known deliberate deviations from the C++ (documented in the report):
  - MC uses pure Poisson counts in every cell (the C++ approximates rate<0.2 cells
    by a Bernoulli, which slightly TRIMS multi-halo coincidences — so the C++ tail
    growth is, if anything, understated relative to this old-arm replica);
  - kappa only (no shear/ellipticity), no filaments (they share the same lambda in
    the C++; omitting them affects both arms identically);
  - no batch-mean anchor (stats below are clip-centered, matching the vark study).
Replicated warts kept ON PURPOSE in the old arm: M_b uses PHYSICAL rmax^2 (not
comoving) and sigma(M_b) is looked up with linear-in-M interpolation clamped at
Mmin (the clamp is what caps sigma_b ~ sigma(1e7) for very thin cells).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "results" / "bias_field_prototype"
MCDIR = OUT / "mc"
PLOTS = REPO / "plots"

# ---------------------------------------------------------------- constants
CLIGHT = 306.535                      # kpc/Gyr (basics.h)
PI = np.pi
GFID = 0.7869370293916                # cosmology.h gfid
DELTAC0 = 3.0 / 5.0 * (3.0 * PI / 2.0) ** (2.0 / 3.0)

H, OM, S8 = 0.674, 0.315, 0.811       # fiducial (matches the vark_nz study)
ZS_LIST = [1.0, 10.0]
NZ_GRID_MC = [25, 100, 200, 400, 800]
NZ_GRID_AN = [25, 50, 100, 200, 400]
KTHR_SWEEP = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
NREAL = 20_000
SEED0 = 900_000_000                   # disjoint namespace (vark_nz uses 8.0-8.2e8)
CHUNK = 200


# ---------------------------------------------------------------- cosmology port
class Cosmo:
    """Python port of the pieces of cpp/cosmology.{h,cpp} the bias layer touches.

    Replicates the C++ numerics (same grids, same trapezoids, same window, same
    bisection tolerances) so tables can be validated against gwlensing helpers.
    """

    def __init__(self, OmegaM=OM, sigma8=S8, h=H, ns=0.965, OmegaB=0.0493,
                 zeq=3402.0, Mmin=1e7, Mmax=1e17, NM=100,
                 zmin=0.01, zmax=10.01, Nz=100, Nk=1000):
        self.OmegaM, self.sigma8, self.h, self.ns = OmegaM, sigma8, h, ns
        self.OmegaB, self.zeq = OmegaB, zeq
        self.Mmin, self.Mmax, self.NM = Mmin, Mmax, NM
        self.zmin, self.zmax, self.Nz = zmin, zmax, Nz
        self.Nk = Nk

        self.OmegaR = OmegaM / (1 + zeq)
        self.OmegaL = 1.0 - OmegaM - self.OmegaR
        self.OmegaC = OmegaM - OmegaB
        self.H0 = 0.000102247 * h
        self.rhoc = 277.394 * h ** 2
        self.rhoM0 = OmegaM * self.rhoc
        self.M8 = 4.0 * PI / 3.0 * (8000.0 / h) ** 3 * self.rhoM0

        self.zlist = np.exp(np.linspace(np.log(zmin), np.log(zmax), Nz))
        self.Mlist = np.exp(np.linspace(np.log(Mmin), np.log(Mmax), NM))
        self.dlogM = (np.log(Mmax) - np.log(Mmin)) / (NM - 1)

        self.deltaH8 = sigma8 / self._sigma_smoothk(self.M8, 1.0)[0]
        self._build_sigmalist()
        self._build_dclist()
        self._build_hmf_bias_nfw()

    # ---- background
    def Az(self, z):
        return (self.OmegaM * (1 + z) ** 3 + self.OmegaR * (1 + z) ** 4
                + self.OmegaL)

    def Hz(self, z):
        return self.H0 * np.sqrt(self.Az(z))

    def Dg(self, z):
        omz = self.OmegaM * (1 + z) ** 3 / self.Az(z)
        olz = self.OmegaL / self.Az(z)
        return (2.5 * omz / (omz ** (4.0 / 7.0) - olz
                             + (1 + omz / 2.0) * (1 + olz / 70.0))
                / (1 + z) / GFID)

    def deltac(self, z):
        return DELTAC0 / self.Dg(z)

    # ---- transfer function (EH98, astro-ph/9709112), vectorized in k
    def TM(self, k):
        h, OmegaM, OmegaB, OmegaC, zeq = (self.h, self.OmegaM, self.OmegaB,
                                          self.OmegaC, self.zeq)
        omh2 = OmegaM * h * h
        obh2 = OmegaB * h * h
        keq = 0.00326227 * self.Hz(zeq) / (1.0 + zeq)
        ksilk = (0.0016 * obh2 ** 0.52 * omh2 ** 0.73
                 * (1 + (10.4 * omh2) ** -0.95))
        a1 = (46.9 * omh2) ** 0.67 * (1 + (32.1 * omh2) ** -0.532)
        a2 = (12.0 * omh2) ** 0.424 * (1 + (45.0 * omh2) ** -0.582)
        b1 = 0.944 / (1 + (458 * omh2) ** -0.708)
        b2 = (0.395 * omh2) ** -0.026
        acnum = a1 ** (-OmegaB / OmegaM) * a2 ** (-(OmegaB / OmegaM) ** 3)
        bcnum = 1.0 / (1 + b1 * ((OmegaC / OmegaM) ** b2 - 1))
        s2 = (44.5 * 1000 * np.log(9.83 / omh2)
              / np.sqrt(1 + 10 * obh2 ** 0.75))
        b3 = 0.313 * omh2 ** -0.419 * (1 + 0.607 * omh2 ** 0.674)
        b4 = 0.238 * omh2 ** 0.223
        zd = (1291 * omh2 ** 0.251 / (1 + 0.659 * omh2 ** 0.828)
              * (1 + b3 * obh2 ** b4))
        T0 = 2.7255
        Rd = 31.5 * obh2 * (T0 / 2.7) ** -4.0 / (zd / 1000)

        def g2(y):
            return y * (-6 * np.sqrt(1 + y) + (2.0 + 3.0 * y)
                        * np.log((np.sqrt(1 + y) + 1) / (np.sqrt(1 + y) - 1)))

        ab = 2.07 * keq * s2 * (1 + Rd) ** -0.75 * g2((1 + zeq) / (1 + zd))
        bb = (0.5 + OmegaB / OmegaM + (3.0 - 2.0 * OmegaB / OmegaM)
              * np.sqrt(1 + (17.2 * omh2) ** 2))
        bnode = 8.41 * omh2 ** 0.435

        k = np.asarray(k, float)
        q = k / (13.41 * keq)
        f = 1.0 / (1 + (k * s2 / 5.4) ** 4)

        def To1(ac, bc):
            L = np.log(np.e + 1.8 * bc * q)
            C1 = 14.2 / ac + 386.0 / (1 + 69.9 * q ** 1.08)
            return L / (L + C1 * q ** 2)

        TC = f * To1(1.0, bcnum) + (1 - f) * To1(acnum, bcnum)
        s3 = s2 / (1 + (bnode / (k * s2)) ** 3) ** (1.0 / 3.0)
        x = k * s3
        j0 = np.where(x != 0.0, np.sin(x) / np.where(x == 0.0, 1.0, x), 1.0)
        TB = ((To1(1.0, 1.0) / (1 + (k * s2 / 5.2) ** 2)
               + ab / (1 + (bb / (k * s2)) ** 3) * np.exp(-(k / ksilk) ** 1.4))
              * j0)
        return OmegaB / OmegaM * TB + OmegaC / OmegaM * TC

    def Delta2(self, k, deltaH=None):
        """Dimensionless z=0 power Delta^2(k) in the code's convention."""
        if deltaH is None:
            deltaH = self.deltaH8
        k = np.asarray(k, float)
        return (CLIGHT * k / self.H0) ** (3.0 + self.ns) * (deltaH
                                                            * self.TM(k)) ** 2

    def Pk(self, k):
        """P(k) [kpc^3] at z=0 (Delta^2 = k^3 P / 2 pi^2)."""
        k = np.asarray(k, float)
        return 2.0 * PI ** 2 * self.Delta2(k) / k ** 3

    # ---- sigma(M): replicate sigmaC's log-k trapezoid with the smooth-k window
    @staticmethod
    def _Ws(x):
        return 1.0 / (1.0 + (0.43 * x) ** 6)

    @staticmethod
    def _DWs(x):
        return -6.0 * (0.43 * x) ** 6 / (x * (1.0 + (0.43 * x) ** 6) ** 2)

    def _sigma_smoothk(self, M, deltaH):
        RM = (3.0 * M / (4.0 * PI * self.rhoM0)) ** (1.0 / 3.0)
        DRM = RM / (3.0 * M)
        kmax = 1000.0 / RM
        kmin = 1.0e-6 * kmax
        dlogk = (np.log(kmax) - np.log(kmin)) / (self.Nk - 1)
        # C++ integrates Nk panels [k_j, k_{j+1}], j=0..Nk-1 (overshoots kmax by
        # one panel) — replicate.
        k = kmin * np.exp(dlogk * np.arange(self.Nk + 1))
        D2 = self.Delta2(k, deltaH)
        f = self._Ws(k * RM) ** 2 * D2 / k
        g = 2.0 * k * DRM * self._DWs(k * RM) * self._Ws(k * RM) * D2 / k
        dk = np.diff(k)
        sigma2 = np.sum(dk * (f[:-1] + f[1:]) / 2.0)
        dsigma2 = np.sum(dk * (g[1:] + g[:-1]) / 2.0)
        s = np.sqrt(sigma2)
        return s, dsigma2 / (2.0 * s)

    def _build_sigmalist(self):
        Nextra = int(np.ceil((self.NM - 1) * np.log(3.0)
                             / np.log(self.Mmax / self.Mmin)))
        n = self.NM + Nextra
        Ms = self.Mmin * np.exp(self.dlogM * np.arange(n))
        sig = np.empty(n)
        dsig = np.empty(n)
        for j, M in enumerate(Ms):
            sig[j], dsig[j] = self._sigma_smoothk(M, self.deltaH8)
        self.sig_M = Ms
        self.sig_s = sig
        self.sig_ds = dsig

    def sigma_of_M(self, M):
        """interpolate(Mb, sigmalist): linear in M, clamped (replicates basics)."""
        return np.interp(M, self.sig_M, self.sig_s)

    # ---- comoving distance (replicates dclist trapezoid-in-log)
    def _build_dclist(self):
        Nz2 = 100 * self.Nz
        dlogz = (np.log(self.zmax) - np.log(self.zmin)) / (Nz2 - 2)
        z = np.concatenate([[0.0], self.zmin * np.exp(dlogz
                                                      * np.arange(Nz2 - 1))])
        invH = 1.0 / self.Hz(np.maximum(z, 1e-300))
        invH[0] = 1.0 / self.Hz(0.0)
        seg = (z[1:] - z[:-1]) * CLIGHT * np.sqrt(invH[1:] * invH[:-1])
        self.dc_z = z
        self.dc_d = np.concatenate([[0.0], np.cumsum(seg)])

    def dc(self, z):
        return np.interp(z, self.dc_z, self.dc_d)

    def DL(self, z):
        return (1 + z) * self.dc(z)

    def Sigmacf(self, zs, zl):
        DsA = self.DL(zs) / (1 + zs) ** 2
        DlA = self.DL(zl) / (1 + zl) ** 2
        DlsA = DsA - DlA * (1 + zl) / (1 + zs)
        return 2.08871e16 * DsA / (4.0 * PI * DlA * DlsA)

    # ---- HMF, halo bias, NFW tables
    @staticmethod
    def _pFC(delta, S):
        from math import gamma as gammaf
        p, q = 0.3, 0.8
        A = 1.0 / (1 + 2.0 ** -p * gammaf(0.5 - p) / np.sqrt(PI))
        nu2 = delta ** 2 / S
        return (A * (1 + (q * nu2) ** -p) * np.sqrt(q * nu2 / (2.0 * PI))
                * np.exp(-q * nu2 / 2.0) / S)

    def halobias(self, z, sigma):
        p, q = 0.3, 0.75
        qnu2 = q * (self.deltac(z) / sigma) ** 2
        return (1.0 + (qnu2 - 1.0) / DELTAC0
                + 2.0 * p / (DELTAC0 * (1.0 + qnu2 ** p)))

    def _cons14(self, z, M):
        a = 0.520 + (0.905 - 0.520) * np.exp(-0.617 * z ** 1.21)
        b = -0.101 + 0.026 * z
        return 10.0 ** (a + b * np.log10(M / (1.0e12 / self.h)))

    def _build_hmf_bias_nfw(self):
        Nz, NM = self.Nz, self.NM
        z = self.zlist[:, None]                       # (Nz,1)
        s = self.sig_s[None, :NM]                     # (1,NM) -> sigmalist[jM]
        ds = self.sig_ds[None, :NM]
        # dn/dlnM (HMFlistf: cell jM uses sigmalist[jM])
        self.HMF0 = (-self.rhoM0 * self._pFC(self.deltac(z), s ** 2)
                     * 2.0 * s * ds)
        self.biaslist = self.halobias(z, s)           # (Nz,NM)
        # NFW rs, rhos (conslistf uses c(z, Mlist[jM]) = cons14 at sigmalist node)
        M = self.Mlist[None, :]
        c = self._cons14(z, M)
        rhoz = self.Az(z) * self.rhoc
        r200 = (3.0 * M / (4.0 * PI * 200 * rhoz)) ** (1.0 / 3.0)
        self.rs = r200 / c
        self.rhos = (200 * rhoz * c ** 3 * (1 + c)
                     / (3.0 * ((1 + c) * np.log(1 + c) - c)))

    # ---- NFW kappa kernel (circular, eps=0): kappa(r) = 2 kappa0 Fg0(r/rs)
    @staticmethod
    def Fg0(x):
        x = np.asarray(x, float)
        out = np.empty_like(x)
        hi = x > 1.0
        lo = x < 1.0
        eq = ~(hi | lo)
        xh = x[hi]
        t = np.arctan(np.sqrt((xh - 1) / (1 + xh))) / np.sqrt(xh * xh - 1)
        out[hi] = (1 - 2 * t) / (xh * xh - 1)
        xl = np.clip(x[lo], 1e-300, None)
        t = np.arctanh(np.sqrt((1 - xl) / (1 + xl))) / np.sqrt(1 - xl * xl)
        out[lo] = (1 - 2 * t) / (xl * xl - 1)
        out[eq] = 1.0 / 3.0
        return out

    # ---- rmax bisection (replicates rmaxfNFW: log bounds 1e-6..1e6, tol 0.02)
    def rmax_grid(self, zs, kappathr):
        """rmax[jz,jM] for all cells; 0 where central kappa <= kappathr."""
        Nz, NM = self.Nz, self.NM
        Sig = self.Sigmacf(zs, self.zlist)            # (Nz,)
        kappa0 = self.rs * self.rhos / Sig[:, None]   # (Nz,NM)
        rs = self.rs
        lo = np.full((Nz, NM), np.log(1.0e-6))
        hi = np.full((Nz, NM), np.log(1.0e6))
        kc = 2.0 * kappa0 * self.Fg0(1.0e-6 / rs)
        alive = kc > kappathr
        while True:
            w = hi - lo
            if w.max() <= 0.02:
                break
            mid = np.exp((lo + hi) / 2.0)
            k = 2.0 * kappa0 * self.Fg0(mid / rs)
            up = k > kappathr
            lo = np.where(up, np.log(mid), lo)
            hi = np.where(up, hi, np.log(mid))
        rmax = np.exp((lo + hi) / 2.0)
        rmax[~alive] = 0.0
        rmax[self.zlist >= zs, :] = 0.0
        rmax[0, :] = 0.0                              # deltaNhf loops jz from 1
        rmax[:, 0] = 0.0                              # ... and jM from 1
        return rmax, kappa0

    def Nh(self, zs, kappathr, rmax=None):
        """NhfNFW: expected explicit-halo count above kappathr."""
        if rmax is None:
            rmax, _ = self.rmax_grid(zs, kappathr)
        zl = self.zlist
        dz = np.concatenate([[0.0], np.diff(zl)])[:, None]
        return np.sum(CLIGHT * PI * ((1 + zl[:, None]) * rmax) ** 2
                      / self.Hz(zl)[:, None] * self.HMF0 * self.dlogM * dz)

    def find_kappathr(self, zs, Nhalos=100):
        """findkappathr bisection replica (log10 bounds -12..0, tol 0.01)."""
        l1, l2 = np.log10(1.0e-12), 0.0
        kt = 10.0 ** ((l1 + l2) / 2.0)
        while l2 - l1 > 0.01:
            if self.Nh(zs, kt) > Nhalos:
                l1 = np.log10(kt)
            else:
                l2 = np.log10(kt)
            kt = 10.0 ** ((l1 + l2) / 2.0)
        return kt

    def sigmakappaW(self, zs, kappathr, eps_floor=0.001):
        """Background sigma_W (Campbell, log-annulus): replicates sigmakappaW."""
        rmax, kappa0 = self.rmax_grid(zs, kappathr)
        # the C++ version has no jz>=1/jM>=1 skip (loops from 1 anyway) and uses
        # r=1e-6 where rmax==0
        zl = self.zlist
        dz = np.concatenate([[0.0], np.diff(zl)])
        mask = np.broadcast_to(zl[:, None] < zs,
                               (self.Nz, self.NM)).copy()
        mask[0, :] = False
        mask[:, 0] = False
        r = np.where(rmax > 0.0, rmax, 1.0e-6)
        pref = (CLIGHT * 2.0 * PI * (1 + zl[:, None]) ** 2
                / self.Hz(zl)[:, None] * self.HMF0 * 0.01 * self.dlogM
                * dz[:, None])
        kappa2 = 0.0
        alive = mask.copy()
        Edlnr = np.exp(0.01)
        while alive.any():
            k = 2.0 * kappa0 * self.Fg0(r / self.rs)
            kappa2 += np.sum((pref * r ** 2 * k ** 2)[alive])
            r = np.where(alive, r * Edlnr, r)
            alive &= k > eps_floor * kappathr
        return np.sqrt(kappa2)

    # ---- the bias-layer cell tables (deltaNhfNFW replica + extras)
    def cell_tables(self, zs, kappathr):
        rmax, kappa0 = self.rmax_grid(zs, kappathr)
        Nz, NM = self.Nz, self.NM
        zl = self.zlist
        dz = np.concatenate([[0.0], np.diff(zl)])
        chi = self.dc(zl)
        jz, jM = np.nonzero(rmax > 0.0)
        r = rmax[jz, jM]
        barN = (CLIGHT * PI * ((1 + zl[jz]) * r) ** 2 / self.Hz(zl[jz])
                * self.HMF0[jz, jM] * self.dlogM * dz[jz])
        keep = barN > 0.0
        jz, jM, r, barN = jz[keep], jM[keep], r[keep], barN[keep]
        # old-model sigma_b: M_b with PHYSICAL rmax^2 (replicated wart), linear-M
        # interp clamped at Mmin (replicated clamp)
        Lseg = chi[jz] - chi[jz - 1]
        Mb = 2.0 * PI * r ** 2 * Lseg * self.rhoM0
        sigma_old = (self.Dg(zl[jz]) * self.sigma_of_M(Mb)
                     * self.biaslist[jz, jM])
        # disc means of kappa and kappa^2 (for analytic clustering sums)
        x = np.exp(np.linspace(np.log(1e-6), 0.0, 240))[None, :]  # r/rmax grid
        xr = r[:, None] * x / self.rs[jz, jM][:, None]
        kap = 2.0 * kappa0[jz, jM][:, None] * self.Fg0(xr)
        w = x ** 2                                    # d(r^2/rmax^2) = 2x^2 dlnx
        dlnx = np.log(x[0, 1] / x[0, 0])
        kbar = np.sum(kap * 2 * w, axis=1) * dlnx
        k2bar = np.sum(kap ** 2 * 2 * w, axis=1) * dlnx
        return dict(jz=jz, jM=jM, rmax=r, barN=barN, sigma_old=sigma_old,
                    Lseg=Lseg, chi_lo=chi[jz - 1], chi_hi=chi[jz],
                    rs=self.rs[jz, jM], kappa0=kappa0[jz, jM],
                    bDg=self.biaslist[jz, jM] * self.Dg(zl[jz]),
                    kbar=kbar, k2bar=k2bar, zl=zl[jz], Mb=Mb)


# ------------------------------------------------------- new-model field machinery
class LOSField:
    """P_1D pencil projection + cylinder-variance table from the code's P(k)."""

    def __init__(self, C: Cosmo):
        self.C = C
        # k grids (kpc^-1)
        self.kpar = np.exp(np.linspace(np.log(1e-8), np.log(3.0), 4096))
        kperp = np.exp(np.linspace(np.log(1e-8), np.log(30.0), 4096))
        from scipy.special import j1
        # R grid for the disc filter (comoving kpc); row 0 ~ pencil limit
        self.Rgrid = np.exp(np.linspace(np.log(0.5), np.log(1e5), 40))
        kk = np.sqrt(self.kpar[None, :] ** 2 + kperp[:, None] ** 2)  # (nkp,nk)
        P = C.Pk(kk)
        x = kperp[:, None] * np.ones_like(self.kpar)[None, :]
        # Q(kpar, R) = (1/2pi) int kperp P Wdisc^2 dkperp   (trapz in ln kperp)
        dlnkp = np.log(kperp[1] / kperp[0])
        base = kperp[:, None] ** 2 * P                # kperp^2 P dln kperp
        self.Q = np.empty((len(self.Rgrid), len(self.kpar)))
        for i, R in enumerate(self.Rgrid):
            xr = kperp * R
            Wd = np.where(xr < 1e-6, 1.0, 2.0 * j1(xr) / np.where(
                xr < 1e-6, 1.0, xr))
            self.Q[i] = np.trapezoid(base * Wd[:, None] ** 2, dx=dlnkp,
                                     axis=0) / (2.0 * PI)
        self.P1D = self.Q[0]                          # R->0 pencil (0.5 kpc)

    def cell_sigma2(self, R, L, RL):
        """Per-cell z=0 environment variance with the counting-geometry window
        AND the parameter-free peak-background-split floor at the halo's own
        Lagrangian radius R_L(M): transverse disc at R_eff = max(R, R_L),
        longitudinal sinc(k L/2) * Ws(k R_L) (separable approximation of the
        isotropic R_L smoothing; both limits correct). Modes shorter than R_L
        ARE the halo — including them would double-count against the Poisson
        term, and their removal is what keeps sigma finite as rmax -> 0
        (the kappa_thr blow-up check)."""
        R_eff = np.clip(np.maximum(R, RL), self.Rgrid[0], self.Rgrid[-1])
        gR = np.log(self.Rgrid)
        lR = np.log(R_eff)
        iR = np.clip(np.searchsorted(gR, lR) - 1, 0, len(gR) - 2)
        fR = ((lR - gR[iR]) / (gR[iR + 1] - gR[iR]))[:, None]
        logQ = np.log(self.Q)
        Qc = np.exp((1 - fR) * logQ[iR] + fR * logQ[iR + 1])  # (nc, nk)
        k = self.kpar[None, :]
        snc = np.sinc(k * L[:, None] / (2.0 * PI))
        WsL = Cosmo._Ws(k * RL[:, None])
        dlnk = np.log(self.kpar[1] / self.kpar[0])
        return (Qc * snc ** 2 * WsL ** 2 * self.kpar[None, :]
                ).sum(axis=1) * dlnk / PI

    def segment_corr(self, chi_lo_jz, chi_hi_jz):
        """Pencil segment-average CORRELATION matrix across z-shells.

        PSD by construction: C = Ac Ac^T + As As^T with
        A*[i,q] = sqrt(P1D w_q / pi) * {cos,sin}(k_q c_i) * sinc(k_q L_i / 2).
        """
        c = (chi_lo_jz + chi_hi_jz) / 2.0
        L = chi_hi_jz - chi_lo_jz
        k = self.kpar
        dlnk = np.log(k[1] / k[0])
        wq = np.sqrt(self.P1D * k * dlnk / PI)        # (nk,)
        snc = np.sinc(k[None, :] * L[:, None] / (2.0 * PI))
        Ac = wq[None, :] * np.cos(k[None, :] * c[:, None]) * snc
        As = wq[None, :] * np.sin(k[None, :] * c[:, None]) * snc
        Cov = Ac @ Ac.T + As @ As.T
        d = np.sqrt(np.diag(Cov))
        corr = Cov / np.outer(d, d)
        return corr, Cov


def new_model_cell_sigma(fld: LOSField, C: Cosmo, T):
    """Per-cell environment std, new model: b*Dg*sigma(cyl window, R_L floor)."""
    R_com = (1 + T["zl"]) * T["rmax"]
    M = C.Mlist[T["jM"]]
    RL = (3.0 * M / (4.0 * PI * C.rhoM0)) ** (1.0 / 3.0)  # Lagrangian, comoving
    return T["bDg"] * np.sqrt(fld.cell_sigma2(R_com, T["Lseg"], RL))


# ------------------------------------------------------------------ stages
def stage_port_check():
    """Validate the port against the C++ helpers at several (zs, Nz)."""
    sys.path.insert(0, str(REPO / "build"))
    import gwlensing as gw
    lines = ["# Port validation vs gwlensing helpers\n",
             "| zs | Nz | quantity | C++ | python | rel.diff |",
             "|--:|--:|:--|--:|--:|--:|"]
    ok = True
    for nz in (100, 400):
        for zs in (1.0, 10.0):
            C = Cosmo(Nz=nz)
            kt_cpp = gw.get_kappa_threshold(zs, H, OM, S8, 100, Nz=nz)
            kt_py = C.find_kappathr(zs, 100)
            n_cpp = gw.get_expected_halo_count(zs, H, OM, S8, kt_cpp, Nz=nz)
            n_py = C.Nh(zs, kt_cpp)
            s_cpp = gw.get_sigma_background(zs, H, OM, S8, kt_cpp, Nz=nz)
            s_py = C.sigmakappaW(zs, kt_cpp)
            for name, a, b in (("kappa_thr", kt_cpp, kt_py),
                               ("<N>", n_cpp, n_py),
                               ("sigma_W", s_cpp, s_py)):
                rd = abs(b - a) / abs(a)
                # kappa_thr granularity is the bisection tol (1.2%/cell)
                tol = 0.025 if name == "kappa_thr" else 0.01
                ok &= rd < tol
                lines.append(f"| {zs:g} | {nz} | {name} | {a:.6g} | {b:.6g} |"
                             f" {rd:.2e} |")
                print(lines[-1], flush=True)
    lines.append(f"\nPASS: {ok}\n")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "port_validation.md").write_text("\n".join(lines))
    if not ok:
        raise SystemExit("port validation FAILED — see port_validation.md")
    print("port validation PASSED")


def _clustering_std(T, sig, corr=None):
    """LINEARIZED std of the bias-modulated mean explicit kappa:
    sqrt(sum_ij w_i w_j sig_i sig_j rho_ij), w = barN * kbar (lam ~ 1 + g).
    Full lognormal moments E[lam^2]=exp(sig^2) DIVERGE here because the
    current layer carries cells with sig_b ~ 30-150 (huge-bias, tiny-barN) —
    itself a symptom of the pathology; the MC is the honest tail statistic.
    corr=None -> iid cells (old model); corr (njz,njz) -> new model."""
    w = T["barN"] * T["kbar"]
    if corr is None:
        return np.sqrt(np.sum(w ** 2 * sig ** 2))
    a = np.bincount(T["jz_local"], weights=w * sig, minlength=corr.shape[0])
    return np.sqrt(a @ corr @ a)


def _wq(sig, w, q=0.99):
    """w-weighted quantile of sig (robust replacement for a raw max)."""
    o = np.argsort(sig)
    cw = np.cumsum(w[o])
    return sig[o][np.searchsorted(cw, q * cw[-1])]


def _localize_jz(T):
    """Map global jz to a compact local index (cells only)."""
    uj = np.unique(T["jz"])
    T["jz_unique"] = uj
    T["jz_local"] = np.searchsorted(uj, T["jz"])
    return uj


def stage_analytic():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    t0 = time.time()
    # ---- (b) default grid comparison + (c) kappa_thr sweep, both at Nz=100
    C = Cosmo(Nz=100)
    fld = LOSField(C)
    # internal consistency: pencil corr diag vs sigma_cyl(R->0)
    for zs in ZS_LIST:
        kt = C.find_kappathr(zs, 100)
        T = C.cell_tables(zs, kt)
        uj = _localize_jz(T)
        corr, cov = fld.segment_corr(C.dc(C.zlist[uj - 1]),
                                     C.dc(C.zlist[uj]))
        Lu = C.dc(C.zlist[uj]) - C.dc(C.zlist[uj - 1])
        s_pencil = np.sqrt(np.diag(cov))
        r0 = np.full_like(Lu, fld.Rgrid[0])
        s_tab = np.sqrt(fld.cell_sigma2(r0, Lu, r0))
        cons = np.abs(s_pencil / s_tab - 1).max()
        print(f"[consistency] zs={zs:g}: pencil-diag vs table R->0 "
              f"max rel dev = {cons:.3f}")
        sig_new = new_model_cell_sigma(fld, C, T)
        res[f"z{zs:g}_sig_old"] = T["sigma_old"]
        res[f"z{zs:g}_sig_new"] = sig_new
        res[f"z{zs:g}_w"] = T["barN"] * T["kbar"]
        res[f"z{zs:g}_Mb"] = T["Mb"]
        res[f"z{zs:g}_zl"] = T["zl"]
        res[f"z{zs:g}_clust_old"] = _clustering_std(T, T["sigma_old"])
        res[f"z{zs:g}_clust_new"] = _clustering_std(T, sig_new, corr)
        wm = lambda x: np.sum(res[f"z{zs:g}_w"] * x) / np.sum(
            res[f"z{zs:g}_w"])
        print(f"[b-check] zs={zs:g} Nz=100 default kthr={kt:.3e}: "
              f"w-mean sigma_old={wm(T['sigma_old']):.3f} "
              f"sigma_new={wm(sig_new):.3f}  "
              f"clust_std old={res[f'z{zs:g}_clust_old']:.4f} "
              f"new={res[f'z{zs:g}_clust_new']:.4f}", flush=True)
        # ---- (c) kappa_thr sweep at this zs
        mx_o, mx_n, cl_o, cl_n = [], [], [], []
        for k_thr in KTHR_SWEEP:
            Tk = C.cell_tables(zs, k_thr)
            _localize_jz(Tk)
            ujk = Tk["jz_unique"]
            ck, _ = fld.segment_corr(C.dc(C.zlist[ujk - 1]),
                                     C.dc(C.zlist[ujk]))
            sn = new_model_cell_sigma(fld, C, Tk)
            wk = Tk["barN"] * Tk["kbar"]
            mx_o.append(_wq(Tk["sigma_old"], wk))
            mx_n.append(_wq(sn, wk))
            cl_o.append(_clustering_std(Tk, Tk["sigma_old"]))
            cl_n.append(_clustering_std(Tk, sn, ck))
            print(f"[c-check] zs={zs:g} kthr={k_thr:.0e}: w-p99 sig_b "
                  f"old={mx_o[-1]:.2f} new={mx_n[-1]:.2f}  clust_std "
                  f"old={cl_o[-1]:.4f} new={cl_n[-1]:.4f}", flush=True)
        res[f"z{zs:g}_kthr_sweep"] = np.array(KTHR_SWEEP)
        res[f"z{zs:g}_kthr_p99sig_old"] = np.array(mx_o)
        res[f"z{zs:g}_kthr_p99sig_new"] = np.array(mx_n)
        res[f"z{zs:g}_kthr_clust_old"] = np.array(cl_o)
        res[f"z{zs:g}_kthr_clust_new"] = np.array(cl_n)
    # ---- vs Nz (analytic clustering std + max sigma), default rule per Nz
    for zs in ZS_LIST:
        mo, mn, co, cn = [], [], [], []
        for nz in NZ_GRID_AN:
            Cn = Cosmo(Nz=nz)
            # P(k) is Nz-independent: reuse the field object (grid enters
            # only through the segment geometry)
            fln = fld
            ktn = Cn.find_kappathr(zs, 100)
            Tn = Cn.cell_tables(zs, ktn)
            _localize_jz(Tn)
            ujn = Tn["jz_unique"]
            ckn, _ = fln.segment_corr(Cn.dc(Cn.zlist[ujn - 1]),
                                      Cn.dc(Cn.zlist[ujn]))
            snn = new_model_cell_sigma(fln, Cn, Tn)
            wn = Tn["barN"] * Tn["kbar"]
            mo.append(_wq(Tn["sigma_old"], wn))
            mn.append(_wq(snn, wn))
            co.append(_clustering_std(Tn, Tn["sigma_old"]))
            cn.append(_clustering_std(Tn, snn, ckn))
            print(f"[nz-analytic] zs={zs:g} Nz={nz}: w-p99 sig_b "
                  f"old={mo[-1]:.2f} new={mn[-1]:.2f}  clust_std "
                  f"old={co[-1]:.4f} new={cn[-1]:.4f}", flush=True)
        res[f"z{zs:g}_nzgrid"] = np.array(NZ_GRID_AN, float)
        res[f"z{zs:g}_nz_p99sig_old"] = np.array(mo)
        res[f"z{zs:g}_nz_p99sig_new"] = np.array(mn)
        res[f"z{zs:g}_nz_clust_old"] = np.array(co)
        res[f"z{zs:g}_nz_clust_new"] = np.array(cn)
    np.savez(OUT / "analytic.npz", **res)
    print(f"analytic stage done in {time.time()-t0:.0f}s ->",
          OUT / "analytic.npz")


def mc_path(zs, nz, arm):
    return MCDIR / f"z{zs:g}_nz{nz}_{arm}.npz"


def run_mc_config(C, fld, zs, nz, arm, seed):
    """Bias+halo layer MC: kappa samples for one (zs, Nz, arm)."""
    kt = C.find_kappathr(zs, 100)
    sigW = C.sigmakappaW(zs, kt)
    T = C.cell_tables(zs, kt)
    _localize_jz(T)
    uj = T["jz_unique"]
    barN, rmax, rs, k0 = T["barN"], T["rmax"], T["rs"], T["kappa0"]
    jloc = T["jz_local"]
    if arm == "old":
        sig = T["sigma_old"]
    else:
        sig = new_model_cell_sigma(fld, C, T)
        corr, _ = fld.segment_corr(C.dc(C.zlist[uj - 1]), C.dc(C.zlist[uj]))
        cholT = np.linalg.cholesky(
            corr + 1e-10 * np.eye(len(corr))).T
    rng = np.random.default_rng(seed)
    kappa = np.empty(NREAL, dtype=np.float32)
    nc = len(barN)
    halved = sig ** 2 / 2.0
    for s0 in range(0, NREAL, CHUNK):
        s1 = min(s0 + CHUNK, NREAL)
        nch = s1 - s0
        if arm == "old":
            g = sig[None, :] * rng.standard_normal((nch, nc))
        else:
            v = rng.standard_normal((nch, len(uj))) @ cholT
            g = sig[None, :] * v[:, jloc]
        rate = np.exp(g - halved[None, :]) * barN[None, :]
        # pure Poisson in every cell (C++ Bernoulli-approximates rate<0.2 —
        # see module docstring)
        Ncnt = rng.poisson(rate)
        rr, cc = np.nonzero(Ncnt)
        reps = Ncnt[rr, cc]
        rid = np.repeat(rr, reps)
        cid = np.repeat(cc, reps)
        u = rng.random(len(cid))
        r = rmax[cid] * np.sqrt(u)
        kap = 2.0 * k0[cid] * C.Fg0(r / rs[cid])
        kchunk = np.bincount(rid, weights=kap, minlength=nch)
        kappa[s0:s1] = kchunk + sigW * rng.standard_normal(nch)
    return kappa, kt, sigW, nc


def stage_mc():
    MCDIR.mkdir(parents=True, exist_ok=True)
    C100 = Cosmo(Nz=100)
    fld = LOSField(C100)      # P(k)-level tables are Nz-independent
    for iz, zs in enumerate(ZS_LIST):
        for inz, nz in enumerate(NZ_GRID_MC):
            C = Cosmo(Nz=nz)
            for ia, arm in enumerate(("old", "new")):
                p = mc_path(zs, nz, arm)
                if p.exists():
                    continue
                t0 = time.time()
                seed = SEED0 + 1_000_000 * iz + 10_000 * inz + 100 * ia
                kappa, kt, sigW, nc = run_mc_config(C, fld, zs, nz, arm, seed)
                np.savez(p, kappa=kappa, kappathr=kt, sigW=sigW,
                         ncells=nc, nreal=NREAL, seed=seed)
                print(f"[mc] zs={zs:g} Nz={nz} {arm}: ncells={nc} "
                      f"kthr={kt:.3e} sigW={sigW:.4f} f(k>1)="
                      f"{(kappa > 1).mean():.2e}  ({time.time()-t0:.0f}s)",
                      flush=True)


def stage_report():
    an = np.load(OUT / "analytic.npz")
    # measured full-model reference
    ref = np.load(REPO / "data" / "results" / "vark_nz" / "vark_vs_nz.npz")
    lines = ["# Bias-layer prototype: grid-decoupled LOS-correlated field\n",
             "Old arm = replica of the current per-cell iid lognormal layer "
             "(lensing.cpp deltaNhfNFW + sampling loop, kappa only, pure "
             "Poisson counts). New arm = same halo layer, environment drawn "
             "from the pencil-projected linear P(k) with per-cell "
             "cylinder-window amplitude (no free scale).\n"]
    # MC table
    lines += ["## (a) MC: tail + body vs Nz (fixed-<N>=100 rule per Nz)\n",
              "| zs | Nz | arm | f(k>1) | f(k>0.5) | q99.9 | Var(|k|<0.5) |",
              "|--:|--:|:--|--:|--:|--:|--:|"]
    mc = {}
    for zs in ZS_LIST:
        for nz in NZ_GRID_MC:
            for arm in ("old", "new"):
                p = mc_path(zs, nz, arm)
                if not p.exists():
                    continue
                k = np.load(p)["kappa"].astype(np.float64)
                x = k[np.abs(k) < 0.5]
                x = x - x.mean()
                mc[(zs, nz, arm)] = dict(
                    f1=(k > 1).mean(), f05=(k > 0.5).mean(),
                    q999=np.quantile(k, 0.999), var05=(x ** 2).mean())
                d = mc[(zs, nz, arm)]
                lines.append(f"| {zs:g} | {nz} | {arm} | {d['f1']:.2e} | "
                             f"{d['f05']:.2e} | {d['q999']:.4f} | "
                             f"{d['var05']:.4e} |")
    # analytic tables
    lines += ["\n## (b) Default grid (Nz=100): per-cell sigma comparison\n",
              "| zs | w-mean sig_old | w-mean sig_new | clust_std old | "
              "clust_std new |", "|--:|--:|--:|--:|--:|"]
    for zs in ZS_LIST:
        w = an[f"z{zs:g}_w"]
        so, sn = an[f"z{zs:g}_sig_old"], an[f"z{zs:g}_sig_new"]
        wm = lambda x: np.sum(w * x) / np.sum(w)
        lines.append(f"| {zs:g} | {wm(so):.3f} | {wm(sn):.3f} | "
                     f"{float(an[f'z{zs:g}_clust_old']):.4f} | "
                     f"{float(an[f'z{zs:g}_clust_new']):.4f} |")
    lines += ["\n## Analytic continuum-limit check: linearized clustering std "
              "vs Nz\n",
              "(std of the bias-modulated mean explicit kappa, lambda ~ 1+g; "
              "old = iid cells, new = correlated field. A grid-decoupled "
              "model must be flat in Nz.)\n",
              "| zs | Nz | clust_std old | clust_std new | p99 sig_b old | "
              "p99 sig_b new |", "|--:|--:|--:|--:|--:|--:|"]
    for zs in ZS_LIST:
        nzg = an[f"z{zs:g}_nzgrid"]
        for i, nz in enumerate(nzg):
            lines.append(
                f"| {zs:g} | {int(nz)} | {an[f'z{zs:g}_nz_clust_old'][i]:.4f}"
                f" | {an[f'z{zs:g}_nz_clust_new'][i]:.4f} | "
                f"{an[f'z{zs:g}_nz_p99sig_old'][i]:.2f} | "
                f"{an[f'z{zs:g}_nz_p99sig_new'][i]:.2f} |")
    lines += ["\n## (c) kappa_thr sweep (Nz=100): w-p99 per-cell sigma_b\n",
              "| zs | kappa_thr | p99 sig_b old | p99 sig_b new |",
              "|--:|--:|--:|--:|"]
    for zs in ZS_LIST:
        ks = an[f"z{zs:g}_kthr_sweep"]
        for i, k_thr in enumerate(ks):
            lines.append(f"| {zs:g} | {k_thr:.0e} | "
                         f"{an[f'z{zs:g}_kthr_p99sig_old'][i]:.2f} | "
                         f"{an[f'z{zs:g}_kthr_p99sig_new'][i]:.2f} |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUT / "report.md")
    _plot(an, mc, ref)


def _plot(an, mc, ref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink = "#334155"
    grid_c = "#e2e8f0"
    # categorical slots (dataviz default palette, validated 2026-07-14:
    # CVD dE 73.6 PASS; aqua contrast WARN relieved by direct labels + tables)
    c_old = "#2a78d6"     # slot 1 blue: old arm (current per-cell iid layer)
    c_new = "#1baf7a"     # slot 2 aqua: new arm (LOS-correlated field)
    c_z1 = "#4a3aa7"      # slot 5 violet: z_s=1 group (panel c)
    c_z10 = "#eb6834"     # slot 8 orange: z_s=10 group (panel c)
    c_ref = "#8a8776"     # muted neutral: measured reference
    plt.rcParams.update({
        "text.color": ink, "axes.labelcolor": ink, "axes.edgecolor": grid_c,
        "xtick.color": ink, "ytick.color": ink, "font.size": 10,
        "axes.grid": True, "grid.color": grid_c, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white"})

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.6))

    # (a) f(k>1) vs Nz, zs=10
    ax = axes[0, 0]
    zs = 10.0
    for arm, c in (("old", c_old), ("new", c_new)):
        xs = [nz for nz in NZ_GRID_MC if (zs, nz, arm) in mc]
        ys = [mc[(zs, nz, arm)]["f1"] for nz in xs]
        ax.plot(xs, ys, "o-", color=c, label=f"{arm} arm (prototype)")
    ax.plot(ref["nz_grid"], ref["frac_kgt1_z10"].mean(axis=1), "s--",
            color=c_ref, label="full C++ model (measured)", zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Nz")
    ax.set_ylabel("f(kappa > 1)")
    ax.set_title("(a) strong tail vs Nz,  z_s = 10", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    # (b) body Var vs Nz, zs=10
    ax = axes[0, 1]
    for arm, c in (("old", c_old), ("new", c_new)):
        xs = [nz for nz in NZ_GRID_MC if (zs, nz, arm) in mc]
        ys = [mc[(zs, nz, arm)]["var05"] for nz in xs]
        ax.plot(xs, ys, "o-", color=c, label=arm)
    ax.set_xscale("log")
    ax.set_xlabel("Nz")
    ax.set_ylabel("Var(kappa), |kappa| < 0.5")
    ax.set_title("(b) body variance vs Nz,  z_s = 10", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    # (c) per-cell sigma_old vs sigma_new at default grid
    ax = axes[1, 0]
    for zs_i, mk, cc in ((1.0, "o", c_z1), (10.0, "^", c_z10)):
        w = an[f"z{zs_i:g}_w"]
        sel = w > np.quantile(w, 0.5)     # show the cells that matter
        ax.scatter(an[f"z{zs_i:g}_sig_old"][sel],
                   an[f"z{zs_i:g}_sig_new"][sel], s=8, alpha=0.4,
                   marker=mk, color=cc, label=f"z_s = {zs_i:g}",
                   linewidths=0)
    lim = [5e-2, 8]
    ax.plot(lim, lim, "-", color=c_ref, lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("sigma_b  (current: sphere M_b, per-cell iid)")
    ax.set_ylabel("sigma_new  (cylinder window)")
    ax.set_title("(c) per-cell amplitude, default grid Nz=100", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    # (d) max sigma_b vs kappa_thr
    ax = axes[1, 1]
    for zs_i, ls in ((1.0, "-"), (10.0, "--")):
        ks = an[f"z{zs_i:g}_kthr_sweep"]
        ax.plot(ks, an[f"z{zs_i:g}_kthr_p99sig_old"], "o" + ls, color=c_old,
                label=f"old, z_s={zs_i:g}")
        ax.plot(ks, an[f"z{zs_i:g}_kthr_p99sig_new"], "o" + ls, color=c_new,
                label=f"new, z_s={zs_i:g}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("kappa_thr (flat)")
    ax.set_ylabel("w-p99 per-cell sigma_b")
    ax.set_title("(d) kappa_thr blow-up check, Nz=100", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    fig.suptitle("Bias-layer prototype: per-cell iid (old) vs "
                 "LOS-correlated field (new)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    PLOTS.mkdir(exist_ok=True)
    fig.savefig(PLOTS / "bias_field_prototype.png", dpi=160)
    print("wrote", PLOTS / "bias_field_prototype.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["port-check", "analytic", "mc",
                                      "report", "all"])
    a = ap.parse_args()
    if a.stage in ("port-check", "all"):
        stage_port_check()
    if a.stage in ("analytic", "all"):
        stage_analytic()
    if a.stage in ("mc", "all"):
        stage_mc()
    if a.stage in ("report", "all"):
        stage_report()
