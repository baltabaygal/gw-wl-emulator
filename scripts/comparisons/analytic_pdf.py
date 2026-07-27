#!/usr/bin/env python
"""
Semi-analytic magnification PDF (halos-only, spherical, unbiased Poisson model).
Implemented from analytic_chain_spec.md (v1). Chain: dn/dM -> R(xi) -> Lambda(k) -> P(mu).

Every stage prints the spec's checkpoints so we can verify conventions before
comparing to the MC. numpy/scipy only.
"""
import numpy as np
from numpy import log, exp, sqrt, sin, cos, pi
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from scipy.special import gamma as Gamma
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

# ============================ 0. constants ==================================
h   = 0.674
Om  = 0.315
Ob  = 0.0493
OL  = 1.0 - Om
s8  = 0.811
ns  = 0.965
TCMB = 2.7255
dc  = 1.686

ckms = 299792.458            # km/s
H0   = 100.0 * h            # km/s/Mpc
rhoc0 = 2.77536627e11 * h*h  # Msun/Mpc^3
rhom  = Om * rhoc0           # comoving matter density
G     = 4.30091e-9           # Mpc (km/s)^2 / Msun

def E(z):   return sqrt(Om*(1+z)**3 + OL)
def Hz(z):  return H0*E(z)
def rhocrit(z): return rhoc0*E(z)**2          # physical

def chi(z):                                    # comoving distance [Mpc]
    return quad(lambda zz: ckms/H0/E(zz), 0, z)[0]
def DA(z):  return chi(z)/(1+z)
def DA_ls(zl, zs): return (chi(zs)-chi(zl))/(1+zs)   # flat

def Sigma_cr(zl, zs):
    return ckms**2 * DA(zs) / (4*pi*G*DA(zl)*DA_ls(zl, zs))   # Msun/Mpc^2

# ---- growth factor D(z), normalised D(0)=1 --------------------------------
def _Ea(a): return sqrt(Om*a**-3 + OL)
def _Dun(a): return _Ea(a)*quad(lambda ap: 1.0/(ap*_Ea(ap))**3, 0, a)[0]
_D0 = _Dun(1.0)
def D(z):
    a = 1.0/(1+z)
    return _Dun(a)/_D0

# ============================ 1. EH98 + sigma(M) ============================
wm = Om*h*h;  wb = Ob*h*h;  th = TCMB/2.7
_s = 44.5*log(9.83/wm)/sqrt(1+10*wb**0.75)
_aG = 1 - 0.328*log(431*wm)*(wb/wm) + 0.38*log(22.3*wm)*(wb/wm)**2
def Teh(k):                                    # k in Mpc^-1
    Geff = Om*h*(_aG + (1-_aG)/(1+(0.43*k*_s)**4))
    q = k*th*th/(Geff*h)
    L = log(2*np.e + 1.8*q)
    C = 14.2 + 731.0/(1+62.5*q)
    return L/(L + C*q*q)

_kg = np.logspace(-5, 3, 3000)
_Pshape = _kg**ns * Teh(_kg)**2

def _sigmaR_shape(R):
    x = _kg*R
    W = 3*(sin(x) - x*cos(x))/x**3
    integ = _kg**2 * _Pshape * W**2
    return sqrt(_trapz(integ, _kg)/(2*pi*pi))

_Anorm = s8/_sigmaR_shape(8.0/h)               # fix sigma(8/h)=sigma8
def sigmaR(R):  return _Anorm*_sigmaR_shape(R)
def RofM(M):    return (3*M/(4*pi*rhom))**(1.0/3.0)

_Mg = np.logspace(4, 17, 400)
_lnsig = np.array([log(sigmaR(RofM(M))) for M in _Mg])
_spl = CubicSpline(log(_Mg), _lnsig)
def sigmaM(M, z=0.0):  return exp(_spl(log(M)))*D(z)
def dlnsig_dlnM(M):    return _spl(log(M), 1)

# ============================ 2. Sheth-Tormen ==============================
p_st, q_st = 0.3, 0.8
A_st = 1.0/(1 + 2**(-p_st)*Gamma(0.5-p_st)/sqrt(pi))
def dndlnM(M, z):
    nu = (dc/sigmaM(M, z))**2                  # note: SQUARE
    qn = q_st*nu
    mult = A_st*(1+qn**(-p_st))*sqrt(qn/(2*pi))*exp(-qn/2)
    dlnnu_dlnM = -2*dlnsig_dlnM(M)             # (z-independent: D cancels in dln)
    return (rhom/M)*mult*dlnnu_dlnM

# ============================ 3. NFW (WB2000) ==============================
def conc(M, z):
    a = 0.520 + (0.905-0.520)*exp(-0.617*z**1.21)
    b = -0.101 + 0.026*z
    return 10.0**(a + b*np.log10(M*h/1e12))

def nfw_params(M, zl, zs):
    C = conc(M, zl)
    r200 = (3*M/(4*pi*200*rhocrit(zl)))**(1.0/3.0)   # Mpc physical
    rs = r200/C
    fC = log(1+C) - C/(1+C)
    rhos = (200.0/3.0)*rhocrit(zl)*C**3/fC
    Scr = Sigma_cr(zl, zs)
    ks = rhos*rs/Scr
    return C, rs, ks, fC

def _F(x):
    x = np.asarray(x, float); out = np.empty_like(x)
    lo = x < 1-1e-6; hi = x > 1+1e-6; mid = ~(lo | hi)
    xl = x[lo]
    out[lo] = (1/(xl*xl-1))*(1 - (2/sqrt(1-xl*xl))*np.arctanh(sqrt((1-xl)/(1+xl))))
    xh = x[hi]
    out[hi] = (1/(xh*xh-1))*(1 - (2/sqrt(xh*xh-1))*np.arctan(sqrt((xh-1)/(xh+1))))
    xm = x[mid]
    out[mid] = 1.0/3.0 - 2.0*(xm-1)/5.0
    return out

def _hfun(x):
    x = np.asarray(x, float); out = np.empty_like(x)
    lo = x < 1-1e-6; hi = x > 1+1e-6; mid = ~(lo | hi)
    xl = x[lo]
    out[lo] = log(xl/2) + np.arccosh(1/xl)/sqrt(1-xl*xl)
    xh = x[hi]
    out[hi] = log(xh/2) + np.arccos(1/xh)/sqrt(xh*xh-1)
    xm = x[mid]
    out[mid] = log(xm/2) + 1 - (xm-1)/3.0
    # small-x stabilisation
    xs = x < 1e-3
    out[xs] = (x[xs]**2/2)*(log(2/x[xs]) - 0.5)
    return out

def kappa_gamma(x, ks):
    k = 2*ks*_F(x)
    kbar = 4*ks*_hfun(x)/x**2
    g = kbar - k
    return k, g

# ============================ 4. single-lens dsigma/dxi =====================
_xg = np.logspace(-6, 3.5, 900)
_lnx = log(_xg)

def dsigma_dxi_curve(M, zl, zs):
    """LEGACY (spec 4 recipe): np.gradient + branch-merging interp. Kept only
    for A/B against the deposit-binned version; known to undercount dsigma/dxi
    in the multibranch caustic band xi >~ 1."""
    C, rs, ks, fC = nfw_params(M, zl, zs)
    k, g = kappa_gamma(_xg, ks)
    detA = (1-k)**2 - g*g
    good = detA > 1e-12
    xi = np.full_like(_xg, np.nan)
    xi[good] = -log(detA[good])
    # dxi/dlnx
    dxi = np.gradient(xi, _lnx)
    ok = good & np.isfinite(dxi) & (np.abs(dxi) > 0)
    dsig = 2*pi*rs*rs*_xg**2/np.abs(dxi)       # Mpc^2
    return xi[ok], dsig[ok]


def _refined_xgrid(ks):
    """Base log grid + adaptive refinement around every zero of detA(x), so the
    near-caustic exponential tail of xi(x) is resolved to detA ~ 1e-12."""
    k, g = kappa_gamma(_xg, ks)
    detA = (1-k)**2 - g*g
    xs = [_xg]
    sgn = np.sign(detA)
    cross = np.nonzero(sgn[1:]*sgn[:-1] < 0)[0]
    for i in cross:
        a, b = _xg[i], _xg[i+1]
        for _ in range(70):                     # bisect detA = 0
            m = 0.5*(a+b)
            km, gm = kappa_gamma(np.array([m]), ks)
            dm = (1-km[0])**2 - gm[0]*gm[0]
            if np.sign(dm) == sgn[i]:
                a = m
            else:
                b = m
        xc = 0.5*(a+b)
        eps = np.logspace(-11, -1.2, 22)
        xs.append(xc*(1.0+eps))
        xs.append(xc*(1.0-eps))
    x = np.unique(np.concatenate(xs))
    return x[x > 0]


def dsigma_bins(M, zl, zs):
    """Deposit-binned cross-section per XI bin (branch-safe, exact area):
    every x-interval deposits its exact annulus area pi rs^2 (x2^2 - x1^2)
    uniformly over the xi it spans. Returns dsigma/dxi on the XI points [Mpc^2].
    Fixes both legacy flaws (branch merging, caustic under-resolution)."""
    C, rs, ks, fC = nfw_params(M, zl, zs)
    x = _refined_xgrid(ks)
    k, g = kappa_gamma(x, ks)
    detA = (1-k)**2 - g*g
    good = detA > 1e-300
    xi = np.where(good, -log(np.where(good, detA, 1.0)), np.nan)
    xa, xb = x[:-1], x[1:]
    fa, fb = xi[:-1], xi[1:]
    ok = np.isfinite(fa) & np.isfinite(fb)
    xa, xb, fa, fb = xa[ok], xb[ok], fa[ok], fb[ok]
    lo = np.minimum(fa, fb)
    hi = np.maximum(fa, fb)
    dsig = pi*rs*rs*(xb*xb - xa*xa)            # Mpc^2, exact annulus area
    keep = (hi > _XIE[0]) & (lo < _XIE[-1]) & (dsig > 0)
    if not np.any(keep):
        return np.zeros_like(XI)
    lo, hi, dsig = lo[keep], hi[keep], dsig[keep]
    den = np.maximum(hi - lo, 1e-300)
    # cumulative deposited area below each bin edge, then per-bin difference
    frac = np.clip((_XIE[:, None] - lo[None, :])/den[None, :], 0.0, 1.0)
    cum = frac @ dsig
    return np.diff(cum)/_XIW

# ============================ 5. jump measure R(xi) =========================
# XI extended to 30 (2026-07-24): the near-caustic exponential floor of the
# single-lens cross-section (cluster Einstein radii) lives at xi ~ 1-30; the
# original spec cap of 4 truncated it. Spec checkpoints (xi <= 0.8) unaffected.
XI = np.logspace(-7, np.log10(30.0), 560)
_lnXI = log(XI)
# geometric bin edges around the XI points, for the deposit-binned R
_XIE = np.empty(XI.size + 1)
_XIE[1:-1] = sqrt(XI[1:] * XI[:-1])
_XIE[0] = XI[0]**2 / _XIE[1]
_XIE[-1] = XI[-1]**2 / _XIE[-2]
_XIW = np.diff(_XIE)

def R_of_xi(zs, Mmin=1e7, Mmax=1e16, Nz=40, NM=48, legacy=False):
    zgrid = np.linspace(1e-3, zs-1e-3, Nz)
    dz = zgrid[1]-zgrid[0]
    Mgrid = np.logspace(np.log10(Mmin), np.log10(Mmax), NM)
    lnM = log(Mgrid); dlnM = lnM[1]-lnM[0]
    R = np.zeros_like(XI)
    for z in zgrid:
        wz = (1+z)**2 * ckms/Hz(z) * dz          # path measure
        for M in Mgrid:
            if legacy:
                xi, dsig = dsigma_dxi_curve(M, z, zs)
                if xi.size < 5:
                    continue
                order = np.argsort(xi)
                xi_s, ds_s = xi[order], dsig[order]
                lnds = np.interp(_lnXI, log(xi_s), log(ds_s),
                                 left=-np.inf, right=-np.inf)
                dsig_XI = np.where(np.isfinite(lnds), exp(lnds), 0.0)
            else:
                dsig_XI = dsigma_bins(M, z, zs)
            R += wz * dndlnM(M, z) * dlnM * dsig_XI
    return R

# ============================ 6. Levy-Khintchine + sigma_DL =================
def sigma_DL_over_DL(zs, R=None):
    if R is None:
        R = R_of_xi(zs)
    Rs = exp(-XI)*R                              # source-plane Esscher tilt
    def Ltil(t):
        integ = Rs*(exp(-t*XI) - 1 + t*XI)
        return _trapz(integ, XI)
    val = exp(Ltil(1.0) - 2*Ltil(0.5)) - 1.0
    return sqrt(val)

# ============================ checkpoints ==================================
def slope_R(R):
    return -np.gradient(log(R), _lnXI)

if __name__ == "__main__":
    print("== 0. growth ==")
    print(f"  D(0.5) = {D(0.5):.4f}   (spec 0.7689)")

    print("== 1. sigma(M) ==")
    s12 = sigmaM(1e12, 0.0)
    print(f"  sigma(1e12,z=0) = {s12:.3f}   (spec 2.223)")
    print(f"  nu(1e12,z=0)    = {(dc/s12)**2:.3f}   (spec 0.575)")

    print("== 3. NFW checkpoint (M=1e12, zl=0.5, zs=2) ==")
    C, rs, ks, fC = nfw_params(1e12, 0.5, 2.0)
    print(f"  C={C:.2f} (6.76)  rs={rs*1e3:.2f} kpc (25.98)  ks={ks:.4f} (0.0495)")

    print("== 5. R(xi) at zs=2 ==")
    R2 = R_of_xi(2.0)
    sl = slope_R(R2)
    tab = [(1e-4,8.35e6,2.02),(1e-3,7.82e4,2.06),(1e-2,618,2.11),
           (0.05,15.1,2.39),(0.1,2.47,2.69),(0.2,0.323,3.46),
           (0.4,2.81e-2,3.90),(0.8,1.25e-3,3.60)]
    print("   xi        R(spec)     R(calc)     ratio    slope(spec) slope(calc)")
    for xiv, Rsp, ssp in tab:
        j = np.argmin(np.abs(XI-xiv))
        print(f"  {xiv:7.0e}  {Rsp:10.3e}  {R2[j]:10.3e}  {R2[j]/Rsp:6.3f}   "
              f"{ssp:5.2f}      {sl[j]:5.2f}")

    print("== 6. sigma_DL/DL(z) ==")
    spec = {0.5:0.0114, 1:0.0233, 2:0.0412, 5:0.0682, 10:0.0848}
    print("   zs   spec     calc     ratio")
    for zs in (0.5, 1, 2, 5, 10):
        v = sigma_DL_over_DL(zs)
        print(f"  {zs:4.1f}  {spec[zs]:.4f}  {v:.4f}  {v/spec[zs]:.3f}")
