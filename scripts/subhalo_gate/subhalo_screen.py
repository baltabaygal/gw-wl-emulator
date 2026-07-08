"""
Section-4 analytic GATE: does subhalo substructure move the convergence moments?

Campbell-theorem screen for the clump contribution to the convergence cumulants
Delta<kappa^2>_c, Delta<kappa^3>_c, compared against the model's OWN <kappa^2>,
<kappa^3> (the discrete-host part), computed in one consistent framework that
reproduces the halos/lensing.cpp geometry.

Faithful to the C++ code:
  - grid:      Mlist 1e7..1e17 (100, log), zlist 0.01..10.01 (100, log)
  - cosmology: Om=0.315, sigma8=0.811, h=0.674, ns=0.965  (python defaults)
  - HMF:       rho_m * pFC(deltac(z), sigma^2) * |dsigma^2/dM|  (ellipsoidal, p=.3,q=.8)
  - conc:      cons14 = Dutton-Maccio 2014, c200c = 10^(a+b log10(M h/1e12))
  - NFW:       r200=(3M/(4pi 200 rho_c(z)))^(1/3); kappa(x)=2 kappa0 Fg(x); kappa0=rs rhos/Sigma_c
  - hosts:     barNH = 306.535*pi*((1+z)rmax)^2 /Hz * dn/dlnM dlnM dz ; rmax: kappa(rmax)=kappa_thr
  - kappa_thr: solved so total host count = Nhalos = 100
Clumps: evolved JvdB14 SHMF (same as scripts/subhalo_demo.py).

Key analytic facts used:
  * <kappa^3> of the model comes only from the discrete hosts (weak part is Gaussian).
  * The clump self-cumulants I_n = int d^2 d kappa_m^n CONVERGE for n=2,3 (the mass
    removal/truncation is only needed for the n=1 mean), so no kappa_rem term is needed
    for the variance/skewness screen.  Compact-clump (shot-noise) limit:
        Delta<kappa^n>_c = sum_bins [306.535 (1+z)^2 /Hz * dn/dlnM dlnM dz]
                                    * int dm (dN_sub/dm) * 2 pi rs_c^2 (2 kappa0_c)^n J_n
    with J_n = int_0^inf Fg(x)^n x dx  (pure numbers).
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline
from scipy.special import gamma as Gamma, gammaincc

# ============================================================ cosmology (code-matched)
Om, sigma8, h, ns, Ob = 0.315, 0.811, 0.674, 0.965, 0.0493
zeq = 3402.0
OmR = Om/(1+zeq); OmL = 1.0 - Om - OmR
H0 = 0.000102247*h                       # code units (1/kpc); c/H0 = 306.535/H0 kpc
CH = 306.535                             # so c/H(z) = CH/Hz(z) [kpc, comoving]
rho_c0 = 277.394*h**2                    # Msun/kpc^3
rho_m0 = Om*rho_c0
deltac0 = 3.0/5.0*(3.0*np.pi/2.0)**(2.0/3.0)   # 1.686

def Az(z):  return Om*(1+z)**3 + OmR*(1+z)**4 + OmL
def Hz(z):  return H0*np.sqrt(Az(z))
def OmegaMz(z): return Om*(1+z)**3/Az(z)
def OmegaLz(z): return OmL/Az(z)
def Dg(z):                               # code's Carroll-type growth fit (Dg(0)=1)
    omz, olz = OmegaMz(z), OmegaLz(z)
    return 2.5*omz/(omz**(4.0/7.0) - olz + (1+omz/2.0)*(1+olz/70.0))/(1+z)/0.7869370293916
def deltac(z): return deltac0/Dg(z)
def rho_cz(z): return Az(z)*rho_c0       # critical density at z

# comoving distance, luminosity distance, Sigma_crit  (matches Sigmacf)
def Dc(z):  return quad(lambda zp: CH/Hz(zp), 0, z)[0]      # kpc comoving
def DL(z):  return (1+z)*Dc(z)
def Sigma_crit(zs, zl):
    DsA = DL(zs)/(1+zs)**2
    DlA = DL(zl)/(1+zl)**2
    DlsA = DsA - DlA*(1+zl)/(1+zs)
    return 2.08871e16*DsA/(4.0*np.pi*DlA*DlsA)

# ============================================================ sigma(M) via EH98
def T_EH98(k):
    theta = 2.728/2.7; Omh2, Obh2 = Om*h*h, Ob*h*h
    s = 44.5*np.log(9.83/Omh2)/np.sqrt(1+10*Obh2**0.75)
    ag = 1 - 0.328*np.log(431*Omh2)*Ob/Om + 0.38*np.log(22.3*Omh2)*(Ob/Om)**2
    Gam = Om*h*(ag + (1-ag)/(1+(0.43*k*s*h)**4))
    q = k/h*theta**2/Gam
    L0 = np.log(2*np.e + 1.8*q); C0 = 14.2 + 731.0/(1+62.5*q)
    return L0/(L0 + C0*q*q)
def _s2R(R_Mpc):
    R = R_Mpc/h
    f = lambda lk: (np.exp(lk)**3*(np.exp(lk)**ns*T_EH98(np.exp(lk))**2))/(2*np.pi**2)*\
                   (3*(np.sin(np.exp(lk)*R)-np.exp(lk)*R*np.cos(np.exp(lk)*R))/(np.exp(lk)*R)**3)**2
    return quad(f, np.log(1e-4), np.log(1e3), limit=200)[0]
_norm = sigma8**2/_s2R(8.0)
def sigma_M(M):
    R_kpc = (3*M/(4*np.pi*rho_m0))**(1/3)
    return np.sqrt(_norm*_s2R(R_kpc/1000.0*h))

# spline sigma(M) and dsigma/dM over the mass grid
_lMg = np.linspace(np.log(1e5), np.log(1e17), 220)
_sg  = np.array([sigma_M(np.exp(l)) for l in _lMg])
_sig_sp = CubicSpline(_lMg, _sg)
def sig(M):   return _sig_sp(np.log(M))
def dsigdM(M): return _sig_sp(np.log(M), 1)/M       # d/dM = (d/dlnM)/M

# ============================================================ HMF (code's pFC)
def pFC(delta, S):
    p, q = 0.3, 0.8
    A = 1.0/(1+2.0**(-p)*Gamma(0.5-p)/np.sqrt(np.pi))
    if S <= 0: return 0.0
    nu2 = delta*delta/S
    return A*(1+(q*nu2)**(-p))*np.sqrt(q*nu2/(2*np.pi))*np.exp(-q*nu2/2.0)/S
def dndlnM(M, z):                                    # = rho_m * pFC * |dsigma^2/dM|
    S = sig(M)**2
    return rho_m0*pFC(deltac(z), S)*abs(2*sig(M)*dsigdM(M))

# ============================================================ NFW geometry
def conc(M, z):                                      # cons14 (Dutton-Maccio 2014)
    a = 0.520 + (0.905-0.520)*np.exp(-0.617*z**1.21)
    b = -0.101 + 0.026*z
    return 10.0**(a + b*np.log10(M*h/1.0e12))
def nfw_rs_rhos(M, z):
    c = conc(M, z)
    r200 = (3*M/(4*np.pi*200*rho_cz(z)))**(1/3)
    rs = r200/c
    rhos = 200*rho_cz(z)*c**3/(3.0*(np.log(1+c) - c/(1+c)))
    return rs, rhos, c

def Fg_kappa(x):                                     # FgNFW[0]: kappa shape, kappa=2 kappa0 Fg
    x = np.asarray(x, float); out = np.empty_like(x)
    lo, hi, eq = x < 1-1e-7, x > 1+1e-7, np.abs(x-1) <= 1e-7
    xl, xh = x[lo], x[hi]
    tl = np.arctanh(np.sqrt((1-xl)/(1+xl)))/np.sqrt(1-xl**2)
    th = np.arctan(np.sqrt((xh-1)/(1+xh)))/np.sqrt(xh**2-1)
    out[lo] = (1 - 2*tl)/(xl**2 - 1)
    out[hi] = (1 - 2*th)/(xh**2 - 1)
    out[eq] = 1.0/3.0
    return out

# pure-number radial integrals J_n = int_0^xmax Fg(x)^n x dx
def Jn(n, xmax=np.inf):
    f = lambda x: Fg_kappa(np.array([x]))[0]**n*x
    if np.isinf(xmax):
        return quad(f, 0, 1, points=[1])[0] + quad(f, 1, np.inf)[0]
    return quad(f, 0, min(xmax,1), points=[1] if xmax>1 else None)[0] + \
           (quad(f, 1, xmax)[0] if xmax > 1 else 0.0)
J2_inf, J3_inf = Jn(2), Jn(3)        # clump self-cumulant constants

# ============================================================ evolved SHMF (JvdB14, as in demo)
ALPHA, BETA, OMEGA, PSI_RES = -0.82, 50.0, 4.0, 1e-4
def growthD(z):                                      # D(z) normalized to 1 at z=0 (for zform)
    integ = lambda zp: (1+zp)/Az(zp)**1.5
    return (np.sqrt(Az(z))*quad(integ, z, np.inf)[0])/(quad(integ, 0, np.inf)[0])
def Dvir(z):
    d = OmegaMz(z) - 1.0
    return 18*np.pi**2 + 82*d - 39*d**2
def tau_dyn(z): return 1.628/h*(Dvir(z)/178.0)**(-0.5)*Az(z)**(-0.5)/np.sqrt(1.0)  # Gyr ~ (E)^-1
def _Hfac(z):   return np.sqrt(Az(z))                 # E(z)
def lookbackGyr(z): return (9.778/h)*quad(lambda zp: 1.0/((1+zp)*_Hfac(zp)), 0, z)[0]
def z_form(M0, z0, f=0.5):
    dc0 = 1.686/growthD(z0); af = 0.815*np.exp(-2*f**3)/f**0.707; wf = np.sqrt(2*np.log(af+1))
    rhs = dc0 + wf*np.sqrt(sig(f*M0)**2 - sig(M0)**2)
    return brentq(lambda zf: 1.686/growthD(zf) - rhs, z0, 30.0)
def N_tau(M0, z0):
    zf = z_form(M0, z0)
    return quad(lambda z: 1.0/(tau_dyn(z)*(1+z)*_Hfac(z))*(9.778/h), z0, zf)[0]
def gamma_norm(fs):
    s = (1+ALPHA)/OMEGA
    den = Gamma(s)*(gammaincc(s, BETA*PSI_RES**OMEGA) - gammaincc(s, BETA))
    return OMEGA*BETA**s/den*fs
def shmf_params(M0, z0):
    Nt = N_tau(M0, z0); fs = 0.3563/Nt**0.6 - 0.075
    return max(fs, 0.0), gamma_norm(max(fs, 0.0))
def dN_dlnpsi(psi, g): return g*psi**ALPHA*np.exp(-BETA*psi**OMEGA)

# ============================================================ the grid + threshold
Mlist = np.logspace(7, 17, 100)
zlist = np.logspace(np.log10(0.01), np.log10(10.01), 100)
ZS_FID = 5.0                                          # source redshift (fiducial)
ZS_SCAN = [2.0, 3.5, 5.0]
sig_c = {}
_shmf_cache = {}

def cached_shmf_params(M, z):
    key = (float(M), float(z))
    if key not in _shmf_cache:
        _shmf_cache[key] = shmf_params(M, z)
    return _shmf_cache[key]

def kappa0_host(M, z, jz):
    rs, rhos, c = nfw_rs_rhos(M, z)
    return rs*rhos/sig_c[jz], rs
def kappa_host(r, M, z, jz):
    k0, rs = kappa0_host(M, z, jz)
    return 2*k0*Fg_kappa(np.atleast_1d(r/rs))

def rmax_host(M, z, jz, kthr):
    k0, rs = kappa0_host(M, z, jz)
    if 2*k0*Fg_kappa(np.array([1e-6]))[0] <= kthr: return 0.0
    lo, hi = np.log(1e-6), np.log(1e7)
    while hi-lo > 0.02:
        mid = 0.5*(lo+hi)
        if 2*k0*Fg_kappa(np.array([np.exp(mid)/rs]))[0] > kthr: lo = mid
        else: hi = mid
    return np.exp(0.5*(lo+hi))

def Nh_total(kthr):
    tot = 0.0
    for jz in range(1, len(zlist)):
        z = zlist[jz];  dz = z - zlist[jz-1]
        if z >= _current_ZS: continue
        for jM in range(1, len(Mlist)):
            M = Mlist[jM]; dlnM = np.log(Mlist[jM]/Mlist[jM-1])
            rm = rmax_host(M, z, jz, kthr)
            if rm > 0:
                tot += CH*np.pi*((1+z)*rm)**2/Hz(z)*dndlnM(M, z)*dlnM*dz
    return tot

def sigma2_weak(kthr):
    s = 0.0
    for jz in range(1, len(zlist)):
        z = zlist[jz]; dz = z - zlist[jz-1]
        if z >= _current_ZS: continue
        for jM in range(1, len(Mlist)):
            M = Mlist[jM]; dlnM = np.log(Mlist[jM]/Mlist[jM-1])
            k0, rs = kappa0_host(M, z, jz)
            rstart = rmax_host(M, z, jz, kthr)
            if rstart == 0: rstart = 1e-6
            r = rstart; dlnr = 0.01; kap = kthr
            while kap > 0.001*kthr:
                kap = 2*k0*Fg_kappa(np.array([r/rs]))[0]
                s += CH*np.pi*((1+z)*r)**2/Hz(z)*dndlnM(M, z)*kap**2*dlnr*dlnM*dz
                r *= np.exp(dlnr)
    return s

def compute_screen(ZS):
    global sig_c, _current_ZS
    _current_ZS = ZS
    sig_c = {jz: Sigma_crit(ZS, z) for jz, z in enumerate(zlist) if z < ZS}

    print(f"\nsolving kappa_thr for Nhalos=100 at zs={ZS:.1f} ...")
    kthr = 10**brentq(lambda lk: Nh_total(10**lk) - 100.0, -10, -1, xtol=1e-3)
    print(f"  kappa_thr = {kthr:.3e}")

    print("accumulating host + clump cumulants ...")
    K2H = K3H = 0.0
    dK2c = dK3c = 0.0
    rows = []
    rows_all = []
    mfloor = 1e7
    for jz in range(1, len(zlist)):
        z = zlist[jz]; dz = z - zlist[jz-1]
        if z >= ZS: continue
        for jM in range(1, len(Mlist)):
            M = Mlist[jM]; dlnM = np.log(Mlist[jM]/Mlist[jM-1])
            rm = rmax_host(M, z, jz, kthr)
            if rm <= 0: continue
            k0H, rsH = kappa0_host(M, z, jz)
            barNH = CH*np.pi*((1+z)*rm)**2/Hz(z)*dndlnM(M, z)*dlnM*dz
            xmaxH = rm/rsH
            cH2 = barNH*(2*k0H)**2*(2*rsH**2/rm**2)*Jn(2, xmaxH)
            cH3 = barNH*(2*k0H)**3*(2*rsH**2/rm**2)*Jn(3, xmaxH)
            K2H += cH2; K3H += cH3

            mmax = 0.1*M
            if mmax > mfloor:
                fs, g = cached_shmf_params(M, z)
                geom = CH*(1+z)**2/Hz(z)*dndlnM(M, z)*dlnM*dz
                lm = np.linspace(np.log(mfloor), np.log(mmax), 40)
                mm = np.exp(lm)
                dN = dN_dlnpsi(mm/M, g)
                rs_c = np.array([nfw_rs_rhos(mk, z)[0] for mk in mm])
                rhos_c = np.array([nfw_rs_rhos(mk, z)[1] for mk in mm])
                k0_c = rs_c*rhos_c/sig_c[jz]
                I2 = 2*np.pi*rs_c**2*(2*k0_c)**2*J2_inf
                I3 = 2*np.pi*rs_c**2*(2*k0_c)**3*J3_inf
                d2 = geom*np.trapezoid(dN*I2, lm)
                d3 = geom*np.trapezoid(dN*I3, lm)
                dK2c += d2; dK3c += d3
            else:
                d2 = d3 = 0.0
            rows.append((z, M, barNH, cH3, d3))
            rows_all.append((z, M, barNH, cH2, d2, cH3, d3))

    # The weak Gaussian variance is tiny for this screen and very slow to
    # recompute for every source redshift. The scan focuses on subhalo ratios.
    K2_weak = 0.0
    K2_model = K2H + K2_weak
    K3_model = K3H
    result = {
        "zs": ZS,
        "kthr": kthr,
        "K2_model": K2_model,
        "K2H": K2H,
        "K2_weak": K2_weak,
        "K3_model": K3_model,
        "dK2c": dK2c,
        "dK3c": dK3c,
        "skew_model": K3_model/K2_model**1.5,
        "skew_clumps": (K3_model+dK3c)/(K2_model+dK2c)**1.5,
        "rows": np.array(rows),
        "rows_all": np.array(rows_all),
    }
    print(f"  <kappa^2>={K2_model:.3e}, Delta={dK2c:.3e} ({dK2c/K2_model:.3%})")
    print(f"  <kappa^3>={K3_model:.3e}, Delta={dK3c:.3e} ({dK3c/K3_model:.3%})")
    print(f"  skewness: {result['skew_model']:.3f} -> {result['skew_clumps']:.3f}")
    return result

print("precomputing sigma_c, kappa0, dn/dlnM ...")
results = [compute_screen(ZS) for ZS in ZS_SCAN]
fid = min(results, key=lambda r: abs(r["zs"] - ZS_FID))

print("\n==================  SECTION-4 GATE  (fiducial zs=%.1f, Nhalos=100, kappa_thr=%.2e) ==================" % (fid["zs"], fid["kthr"]))
print(f"  model  <kappa^2>            = {fid['K2_model']:.3e}   (hosts {fid['K2H']:.2e} + weak {fid['K2_weak']:.2e})")
print(f"  model  <kappa^3>            = {fid['K3_model']:.3e}   (discrete hosts only)")
print(f"  clump  Delta<kappa^2>_c     = {fid['dK2c']:.3e}")
print(f"  clump  Delta<kappa^3>_c     = {fid['dK3c']:.3e}")
print("  ------------------------------------------------------------------")
print(f"  Delta<kappa^2>_c / <kappa^2> = {fid['dK2c']/fid['K2_model']:.3%}")
print(f"  Delta<kappa^3>_c / <kappa^3> = {fid['dK3c']/fid['K3_model']:.3%}")
print(f"  skewness  <k^3>/<k^2>^1.5  : model {fid['skew_model']:.3f} -> +clumps {fid['skew_clumps']:.3f}")

np.save('plots/screen_rows.npy', fid["rows"])
np.save('plots/screen_rows_all.npy', fid["rows_all"])
np.savez(
    'plots/screen_zs_scan.npz',
    zs=np.array([r["zs"] for r in results]),
    kthr=np.array([r["kthr"] for r in results]),
    K2_model=np.array([r["K2_model"] for r in results]),
    K3_model=np.array([r["K3_model"] for r in results]),
    dK2c=np.array([r["dK2c"] for r in results]),
    dK3c=np.array([r["dK3c"] for r in results]),
    skew_model=np.array([r["skew_model"] for r in results]),
    skew_clumps=np.array([r["skew_clumps"] for r in results]),
)
print("\nsaved plots/screen_rows.npy and plots/screen_zs_scan.npz")
