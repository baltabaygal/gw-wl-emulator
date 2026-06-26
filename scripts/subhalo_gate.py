"""
Section-4 gate, baseline 2: the M_min (abundance) degeneracy.

Vectorized/spline reimplementation of the host convergence cumulants so we can
re-solve kappa_thr (Nhalos=100) and recompute <kappa^2>,<kappa^3> across a range
of host-mass cutoffs M_min -- and ask whether the clump signal Delta<kappa^n>_c is
distinguishable from simply moving M_min (rerunning Plnmuf with a different cutoff).

Validated against scripts/subhalo_screen.py at the fiducial M_min=1e7:
   kappa_thr ~ 1.276e-4,  <k^2>_host ~ 7.30e-4,  <k^3>_host ~ 7.42e-5.
Clump correction (from that run): Delta<k^2>_c=2.85e-5, Delta<k^3>_c=1.33e-6.
"""
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

# ---------------- cosmology (code-matched) ----------------
Om, sigma8, h, ns, Ob = 0.315, 0.811, 0.674, 0.965, 0.0493
zeq = 3402.0; OmR = Om/(1+zeq); OmL = 1-Om-OmR
H0 = 0.000102247*h; CH = 306.535
rho_c0 = 277.394*h**2; rho_m0 = Om*rho_c0
deltac0 = 3.0/5.0*(3*np.pi/2)**(2/3)
def Az(z):  return Om*(1+z)**3 + OmR*(1+z)**4 + OmL
def Hz(z):  return H0*np.sqrt(Az(z))
def OmegaMz(z): return Om*(1+z)**3/Az(z)
def OmegaLz(z): return OmL/Az(z)
def Dg(z):
    omz, olz = OmegaMz(z), OmegaLz(z)
    return 2.5*omz/(omz**(4/7)-olz+(1+omz/2)*(1+olz/70))/(1+z)/0.7869370293916
def deltac(z): return deltac0/Dg(z)
def rho_cz(z): return Az(z)*rho_c0
def Dc(z): return quad(lambda zp: CH/Hz(zp), 0, z)[0]
def DL(z): return (1+z)*Dc(z)
def Sigma_crit(zs, zl):
    DsA = DL(zs)/(1+zs)**2; DlA = DL(zl)/(1+zl)**2
    DlsA = DsA - DlA*(1+zl)/(1+zs)
    return 2.08871e16*DsA/(4*np.pi*DlA*DlsA)

# ---------------- sigma(M) EH98 ----------------
def T_EH98(k):
    th = 2.728/2.7; Omh2, Obh2 = Om*h*h, Ob*h*h
    s = 44.5*np.log(9.83/Omh2)/np.sqrt(1+10*Obh2**0.75)
    ag = 1-0.328*np.log(431*Omh2)*Ob/Om+0.38*np.log(22.3*Omh2)*(Ob/Om)**2
    Gam = Om*h*(ag+(1-ag)/(1+(0.43*k*s*h)**4)); q = k/h*th**2/Gam
    L0 = np.log(2*np.e+1.8*q); C0 = 14.2+731/(1+62.5*q)
    return L0/(L0+C0*q*q)
def _s2R(R_Mpc):
    R = R_Mpc/h
    f = lambda lk: (np.exp(lk)**3*(np.exp(lk)**ns*T_EH98(np.exp(lk))**2))/(2*np.pi**2)*\
        (3*(np.sin(np.exp(lk)*R)-np.exp(lk)*R*np.cos(np.exp(lk)*R))/(np.exp(lk)*R)**3)**2
    return quad(f, np.log(1e-4), np.log(1e3), limit=200)[0]
_norm = sigma8**2/_s2R(8.0)
_lMg = np.linspace(np.log(1e5), np.log(1e17), 240)
_sg = np.array([np.sqrt(_norm*_s2R((3*np.exp(l)/(4*np.pi*rho_m0))**(1/3)/1000*h)) for l in _lMg])
_sig = CubicSpline(_lMg, _sg)
def sigA(M): return _sig(np.log(M))
def dsigdMA(M): return _sig(np.log(M), 1)/M

# ---------------- HMF pFC ----------------
from scipy.special import gamma as _G
def pFC(delta, S):
    p, q = 0.3, 0.8
    A = 1.0/(1+2.0**(-p)*_G(0.5-p)/np.sqrt(np.pi))
    nu2 = delta**2/S
    return A*(1+(q*nu2)**(-p))*np.sqrt(q*nu2/(2*np.pi))*np.exp(-q*nu2/2)/S

# ---------------- NFW + kappa shape ----------------
def conc(M, z):
    a = 0.520+(0.905-0.520)*np.exp(-0.617*z**1.21); b = -0.101+0.026*z
    return 10.0**(a+b*np.log10(M*h/1e12))
def Fg_kappa(x):
    x = np.asarray(x, float); out = np.empty_like(x)
    lo, hi, eq = x < 1-1e-7, x > 1+1e-7, np.abs(x-1) <= 1e-7
    xl, xh = x[lo], x[hi]
    out[lo] = (1-2*np.arctanh(np.sqrt((1-xl)/(1+xl)))/np.sqrt(1-xl**2))/(xl**2-1)
    out[hi] = (1-2*np.arctan(np.sqrt((xh-1)/(1+xh)))/np.sqrt(xh**2-1))/(xh**2-1)
    out[eq] = 1.0/3.0
    return out

# spline: cumulative J_n(xmax)=int_0^xmax Fg^n x dx  (integrate in ln x: Fg^n x^2 dlnx)
_xg = np.logspace(-7, 3, 6000); _lxg = np.log(_xg); _fg = Fg_kappa(_xg)
def _cumJ(n):
    integ = _fg**n*_xg**2
    c = np.concatenate([[0], np.cumsum(0.5*(integ[1:]+integ[:-1])*np.diff(_lxg))])
    return CubicSpline(_lxg, c)
_J2, _J3 = _cumJ(2), _cumJ(3)
def J2(xmax): return _J2(np.log(np.clip(xmax, _xg[0], _xg[-1])))
def J3(xmax): return _J3(np.log(np.clip(xmax, _xg[0], _xg[-1])))
# inverse of Fg (monotonic decreasing): x given Fg-value
_fr = _fg[::-1]; _xr = _xg[::-1]
def Fg_inv(T):  # returns x s.t. Fg(x)=T ; clip to grid
    return np.interp(np.clip(T, _fr[0], _fr[-1]), _fr, _xr)

# ---------------- build static grid (z<zs) ----------------
ZS = 1.0
Mlist = np.logspace(7, 17, 100); zlist = np.logspace(np.log10(0.01), np.log10(10.01), 100)
dlnM = np.log(Mlist[1]/Mlist[0])
idx = np.where(zlist < ZS)[0]; idx = idx[idx >= 1]      # mirror C++ loop (jz from 1)
Z = zlist[idx]; dz = zlist[idx] - zlist[idx-1]
nz, nM = len(Z), len(Mlist)
ZZ, MM = np.meshgrid(Z, Mlist, indexing='ij')        # (nz, nM)
sigM = sigA(Mlist); dsigM = dsigdMA(Mlist)
deltacZ = deltac(Z); SigcZ = np.array([Sigma_crit(ZS, z) for z in Z]); HzZ = Hz(Z)
cc = conc(MM, ZZ)
r200 = (3*MM/(4*np.pi*200*rho_cz(ZZ)))**(1/3)
rs = r200/cc
rhos = 200*rho_cz(ZZ)*cc**3/(3*(np.log(1+cc)-cc/(1+cc)))
k0 = rs*rhos/SigcZ[:, None]
SS = (sigM**2)[None, :]
nu2 = deltacZ[:, None]**2/SS
p, q = 0.3, 0.8; Apf = 1.0/(1+2.0**(-p)*_G(0.5-p)/np.sqrt(np.pi))
pfc = Apf*(1+(q*nu2)**(-p))*np.sqrt(q*nu2/(2*np.pi))*np.exp(-q*nu2/2)/SS
dndlnM = rho_m0*pfc*np.abs(2*sigM*dsigM)[None, :]      # (nz,nM)

def host_moments(Mmin, Mmax=1e17, kthr=None, solve=True):
    msel = (Mlist >= Mmin) & (Mlist <= Mmax)
    k0s, rss = k0[:, msel], rs[:, msel]
    dn = dndlnM[:, msel]; Ms = Mlist[msel]
    dzc = dz[:, None]; oneZ = (1+Z)[:, None]; Hz_c = HzZ[:, None]
    def counts_and_moments(kt):
        T = kt/(2*k0s)
        x = Fg_inv(T); rmax = rss*x                       # (nz,nMsel)
        good = (T < _fr[-1]) & (rmax > 0)
        barNH = np.where(good, CH*np.pi*(oneZ*rmax)**2/Hz_c*dn*dlnM*dzc, 0.0)
        xmax = np.where(good, rmax/rss, _xg[0])
        c2 = barNH*(2*k0s)**2*2*(rss/np.where(rmax>0, rmax, 1))**2*J2(xmax)
        c3 = barNH*(2*k0s)**3*2*(rss/np.where(rmax>0, rmax, 1))**2*J3(xmax)
        return barNH.sum(), c2.sum(), c3.sum()
    if solve:
        kthr = 10**brentq(lambda lk: counts_and_moments(10**lk)[0]-100.0, -8, -1, xtol=1e-4)
    N, K2, K3 = counts_and_moments(kthr)
    return kthr, N, K2, K3

# ---------------- fiducial validation ----------------
kf, Nf, K2f, K3f = host_moments(1e7)
print("VALIDATION (M_min=1e7):  kappa_thr=%.3e  N=%.1f  <k2>_H=%.3e  <k3>_H=%.3e"
      % (kf, Nf, K2f, K3f))
print("   (screen.py gave        kappa_thr=1.276e-4         <k2>_H=7.30e-4   <k3>_H=7.42e-5)\n")

# clump correction (from subhalo_screen.py run)
dK2c, dK3c = 2.850e-5, 1.327e-6

# ---------------- M_min scan ----------------
print("M_min scan  (kappa_thr re-solved for Nhalos=100 each time):")
print(" %-9s %-11s %-11s %-11s %-9s %-9s" % ("Mmin", "kappa_thr", "<k2>_H", "<k3>_H",
                                              "d<k2>/k2", "d<k3>/k3"))
Mmins = [1e6, 1e7, 3e7, 1e8, 3e8, 1e9, 3e9, 1e10, 1e11, 1e12]
res = []
for Mm in Mmins:
    kt, N, K2, K3 = host_moments(Mm)
    res.append((Mm, kt, K2, K3))
    print(" %-9.0e %-11.3e %-11.3e %-11.3e %+8.2f%% %+8.2f%%"
          % (Mm, kt, K2, K3, 100*(K2/K2f-1), 100*(K3/K3f-1)))

print("\n----- interpretation -----")
print("clump  Delta<k2>_c/<k2> = %+.2f%%   Delta<k3>_c/<k3> = %+.2f%%" %
      (100*dK2c/(K2f), 100*dK3c/K3f))
# equivalent M_min change that reproduces the clump <k3> shift
K3s = np.array([r[3] for r in res]); Ms = np.array([r[0] for r in res])
print("how much does <k3> move per decade of M_min near the lensing scale?")
for i in range(len(res)-1):
    if Ms[i] >= 1e9:
        dlogM = np.log10(Ms[i+1]/Ms[i])
        dk3 = (K3s[i+1]-K3s[i])/K3f
        print("   M_min %.0e->%.0e (%.2f dex):  d<k3>/<k3> = %+.2f%%   (clump = %+.2f%%)"
              % (Ms[i], Ms[i+1], dlogM, 100*dk3, 100*dK3c/K3f))
np.save('plots/mmin_scan.npy', np.array(res))
print("\nsaved plots/mmin_scan.npy")
