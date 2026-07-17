"""
Option-A external validation, layer 1 (internal exactness):
substructure convergence power spectrum P_sub(k) for one host —
MC point-process estimator vs our analytic Campbell-in-Fourier prediction.

Population spec == production model (psi_max=1, thinned SHMF, Han+16 anti-biased radial
profile, cons14 concentrations, corrected Giocoli w_f). Clump kappa profile: untruncated
NFW, as in cpp/subhalo.cpp; its 2D Fourier transform is computed by numerical Hankel
transform of 2*kappa0*f(R/rs).

Layer 2 (external overlay vs Diaz Rivero+ 2018 / ETHOS) is added after pinning their
aperture + normalization conventions from the papers.
"""
import numpy as np
from scipy.integrate import quad
from scipy.special import j0, gammaincc, gamma as G

# ---------------- cosmology (code-matched, as in subhalo_gate.py) ----------------
Om, sigma8, h, ns, Ob = 0.315, 0.811, 0.674, 0.965, 0.0493
zeq = 3402.0; OmR = Om/(1+zeq); OmL = 1-Om-OmR
H0 = 0.000102247*h; CH = 306.535
rho_c0 = 277.394*h**2; rho_m0 = Om*rho_c0
def Az(z):  return Om*(1+z)**3 + OmR*(1+z)**4 + OmL
def Hz(z):  return H0*np.sqrt(Az(z))
def OmegaMz(z): return Om*(1+z)**3/Az(z)
def OmegaLz(z): return OmL/Az(z)
def Dg(z):
    omz, olz = OmegaMz(z), OmegaLz(z)
    return 2.5*omz/(omz**(4/7)-olz+(1+omz/2)*(1+olz/70))/(1+z)/0.7869370293916
def deltac(z): return (3/5)*(3*np.pi/2)**(2/3)/Dg(z)
def Dc(z): return quad(lambda zp: CH/Hz(zp), 0, z)[0]
def DL(z): return (1+z)*Dc(z)
def Sigma_crit(zs, zl):
    DsA = DL(zs)/(1+zs)**2; DlA = DL(zl)/(1+zl)**2
    DlsA = DsA - DlA*(1+zl)/(1+zs)
    return 2.08871e16*DsA/(4*np.pi*DlA*DlsA)

# ---------------- sigma(M): EH98, code-matched ----------------
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
_lMg = np.linspace(np.log(1e5), np.log(1e17), 200)
_sg = np.array([np.sqrt(_norm*_s2R((3*np.exp(l)/(4*np.pi*rho_m0))**(1/3)/1000*h)) for l in _lMg])
def sig(M): return np.interp(np.log(M), _lMg, _sg)

# ---------------- SHMF (corrected w_f; JvdB14 with exp cutoff; psi_max=1) --------
ALPHA, BETA, OMEGA, PSI_RES, PSI_MAX = -0.82, 50.0, 4.0, 1e-4, 1.0
MFLOOR = 1e7
def z_form(M0, z0, f=0.5):
    af = 0.815*np.exp(-2*f**3)/f**0.707; wf = np.sqrt(2*np.log(af+1))
    rhs = deltac(z0) + wf*np.sqrt(sig(f*M0)**2 - sig(M0)**2)
    from scipy.optimize import brentq
    return brentq(lambda zf: deltac(zf) - rhs, z0, 30.0)
def N_tau(M0, z0):
    zf = z_form(M0, z0)
    f = lambda z: 6.006*np.sqrt((18*np.pi**2+82*(OmegaMz(z)-1)-39*(OmegaMz(z)-1)**2)/178)/(1+z)
    return quad(f, z0, zf)[0]
def gnorm_of(M0, z0):
    fs = 0.3563/N_tau(M0, z0)**0.6 - 0.075
    if fs <= 0: return 0.0, 0.0
    s = (1+ALPHA)/OMEGA
    den = G(s)*(gammaincc(s, BETA*PSI_RES**OMEGA) - gammaincc(s, BETA))
    return OMEGA*BETA**s/den*fs, fs

# ---------------- NFW structure (cons14 Dutton-Maccio, code-matched) -------------
def cons14(M, z):
    b = -0.101 + 0.026*z
    a = 0.520 + (0.905 - 0.520)*np.exp(-0.617*z**1.21)
    return 10**(a + b*np.log10(M/(1e12/h)))
def nfw_rs_rhos(m, z):
    rhoz = Az(z)*rho_c0
    c = cons14(m, z)
    r200 = (3*m/(4*np.pi*200*rhoz))**(1/3)
    rs = r200/c
    rhos = 200*rhoz*c**3*(1+c)/(3*((1+c)*np.log(1+c) - c))
    return rs, rhos, c, r200

def fNFW(x):
    x = np.atleast_1d(np.asarray(x, float)); out = np.empty_like(x)
    hi = x > 1; lo = x < 1; eq = ~hi & ~lo
    t = np.empty_like(x)
    t[hi] = np.arctan(np.sqrt((x[hi]-1)/(1+x[hi])))/np.sqrt(x[hi]**2-1)
    t[lo] = np.arctanh(np.sqrt((1-x[lo])/(1+x[lo])))/np.sqrt(1-x[lo]**2)
    out[hi] = (1-2*t[hi])/(x[hi]**2-1); out[lo] = (1-2*t[lo])/(x[lo]**2-1); out[eq] = 1/3
    return out

def u_tilde(k, rs, rhos, Sigc, xmax=200.0, nx=4000):
    """2D FT of the untruncated-NFW kappa profile, kappa(R)=2 kappa0 f(R/rs):
       u(k) = 2 pi rs^2 * 2 kappa0 * int_0^xmax f(x) J0(k rs x) x dx   [dimension: kpc^2·kappa]
       (xmax regulates the log tail; convergence checked vs xmax)."""
    k0 = rs*rhos/Sigc
    x = np.logspace(-4, np.log10(xmax), nx)
    fx = fNFW(x)
    kk = np.atleast_1d(k)
    out = np.array([np.trapezoid(fx*j0(ki*rs*x)*x, x) for ki in kk])
    return 2*np.pi*rs**2*2*k0*out

# ---------------- Han+16 anti-biased radial profile (code-matched) ---------------
def radial_cdf(c, Nx=4000):
    xs = np.linspace(0, 1, Nx)
    B = np.where(xs > 0, 1/np.sqrt((xs/0.54)**(-2.5)+1), 0.0)
    w = xs**2/(1+c*xs)**2*B
    cdf = np.concatenate([[0], np.cumsum(0.5*(w[1:]+w[:-1])*np.diff(xs))])
    return xs, cdf/cdf[-1]

# ================= configuration =================
M_HOST, ZL, ZS = 1e13, 0.5, 2.0
SEED, NREAL = 7, 1200
rng = np.random.default_rng(SEED)

Sigc = Sigma_crit(ZS, ZL)
g, fs_tot = gnorm_of(M_HOST, ZL)
rs_h, rhos_h, c_h, r200_h = nfw_rs_rhos(M_HOST, ZL)
xs_grid, cdf_grid = radial_cdf(c_h)
psi_lo = MFLOOR/M_HOST
pa_lo, pa_hi = psi_lo**ALPHA, PSI_MAX**ALPHA
Nprop = (g/ALPHA)*(pa_hi - pa_lo)
print(f"host M={M_HOST:.1e} zl={ZL} zs={ZS}: g={g:.3f}, f_s={fs_tot:.3f}, r200={r200_h:.1f} kpc, "
      f"proposal N={Nprop:.1f}")

# aperture: annulus 5..40 kpc from host center (avoids both the exact center and the outskirts)
R_IN, R_OUT = 5.0, 40.0
AREA = np.pi*(R_OUT**2 - R_IN**2)
# window-safe range: k >~ 2 pi / (R_OUT - R_IN); below that, aperture windowing is O(1)
KS = np.logspace(np.log10(2*np.pi/(R_OUT - R_IN)), np.log10(10.0), 12)   # 1/kpc

# ---------------- MC: draw populations, direct mode sum --------------------------
# P_hat(k) = < |sum_j u_j(k) e^{-i k.x_j}|^2 > / A  over clumps in the aperture,
# k directions averaged (8 azimuths).
mass_grid = np.logspace(np.log10(MFLOOR), np.log10(PSI_MAX*M_HOST), 40)
u_tab = {}   # per mass-grid point: u(k) table
for m in mass_grid:
    rs_c, rhos_c, _, _ = nfw_rs_rhos(m, ZL)
    u_tab[m] = u_tilde(KS, rs_c, rhos_c, Sigc)
def u_interp(m):
    i = np.clip(np.searchsorted(mass_grid, m), 1, len(mass_grid)-1)
    w = (np.log(m) - np.log(mass_grid[i-1]))/(np.log(mass_grid[i]) - np.log(mass_grid[i-1]))
    return (1-w)*u_tab[mass_grid[i-1]] + w*u_tab[mass_grid[i]]

phis = np.linspace(0, np.pi, 8, endpoint=False)
# accumulate complex mode amplitudes per (k, azimuth) so the MEAN FIELD (the smooth
# radial gradient of substructure density across the annulus) can be subtracted:
# P_connected = < |A - <A>|^2 > / Area,  A(k,phi) = sum_j u_j(k) e^{-i k.x_j}
A_sum = np.zeros((len(phis), len(KS)), dtype=complex)
A2_sum = np.zeros((len(phis), len(KS)))
n_in_acc = 0.0
for r in range(NREAL):
    Nc = rng.poisson(Nprop)
    if Nc == 0: continue
    u = rng.random(Nc)
    psi = (pa_lo + u*(pa_hi-pa_lo))**(1/ALPHA)
    keep = rng.random(Nc) <= np.exp(-BETA*psi**OMEGA)
    psi = psi[keep]
    if psi.size == 0: continue
    m = psi*M_HOST
    x3 = np.interp(rng.random(psi.size), cdf_grid, xs_grid)*r200_h
    cth = rng.uniform(-1, 1, psi.size)
    az = rng.uniform(0, 2*np.pi, psi.size)
    R2 = x3*np.sqrt(1-cth**2)
    X, Y = R2*np.cos(az), R2*np.sin(az)
    Rr = np.hypot(X, Y)
    sel = (Rr > R_IN) & (Rr < R_OUT)
    if not sel.any(): continue
    n_in_acc += sel.sum()
    uk = np.array([u_interp(mi) for mi in m[sel]])     # (Nsel, Nk)
    for ip, ph in enumerate(phis):
        kx, ky = np.cos(ph), np.sin(ph)
        phase = np.exp(-1j*np.outer(X[sel]*kx + Y[sel]*ky, KS))
        A = (uk*phase).sum(axis=0)                     # (Nk,) complex amplitude
        A_sum[ip] += A
        A2_sum[ip] += np.abs(A)**2
A_mean = A_sum/NREAL
P_mc = ((A2_sum/NREAL - np.abs(A_mean)**2)/AREA).mean(axis=0)   # connected, azimuth-avg
P_mc_err = ((A2_sum/NREAL - np.abs(A_mean)**2)/AREA).std(axis=0)/np.sqrt(len(phis))
print(f"MC: <N in aperture> = {n_in_acc/NREAL:.2f}")

# ---------------- analytic: Campbell-in-Fourier over the aperture ----------------
# P(k) = (1/A) int_ap d^2x  int dm  d2N/dm dA (x)  |u(k;m)|^2
# with d2N/dmdA(x) = (dN/dm)(m) * Sigma_n(R)/int Sigma_n dA  (projected number profile)
# projected number density profile Sigma_n(R): project the 3D anti-biased profile.
# dense inside the aperture (a coarse grid here caused a 9% count bias in v1)
Rg = np.concatenate([np.linspace(0.2, 60, 400), np.linspace(60.5, r200_h, 400)])
w3 = lambda x: np.where(x > 0, x**2/(1+c_h*x)**2/np.sqrt((x/0.54)**(-2.5)+1), 0.0)  # dN/dx (shell)
# 3D number density nu(x) ∝ w3(x)/x^2; project: Sigma_n(R) ∝ int nu(sqrt(R^2+z^2)) dz
xs3 = np.linspace(1e-4, 1, 2000)
nu3 = w3(xs3)/xs3**2
def Sigma_n(R):
    zmax = np.sqrt(max(r200_h**2 - R**2, 0.0))
    if zmax <= 0: return 0.0
    zz = np.linspace(0, zmax, 2000)
    x = np.sqrt(R**2 + zz**2)/r200_h
    return 2*np.trapezoid(np.interp(x, xs3, nu3), zz)
Sn = np.array([Sigma_n(R) for R in Rg])
norm3 = np.trapezoid(Sn*2*np.pi*Rg, Rg)     # total number normalization
sel_ap = (Rg > R_IN) & (Rg < R_OUT)
frac_ap = np.trapezoid(Sn[sel_ap]*2*np.pi*Rg[sel_ap], Rg[sel_ap])/norm3
# exact mean count in aperture (thinned SHMF):
lm = np.linspace(np.log(MFLOOR), np.log(PSI_MAX*M_HOST), 400)
mm = np.exp(lm); psim = mm/M_HOST
dNdlnm = g*psim**ALPHA*np.exp(-BETA*psim**OMEGA)
N_exact = np.trapezoid(dNdlnm, lm)
print(f"analytic: <N total> = {N_exact:.2f}, aperture fraction = {frac_ap:.4f}, "
      f"<N in aperture> = {N_exact*frac_ap:.2f}")
u2 = np.array([u_interp(mi)**2 for mi in mm])          # (Nm, Nk)
P_an = frac_ap*np.trapezoid(dNdlnm[:, None]*u2, lm, axis=0)/AREA

print(f"\n{'k [1/kpc]':>10} {'P_MC [kpc^2]':>13} {'+-':>9} {'P_analytic':>12} {'ratio':>7}")
for i, k in enumerate(KS):
    print(f"{k:10.4f} {P_mc[i]:13.4e} {P_mc_err[i]:9.1e} {P_an[i]:12.4e} {P_mc[i]/P_an[i]:7.3f}")

np.savez("/Users/baltabay/Desktop/gw-wl-emulator/tmp/psub_check.npz",
         ks=KS, P_mc=P_mc, P_an=P_an, M_host=M_HOST, zl=ZL, zs=ZS,
         R_in=R_IN, R_out=R_OUT, Nreal=NREAL)
print("\nsaved tmp/psub_check.npz")
