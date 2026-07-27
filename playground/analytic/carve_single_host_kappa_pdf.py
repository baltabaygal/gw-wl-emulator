#!/usr/bin/env python
"""
Single-host convergence-PDF comparison of the host-carving schemes, with a REAL
NFW kernel, against a CARVED BRUTE reference (all clumps explicit, host = M - sum_all).

Methods (all share the SAME clump model, so brute<->analytic agree by construction
in mean/var; the test isolates the mass-bookkeeping layer):

  (0) current model 3 : host at fixed (1-f_b)M ; resolved clumps explicit ;
                        unresolved = mu_U(r) + sigma_U(r) N(0,1).  [total mass wiggles]
  (A) realized carve  : host at M - sum_res - <M_U> ; resolved explicit ; kappa_U marginal.
  (C) proportional    : host at M - sum_res - M_U_target ; kappa_U conditioned on M_U_target.
  (brute, carved)     : host at M - sum_ALL ; ALL clumps (resolved+unresolved) explicit.

The analytic kappa_U moments (mu_U, sigma_U, Cov(kU,M_U), Var(M_U)) are estimated by
single-clump Monte Carlo of the SAME kernel/position model used by brute -> exact
consistency, so any PDF difference is the carve/conditioning, not a kernel mismatch.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

rng = np.random.default_rng(7)

# ---- cosmology (CLAUDE.md internal constants) -----------------------------
Om, OL, h = 0.315, 0.685, 0.674
RHOC0 = 277.394 * h*h                      # M_sun/kpc^3
CKMS  = 2.998e5                            # km/s
G_KPC = 4.30091e-6                         # kpc (km/s)^2 / Msun
C2_4piG = CKMS**2/(4*np.pi*G_KPC)          # Msun/kpc
def Ez(z): return np.sqrt(Om*(1+z)**3+OL)
def Dc(z):                                  # comoving distance [kpc]
    return (CKMS/(100*h)) * 1e3 * integrate.quad(lambda zz: 1/Ez(zz), 0, z)[0]

ZL, ZS = 0.5, 1.0
DcL, DcS = Dc(ZL), Dc(ZS)
DAL, DAS = DcL/(1+ZL), DcS/(1+ZS)
DALS = (DcS-DcL)/(1+ZS)
SIGMA_C = C2_4piG * DAS/(DAL*DALS)         # Msun/kpc^2 (physical)
rhoc_zl = RHOC0*Ez(ZL)**2

# ---- NFW host / clump geometry --------------------------------------------
def nfw_params(M, c=6.0):
    r200 = (3*M/(4*np.pi*200*rhoc_zl))**(1/3.)
    rs   = r200/c
    mc   = np.log(1+c)-c/(1+c)
    rhos = (200/3.)*rhoc_zl*c**3/mc
    return rs, rhos, r200

def Fg(x):                                  # NFW convergence profile f(x), kappa=2 kappa0 f(x)
    x = np.asarray(x, float); out = np.empty_like(x)
    lo, hi = x < 1-1e-6, x > 1+1e-6; mid = ~(lo|hi)
    xl, xh = x[lo], x[hi]
    out[lo] = (1 - 2/np.sqrt(1-xl**2)*np.arctanh(np.sqrt((1-xl)/(1+xl))))/(xl**2-1)
    out[hi] = (1 - 2/np.sqrt(xh**2-1)*np.arctan (np.sqrt((xh-1)/(xh+1))))/(xh**2-1)
    out[mid] = 1/3.
    return out

def kappa_nfw(M, R, c=6.0):                 # convergence of an NFW halo mass M at 2D sep R [kpc]
    rs, rhos, _ = nfw_params(M, c)
    kappa0 = rs*rhos/SIGMA_C
    return 2*kappa0*Fg(np.maximum(R/rs, 1e-6))

# host concentration ~6; clumps use same c-M (weak); keep c=6 for clumps too (POC).
CH = 6.0
M_HOST = 1.0e13
_, _, R200 = nfw_params(M_HOST, CH)

# ---- SHMF (JvdB14) ---------------------------------------------------------
ALPHA, BETA, OMEGA = -0.82, 50.0, 4.0
PSI_MAX, PSI_MIN = 1.0, 1.0e7/M_HOST
PSI_RES = 1.0e-3
F_B_TARGET = 0.14
def dN_dpsi(psi): return psi**(ALPHA-1.0)*np.exp(-BETA*psi**OMEGA)
mass_int = integrate.quad(lambda p: p*dN_dpsi(p), PSI_MIN, PSI_MAX)[0]
GAMMA = F_B_TARGET/mass_int
def band(a,b):
    Nm  = integrate.quad(lambda p: GAMMA*dN_dpsi(p),      a,b)[0]
    Mf  = integrate.quad(lambda p: GAMMA*p*dN_dpsi(p),    a,b)[0]
    return Nm, Mf
N_res, f_res = band(PSI_RES, PSI_MAX)
N_unr, f_unr = band(PSI_MIN, PSI_RES)
N_all, f_b   = band(PSI_MIN, PSI_MAX)
M_U_MEAN = f_unr*M_HOST
print(f"f_b={f_b:.4f} f_res={f_res:.4f} f_unr={f_unr:.5f}  N_res={N_res:.1f} N_unr={N_unr:.0f}")
print(f"R200={R200:.0f} kpc  Sigma_c={SIGMA_C:.3e} Msun/kpc^2  host kappa(0.3 R200)={kappa_nfw(M_HOST,0.3*R200):.4f}")

# psi CDF samplers for the three bands
def make_cdf(a,b,n=4000):
    pg = np.logspace(np.log10(a), np.log10(b), n); w = dN_dpsi(pg)
    cdf = np.concatenate([[0], np.cumsum(0.5*(w[1:]+w[:-1])*np.diff(pg))]); cdf/=cdf[-1]
    return pg, cdf
pg_res, cdf_res = make_cdf(PSI_RES, PSI_MAX)
pg_unr, cdf_unr = make_cdf(PSI_MIN, PSI_RES)
pg_all, cdf_all = make_cdf(PSI_MIN, PSI_MAX)

# clump 3D position: radius ~ NFW mass profile within r200, isotropic; projected to 2D sep from ray.
def sample_clump_2d_sep(n, r_ray):
    # inverse-CDF of NFW enclosed-mass shape m(cx)/m(c) for x in (0,1]
    xs = np.linspace(1e-4,1,4000); mc = np.log(1+CH*xs)-CH*xs/(1+CH*xs); mc/=mc[-1]
    u = rng.random(n); x = np.interp(u, mc, xs); r3d = x*R200
    cth = rng.uniform(-1,1,n); az = rng.uniform(0,2*np.pi,n)
    R2 = r3d*np.sqrt(1-cth**2)
    dx = r_ray - R2*np.cos(az); dy = -R2*np.sin(az)
    return np.sqrt(dx*dx+dy*dy)

# ---- analytic kappa_U Campbell moments on an r-grid (single-clump MC) ------
RGRID = np.linspace(0.05*R200, 1.2*R200, 24)
NS = 200000     # single-clump MC pool per r
mu_U_g=np.zeros_like(RGRID); s2_U_g=np.zeros_like(RGRID)
cov_g =np.zeros_like(RGRID); varM_g=np.zeros_like(RGRID); muM_g=np.zeros_like(RGRID)
for i,r in enumerate(RGRID):
    psi = np.interp(rng.random(NS), cdf_unr, pg_unr); m = psi*M_HOST
    d = sample_clump_2d_sep(NS, r); kc = kappa_nfw(m, d)   # vector over masses (kappa_nfw handles array M via rs)
    mu_U_g[i]=N_unr*kc.mean(); s2_U_g[i]=N_unr*(kc**2).mean()
    cov_g[i]=N_unr*(kc*m).mean(); varM_g[i]=N_unr*(m**2).mean(); muM_g[i]=N_unr*m.mean()
def interp_r(g,r): return np.interp(r, RGRID, g)

# ---- main comparison loop --------------------------------------------------
NRAY = 40000
# area-weighted sightlines within the host cross-section
r_ray = R200*np.sqrt(rng.uniform((0.05)**2, 1.0, NRAY))
k0=np.empty(NRAY); kA=np.empty(NRAY); kC=np.empty(NRAY); kBr=np.empty(NRAY)
w_h = (1-f_b)/(1-f_res); w_u = f_unr/(1-f_res)
neg_hostC=0; neg_hostBr=0
for j in range(NRAY):
    r = r_ray[j]
    # resolved clumps (shared by 0,A,C and part of brute)
    nR = rng.poisson(N_res)
    if nR:
        psiR = np.interp(rng.random(nR), cdf_res, pg_res); mR = psiR*M_HOST
        dR = sample_clump_2d_sep(nR, r); kR = kappa_nfw(mR, dR).sum(); sumR = mR.sum()
    else:
        kR = 0.0; sumR = 0.0
    delta = sumR - f_res*M_HOST
    muU=interp_r(mu_U_g,r); s2U=interp_r(s2_U_g,r); sU=np.sqrt(max(s2U,0))
    covU=interp_r(cov_g,r); varM=interp_r(varM_g,r); muM=interp_r(muM_g,r)
    slope = covU/varM; cond_s = np.sqrt(max(s2U - slope*covU, 0.0))   # s2U*(1-rho^2)

    g = rng.standard_normal()
    # (0) current: fixed host, marginal kappa_U
    k0[j] = kappa_nfw((1-f_b)*M_HOST, r) + kR + (muU + sU*g)
    # (A) realized carve, marginal kappa_U
    MhA = M_HOST - sumR - M_U_MEAN
    kA[j] = kappa_nfw(max(MhA,1.0), r) + kR + (muU + sU*g)
    # (C) proportional carve + conditioned kappa_U
    MhC = M_HOST - sumR - (M_U_MEAN - w_u*delta)
    MUtarget = M_U_MEAN - w_u*delta
    kU_c = muU + slope*(MUtarget - muM) + cond_s*g
    if MhC < 0: neg_hostC += 1
    kC[j] = kappa_nfw(max(MhC,1.0), r) + kR + kU_c
    # (brute, carved): all clumps explicit, host = M - sum_all
    nU = rng.poisson(N_unr)
    if nU:
        psiU = np.interp(rng.random(nU), cdf_unr, pg_unr); mU = psiU*M_HOST
        dU = sample_clump_2d_sep(nU, r); kUbr = kappa_nfw(mU, dU).sum(); sumU = mU.sum()
    else:
        kUbr = 0.0; sumU = 0.0
    MhBr = M_HOST - sumR - sumU
    if MhBr < 0: neg_hostBr += 1
    kBr[j] = kappa_nfw(max(MhBr,1.0), r) + kR + kUbr

print(f"neg host: C={neg_hostC}  brute={neg_hostBr}  (of {NRAY})")
for nm,k in [('(0) current',k0),('(A) carve',kA),('(C) carve+cond',kC),('brute',kBr)]:
    print(f"  {nm:16s} <k>={k.mean():.5f}  Var={k.var():.3e}  clipVar(|k|<0.3)={k[np.abs(k)<0.3].var():.3e}")

# JSD vs carved brute (shared bins), plus a brute self-split floor
def jsd(a,b,bins):
    pa,_=np.histogram(a,bins=bins,density=True); pb,_=np.histogram(b,bins=bins,density=True)
    pa=pa/pa.sum()+1e-12; pb=pb/pb.sum()+1e-12; m=0.5*(pa+pb)
    return 0.5*(np.sum(pa*np.log(pa/m))+np.sum(pb*np.log(pb/m)))
bins=np.linspace(np.percentile(kBr,0.2),np.percentile(kBr,99.8),120)
half=NRAY//2; floor=jsd(kBr[:half],kBr[half:],bins)
print(f"\nJSD vs carved brute (floor={floor:.2e}):")
for nm,k in [('(0) current',k0),('(A) carve',kA),('(C) carve+cond',kC)]:
    print(f"  {nm:16s} JSD={jsd(k,kBr,bins):.2e}  ({jsd(k,kBr,bins)/floor:.1f}x floor)")

# ---- figure ----------------------------------------------------------------
plt.rcParams.update({'font.size':10.5})
fig,ax=plt.subplots(1,2,figsize=(12,4.6))
b=np.linspace(np.percentile(kBr,0.3),np.percentile(kBr,99.5),90)
for nm,k,c,ls in [('brute (carved)',kBr,'#000000','-'),('(0) current',k0,'#cc4444','--'),
                  ('(A) carve',kA,'#228833',':'),('(C) carve+cond',kC,'#3366cc','-.')]:
    ax[0].hist(k,bins=b,histtype='step',density=True,color=c,ls=ls,lw=1.8,label=nm)
ax[0].set_xlabel(r'host convergence $\kappa$'); ax[0].set_ylabel('density')
ax[0].set_title('(a)  single-host $\\kappa$ PDF',loc='left'); ax[0].legend(fontsize=8.6,frameon=False)
# residuals vs brute in the body
cen=0.5*(b[1:]+b[:-1]); hb,_=np.histogram(kBr,bins=b,density=True)
for nm,k,c,ls in [('(0) current',k0,'#cc4444','--'),('(A) carve',kA,'#228833',':'),
                  ('(C) carve+cond',kC,'#3366cc','-.')]:
    hk,_=np.histogram(k,bins=b,density=True)
    ax[1].plot(cen,(hk-hb),color=c,ls=ls,lw=1.8,label=nm)
ax[1].axhline(0,color='k',lw=0.8)
ax[1].set_xlabel(r'host convergence $\kappa$'); ax[1].set_ylabel(r'density $-$ brute')
ax[1].set_title('(b)  residual vs carved brute',loc='left'); ax[1].legend(fontsize=8.6,frameon=False)
fig.suptitle(f'Single host $M={M_HOST:.0e}\\,M_\\odot$, $z_l{{=}}0.5$, $z_s{{=}}1$ '
             f'(carved brute reference; {NRAY} sightlines)',y=1.0)
fig.tight_layout()
fig.savefig('plots/single_host_kappa_pdf.png',dpi=140,bbox_inches='tight')
print("\nwrote plots/single_host_kappa_pdf.png")
