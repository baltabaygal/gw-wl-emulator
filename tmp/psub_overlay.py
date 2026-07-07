"""
Option-A Layer 2: overlay of our subhalo model's P_1sh(k) against Diaz Rivero,
Cyr-Racine & Dvorkin 2018 (arXiv:1707.04590), their fiducial configuration.

Their convention (eqs. 4, 28, 50): kappa-hat is the mass-normalized profile,
P_1sh(k) = kbar_sub/(<m> Sigma_cr) * int dm m^2 P_m(m) |ktilde(k,m)|^2, with
ktilde(k->0)=1 for finite-mass profiles. Fiducial: z_l=0.5, z_s=1, kbar_sub=0.02,
dN/dm ~ m^-1.9 on [1e5,1e8] Msun, tNFW with tau=r_t/r_s=15, r_s=0.1(m/1e6)^(1/3) kpc.
Self-check targets from their Fig 3a: plateau 1.2e-4 kpc^2, k_trunc=0.14/kpc,
k_scale=21.5/kpc, P ~ 1/k^4 beyond.

Our model curve: same kbar_sub, same mass window (isolates profile+slope physics),
slope dN/dm ~ m^-1.82, untruncated NFW with cons14 c(m,z) (profile extent regulated
at x_max=200 r_s as in the production kappa profile usage; sensitivity reported).
"""
import numpy as np
from scipy.integrate import quad

# ---- cosmology pieces (code-matched) ----
Om, h = 0.315, 0.674
zeq = 3402.0; OmR = Om/(1+zeq); OmL = 1-Om-OmR
H0 = 0.000102247*h; CH = 306.535
rho_c0 = 277.394*h**2
def Az(z):  return Om*(1+z)**3 + OmR*(1+z)**4 + OmL
def Hz(z):  return H0*np.sqrt(Az(z))
def Dc(z): return quad(lambda zp: CH/Hz(zp), 0, z)[0]
def DL(z): return (1+z)*Dc(z)
def Sigma_crit(zs, zl):
    DsA = DL(zs)/(1+zs)**2; DlA = DL(zl)/(1+zl)**2
    DlsA = DsA - DlA*(1+zl)/(1+zs)
    return 2.08871e16*DsA/(4*np.pi*DlA*DlsA)

ZL, ZS, KBAR = 0.5, 1.0, 0.02
Sigcr = Sigma_crit(ZS, ZL)
print(f"Sigma_crit(zl={ZL}, zs={ZS}) = {Sigcr:.4e} Msun/kpc^2  (they quote ~3e9)")

KS = np.logspace(-2, 2, 60)   # 1/kpc

# ---- their tNFW profile: rho = mNFW/(4 pi R (R+rs)^2) * rt^2/(R^2+rt^2) ----
def util_tnfw(ks, rs, tau):
    rt = tau*rs
    R = np.logspace(-5, np.log10(rt*300), 3000)*rs if False else np.logspace(np.log10(1e-4*rs), np.log10(300*rt), 3000)
    rho = 1.0/(R*(R+rs)**2) * rt**2/(R**2+rt**2)
    mtot = np.trapezoid(4*np.pi*R**2*rho, R)
    out = np.array([np.trapezoid(4*np.pi*R**2*rho*np.sinc(k*R/np.pi), R) for k in ks])
    return out/mtot   # unit-normalized: util(0)=1

# ---- our untruncated-NFW kappa profile FT (2D Hankel), mass-normalized by m200 ----
from scipy.special import j0
def fNFW(x):
    x = np.atleast_1d(np.asarray(x, float)); out = np.empty_like(x)
    hi = x > 1; lo = x < 1; eq = ~hi & ~lo
    t = np.empty_like(x)
    t[hi] = np.arctan(np.sqrt((x[hi]-1)/(1+x[hi])))/np.sqrt(x[hi]**2-1)
    t[lo] = np.arctanh(np.sqrt((1-x[lo])/(1+x[lo])))/np.sqrt(1-x[lo]**2)
    out[hi] = (1-2*t[hi])/(x[hi]**2-1); out[lo] = (1-2*t[lo])/(x[lo]**2-1); out[eq] = 1/3
    return out
def cons14(M, z):
    b = -0.101 + 0.026*z
    a = 0.520 + (0.905-0.520)*np.exp(-0.617*z**1.21)
    return 10**(a + b*np.log10(M/(1e12/h)))
def util_ours(ks, m, z, xmax=200.0):
    rhoz = Az(z)*rho_c0
    c = cons14(m, z)
    r200 = (3*m/(4*np.pi*200*rhoz))**(1/3)
    rs = r200/c
    rhos = 200*rhoz*c**3*(1+c)/(3*((1+c)*np.log(1+c) - c))
    x = np.logspace(-4, np.log10(xmax), 3000)
    fx = fNFW(x)
    # kappa(R) = 2 rs rhos f(R/rs)/Sigcr; projected mass integral = 2pi rs^2 * 2 rs rhos * int f x dx
    out = np.array([np.trapezoid(fx*j0(k*rs*x)*x, x) for k in ks])
    return 2*np.pi*rs**3*2*rhos*out/m     # normalized by m200 (code's bookkeeping)

def P1sh(ks, mlo, mhi, beta_or_alpha, util_fn, nm=50):
    """kbar/( <m> Sigcr) * int dm m^2 P_m |util|^2 ; P_m ~ m^slope (dN/dm)."""
    lm = np.linspace(np.log(mlo), np.log(mhi), nm)
    m = np.exp(lm)
    w = m**(beta_or_alpha+1)                     # dN/dlnm
    mmean = np.trapezoid(w*m, lm)/np.trapezoid(w, lm)
    u2 = np.array([util_fn(ks, mi)**2 for mi in m])   # (nm, nk)
    num = np.trapezoid((w*m*m)[:, None]*u2, lm, axis=0)/np.trapezoid(w, lm)
    meff = np.trapezoid(w*m*m, lm)/np.trapezoid(w, lm)/mmean
    return KBAR/(mmean*Sigcr)*num, mmean, meff

# ---- their fiducial ----
MLO, MHI = 1e5, 1e8
P_drcd, mmean_d, meff_d = P1sh(KS, MLO, MHI, -1.9,
                               lambda ks, m: util_tnfw(ks, 0.1*(m/1e6)**(1/3), 15.0))
g0 = KBAR*meff_d/Sigcr
print(f"DRCD fiducial: <m>={mmean_d:.3e}, m_eff={meff_d:.3e}, g0=kbar*m_eff/Sigcr={g0:.3e} kpc^2")
print(f"  self-check: plateau target 1.2e-4; our P(kmin)={P_drcd[0]:.3e}")
rt_max = 15*0.1*(MHI/1e6)**(1/3); rs_min = 0.1*(MLO/1e6)**(1/3)
print(f"  k_trunc target 0.14: 1/rt_max = {1/rt_max:.3f};  k_scale target 21.5: 1/rs_min = {1/rs_min:.1f}")
i1, i2 = np.argmin(np.abs(KS-40)), np.argmin(np.abs(KS-90))
slope = np.log(P_drcd[i2]/P_drcd[i1])/np.log(KS[i2]/KS[i1])
print(f"  high-k slope target -4: measured {slope:.2f}")

# ---- ours, matched window ----
P_ours, mmean_o, meff_o = P1sh(KS, MLO, MHI, -1.82,
                               lambda ks, m: util_ours(ks, m, ZL))
P_ours_x50, _, _ = P1sh(KS, MLO, MHI, -1.82,
                        lambda ks, m: util_ours(ks, m, ZL, xmax=50.0), nm=30)
print(f"\nOurs (matched window, kbar=0.02): <m>={mmean_o:.3e}, m_eff(m200)={meff_o:.3e}")

print(f"\n{'k':>8} {'P_DRCD':>11} {'P_ours':>11} {'ratio':>7} {'P_ours(x50)':>12}")
for i in range(0, len(KS), 6):
    print(f"{KS[i]:8.3f} {P_drcd[i]:11.3e} {P_ours[i]:11.3e} {P_ours[i]/P_drcd[i]:7.2f} {P_ours_x50[i]:12.3e}")

np.savez("/Users/baltabay/Desktop/gw-wl-emulator/tmp/psub_overlay.npz",
         ks=KS, P_drcd=P_drcd, P_ours=P_ours, P_ours_x50=P_ours_x50,
         g0=g0, meff_d=meff_d, meff_o=meff_o, Sigcr=Sigcr)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.4, 4.6))
ax.loglog(KS, P_drcd, "k-", lw=2, label=r"DRCD18 fiducial (tNFW, $\beta=-1.9$)")
ax.loglog(KS, P_ours, "-", color="#3b5bdb", lw=2,
          label=r"this work (untrunc. NFW, $-1.82$, cons14), $x_{\max}=200$")
ax.loglog(KS, P_ours_x50, "--", color="#3b5bdb", lw=1.2, label=r"same, $x_{\max}=50$")
ax.axhline(1.2e-4, color="0.6", ls=":", lw=1, label=r"their quoted plateau $1.2\times10^{-4}$")
ax.axvline(0.14, color="0.75", ls=":", lw=1)
ax.axvline(21.5, color="0.75", ls=":", lw=1)
ax.set_xlabel(r"$k\ [\mathrm{kpc}^{-1}]$"); ax.set_ylabel(r"$P_{\rm 1sh}(k)\ [\mathrm{kpc}^2]$")
ax.set_title(r"$P_{\rm sub}$: matched window $[10^5,10^8]\,M_\odot$, $\bar\kappa_{\rm sub}=0.02$, $z_l=0.5\to z_s=1$")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("/Users/baltabay/Desktop/gw-wl-emulator/plots/figures/psub_overlay_drcd18.png", dpi=200)
print("\nsaved plots/figures/psub_overlay_drcd18.png and tmp/psub_overlay.npz")
