"""
One host halo + its evolved subhalo population, drawn and plotted.

Physics lifted from the papers in papers/misc (no memory):
  - Evolved SHMF: Jiang & van den Bosch 2014 (1403.6827), Eq. (22):
        dN/dln(psi) = gamma * psi^alpha * exp(-beta * psi^omega),   psi = m/M0
    all-orders best fit: alpha = -0.82, (beta, omega) = (50, 4).
  - Normalization gamma from subhalo mass fraction fs, Eq. (23), psi_res = 1e-4.
  - fs from halo 'dynamical age' Ntau, Eq. (26): fs = 0.3563/Ntau^0.6 - 0.075.
  - Ntau, Eq. (24), uses tau_dyn (Eq. 2, Bryan & Norman 1998 Delta_vir) and the
    formation redshift zform from the Giocoli+2012 model, Eq. (25).

Spatial placement is the spec's first-pass proxy (subhalos trace the host NFW;
sec 2b). Concentration is a field c(M,z) (Duffy+2008) used for host and clumps.

This is a standalone sanity render -- it does NOT touch the C++ pipeline.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.special import gamma as Gamma, gammaincc
from scipy.integrate import quad
from scipy.optimize import brentq

# ---------------------------------------------------------------- cosmology
# matches the project (python/diagnostics.py, python_bindings.cpp defaults)
Om, OL = 0.315, 0.685
sigma8, ns, h = 0.811, 0.965, 0.674
rho_crit0 = 277.394 * h**2          # Msun / kpc^3  (code's internal constant)
rho_m0    = Om * rho_crit0          # mean matter density today, Msun/kpc^3
H0_inv_Gyr = 9.778 / h              # 1/H0 in Gyr

def E(z):  return np.sqrt(Om*(1+z)**3 + OL)
def Omz(z): return Om*(1+z)**3 / E(z)**2

def Dvir(z):                         # Bryan & Norman 1998, wrt critical density
    d = Omz(z) - 1.0
    return 18*np.pi**2 + 82*d - 39*d**2

def growth(z):                       # linear growth D(z), normalized D(0)=1
    integ = lambda zp: (1+zp)/E(zp)**3
    num, _ = quad(integ, z, np.inf)
    den, _ = quad(integ, 0, np.inf)
    return (E(z)*num) / (E(0)*den)

def lookback_Gyr(z):                 # lookback time to redshift z
    val, _ = quad(lambda zp: 1.0/((1+zp)*E(zp)), 0, z)
    return H0_inv_Gyr * val

def tau_dyn_Gyr(z):                  # JvdB14 Eq. (2): dynamical time
    return 1.628/h * (Dvir(z)/178.0)**(-0.5) * (E(z))**(-1.0)

# ----------------------------------------------- mass variance sigma(M)
# Eisenstein & Hu 1998 no-wiggle transfer function -> P(k) -> sigma(R)
def T_EH98(k):                       # k in 1/Mpc (physical), no-wiggle form
    Ob = 0.0493
    theta = 2.728/2.7
    Omh2, Obh2 = Om*h*h, Ob*h*h
    s = 44.5*np.log(9.83/Omh2) / np.sqrt(1+10*Obh2**0.75)   # sound horizon, Mpc
    alpha_g = 1 - 0.328*np.log(431*Omh2)*Ob/Om + 0.38*np.log(22.3*Omh2)*(Ob/Om)**2
    Gamma_eff = Om*h*(alpha_g + (1-alpha_g)/(1+(0.43*k*s*h)**4))
    q = k/h * theta**2 / Gamma_eff
    L0 = np.log(2*np.e + 1.8*q)
    C0 = 14.2 + 731.0/(1+62.5*q)
    return L0/(L0 + C0*q*q)

def _sigma2_R(R_Mpc):                # R in Mpc/h -> top-hat variance (unnormalized)
    R = R_Mpc/h                      # Mpc
    def integrand(lnk):
        k = np.exp(lnk)
        x = k*R
        W = 3*(np.sin(x)-x*np.cos(x))/x**3
        Pk = k**ns * T_EH98(k)**2
        return (k**3*Pk)/(2*np.pi**2) * W**2
    val, _ = quad(integrand, np.log(1e-4), np.log(1e3), limit=200)
    return val

_norm = sigma8**2 / _sigma2_R(8.0)   # fix amplitude so sigma(8 Mpc/h)=sigma8

def sigma_M(M):                      # M in Msun ; top-hat radius from rho_m0
    R_kpc = (3*M/(4*np.pi*rho_m0))**(1/3)         # comoving Lagrangian radius, kpc
    R_Mpc_over_h = R_kpc/1000.0*h                 # -> Mpc/h
    return np.sqrt(_norm*_sigma2_R(R_Mpc_over_h))

# --------------------------------------------------- formation redshift (Eq. 25)
def z_form(M0, z0, f=0.5):
    dc0 = 1.686/growth(z0)
    af  = 0.815*np.exp(-2*f)/f**0.707
    wf  = np.sqrt(2*np.log(af+1))
    rhs = dc0 + wf*np.sqrt(sigma_M(f*M0)**2 - sigma_M(M0)**2)
    g   = lambda zf: 1.686/growth(zf) - rhs       # solve dc(zf)=rhs
    return brentq(g, z0, 30.0)

def N_tau(M0, z0):                   # Eq. (24): dynamical age
    zf = z_form(M0, z0)
    t0, tf = lookback_Gyr(z0), lookback_Gyr(zf)
    val, _ = quad(lambda z: 1.0/(tau_dyn_Gyr(z)*(1+z)*E(z)) * H0_inv_Gyr,
                  z0, zf)            # change var dt = H0_inv/((1+z)E) dz
    return val, zf

# ----------------------------------------------- evolved SHMF (JvdB14 Eq.22/23)
ALPHA, BETA, OMEGA, PSI_RES = -0.82, 50.0, 4.0, 1e-4
def gamma_norm(fs):                  # Eq. (23)
    s = (1+ALPHA)/OMEGA
    denom = Gamma(s)*(gammaincc(s, BETA*PSI_RES**OMEGA) - gammaincc(s, BETA))
    return OMEGA*BETA**s/denom * fs
def dN_dlnpsi(psi, gamma):
    return gamma*psi**ALPHA*np.exp(-BETA*psi**OMEGA)

# ----------------------------------------------- concentration & NFW geometry
def conc(M, z):                      # Duffy+2008 (200c)
    return 5.71*(M/(2e12/h))**(-0.084)*(1+z)**(-0.47)
def r200_kpc(M, z):
    return (3*M/(4*np.pi*200*E(z)**2*rho_crit0))**(1/3)

def sample_nfw_radii(n, rs, c, rng):
    # Draw 3D radii from host NFW mass profile
    if n == 0:
        return np.array([])
    mu = lambda x: np.log(1+x) - x/(1+x)
    u  = rng.uniform(0, 1, n)
    target = u*mu(c)
    out = np.empty(n)
    for i, t in enumerate(target):
        out[i] = brentq(lambda x: mu(x)-t, 1e-6, c)*rs
    return out

def sample_biased_radii(n, rs, c, rng):
    # Draw 3D radii from host NFW profile modified by the SatGen radial bias function
    # B(x) = 1 / sqrt( (x/0.54)**(-2.5) + 1.0 ) where x = r/rvir
    if n == 0:
        return np.array([])
    lx = np.linspace(np.log(1e-6), np.log(1.0), 4000)
    x_grid = np.exp(lx)
    B_bias = 1.0 / np.sqrt((x_grid / 0.54)**(-2.5) + 1.0)
    integrand = (x_grid**2 / (1.0 + c * x_grid)**2) * B_bias
    cdf = np.cumsum(integrand)
    cdf /= cdf[-1]
    u = rng.uniform(0, 1, n)
    x_sampled = np.exp(np.interp(u, cdf, lx))
    return x_sampled * (rs * c)


# =====================================================================  RUN
M0, z0 = 1e15, 0.5                    # one massive cluster-scale lens at z=0.5
psi_min, psi_max = PSI_RES, 1.0      # clump range: JvdB resolution .. M0

# host-level quantities (same for every realization)
Nt, zf = N_tau(M0, z0)
fs = 0.3563/Nt**0.6 - 0.075
g  = gamma_norm(fs)
Nmean, _ = quad(lambda lp: dN_dlnpsi(np.exp(lp), g), np.log(psi_min), np.log(psi_max))
cH  = conc(M0, z0)
r200H = r200_kpc(M0, z0); rsH = r200H/cH

# Larger-host comparison for the high-mass tail of the SHMF.
comparison_masses = [1e14, 3e14, M0, 3e15]
host_comparison = []
for Mh in comparison_masses:
    Nt_h, zf_h = N_tau(Mh, z0)
    fs_h = 0.3563/Nt_h**0.6 - 0.075
    g_h = gamma_norm(fs_h)
    N_h, _ = quad(lambda lp: dN_dlnpsi(np.exp(lp), g_h),
                  np.log(psi_min), np.log(psi_max))
    N_tail_h, _ = quad(lambda lp: dN_dlnpsi(np.exp(lp), g_h),
                       np.log(0.1), np.log(psi_max))
    host_comparison.append((Mh, Nt_h, zf_h, fs_h, g_h, N_h, N_tail_h))

# inverse-CDF table for sampling subhalo masses from dN/dlnpsi
lp  = np.linspace(np.log(psi_min), np.log(psi_max), 4000)
cdf = np.cumsum(dN_dlnpsi(np.exp(lp), g)); cdf /= cdf[-1]

def realize(seed):
    """One Monte-Carlo realization of the host's subhalo population."""
    rng  = np.random.default_rng(seed)
    Nsub = rng.poisson(Nmean)                                  # shot noise (grand-canonical)
    psi  = np.exp(np.interp(rng.uniform(0, 1, Nsub), cdf, lp)) # clump masses
    m    = psi*M0
    cth  = rng.uniform(-1, 1, Nsub)
    ph   = rng.uniform(0, 2*np.pi, Nsub)
    
    # Spawn synchronized generators to align the radial percentile u
    rng_nfw = np.random.default_rng(seed + 1000)
    rng_biased = np.random.default_rng(seed + 1000)
    
    r3d_nfw = sample_nfw_radii(Nsub, rsH, cH, rng_nfw)
    r3d_biased = sample_biased_radii(Nsub, rsH, cH, rng_biased)
    
    xs_nfw = r3d_nfw*np.sqrt(1-cth**2)*np.cos(ph)
    ys_nfw = r3d_nfw*np.sqrt(1-cth**2)*np.sin(ph)
    
    xs_biased = r3d_biased*np.sqrt(1-cth**2)*np.cos(ph)
    ys_biased = r3d_biased*np.sqrt(1-cth**2)*np.sin(ph)
    
    return m, xs_nfw, ys_nfw, xs_biased, ys_biased, Nsub


seeds = list(range(1, 201))
reals = [realize(s) for s in seeds]
shown = list(zip(seeds[:3], reals[:3]))
cols  = ['tab:red', 'tab:green', 'tab:purple']

print(f"host M0={M0:.1e} Msun  z0={z0}")
print(f"zform={zf:.2f}  Ntau={Nt:.2f}  fs={fs:.3f}  gamma={g:.4f}")
drawn_N = np.array([r[5] for r in reals])
drawn_tail = np.array([int(np.sum(r[0]/M0 > 0.1)) for r in reals])
print(f"<Nsub>(psi>{psi_min:g})={Nmean:.1f}  ->  drawn N mean={drawn_N.mean():.1f} "
      f"std={drawn_N.std(ddof=1):.1f}  first 3={drawn_N[:3].tolist()}")
print(f"drawn N(psi>0.1): total={drawn_tail.sum()} mean={drawn_tail.mean():.3f} "
      f"nonzero realizations={np.count_nonzero(drawn_tail)}/{len(reals)}  first 3={drawn_tail[:3].tolist()}")
print(f"host r200={r200H:.0f} kpc  c={cH:.2f}  rs={rsH:.0f} kpc")
print("larger-host comparison at same z0:")
for Mh, Nt_h, zf_h, fs_h, g_h, N_h, N_tail_h in host_comparison:
    print(f"  M0={Mh:.1e}: zform={zf_h:.2f} Ntau={Nt_h:.2f} fs={fs_h:.3f} "
          f"<Nsub>={N_h:.1f} <Nsub>(psi>0.1)={N_tail_h:.3f}")

# ================================================================  PLOT
fig = plt.figure(figsize=(14, 13.5))
gs  = fig.add_gridspec(3, 6, height_ratios=[1.05, 1.0, 1.05],
                       hspace=0.42, wspace=0.28)

# NFW projected surface density (Wright & Brainerd 2000), arbitrary units
def nfw_sigma(x):
    x = np.maximum(x, 1e-4)
    out = np.empty_like(x)
    lo, hi, eq = x < 1-1e-6, x > 1+1e-6, np.abs(x-1) <= 1e-6
    xl, xh = x[lo], x[hi]
    out[lo] = (1 - 2/np.sqrt(1-xl**2)*np.arctanh(np.sqrt((1-xl)/(1+xl))))/(xl**2-1)
    out[hi] = (1 - 2/np.sqrt(xh**2-1)*np.arctan(np.sqrt((xh-1)/(xh+1))))/(xh**2-1)
    out[eq] = 1.0/3.0
    return out

def nfw_rhos(M, rs, c):
    mu = np.log(1+c) - c/(1+c)
    return M/(4*np.pi*rs**3*mu)

def projected_nfw_sigma(R, rs, rhos):
    return 2*rhos*rs*nfw_sigma(R/rs)

L = 1.15*r200H
gx = np.linspace(-L, L, 400); X, Y = np.meshgrid(gx, gx)
rhosH = nfw_rhos(M0, rsH, cH)
Sig = projected_nfw_sigma(np.sqrt(X**2+Y**2), rsH, rhosH)

# ---- top row: three realizations of the projected map -----------------------
for k, ((seed, (m, xs_nfw, ys_nfw, xs_biased, ys_biased, Nsub)), col) in enumerate(zip(shown, cols)):
    ax = fig.add_subplot(gs[0, 2*k:2*k+2])
    ax.imshow(Sig, extent=[-L, L, -L, L], origin='lower',
              norm=LogNorm(vmin=Sig.min()*5, vmax=Sig.max()), cmap='bone_r')
    ax.scatter(xs_biased, ys_biased, s=30*(m/1e11)**(1/3)+8, c=np.log10(m), cmap='autumn',
               vmin=10, vmax=14, edgecolor='k', linewidth=0.4, alpha=0.9, zorder=3)
    ax.add_patch(plt.Circle((0, 0), r200H, fill=False, ec='tab:cyan', ls='--', lw=1.2))
    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect('equal')
    ax.set_title(f'realization {k+1} (seed {seed}):  N = {Nsub}\n(biased spatial profile)',
                 color=col, fontsize=10)
    ax.set_xlabel('x [kpc]')
    if k == 0: ax.set_ylabel('y [kpc]')
fig.text(0.5, 0.96, f'Same host  $M_0={M0:.1e}\\,M_\\odot$, $z={z0}$  '
         f'($f_s={fs:.3f}$, $N_\\tau={Nt:.2f}$, $z_f={zf:.2f}$, $\\langle N\\rangle={Nmean:.0f}$)'
         '  —  200 Monte-Carlo realizations; first three maps shown', ha='center', fontsize=13)

# ---- middle row: SHMF in JvdB native form, log[dN/dlog psi] vs log psi ------
LN10 = np.log(10.0)
ax = fig.add_subplot(gs[1, :])
# analytic curve  dN/dlog10(psi) = ln10 * dN/dln(psi)
lpc = np.linspace(-4, 0, 400)                       # calibrated range
lpe = np.linspace(-7, -4, 120)                      # extrapolation to lensing floor
ax.plot(lpc, np.log10(LN10*dN_dlnpsi(10**lpc, g)), 'k-', lw=2.2,
        label=r'JvdB14 evolved SHMF (all orders)')
ax.plot(lpe, np.log10(LN10*dN_dlnpsi(10**lpe, g)), color='0.55', ls='--', lw=1.6,
        label=r'extrapolated below $\psi_{\rm res}=10^{-4}$')
# sampled points from the first three realizations, in log[dN/dlog psi]
edges = np.linspace(-4, 0, 21); dlog = np.diff(edges); ctr = 0.5*(edges[:-1]+edges[1:])
for (seed, (m, *_)), col in zip(shown, cols):
    cnt, _ = np.histogram(np.log10(m/M0), bins=edges)
    ok = cnt > 0
    y  = np.log10(cnt[ok]/dlog[ok])
    yerr = 0.434/np.sqrt(cnt[ok])                   # dex error from sqrt(N)
    ax.errorbar(ctr[ok], y, yerr=yerr, fmt='o', color=col, ms=5, capsize=2,
                alpha=0.85, label=f'sampled, seed {seed}')
all_psi = np.concatenate([r[0]/M0 for r in reals])
cnt_all, _ = np.histogram(np.log10(all_psi), bins=edges)
ok_all = cnt_all > 0
y_all = np.log10((cnt_all[ok_all]/len(reals))/dlog[ok_all])
yerr_all = 0.434/np.sqrt(cnt_all[ok_all])
ax.errorbar(ctr[ok_all], y_all, yerr=yerr_all, fmt='s', color='tab:blue',
            ms=4, capsize=2, alpha=0.9, label='200-realization mean')
ax.axvline(-7, color='tab:blue', ls=':', lw=1.5)
ax.text(-6.9, -1.3, r'lensing floor  $m\sim10^7\,M_\odot$  ($\psi=10^{-7}$)',
        color='tab:blue', fontsize=9)
ax.axvline(0, color='gray', ls=':', lw=1)
ax.text(-0.08, -1.3, r'$M_0$', color='gray', fontsize=9, ha='right')
ax.set_xlabel(r'$\log_{10}(m/M_0)$', fontsize=12)
ax.set_ylabel(r'$\log_{10}\left[\,dN/d\log_{10}(m/M_0)\,\right]$', fontsize=12)
ax.set_title('Evolved subhalo mass function (JvdB14 Eq. 22) vs. sampled clumps')
ax.set_xlim(-5, 0); ax.set_ylim(-1, 3.5); ax.grid(alpha=0.3)
ax.legend(fontsize=9, ncol=1, loc='center left', bbox_to_anchor=(1.01, 0.5),
          borderaxespad=0.0)

# ---- bottom row: host + NFW subhalos vs host + biased subhalos -----------
seed0, (m0, xs0_nfw, ys0_nfw, xs0_biased, ys0_biased, N0) = shown[0]

Sig_sub_nfw = np.zeros_like(Sig)
Sig_sub_biased = np.zeros_like(Sig)

for mi, xi_n, yi_n, xi_b, yi_b in zip(m0, xs0_nfw, ys0_nfw, xs0_biased, ys0_biased):
    ci = conc(mi, z0)
    r200i = r200_kpc(mi, z0)
    rsi = r200i/ci
    rhosi = nfw_rhos(mi, rsi, ci)
    
    # Accumulate NFW
    Ri_n = np.sqrt((X-xi_n)**2 + (Y-yi_n)**2)
    Sig_sub_nfw += projected_nfw_sigma(Ri_n, rsi, rhosi)
    
    # Accumulate Biased
    Ri_b = np.sqrt((X-xi_b)**2 + (Y-yi_b)**2)
    Sig_sub_biased += projected_nfw_sigma(Ri_b, rsi, rhosi)

Sig_total_nfw = Sig + Sig_sub_nfw
Sig_total_biased = Sig + Sig_sub_biased

field_norm = LogNorm(vmin=np.percentile(Sig, 1), 
                     vmax=max(np.percentile(Sig_total_nfw, 99.9), np.percentile(Sig_total_biased, 99.9)))

for ax, field, title in [
    (fig.add_subplot(gs[2, :3]), Sig_total_nfw, f'host + NFW subhalos (seed {seed0}, N = {N0})'),
    (fig.add_subplot(gs[2, 3:]), Sig_total_biased, r'host + biased subhalos $[(x/0.54)^{-2.5}+1]^{-1/2}$' + f' (seed {seed0}, N = {N0})'),
]:
    im = ax.imshow(field, extent=[-L, L, -L, L], origin='lower',
                   norm=field_norm, cmap='magma')
    ax.add_patch(plt.Circle((0, 0), r200H, fill=False, ec='tab:cyan', ls='--', lw=1.1))
    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect('equal')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('x [kpc]')
fig.axes[-2].set_ylabel('y [kpc]')
cbar = fig.colorbar(im, ax=fig.axes[-2:], fraction=0.026, pad=0.02)
cbar.set_label(r'projected NFW surface density proxy  $\Sigma$')

out = 'plots/subhalo_demo.png'
fig.savefig(out, dpi=140, bbox_inches='tight')
print('wrote', out)
