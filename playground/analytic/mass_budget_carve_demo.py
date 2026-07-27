#!/usr/bin/env python
"""
Mass-budget demonstration for the subhalo host-carving decision.

Compares, for a single fiducial host (M, z_l), the PER-REALIZATION total mass
laid down by:

  (0) current model 3      : host carved by the EXPECTATION f_b; resolved clumps
                             float; unresolved = mu_U + N(0, sigma_U^2).
                             -> total mass DRIFTS by the resolved-clump shot noise.
  (A) realized carve       : host = M - sum_i m_i - M_U_mean.  Keep kappa_U marginal.
                             -> total mass = M exactly, every draw. (my proposal)
  (B) conditional kappa_U  : host fixed at the mean (1-f_b)M; the UNRESOLVED band
                             absorbs the residual mass M_rem = M - host - sum_i m_i,
                             and kappa_U is drawn from the CLOSED-FORM conditional
                             p(kappa_U | mass_U = M_rem).  (the user's idea)

Also demonstrates that (B) is computable in closed form: kappa_U and the unresolved
mass M_U are two linear (Campbell) functionals of the SAME marked Poisson process,
hence jointly Gaussian, hence the conditional is an exact Gaussian regression.

SHMF (JvdB14, subhalo.h): dN/dlnpsi = gamma psi^alpha exp(-beta psi^omega), psi=m/M.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

rng = np.random.default_rng(20260722)

# ---- model constants (subhalo.h) ------------------------------------------
ALPHA, BETA, OMEGA = -0.82, 50.0, 4.0
PSI_MAX = 1.0
M_HOST  = 1.0e13            # fiducial host mass [Msun]
M_FLOOR = 1.0e7            # SHMF lower clump mass
PSI_MIN = M_FLOOR / M_HOST  # = 1e-6
F_B_TARGET = 0.14          # total bound fraction over [psi_min, psi_max] (fiducial)

# resolution floor separating RESOLVED (explicit clumps) from UNRESOLVED (analytic).
# In production this is the dynamic m_res(r); here a representative fixed value.
PSI_RES = 1.0e-3           # m_res = 1e10 Msun

# ---- SHMF intensity & normalization ---------------------------------------
def dN_dpsi_unit(psi):
    """intensity shape per unit psi (gamma factored out): psi^(alpha-1) exp(-beta psi^omega)"""
    return psi**(ALPHA - 1.0) * np.exp(-BETA * psi**OMEGA)

# gamma set so the mean bound fraction over [PSI_MIN, PSI_MAX] equals F_B_TARGET:
#   f_b = gamma * int psi * psi^(alpha-1) exp(-beta psi^omega) dpsi
mass_integral, _ = integrate.quad(lambda p: p * dN_dpsi_unit(p), PSI_MIN, PSI_MAX)
GAMMA = F_B_TARGET / mass_integral

def band_moments(psi_a, psi_b):
    """Campbell moments over a psi band for the SHMF:
       N_mean  = int gamma dN_dpsi
       Mfrac   = int gamma psi dN_dpsi           (mean mass fraction of band)
       M2frac  = int gamma psi^2 dN_dpsi         (Var of band mass fraction = this)
    """
    f = lambda p, k: GAMMA * p**k * dN_dpsi_unit(p)
    Nm  = integrate.quad(lambda p: f(p, 0), psi_a, psi_b)[0]
    Mf  = integrate.quad(lambda p: f(p, 1), psi_a, psi_b)[0]
    M2f = integrate.quad(lambda p: f(p, 2), psi_a, psi_b)[0]
    return Nm, Mf, M2f

# full-band bound fraction f_b (host reduction), and split into resolved / unresolved
N_all, fb_frac,  _        = band_moments(PSI_MIN, PSI_MAX)
N_res, fres_frac, _       = band_moments(PSI_RES, PSI_MAX)
N_unr, funr_frac, v_unr_f = band_moments(PSI_MIN, PSI_RES)   # v_unr_f = Var(mass frac) of unres band

M_U_MEAN = funr_frac * M_HOST            # mean unresolved-band mass (carried by mu_U)
print(f"f_b={fb_frac:.4f}  f_res={fres_frac:.4f}  f_unr={funr_frac:.5f}  "
      f"(f_res+f_unr={fres_frac+funr_frac:.4f})")
print(f"mean # resolved clumps={N_res:.2f}   mean # unresolved={N_unr:.1f}")

# ---- sampler: resolved clumps of one host ---------------------------------
# inverse-CDF over psi in [PSI_RES, PSI_MAX] for the (thinned) SHMF via rejection.
def sample_resolved(nreal):
    """returns array of sum_i m_i (resolved clump mass) for nreal host realizations."""
    sums = np.empty(nreal)
    # build a fine CDF for the resolved band to draw masses
    pg = np.logspace(np.log10(PSI_RES), np.log10(PSI_MAX), 4000)
    w  = dN_dpsi_unit(pg)
    cdf = np.concatenate([[0], np.cumsum(0.5*(w[1:]+w[:-1])*np.diff(pg))])
    cdf /= cdf[-1]
    for i in range(nreal):
        n = rng.poisson(N_res)
        if n == 0:
            sums[i] = 0.0
            continue
        u = rng.random(n)
        psi = np.interp(u, cdf, pg)
        sums[i] = psi.sum() * M_HOST
    return sums

# ---- unresolved band as a marked Poisson process (for the joint (kappa_U, M_U)) ----
# convergence kernel proxy: kappa_c = A * m * g(x),  x ~ clump-ray geometry, g in (0,1].
# (production uses the tabulated NFW kappa_c; the joint-Gaussian argument is identical.)
G_SCALE = 1.0e-13          # arbitrary kappa units; only ratios/correlation matter
def sample_unresolved_joint(nreal):
    pg = np.logspace(np.log10(PSI_MIN), np.log10(PSI_RES), 4000)
    w  = dN_dpsi_unit(pg)
    cdf = np.concatenate([[0], np.cumsum(0.5*(w[1:]+w[:-1])*np.diff(pg))])
    cdf /= cdf[-1]
    MU = np.empty(nreal); KU = np.empty(nreal)
    for i in range(nreal):
        n = rng.poisson(N_unr)
        if n == 0:
            MU[i] = 0.0; KU[i] = 0.0; continue
        u = rng.random(n)
        m = np.interp(u, cdf, pg) * M_HOST
        # geometry weight g in (0,1]: nearer clumps weigh more. draw g ~ Beta-ish.
        g = rng.random(n)**2          # skewed toward 0 (most clumps far) -> rho<1
        MU[i] = m.sum()
        KU[i] = (G_SCALE * m * g).sum()
    return MU, KU

# ===========================================================================
NREAL = 40000
sum_res = sample_resolved(NREAL)

# --- total mass per realization for the three methods ---
host_expect = (1.0 - fb_frac) * M_HOST                      # model 3 & method B host
M_tot_cur = host_expect + sum_res + M_U_MEAN                # (0) drifts
M_tot_A   = np.full(NREAL, M_HOST)                          # (A) realized carve: exact
M_rem_B   = M_HOST - host_expect - sum_res                 # residual for method B
M_tot_B   = host_expect + sum_res + M_rem_B                 # (B) = M exactly by construction

print(f"\nTotal mass / M_host  (mean +/- std):")
print(f"  (0) current model 3 : {M_tot_cur.mean()/M_HOST:.4f} +/- {M_tot_cur.std()/M_HOST:.4f}")
print(f"  (A) realized carve  : {M_tot_A.mean()/M_HOST:.4f} +/- {M_tot_A.std()/M_HOST:.4f}")
print(f"  (B) conditional kU  : {M_tot_B.mean()/M_HOST:.4f} +/- {M_tot_B.std()/M_HOST:.4f}")

# reservoir demand: which reservoir must absorb the resolved shot noise?
host_A_mass = M_HOST - sum_res - M_U_MEAN               # (A) smooth host does it: mass ~ 0.82 M
frac_host_neg = (host_A_mass < 0).mean()
frac_rem_neg  = (M_rem_B < 0).mean()                    # (B) unresolved band does it
print(f"\nReservoir demand (resolved shot noise std = {sum_res.std()/M_HOST:.3f} M):")
print(f"  smooth host mean (1-f_b)M = {(1-fb_frac):.3f} M ; unresolved band mean f_unr M = {funr_frac:.3f} M")
print(f"  (A) host mass < 0  in {frac_host_neg*100:.1f}% of draws")
print(f"  (B) residual unresolved mass < 0 (UNPHYSICAL) in {frac_rem_neg*100:.1f}% of draws")

# --- joint (kappa_U, M_U): demonstrate the closed-form conditional ---
MU, KU = sample_unresolved_joint(NREAL)
# Campbell (analytic) joint moments over the unresolved band:
#   mu_M   = int gamma psi dN M            = funr_frac * M
#   var_M  = int gamma (psi M)^2 dN        = v_unr_f * M^2   (Poisson: Var = int lambda m^2)
#   mu_K   = int gamma A m <g> dN
#   var_K  = int gamma (A m)^2 <g^2> dN
#   cov    = int gamma (A m^2) <g> dN
Eg, Eg2 = 1/3.0, 1/5.0     # E[g], E[g^2] for g=U^2, U~Unif(0,1)
mu_M  = funr_frac * M_HOST
var_M = v_unr_f * M_HOST**2
mu_K  = integrate.quad(lambda p: GAMMA*dN_dpsi_unit(p)*(G_SCALE*p*M_HOST)*Eg,      PSI_MIN, PSI_RES)[0]
var_K = integrate.quad(lambda p: GAMMA*dN_dpsi_unit(p)*(G_SCALE*p*M_HOST)**2*Eg2,  PSI_MIN, PSI_RES)[0]
cov_KM= integrate.quad(lambda p: GAMMA*dN_dpsi_unit(p)*(G_SCALE*p*M_HOST)*(p*M_HOST)*Eg, PSI_MIN, PSI_RES)[0]
rho   = cov_KM/np.sqrt(var_K*var_M)
print(f"\nUnresolved band joint (analytic Campbell):")
print(f"  MC   mu_M={MU.mean():.3e} var_M={MU.var():.3e}  mu_K={KU.mean():.3e} var_K={KU.var():.3e} corr={np.corrcoef(MU,KU)[0,1]:.3f}")
print(f"  ana  mu_M={mu_M:.3e} var_M={var_M:.3e}  mu_K={mu_K:.3e} var_K={var_K:.3e} rho ={rho:.3f}")

# closed-form conditional  E[kU | M_U = m*] and its residual std
slope = cov_KM/var_M
cond_std = np.sqrt(var_K*(1-rho**2))
def kappaU_conditional(m_star):
    return mu_K + slope*(m_star - mu_M)

# ---- method C: proportional-split hybrid ----------------------------------
# share the resolved deficit delta between the two reservoirs by their mean mass.
delta   = sum_res - fres_frac*M_HOST              # resolved excess [Msun]
w_h     = (1.0-fb_frac)/(1.0-fres_frac)           # host share of the deficit
w_u     = funr_frac /(1.0-fres_frac)              # unresolved-band share
print(f"\nProportional-split weights: w_host={w_h:.3f}  w_unres={w_u:.3f}")
host_C     = (1.0-fb_frac)*M_HOST - w_h*delta      # host mass, method C
MU_targ_C  = funr_frac*M_HOST     - w_u*delta      # unresolved target mass, method C
M_tot_C    = host_C + sum_res + MU_targ_C          # == M exactly (w_h+w_u=1)
frac_C_neg = (MU_targ_C < 0).mean()
print(f"  (C) total mass/M = {M_tot_C.mean()/M_HOST:.4f} +/- {M_tot_C.std()/M_HOST:.4f}"
      f" ;  unresolved residual < 0 in {frac_C_neg*100:.2f}% of draws")
# kappa_U treatments: A independent, B full-conditional, C proportional-conditional
kU_A = rng.normal(mu_K, np.sqrt(var_K), NREAL)
kU_B = rng.normal(kappaU_conditional(M_rem_B),  cond_std)
kU_C = rng.normal(kappaU_conditional(MU_targ_C), cond_std)

# ===========================================================================
# FIGURE  (2x2)
plt.rcParams.update({'font.size': 10.5})
fig = plt.figure(figsize=(12.4, 8.4))
gs = fig.add_gridspec(2, 2, wspace=0.24, hspace=0.34)
cA, cB, cC = '#228833', '#aa3377', '#3366cc'   # A green, B magenta, C blue

# panel (a): total mass histograms
axa = fig.add_subplot(gs[0,0])
bins = np.linspace(0.85, 1.15, 80)
axa.hist(M_tot_cur/M_HOST, bins=bins, color='#c44', alpha=0.7,
         label=f'(0) current model 3\nstd={M_tot_cur.std()/M_HOST*100:.1f}%')
axa.axvline(1.0, color='#333', lw=2.2)
axa.annotate('(A) realized carve\n(B) conditional $\\kappa_U$\n(C) proportional hybrid\nall $\\equiv M$ exactly',
             xy=(1.0, axa.get_ylim()[1]*0.60), xytext=(1.043, axa.get_ylim()[1]*0.58),
             fontsize=8.8, ha='left', va='center',
             arrowprops=dict(arrowstyle='->', color='#333'))
axa.set_xlabel(r'total mass laid down $/\,M_{\rm host}$')
axa.set_ylabel('realizations')
axa.set_title('(a)  per-realization mass budget', fontsize=10.5, loc='left')
axa.legend(loc='upper left', fontsize=8.6, frameon=False)

# panel (b): joint (M_U, kappa_U) + conditional regression
axb = fig.add_subplot(gs[0,1])
sub = slice(0, 6000)
axb.scatter(MU[sub]/M_U_MEAN, KU[sub]/mu_K, s=3, alpha=0.16, color='#4477aa', rasterized=True)
mx = np.linspace(np.percentile(MU,0.3), np.percentile(MU, 99.7), 100)
axb.plot(mx/M_U_MEAN, kappaU_conditional(mx)/mu_K, color='#cc3311', lw=2.4,
         label='closed-form\n$E[\\kappa_U\\,|\\,M_U{=}m^*]$')
axb.fill_between(mx/M_U_MEAN,
                 (kappaU_conditional(mx)-cond_std)/mu_K,
                 (kappaU_conditional(mx)+cond_std)/mu_K,
                 color='#cc3311', alpha=0.18, label=r'$\pm\sigma_U\sqrt{1-\rho^2}$')
axb.set_xlabel(r'unresolved-band mass $M_U\,/\,\langle M_U\rangle$')
axb.set_ylabel(r'$\kappa_U\,/\,\langle\kappa_U\rangle$')
axb.set_title(f'(b)  $\\kappa_U$ and $M_U$ jointly Gaussian ($\\rho={rho:.2f}$)',
              fontsize=10.5, loc='left')
axb.legend(loc='upper left', fontsize=8.6, frameon=False)
axb.set_xlim(0.7, 1.3); axb.set_ylim(0.3, 1.5)

# panel (c): the reservoir problem, now with method C's unresolved residual.
axc = fig.add_subplot(gs[1,0])
b2 = np.linspace(-0.20, 1.0, 90)
axc.hist(host_A_mass/M_HOST, bins=b2, color=cA, alpha=0.55,
         label=f'(A) host mass ($\\langle\\rangle$=0.82$M$, neg {frac_host_neg*100:.1f}%)')
axc.hist(M_rem_B/M_HOST, bins=b2, color=cB, alpha=0.55,
         label=f'(B) unres. residual ($\\langle\\rangle$=0.04$M$, neg {frac_rem_neg*100:.0f}%)')
axc.hist(MU_targ_C/M_HOST, bins=b2, color=cC, alpha=0.6,
         label=f'(C) unres. residual ($\\langle\\rangle$=0.04$M$, neg {frac_C_neg*100:.2f}%)')
axc.axvspan(-0.20, 0.0, color='#cc2222', alpha=0.10)
axc.axvline(0.0, color='#cc2222', lw=1.6)
axc.text(-0.10, axc.get_ylim()[1]*0.55, 'unphysical\n(neg. mass)', color='#992222',
         fontsize=8.2, ha='center', va='center', rotation=90)
axc.set_xlabel(r'reservoir mass that absorbs the shot noise $/M$')
axc.set_ylabel('realizations')
axc.set_title('(c)  can the reservoir hold it?', fontsize=10.5, loc='left')
axc.legend(loc='upper right', fontsize=8.0, frameon=False)
axc.set_xlim(-0.20, 1.0)

# panel (d): the induced kappa_U <-> resolved-excess correlation for A / B / C.
axd = fig.add_subplot(gs[1,1])
ex = delta/M_HOST
axd.scatter(ex[:5000], kU_A[:5000]/mu_K, s=3, alpha=0.12, color=cA, rasterized=True)
axd.scatter(ex[:5000], kU_C[:5000]/mu_K, s=3, alpha=0.14, color=cC, rasterized=True)
xg = np.linspace(np.percentile(ex,0.5), np.percentile(ex,99.5), 50)
axd.plot(xg, np.ones_like(xg), color=cA, lw=2.2, label='(A) independent (no corr.)')
axd.plot(xg, (mu_K - slope*xg*M_HOST)/mu_K, color=cB, lw=2.2, ls='--',
         label='(B) full-conditional (over-steep)')
axd.plot(xg, (mu_K - slope*w_u*xg*M_HOST)/mu_K, color=cC, lw=2.6,
         label='(C) proportional (correct slope)')
axd.axhline(1.0, color='#bbb', lw=0.8, ls=':')
lim = np.percentile(np.abs(ex), 99)
axd.set_xlabel(r'resolved-clump excess $\delta/M$')
axd.set_ylabel(r'$\kappa_U\,/\,\langle\kappa_U\rangle$')
axd.set_title('(d)  induced clump$-\\kappa_U$ correlation', fontsize=10.5, loc='left')
axd.legend(loc='upper right', fontsize=8.0, frameon=False)
axd.set_xlim(-lim, lim); axd.set_ylim(0.4, 1.6)

fig.suptitle(r'Mass-conserving host carving: realized carve (A), conditional $\kappa_U$ (B), '
             r'proportional hybrid (C)  '
             f'[$M={M_HOST:.0e}\\,M_\\odot$, $f_b={fb_frac:.2f}$, $w_u={w_u:.3f}$]',
             fontsize=11, y=0.975)
fig.savefig('plots/subhalo_mass_budget.png', dpi=140, bbox_inches='tight')
print("\nwrote plots/subhalo_mass_budget.png")
