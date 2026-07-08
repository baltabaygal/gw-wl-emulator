import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import gamma as Gamma, gammaincc

# Fiducial host values copied from scripts/subhalo_demo.py output.
M0 = 1.0e15
z0 = 0.5
fs = 0.286

ALPHA, BETA, OMEGA = -0.82, 50.0, 4.0
PSI_RES = 1.0e-4

def gamma_norm(fs, psi_res):
    s = (1 + ALPHA) / OMEGA
    denom = Gamma(s) * (gammaincc(s, BETA*psi_res**OMEGA) - gammaincc(s, BETA))
    return OMEGA * BETA**s / denom * fs

def dN_dlnpsi(psi, gamma):
    return gamma * psi**ALPHA * np.exp(-BETA * psi**OMEGA)

def nsub_above(psi_min, gamma):
    val, _ = quad(lambda lp: dN_dlnpsi(np.exp(lp), gamma),
                  np.log(psi_min), 0.0, limit=200)
    return val

def mass_fraction_above(psi_min, gamma):
    val, _ = quad(lambda lp: np.exp(lp) * dN_dlnpsi(np.exp(lp), gamma),
                  np.log(psi_min), 0.0, limit=200)
    return val

psi_floors = np.logspace(-21, -2, 240)

# Case A: keep the original JvdB normalization fixed at psi_res=1e-4 and
# extrapolate the same SHMF below that floor.
gamma_fixed = gamma_norm(fs, PSI_RES)
n_fixed = np.array([nsub_above(p, gamma_fixed) for p in psi_floors])
f_fixed = np.array([mass_fraction_above(p, gamma_fixed) for p in psi_floors])

# Case B: if the lower bound itself is redefined, renormalize gamma at each
# floor so the total subhalo mass fraction remains fs above that new floor.
gamma_renorm = np.array([gamma_norm(fs, p) for p in psi_floors])
n_renorm = np.array([nsub_above(p, g) for p, g in zip(psi_floors, gamma_renorm)])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))

ax1.plot(psi_floors, n_fixed, lw=2.4, label=r'fixed normalization at $\psi_{\rm res}=10^{-4}$')
ax1.plot(psi_floors, n_renorm, lw=2.0, ls='--',
         label=r'renormalized to same $f_s$ at each floor')
ax1.axvline(PSI_RES, color='0.35', ls=':', lw=1.5)
ax1.text(PSI_RES*1.1, 6.0, r'current $\psi_{\rm res}=10^{-4}$',
         rotation=90, va='bottom', fontsize=9, color='0.35')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.invert_xaxis()
ax1.set_xlabel(r'lower mass-ratio bound $\psi_{\min}=m_{\rm sub}/M_0$')
ax1.set_ylabel(r'expected count $N_{\rm sub}(>\psi_{\min})$')
ax1.set_title('Subhalo count grows as the mass floor is lowered')
ax1.grid(alpha=0.3, which='both')
ax1.legend(fontsize=9)

ax2.plot(psi_floors, f_fixed, lw=2.4, color='tab:green',
         label=r'mass fraction above $\psi_{\min}$, fixed normalization')
ax2.axhline(fs, color='0.25', ls='--', lw=1.3, label=rf'fiducial $f_s={fs:.3f}$')
ax2.axvline(PSI_RES, color='0.35', ls=':', lw=1.5)
ax2.set_xscale('log')
ax2.invert_xaxis()
ax2.set_xlabel(r'lower mass-ratio bound $\psi_{\min}=m_{\rm sub}/M_0$')
ax2.set_ylabel(r'$\int_{\psi_{\min}}^1 \psi\,dN/d\ln\psi\,d\ln\psi$')
ax2.set_title('Mass fraction changes slowly; number does not plateau')
ax2.grid(alpha=0.3, which='both')
ax2.legend(fontsize=9)

fig.suptitle(rf'Fiducial host $M_0={M0:.1e}\,M_\odot$, $z={z0}$; JvdB14 evolved SHMF',
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = 'plots/shmf_psi_floor.png'
fig.savefig(out, dpi=140, bbox_inches='tight')

print(f'Nsub(>1e-4) = {nsub_above(1e-4, gamma_fixed):.1f}')
print(f'Nsub(>1e-5) = {nsub_above(1e-5, gamma_fixed):.1f}')
print(f'Nsub(>1e-6) = {nsub_above(1e-6, gamma_fixed):.1f}')
print(f'Nsub(>1e-7) = {nsub_above(1e-7, gamma_fixed):.1f}')
print(f'Nsub(>1e-9) = {nsub_above(1e-9, gamma_fixed):.3e}')
print(f'Nsub(>1e-12) = {nsub_above(1e-12, gamma_fixed):.3e}')
print(f'Nsub(>1e-15) = {nsub_above(1e-15, gamma_fixed):.3e}')
print(f'Nsub(>1e-18) = {nsub_above(1e-18, gamma_fixed):.3e}')
print(f'Nsub(>1e-21) = {nsub_above(1e-21, gamma_fixed):.3e}')
print('wrote', out)
