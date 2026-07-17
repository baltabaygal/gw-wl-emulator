import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import gamma as Gamma, gammaincc

M_FLOOR = 1.0e7
PSI_RES = 1.0e-4
ALPHA, BETA, OMEGA = -0.82, 50.0, 4.0

# Fiducial normalization used for the count-scaling diagnostic. The lensing
# panels below use the actual screen_rows.npy output from subhalo_screen.py.
FS_FID = 0.286

def gamma_norm(fs, psi_res):
    s = (1 + ALPHA) / OMEGA
    den = Gamma(s) * (gammaincc(s, BETA*psi_res**OMEGA) - gammaincc(s, BETA))
    return OMEGA * BETA**s / den * fs

def dN_dlnpsi(psi, gamma):
    return gamma * psi**ALPHA * np.exp(-BETA * psi**OMEGA)

def nsub_for_host_mass(M, gamma):
    psi_min = M_FLOOR / M
    psi_max = 0.1
    if psi_min >= psi_max:
        return 0.0
    val, _ = quad(lambda lp: dN_dlnpsi(np.exp(lp), gamma),
                  np.log(psi_min), np.log(psi_max), limit=200)
    return val

rows = np.load('plots/screen_rows.npy')          # (z, M, barNH, cH3, d3)
z, M, barNH, cH3, d3 = rows.T

Mgrid = np.logspace(8, 16, 240)
psi_min = M_FLOOR / Mgrid
gamma_fid = gamma_norm(FS_FID, PSI_RES)
Nsub = np.array([nsub_for_host_mass(Mh, gamma_fid) for Mh in Mgrid])

mbins = np.logspace(8, 16, 45)
ctr = np.sqrt(mbins[:-1] * mbins[1:])
host_k3, _ = np.histogram(M, bins=mbins, weights=cH3)
sub_k3, _ = np.histogram(M, bins=mbins, weights=d3)
host_counts, _ = np.histogram(M, bins=mbins, weights=barNH)
frac = np.divide(sub_k3, host_k3, out=np.full_like(sub_k3, np.nan), where=host_k3 > 0)

fig, axs = plt.subplots(2, 2, figsize=(13.8, 9.6))
ax1, ax2, ax3, ax4 = axs.ravel()

ax1.plot(Mgrid, psi_min, lw=2.4)
ax1.axhline(PSI_RES, color='0.35', ls='--', lw=1.4, label=r'JvdB $\psi_{\rm res}=10^{-4}$')
ax1.axvline(M_FLOOR/PSI_RES, color='0.35', ls=':', lw=1.4)
ax1.text(M_FLOOR/PSI_RES*1.1, 2e-8, r'$M=10^{11}M_\odot$', rotation=90,
         va='bottom', fontsize=9, color='0.35')
ax1.set_xscale('log'); ax1.set_yscale('log')
ax1.set_xlabel(r'host mass $M_{\rm host}\ [M_\odot]$')
ax1.set_ylabel(r'$\psi_{\min}=10^7M_\odot/M_{\rm host}$')
ax1.set_title('Absolute subhalo floor becomes smaller in massive hosts')
ax1.grid(alpha=0.3, which='both')
ax1.legend(fontsize=9)

ax2.plot(Mgrid, Nsub, lw=2.4, color='tab:orange')
ax2.axvline(M_FLOOR/PSI_RES, color='0.35', ls=':', lw=1.4)
ax2.set_xscale('log'); ax2.set_yscale('log')
ax2.set_xlabel(r'host mass $M_{\rm host}\ [M_\odot]$')
ax2.set_ylabel(r'expected $N_{\rm sub}(10^7M_\odot<m<0.1M_{\rm host})$')
ax2.set_title('Expected resolved subhalo count rises with host mass')
ax2.grid(alpha=0.3, which='both')

ax3.plot(ctr, host_k3, color='0.25', lw=2.2, label=r'host-only $d\langle\kappa^3\rangle$')
ax3.plot(ctr, sub_k3, color='tab:red', lw=2.2, label=r'subhalo $\Delta d\langle\kappa^3\rangle$')
ax3b = ax3.twinx()
ax3b.bar(ctr, host_counts, width=np.diff(mbins), align='center',
         color='tab:blue', alpha=0.18, label='expected host encounters')
ax3.set_xscale('log'); ax3.set_yscale('log'); ax3b.set_yscale('log')
ax3.set_xlabel(r'host mass $M_{\rm host}\ [M_\odot]$')
ax3.set_ylabel(r'mass-bin contribution to $\langle\kappa^3\rangle$')
ax3b.set_ylabel('expected host encounters per bin', color='tab:blue')
ax3.set_title(r'Absolute $\kappa^3$ contribution by host mass, $z_s=1$')
ax3.grid(alpha=0.3, which='both')
ax3.legend(fontsize=9, loc='upper left')

ax4.plot(ctr, 100*frac, color='tab:purple', lw=2.4)
ax4.axhline(100*np.nansum(sub_k3)/np.nansum(host_k3), color='0.35',
            ls='--', lw=1.4, label='global weighted ratio')
ax4.set_xscale('log')
ax4.set_xlabel(r'host mass $M_{\rm host}\ [M_\odot]$')
ax4.set_ylabel(r'$\Delta d\langle\kappa^3\rangle_{\rm sub}/d\langle\kappa^3\rangle_{\rm host}$ [%]')
ax4.set_title('Fractional subhalo boost by host mass')
ax4.grid(alpha=0.3, which='both')
ax4.legend(fontsize=9)

fig.suptitle(r'Fixed physical subhalo floor: $m_{\rm sub,min}=C.Mmin=10^7M_\odot$',
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = 'plots/subhalo_mfloor_effect.png'
fig.savefig(out, dpi=140, bbox_inches='tight')

for Mh in [1e11, 1e12, 1e13, 1e14, 1e15]:
    print(f'Mhost={Mh:.1e}: psi_min={M_FLOOR/Mh:.1e}, Nsub={nsub_for_host_mass(Mh, gamma_fid):.2e}')
print('wrote', out)
