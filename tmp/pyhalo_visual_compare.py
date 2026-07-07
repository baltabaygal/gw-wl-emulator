"""
Visual realization comparison: our subhalo model vs pyHalo, same host
(M=1e13 Msun, z_l=0.5), in the style of subhalo_resolved_demo_default_factor.png.

Top row:    (a) OUR realization, full host disc r200=378 kpc (brute, m>=1e7);
            (b) OUR realization, zoomed to pyHalo's rendering region R<38 kpc;
            (c) pyHalo realization (defaults, infall window to 10^11.5), R<38 kpc,
                colored by BOUND mass (what actually lenses).
Bottom row: (d) projected SHMF comparison (from tmp/pyhalo_vs_ours.npz);
            (e) mean N(>m) inside R<38 kpc per realization, both codes.

Run: .venv_pyhalo/bin/python tmp/pyhalo_visual_compare.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import Normalize

RNG = np.random.default_rng(11)
MHOST, ZL, ZS = 1e13, 0.5, 2.0
G_NORM = 0.040                      # g(1e13, z=0.5), corrected wf
ALPHA, BETA, OMEGA, PSI_MAX = -0.82, 50.0, 4.0, 1.0
MFLOOR = 1e7
C_H, R200 = 5.52, 378.4             # cons14 host
R_AP = 37.9                         # pyHalo 12'' cone radius at z=0.5 [kpc]

# ---------------- our realization (production spec, brute to m_floor) ----------------
pa_lo, pa_hi = (MFLOOR/MHOST)**ALPHA, PSI_MAX**ALPHA
Nprop = (G_NORM/ALPHA)*(pa_hi - pa_lo)
Nc = RNG.poisson(Nprop)
u = RNG.random(Nc)
psi = (pa_lo + u*(pa_hi - pa_lo))**(1/ALPHA)
keep = RNG.random(Nc) <= np.exp(-BETA*psi**OMEGA)
m_our = psi[keep]*MHOST
# Han+16 radial CDF
xs = np.linspace(0, 1, 4000)
with np.errstate(divide='ignore'):
    B = np.where(xs > 0, 1/np.sqrt((xs/0.54)**(-2.5)+1), 0.0)
w = xs**2/(1+C_H*xs)**2*B
cdf = np.concatenate([[0], np.cumsum(0.5*(w[1:]+w[:-1])*np.diff(xs))]); cdf /= cdf[-1]
x3 = np.interp(RNG.random(m_our.size), cdf, xs)*R200
cth = RNG.uniform(-1, 1, m_our.size); az = RNG.uniform(0, 2*np.pi, m_our.size)
R2 = x3*np.sqrt(1-cth**2)
X_our, Y_our = R2*np.cos(az), R2*np.sin(az)

# ---------------- pyHalo realization ----------------
from pyHalo.preset_models import preset_model_from_name
CDM = preset_model_from_name('CDM')
real = CDM(z_lens=ZL, z_source=ZS, log_m_host=np.log10(MHOST),
           log_mlow=7.0, log_mhigh=11.5, LOS_normalization=0.0,
           cone_opening_angle_arcsec=12.0, two_halo_contribution=False)
kpa = real.halos[0].lens_cosmo.cosmo.kpc_per_asec(ZL) if hasattr(real.halos[0].lens_cosmo.cosmo, 'kpc_per_asec') \
      else real.halos[0].lens_cosmo.cosmo.kpc_proper_per_asec(ZL)
m_inf = np.array([h.mass for h in real.halos])
m_bnd = np.array([getattr(h, 'bound_mass', np.nan) for h in real.halos])
X_py = np.array([h.x for h in real.halos])*kpa
Y_py = np.array([h.y for h in real.halos])*kpa
ok = np.isfinite(m_bnd) & (m_bnd > 0)

# ---------------- figure ----------------
norm = Normalize(vmin=7, vmax=11.5)
cmap = plt.cm.autumn_r
def sizes(m): return np.clip(4 + 26*(np.log10(m)-7)/4.5, 0.8, None)

fig = plt.figure(figsize=(13.2, 8.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0], hspace=0.32, wspace=0.28)

axA = fig.add_subplot(gs[0, 0])
axA.scatter(X_our, Y_our, c=np.log10(m_our), s=sizes(m_our), cmap=cmap, norm=norm,
            edgecolors='k', linewidths=0.15, zorder=3)
axA.add_patch(Circle((0, 0), R200, fill=False, ls='--', color='#1c7ed6', lw=1.4))
axA.add_patch(Circle((0, 0), R_AP, fill=False, ls='-', color='0.25', lw=1.2))
axA.set_xlim(-1.1*R200, 1.1*R200); axA.set_ylim(-1.1*R200, 1.1*R200)
axA.set_aspect('equal')
axA.set_title(f"(a) this work: full host\nN={m_our.size} with $m\\geq10^7$", fontsize=10)
axA.set_xlabel("x [kpc]"); axA.set_ylabel("y [kpc]")
axA.annotate("pyHalo renders\nonly this region", xy=(R_AP*0.7, -R_AP*0.7), xytext=(150, -300),
             fontsize=8.5, color='0.25', arrowprops=dict(arrowstyle='->', color='0.25', lw=0.9))

axB = fig.add_subplot(gs[0, 1])
inb = np.hypot(X_our, Y_our) < R_AP
axB.scatter(X_our[inb], Y_our[inb], c=np.log10(m_our[inb]), s=1.6*sizes(m_our[inb]),
            cmap=cmap, norm=norm, edgecolors='k', linewidths=0.15)
axB.add_patch(Circle((0, 0), R_AP, fill=False, ls='-', color='0.25', lw=1.2))
axB.set_xlim(-R_AP*1.08, R_AP*1.08); axB.set_ylim(-R_AP*1.08, R_AP*1.08)
axB.set_aspect('equal')
axB.set_title(f"(b) this work: inner {R_AP:.0f} kpc\nN={inb.sum()} with $m\\geq10^7$ (bound)", fontsize=10)
axB.set_xlabel("x [kpc]")

axC = fig.add_subplot(gs[0, 2])
sc = axC.scatter(X_py[ok], Y_py[ok], c=np.log10(m_bnd[ok]), s=1.6*sizes(m_bnd[ok]),
                 cmap=cmap, norm=norm, edgecolors='k', linewidths=0.15)
axC.add_patch(Circle((0, 0), R_AP, fill=False, ls='-', color='0.25', lw=1.2))
axC.set_xlim(-R_AP*1.08, R_AP*1.08); axC.set_ylim(-R_AP*1.08, R_AP*1.08)
axC.set_aspect('equal')
nb7 = int(((m_bnd[ok]) >= 1e7).sum())
axC.set_title(f"(c) pyHalo: inner {R_AP:.0f} kpc, bound masses\n"
              f"{ok.sum()} remnants, only N={nb7} with $m_b\\geq10^7$", fontsize=10)
axC.set_xlabel("x [kpc]")
cb = fig.colorbar(sc, ax=[axA, axB, axC], shrink=0.8, pad=0.015)
cb.set_label(r"$\log_{10} m\ [M_\odot]$")

# (d) projected SHMF comparison, from the saved comparison run
d = np.load('/Users/baltabay/Desktop/gw-wl-emulator/tmp/pyhalo_vs_ours.npz')
axD = fig.add_subplot(gs[1, 0:2])
axD.loglog(d['ctr'], d['h_inf'], 'o-', color='0.55', ms=4, label='pyHalo infall masses (their SHMF)')
axD.loglog(d['ctr'], d['h_bnd'], 's-', color='#e8590c', ms=4, label='pyHalo bound masses (after Galacticus stripping)')
axD.loglog(d['ctr'], d['ours'], '-', color='#3b5bdb', lw=2, label='this work: evolved SHMF (JvdB14, stripping pre-integrated)')
axD.set_xlabel(r"$m\ [M_\odot]$"); axD.set_ylabel(r"$d^2N/(d\ln m\, dA)\ [\mathrm{kpc}^{-2}]$")
axD.set_title(f"(d) projected subhalo mass function inside R<{R_AP:.0f} kpc", fontsize=10)
axD.legend(fontsize=8.5); axD.grid(alpha=0.3, which='both')

# (e) cumulative N(>m) inside the aperture (single realizations shown above)
axE = fig.add_subplot(gs[1, 2])
mg = np.logspace(7, 11, 40)
axE.loglog(mg, [(m_our[inb] > m).sum() for m in mg], color='#3b5bdb', lw=2, label='this work')
axE.loglog(mg, [(m_bnd[ok] > m).sum() for m in mg], color='#e8590c', lw=2, label='pyHalo (bound)')
axE.loglog(mg, [(m_inf > m).sum() for m in mg], color='0.55', lw=1.2, ls='--', label='pyHalo (infall)')
axE.set_xlabel(r"$m\ [M_\odot]$"); axE.set_ylabel(r"$N(>m)$, $R<38$ kpc")
axE.set_title("(e) cumulative counts (one realization)", fontsize=10)
axE.legend(fontsize=8.5); axE.grid(alpha=0.3, which='both')

fig.suptitle(r"Same host ($M=10^{13}M_\odot$, $z_l=0.5$): this work vs pyHalo — "
             "different routes to the bound-subhalo population", fontsize=12, y=0.99)
fig.savefig('/Users/baltabay/Desktop/gw-wl-emulator/plots/figures/pyhalo_visual_compare.png',
            dpi=200, bbox_inches='tight')
print(f"ours: N_total={m_our.size}, N(<38kpc)={inb.sum()}")
print(f"pyHalo: N={ok.sum()} in cone; median stripping {np.nanmedian(m_bnd/m_inf):.4f}")
print("saved plots/figures/pyhalo_visual_compare.png")
