"""
Comparison figure: pyHalo (defaults, infall SHMF + Galacticus stripping) vs our
evolved-SHMF model, matched host M=1e13 at z_l=0.5 (z_s=2), aperture R<38 kpc
(pyHalo 12'' full-opening cone).

Panel (a): projected subhalo mass function d^2N/(dln m dA):
  - pyHalo INFALL masses (their SHMF as parametrized)
  - pyHalo BOUND masses (after their tidal evolution)  <- maps onto OUR evolved SHMF
  - our evolved JvdB14 SHMF x Han+16 aperture fraction
Panel (b): normalized projected radial profiles.

Run with:
  .venv_pyhalo/bin/python tmp/pyhalo_vs_ours_plot.py
  .venv_pyhalo/bin/python tmp/pyhalo_vs_ours_plot.py --mhi 1e13 --suffix mhi1e13
"""
import argparse
import numpy as np
from pyHalo.preset_models import preset_model_from_name

parser = argparse.ArgumentParser()
parser.add_argument("--mhost", type=float, default=1e13)
parser.add_argument("--zl", type=float, default=0.5)
parser.add_argument("--zs", type=float, default=2.0)
parser.add_argument("--mlo", type=float, default=1e7)
parser.add_argument("--mhi", type=float, default=10**11.5)
parser.add_argument("--nreal", type=int, default=30)
parser.add_argument("--suffix", default="")
args = parser.parse_args()

MHOST, ZL, ZS = args.mhost, args.zl, args.zs
MLO, MHI = args.mlo, args.mhi
NREAL = args.nreal
suffix = f"_{args.suffix}" if args.suffix else ""
CDM = preset_model_from_name('CDM')

m_inf, m_bnd, R_all = [], [], []
kpc_per_asec = None
for i in range(NREAL):
    real = CDM(z_lens=ZL, z_source=ZS, log_m_host=np.log10(MHOST),
               log_mlow=np.log10(MLO), log_mhigh=np.log10(MHI),
               LOS_normalization=0.0, cone_opening_angle_arcsec=12.0,
               two_halo_contribution=False)          # ALL OTHER DEFAULTS
    for h in real.halos:
        m_inf.append(h.mass)
        try:
            m_bnd.append(h.bound_mass)
        except Exception:
            m_bnd.append(np.nan)
        R_all.append(np.hypot(h.x, h.y))
        if kpc_per_asec is None:
            kpc_per_asec = h.lens_cosmo.cosmo.kpc_proper_per_asec(ZL)
m_inf = np.array(m_inf); m_bnd = np.array(m_bnd); R_kpc = np.array(R_all)*kpc_per_asec
R_ap = 6.0*kpc_per_asec
A_ap = np.pi*R_ap**2
print(f"kpc/arcsec={kpc_per_asec:.3f}, aperture R={R_ap:.1f} kpc, A={A_ap:.0f} kpc^2")
print(f"N/real: {len(m_inf)/NREAL:.0f}; bound/infall mass ratio median: "
      f"{np.nanmedian(m_bnd/m_inf):.3f}")

# histograms per dlnm per area (per realization)
bins = np.logspace(np.log10(MLO), np.log10(MHI), 20)
lnw = np.diff(np.log(bins))
ctr = np.sqrt(bins[1:]*bins[:-1])
h_inf = np.histogram(m_inf, bins=bins)[0]/NREAL/lnw/A_ap
mb = m_bnd[np.isfinite(m_bnd) & (m_bnd > 0)]
h_bnd = np.histogram(mb, bins=bins)[0]/NREAL/lnw/A_ap

# ---- our evolved SHMF, projected into the same aperture ----
G_NORM, FS = 0.040, 0.139           # g(1e13, z=0.5), f_s  (from psub_check run)
ALPHA, BETA, OMEGA = -0.82, 50.0, 4.0
# Han+16 projected aperture fraction for R < R_ap (c_host from cons14 = 5.52, r200=378 kpc)
c_h, r200 = 5.52, 378.4
xs3 = np.linspace(1e-4, 1, 3000)
nu3 = (1/np.sqrt((xs3/0.54)**(-2.5)+1))/(1+c_h*xs3)**2
def SigN(Rk):
    zmax = np.sqrt(max(r200**2-Rk**2, 0))
    if zmax <= 0: return 0.0
    zz = np.linspace(0, zmax, 1500)
    return 2*np.trapezoid(np.interp(np.sqrt(Rk**2+zz**2)/r200, xs3, nu3), zz)
Rg = np.concatenate([np.linspace(0.2, 40, 300), np.linspace(40.5, r200, 300)])
Sn = np.array([SigN(R) for R in Rg])
tot = np.trapezoid(Sn*2*np.pi*Rg, Rg)
sel = Rg < R_ap
f_ap = np.trapezoid(Sn[sel]*2*np.pi*Rg[sel], Rg[sel])/tot
print(f"our aperture fraction (R<{R_ap:.0f} kpc) = {f_ap:.4f}")
ours = G_NORM*(ctr/MHOST)**ALPHA*np.exp(-BETA*(ctr/MHOST)**OMEGA)*f_ap/A_ap

print(f"\n{'m [Msun]':>10} {'pyH infall':>11} {'pyH bound':>11} {'ours(evolved)':>13} {'ours/bound':>10}")
for i in range(0, len(ctr), 3):
    r = ours[i]/h_bnd[i] if h_bnd[i] > 0 else np.nan
    print(f"{ctr[i]:10.2e} {h_inf[i]:11.3e} {h_bnd[i]:11.3e} {ours[i]:13.3e} {r:10.2f}")

# radial profile (area-normalized) within aperture
pr_h, pr_e = np.histogram(R_kpc, bins=np.linspace(1, R_ap, 10))
pr_ctr = 0.5*(pr_e[1:]+pr_e[:-1]); pr_area = np.pi*(pr_e[1:]**2-pr_e[:-1]**2)
py_n = pr_h/pr_area; py_n = py_n/np.trapezoid(py_n*2*np.pi*pr_ctr, pr_ctr)
our_n = np.array([SigN(R) for R in pr_ctr])
our_n = our_n/np.trapezoid(our_n*2*np.pi*pr_ctr, pr_ctr)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_positive(ax, x, y, *args, **kwargs):
    y = np.asarray(y)
    ok = np.isfinite(y) & (y > 0)
    return ax.plot(np.asarray(x)[ok], y[ok], *args, **kwargs)

fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2))
plot_positive(a, ctr, h_inf, "o-", color="0.55", ms=4, label="pyHalo infall masses (their SHMF)")
plot_positive(a, ctr, h_bnd, "s-", color="#e8590c", ms=4, label="pyHalo bound masses (after stripping)")
plot_positive(a, ctr, ours, "-", color="#3b5bdb", lw=2, label="this work: evolved SHMF (JvdB14)")
a.set_xscale("log")
a.set_yscale("log")
a.set_xlabel(r"$m\ [M_\odot]$"); a.set_ylabel(r"$d^2N/(d\ln m\, dA)\ [\mathrm{kpc}^{-2}]$")
a.set_title(f"projected SHMF, $R<{R_ap:.0f}$ kpc of $10^{{13}}M_\\odot$ host, $z_l=0.5$")
a.legend(fontsize=8); a.grid(alpha=0.3, which="both")
b.plot(pr_ctr, py_n, "s-", color="#e8590c", ms=4, label="pyHalo (defaults)")
b.plot(pr_ctr, our_n, "-", color="#3b5bdb", lw=2, label="this work (Han+16 projected)")
b.set_xlabel(r"$R\ [\mathrm{kpc}]$"); b.set_ylabel(r"$n(R)$ (area-normalized)")
b.set_title("projected radial profile shape"); b.legend(fontsize=8); b.grid(alpha=0.3)
b.set_ylim(bottom=0)
fig.tight_layout()
main_png = f"/Users/baltabay/Desktop/gw-wl-emulator/plots/figures/pyhalo_vs_ours{suffix}.png"
fig.savefig(main_png, dpi=200)

psi = ctr / MHOST
x_lnpsi = np.log(psi)
fig2, ax = plt.subplots(figsize=(7.2, 4.8))
plot_positive(ax, x_lnpsi, h_inf * A_ap, "o-", color="0.55", ms=4, label="pyHalo infall masses")
plot_positive(ax, x_lnpsi, h_bnd * A_ap, "s-", color="#e8590c", ms=4, label="pyHalo bound masses")
plot_positive(ax, x_lnpsi, ours * A_ap, "-", color="#3b5bdb", lw=2, label="this work: evolved SHMF")
ax.set_yscale("log")
ax.set_xlabel(r"$\ln(m/M_{\rm host})$")
ax.set_ylabel(r"$dN/d\ln(m/M_{\rm host})$")
ax.set_title(
    rf"SHMF in $R<{R_ap:.0f}$ kpc aperture; $M={MHOST:.0e}M_\odot$, "
    rf"$m_{{\rm inf,max}}={MHI:.1e}M_\odot$"
)
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8)
fig2.tight_layout()
psi_png = f"/Users/baltabay/Desktop/gw-wl-emulator/plots/figures/pyhalo_vs_ours_lnpsi{suffix}.png"
fig2.savefig(psi_png, dpi=200)

npz_path = f"/Users/baltabay/Desktop/gw-wl-emulator/tmp/pyhalo_vs_ours{suffix}.npz"
np.savez(npz_path,
         ctr=ctr, h_inf=h_inf, h_bnd=h_bnd, ours=ours, pr_ctr=pr_ctr, py_n=py_n, our_n=our_n,
         f_ap=f_ap, R_ap=R_ap, MHOST=MHOST, ZL=ZL, ZS=ZS, MLO=MLO, MHI=MHI,
         psi=psi, x_lnpsi=x_lnpsi,
         dndlnpsi_infall=h_inf * A_ap, dndlnpsi_bound=h_bnd * A_ap,
         dndlnpsi_ours=ours * A_ap)
print(f"\nsaved {main_png}")
print(f"saved {psi_png}")
print(f"saved {npz_path}")
