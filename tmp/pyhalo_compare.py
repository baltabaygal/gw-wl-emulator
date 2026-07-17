"""
Option B: independent-code cross-check of the subhalo population against pyHalo (Gilman+).

Compares the population statistics that drive our lensing signal, at a MATCHED host
(M=1e13, z_l=0.5, z_s=2) and mass window [1e7, 1e10] Msun, subhalos only:
 1. SHMF slope  dN/dlnm  (ours: psi^alpha, alpha=-0.82  =>  dN/dm ~ m^-1.82)
 2. effective mass  m_eff = <m^2>/<m>  (sets P_1sh amplitude; ~normalization independent)
 3. projected radial profile shape  n(R)  (ours: Han+16 anti-biased; theirs: default)

Run with the pyHalo venv:  .venv_pyhalo/bin/python tmp/pyhalo_compare.py
Writes tmp/pyhalo_compare.txt + .npz.
"""
import numpy as np
from pyHalo.preset_models import preset_model_from_name
CDM = preset_model_from_name('CDM')

MLO, MHI, MHOST = 1e7, 1e10, 1e13
NREAL = 40
SLOPE_DNDM = -1.82        # match OUR dN/dm slope (psi^-0.82 => m^-1.82)

def draw(shmf_slope, trunc=None):
    kw = dict(z_lens=0.5, z_source=2.0, log_m_host=np.log10(MHOST),
              log_mlow=np.log10(MLO), log_mhigh=np.log10(MHI),
              LOS_normalization=0.0, cone_opening_angle_arcsec=12.0,
              two_halo_contribution=False, shmf_log_slope=shmf_slope)
    if trunc is not None:
        kw['truncation_model_subhalos'] = trunc
    m_all, R_all = [], []
    lens_cosmo = None
    for i in range(NREAL):
        real = CDM(**kw)
        h = real.halos
        m_all.append(np.array([x.mass for x in h]))
        R_all.append(np.hypot([x.x for x in h], [x.y for x in h]))  # arcsec
        if lens_cosmo is None and h:
            lens_cosmo = h[0].lens_cosmo
    m = np.concatenate(m_all); R = np.concatenate(R_all)
    # arcsec -> kpc at the lens
    kpc_per_arcsec = lens_cosmo.cosmo.kpc_proper_per_asec(0.5)
    return m, R*kpc_per_arcsec

def meff(m):  return (m**2).mean()/m.mean()

# ---- pyHalo, matched slope, default (tidal-truncated) profile ----
m_py, R_py = draw(SLOPE_DNDM)
# measure the drawn dN/dlnm slope
lm = np.log(m_py); hist, edges = np.histogram(lm, bins=18)
ctr = 0.5*(edges[1:]+edges[:-1]); good = hist > 5
p = np.polyfit(ctr[good], np.log(hist[good]), 1)   # d ln N / d ln m
print(f"pyHalo: N/realization = {len(m_py)/NREAL:.1f}, mass [{m_py.min():.2e},{m_py.max():.2e}]")
print(f"pyHalo measured dN/dlnm slope = {p[0]:.3f}  (expected {1+SLOPE_DNDM:.3f})")
print(f"pyHalo m_eff = <m^2>/<m> = {meff(m_py):.3e} Msun")

# ---- ours: analytic SHMF (thinned, psi_max=1) over the same window ----
ALPHA, BETA, OMEGA = -0.82, 50.0, 4.0
lmm = np.linspace(np.log(MLO), np.log(MHI), 400); mm = np.exp(lmm); psi = mm/MHOST
dNdlnm = psi**ALPHA*np.exp(-BETA*psi**OMEGA)
m1 = np.trapezoid(dNdlnm*mm, lmm)/np.trapezoid(dNdlnm, lmm)
m2 = np.trapezoid(dNdlnm*mm**2, lmm)/np.trapezoid(dNdlnm, lmm)
print(f"\nours:   analytic dN/dlnm slope = {ALPHA:.3f} (m^{1+ALPHA-2:.2f} in dN/dm... = {ALPHA:.2f} in dN/dlnm)")
print(f"ours:   m_eff = <m^2>/<m> = {m2/m1:.3e} Msun")
print(f"\nm_eff ratio (pyHalo/ours) = {meff(m_py)/(m2/m1):.3f}")

# ---- radial profile shapes (normalized), our Han+16 projected vs pyHalo ----
# our projected n(R): project 3D anti-biased profile for a 1e13 host (c from cons14)
def cons14(M, z, h=0.674):
    b=-0.101+0.026*z; a=0.520+(0.905-0.520)*np.exp(-0.617*z**1.21)
    return 10**(a+b*np.log10(M/(1e12/h)))
c_h = cons14(MHOST, 0.5)
rho_c0=277.394*0.674**2; Az=lambda z:0.315*(1+z)**3+(1-0.315-0.315/(1+3402))+0.315/(1+3402)*(1+z)**4
r200=(3*MHOST/(4*np.pi*200*Az(0.5)*rho_c0))**(1/3)
xs3=np.linspace(1e-4,1,3000)
nu3=(1/np.sqrt((xs3/0.54)**(-2.5)+1))/(1+c_h*xs3)**2
def SigmaN(Rk):
    zmax=np.sqrt(max(r200**2-Rk**2,0));
    if zmax<=0: return 0.0
    zz=np.linspace(0,zmax,1500); x=np.sqrt(Rk**2+zz**2)/r200
    return 2*np.trapezoid(np.interp(x,xs3,nu3),zz)
Rgrid=np.linspace(2,38,25)
our_n=np.array([SigmaN(R) for R in Rgrid]); our_n/=np.trapezoid(our_n*2*np.pi*Rgrid,Rgrid)
py_h,py_e=np.histogram(R_py,bins=np.linspace(2,38,13),density=False)
py_ctr=0.5*(py_e[1:]+py_e[:-1]); py_area=np.pi*(py_e[1:]**2-py_e[:-1]**2)
py_n=py_h/py_area; py_n=py_n/np.trapezoid(py_n*2*np.pi*py_ctr,py_ctr)
print("\nprojected radial profile n(R) (both area-normalized):")
print(f"{'R[kpc]':>8} {'ours':>11} {'pyHalo':>11} {'ratio':>7}")
for R in [4,8,14,22,32]:
    o=np.interp(R,Rgrid,our_n); pv=np.interp(R,py_ctr,py_n)
    print(f"{R:8.0f} {o:11.3e} {pv:11.3e} {pv/o:7.2f}")

np.savez('/Users/baltabay/Desktop/gw-wl-emulator/tmp/pyhalo_compare.npz',
         m_py=m_py, R_py=R_py, Rgrid=Rgrid, our_n=our_n, py_ctr=py_ctr, py_n=py_n,
         meff_py=meff(m_py), meff_ours=m2/m1, slope_py=p[0])
print("\nsaved tmp/pyhalo_compare.npz")
