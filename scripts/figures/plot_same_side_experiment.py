import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Analytical NFW profile functions for lensing convergence (kappa)
def Fg(x):
    """Radial profile factor F(x) for NFW lensing convergence."""
    x = np.atleast_1d(x)
    res = np.zeros_like(x)
    
    idx = x > 1
    if np.any(idx):
        xx = x[idx]
        t = np.arctan(np.sqrt((xx - 1.0) / (xx + 1.0))) / np.sqrt(xx**2 - 1.0)
        res[idx] = (1.0 - 2.0 * t) / (xx**2 - 1.0)
        
    idx = x < 1
    if np.any(idx):
        xx = x[idx]
        t = np.arctanh(np.sqrt((1.0 - xx) / (xx + 1.0))) / np.sqrt(1.0 - xx**2)
        res[idx] = (1.0 - 2.0 * t) / (xx**2 - 1.0)
        
    idx = np.isclose(x, 1.0)
    if np.any(idx):
        res[idx] = 1.0 / 3.0
        
    return res

def get_nfw_params(M, z, H0=70.0, Om=0.3):
    """Computes rs (Mpc) and rhos (M_sun/Mpc^3) for a halo of mass M."""
    Ez = np.sqrt(Om * (1.0 + z)**3 + (1.0 - Om))
    G = 4.30091e-9
    H = H0 * Ez
    rho_crit_z = 3.0 * H**2 / (8.0 * np.pi * G)
    
    r200 = (3.0 * M / (4.0 * np.pi * 200.0 * rho_crit_z))**(1.0 / 3.0)
    c = 5.71 * (M / 2e12)**(-0.084) * (1.0 + z)**(-0.47)
    rs = r200 / c
    rhos = 200.0 * rho_crit_z * (c**3) / (3.0 * (np.log(1.0 + c) - c / (1.0 + c)))
    
    return rs, rhos

def get_sigmac(zl, zs, H0=70.0, Om=0.3):
    """Critical surface mass density Sigma_c in M_sun/Mpc^2."""
    def comoving_distance(z):
        from scipy.integrate import quad
        integrand = lambda zp: 1.0 / np.sqrt(Om * (1.0 + zp)**3 + (1.0 - Om))
        dist, _ = quad(integrand, 0.0, z)
        return (299792.458 / H0) * dist

    Dl = comoving_distance(zl) / (1.0 + zl)
    Ds = comoving_distance(zs) / (1.0 + zs)
    Dls = Ds - Dl * (1.0 + zl) / (1.0 + zs)
    
    c_light = 299792.458
    G = 4.30091e-9
    return (c_light**2 * Ds) / (4.0 * np.pi * G * Dl * Dls)

def kappa_NFW(d_mpc, M, rs, rhos, sigmac):
    """Calculates convergence at a distance d_mpc (Mpc) given rs, rhos."""
    kappa0 = rs * rhos / sigmac
    x = d_mpc / rs
    return 2.0 * kappa0 * Fg(x)

# 2. Main Execution
if __name__ == "__main__":
    zl = 0.5
    zs = 2.0
    sigmac = get_sigmac(zl, zs)
    
    # Masses
    M_total = 1e14   # Total mass of the system (M_sun)
    m_sub = 1e13     # Subhalo mass (M_sun)
    
    # Subhalo offset from host center (kpc)
    delta_x_kpc = 100.0
    delta_x_mpc = delta_x_kpc / 1000.0
    
    # We sweep the distance D (from LoS to host center) from 120 kpc to 2000 kpc.
    # Since the subhalo is offset by +100 kpc towards the LoS, its distance to the LoS is D - 100 kpc.
    # When D > 100 kpc, both host and subhalo are on the left side of the LoS.
    D_kpc = np.linspace(120.0, 2000.0, 1000)
    D_mpc = D_kpc / 1000.0
    
    # NFW parameters
    rs_host_full, rhos_host_full = get_nfw_params(M_total, zl)
    rs_host_red, rhos_host_red = get_nfw_params(M_total - m_sub, zl)
    rs_sub, rhos_sub = get_nfw_params(m_sub, zl)
    
    # 1. Total Mass Halo (Ideal limit where host + subhalo is modeled as one big halo of M_total)
    kappa_total = kappa_NFW(D_mpc, M_total, rs_host_full, rhos_host_full, sigmac)
    
    # 2. Option B: Reduced host (M_total - m_sub) + Subhalo (m_sub)
    kappa_B_host = kappa_NFW(D_mpc, M_total - m_sub, rs_host_red, rhos_host_red, sigmac)
    kappa_B_sub = kappa_NFW(D_mpc - delta_x_mpc, m_sub, rs_sub, rhos_sub, sigmac)
    kappa_B = kappa_B_host + kappa_B_sub
    
    # 3. Option A: Full host + Subhalo - m * gslope(D)
    # Compute gslope = dkappa_host/dM at each D
    Me = 1.02 * M_total
    rs_host_e, rhos_host_e = get_nfw_params(Me, zl)
    kappa_host_e = kappa_NFW(D_mpc, Me, rs_host_e, rhos_host_e, sigmac)
    gslope = (kappa_host_e - kappa_total) / (Me - M_total)
    kappa_A = kappa_total + kappa_B_sub - m_sub * gslope
    
    # Plotting
    plt.figure(figsize=(10, 6.5))
    
    plt.plot(D_kpc, kappa_total, color='#4b5563', linestyle='--', linewidth=2.0,
             label=r'Total Mass Halo ($M_{total} = 10^{14} \, M_\odot$)')
    plt.plot(D_kpc, kappa_B, color='#0891b2', linewidth=2.5,
             label='Option B (Globally Reduced Host + Subhalo)')
    plt.plot(D_kpc, kappa_A, color='#0d9488', linewidth=2.5,
             label=r'Option A (Full Host + Subhalo $- m\cdot g_{slope}$)')
    
    plt.yscale('log')
    plt.xscale('log')
    plt.title('Convergence at LoS vs. Distance to Host (Both on Same Side of LoS)', 
              fontsize=13, fontweight='bold', pad=15, color='#111827')
    plt.xlabel('Distance from LoS to Host Center $D$ [kpc]', fontsize=12, labelpad=10)
    plt.ylabel(r'Convergence at LoS $\kappa$', fontsize=12, labelpad=10)
    
    plt.grid(True, which="both", ls="-", color='#e5e7eb', alpha=0.8)
    plt.legend(frameon=True, facecolor='#f9fafb', edgecolor='#e5e7eb', fontsize=10)
    
    plt.xlim(120.0, 2000.0)
    plt.ylim(1e-4, 1.0)
    
    # Annotations to show convergence behavior
    plt.annotate('Close to LoS:\nProfiles differ', xy=(180.0, 0.08), xytext=(300.0, 0.2),
                 arrowprops=dict(arrowstyle="->", color='#374151', lw=1.5),
                 fontsize=9.5, color='#374151')
                 
    plt.annotate('Far away:\nAll profiles converge to\nthe Total Mass Halo', 
                 xy=(1200.0, 0.0015), xytext=(300.0, 0.0003),
                 arrowprops=dict(arrowstyle="->", color='#374151', lw=1.5),
                 fontsize=9.5, color='#374151')
                 
    plt.tight_layout()
    
    os.makedirs('plots', exist_ok=True)
    plot_path = 'plots/same_side_experiment.png'
    plt.savefig(plot_path, dpi=300, facecolor='#ffffff')
    print(f"Saved plot to {plot_path}")
