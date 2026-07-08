import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Analytical NFW profile functions for lensing convergence (kappa)
def Fg(x):
    """Radial profile factor F(x) for NFW lensing convergence."""
    x = np.atleast_1d(x)
    res = np.zeros_like(x)
    
    # x > 1 (outside scale radius)
    idx = x > 1
    if np.any(idx):
        xx = x[idx]
        t = np.arctan(np.sqrt((xx - 1.0) / (xx + 1.0))) / np.sqrt(xx**2 - 1.0)
        res[idx] = (1.0 - 2.0 * t) / (xx**2 - 1.0)
        
    # x < 1 (inside scale radius)
    idx = x < 1
    if np.any(idx):
        xx = x[idx]
        t = np.arctanh(np.sqrt((1.0 - xx) / (xx + 1.0))) / np.sqrt(1.0 - xx**2)
        res[idx] = (1.0 - 2.0 * t) / (xx**2 - 1.0)
        
    # x == 1 (at scale radius)
    idx = np.isclose(x, 1.0)
    if np.any(idx):
        res[idx] = 1.0 / 3.0
        
    return res

def get_nfw_params(M, z, H0=70.0, Om=0.3):
    """Computes rs (Mpc) and rhos (M_sun/Mpc^3) for a halo of mass M."""
    Ez = np.sqrt(Om * (1.0 + z)**3 + (1.0 - Om))
    G = 4.30091e-9  # Mpc * M_sun^-1 * (km/s)^2
    H = H0 * Ez
    rho_crit_z = 3.0 * H**2 / (8.0 * np.pi * G)
    
    r200 = (3.0 * M / (4.0 * np.pi * 200.0 * rho_crit_z))**(1.0 / 3.0)
    
    # Simple concentration relation
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
    
    # Host and Subhalo definitions
    M_host = 1e14  # Host mass (M_sun)
    m_sub = 1e13   # Subhalo mass (M_sun, 10% of host)
    
    # Subhalo position: placed on the "West" side (x = -200 kpc)
    # We will let the ray sweep from East (+500 kpc) to West (-500 kpc)
    # i.e., x_ray goes from -500 kpc to +500 kpc. 
    # Host center is at x = 0. Subhalo center is at x = -200 kpc.
    x_sub_kpc = -200.0
    x_sub_mpc = x_sub_kpc / 1000.0
    
    # Ray x-coordinates from -500 kpc to +500 kpc
    x_ray_kpc = np.linspace(-500.0, 500.0, 1000)
    x_ray_mpc = x_ray_kpc / 1000.0
    
    # NFW parameters
    rs_host_full, rhos_host_full = get_nfw_params(M_host, zl)
    rs_host_red, rhos_host_red = get_nfw_params(M_host - m_sub, zl)
    rs_sub, rhos_sub = get_nfw_params(m_sub, zl)
    
    # 1. Smooth Host only (No Subhalos)
    kappa_smooth = kappa_NFW(np.abs(x_ray_mpc), M_host, rs_host_full, rhos_host_full, sigmac)
    
    # 2. Option B: Globally reduced host + subhalo
    kappa_B_host = kappa_NFW(np.abs(x_ray_mpc), M_host - m_sub, rs_host_red, rhos_host_red, sigmac)
    kappa_sub = kappa_NFW(np.abs(x_ray_mpc - x_sub_mpc), m_sub, rs_sub, rhos_sub, sigmac)
    kappa_B = kappa_B_host + kappa_sub
    
    # 3. Option A: Full host + subhalo - m * gslope(x)
    # Compute gslope = dkappa_host/dM at each ray position
    Me = 1.02 * M_host
    rs_host_e, rhos_host_e = get_nfw_params(Me, zl)
    kappa_host_e = kappa_NFW(np.abs(x_ray_mpc), Me, rs_host_e, rhos_host_e, sigmac)
    gslope = (kappa_host_e - kappa_smooth) / (Me - M_host)
    kappa_A = kappa_smooth + kappa_sub - m_sub * gslope
    
    # Plotting
    plt.figure(figsize=(10, 6.5))
    
    # Subhalo position indicator
    plt.axvline(x=x_sub_kpc, color='#dc2626', linestyle=':', alpha=0.8, linewidth=1.5,
                label='Subhalo Center (-200 kpc)')
    # Host position indicator
    plt.axvline(x=0, color='#9ca3af', linestyle='--', alpha=0.6, linewidth=1.2,
                label='Host Center (0 kpc)')
    
    # Plot profiles
    plt.plot(x_ray_kpc, kappa_smooth, color='#4b5563', linestyle='--', linewidth=2.0,
             label='Smooth Host (No Subhalo)')
    plt.plot(x_ray_kpc, kappa_B, color='#0891b2', linewidth=2.5,
             label='Option B (Globally Reduced Host + Subhalo)')
    plt.plot(x_ray_kpc, kappa_A, color='#0d9488', linewidth=2.5,
             label=r'Option A (Full Host + Subhalo $- m\cdot g_{slope}$)')
    
    plt.yscale('log')
    plt.title('Lensing Convergence Profile with a Subhalo at -200 kpc', 
              fontsize=14, fontweight='bold', pad=15, color='#111827')
    plt.xlabel('Ray Position along x-axis [kpc] (West < 0 < East)', fontsize=12, labelpad=10)
    plt.ylabel(r'Convergence $\kappa(r)$', fontsize=12, labelpad=10)
    
    plt.grid(True, which="both", ls="-", color='#e5e7eb', alpha=0.8)
    plt.legend(frameon=True, facecolor='#f9fafb', edgecolor='#e5e7eb', fontsize=10, loc='upper right')
    
    plt.xlim(-500.0, 500.0)
    plt.ylim(1e-4, 5.0)
    
    # Annotate the East side "hole" for Option B
    plt.annotate('Option B mass deficit ("hole")\non the East side (opposite to subhalo)',
                 xy=(250.0, 0.009), xytext=(80.0, 0.001),
                 arrowprops=dict(facecolor='#0891b2', shrink=0.08, width=1.5, headwidth=6, headlength=6),
                 fontsize=9, color='#0891b2', bbox=dict(boxstyle="round,pad=0.3", fc="#ecfeff", ec="#c5f2f7", alpha=0.9))
                 
    plt.tight_layout()
    
    os.makedirs('plots', exist_ok=True)
    plot_path = 'plots/nfw_with_subhalo_profile.png'
    plt.savefig(plot_path, dpi=300, facecolor='#ffffff')
    print(f"Saved plot to {plot_path}")
