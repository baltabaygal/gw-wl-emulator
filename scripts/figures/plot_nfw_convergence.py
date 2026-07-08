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
    """
    Computes scale radius r_s and characteristic density rho_s.
    M: halo mass in M_sun
    z: lens redshift
    """
    # Critical density at z: rho_crit(z) = rho_crit,0 * E(z)^2
    # E(z) = sqrt(Om*(1+z)^3 + (1-Om))
    Ez = np.sqrt(Om * (1.0 + z)**3 + (1.0 - Om))
    G = 4.30091e-9  # Mpc * M_sun^-1 * (km/s)^2
    H = H0 * Ez     # km/s/Mpc
    rho_crit_z = 3.0 * H**2 / (8.0 * np.pi * G)  # M_sun / Mpc^3
    
    # R200 definition: mass enclosed within a sphere of density 200 * rho_crit
    r200 = (3.0 * M / (4.0 * np.pi * 200.0 * rho_crit_z))**(1.0 / 3.0)  # Mpc
    
    # Concentration relation c(M, z) from Duffy et al. 2008
    # c = 5.71 * (M / 2e12)^-0.084 * (1+z)^-0.47
    c = 5.71 * (M / 2e12)**(-0.084) * (1.0 + z)**(-0.47)
    
    rs = r200 / c  # Mpc
    
    # rho_s calculation
    rhos = 200.0 * rho_crit_z * (c**3) / (3.0 * (np.log(1.0 + c) - c / (1.0 + c)))  # M_sun / Mpc^3
    
    return rs, rhos

def get_sigmac(zl, zs, H0=70.0, Om=0.3):
    """
    Critical surface mass density Sigma_c in M_sun / Mpc^2.
    Simple flat LambdaCDM angular diameter distance calculations.
    """
    # Simple comoving distance integration (in Mpc)
    def comoving_distance(z):
        from scipy.integrate import quad
        integrand = lambda zp: 1.0 / np.sqrt(Om * (1.0 + zp)**3 + (1.0 - Om))
        dist, _ = quad(integrand, 0.0, z)
        dh = 299792.458 / H0  # Hubble distance in Mpc
        return dh * dist

    # Angular diameter distance D_A = D_comoving / (1 + z)
    Dl = comoving_distance(zl) / (1.0 + zl)
    Ds = comoving_distance(zs) / (1.0 + zs)
    Dls = Ds - Dl * (1.0 + zl) / (1.0 + zs)
    
    c_light = 299792.458  # km/s
    G = 4.30091e-9        # Mpc * M_sun^-1 * (km/s)^2
    
    # Sigma_c = c^2 * Ds / (4 * pi * G * Dl * Dls)
    sigmac = (c_light**2 * Ds) / (4.0 * np.pi * G * Dl * Dls)  # M_sun / Mpc^2
    return sigmac

# 2. Main execution and plotting
if __name__ == "__main__":
    zl = 0.5
    zs = 2.0
    
    # Calculate Sigma_c
    sigmac = get_sigmac(zl, zs)
    
    # Projected distances (impact parameters) from 1 kpc to 3000 kpc (in Mpc)
    r_kpc = np.logspace(0, 3.5, 500)  # 1 kpc to ~3162 kpc
    r_mpc = r_kpc / 1000.0            # convert to Mpc
    
    # Masses to plot (10^13, 10^14, 10^15 M_sun)
    masses = [1e13, 1e14, 1e15]
    colors = ['#4f46e5', '#0891b2', '#0d9488']
    
    plt.figure(figsize=(9, 6))
    
    # Dark mode / premium style
    plt.rcParams['text.color'] = '#1f2937'
    plt.rcParams['axes.labelcolor'] = '#1f2937'
    plt.rcParams['xtick.color'] = '#4b5563'
    plt.rcParams['ytick.color'] = '#4b5563'
    
    for M, color in zip(masses, colors):
        rs, rhos = get_nfw_params(M, zl)
        kappa0 = rs * rhos / sigmac
        
        # Compute kappa(r)
        x = r_mpc / rs
        kappa = 2.0 * kappa0 * Fg(x)
        
        plt.loglog(r_kpc, kappa, label=rf'$M = 10^{{{int(np.log10(M))}}} \, M_\odot$', 
                   color=color, linewidth=2.5)
        
        # Draw a vertical dashed line for each scale radius r_s (in kpc)
        rs_kpc = rs * 1000.0
        plt.axvline(x=rs_kpc, color=color, linestyle='--', alpha=0.5, 
                    label=rf'$r_s \approx {rs_kpc:.1f}$ kpc' if M == 1e14 else "")

    plt.title(r'Lensing Convergence $\kappa(r)$ vs. Projected Distance $r$', 
              fontsize=14, fontweight='bold', pad=15, color='#111827')
    plt.xlabel('Projected distance to Line of Sight $r$ [kpc]', fontsize=12, labelpad=10)
    plt.ylabel(r'Convergence $\kappa(r)$', fontsize=12, labelpad=10)
    
    plt.grid(True, which="both", ls="-", color='#e5e7eb', alpha=0.8)
    plt.legend(frameon=True, facecolor='#f9fafb', edgecolor='#e5e7eb', fontsize=10)
    
    # Adjust limits
    plt.xlim(1.0, 3000.0)
    plt.ylim(1e-5, 1e1)
    
    plt.tight_layout()
    
    # Save directory
    os.makedirs('plots', exist_ok=True)
    plot_path = 'plots/nfw_convergence_profile.png'
    plt.savefig(plot_path, dpi=300, facecolor='#ffffff')
    print(f"Saved plot to {plot_path}")
