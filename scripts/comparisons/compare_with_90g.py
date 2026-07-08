import numpy as np
import matplotlib.pyplot as plt
import os

# 1. NFW profile function
def Fg(x):
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
    def comoving_distance(z):
        from scipy.integrate import quad
        integrand = lambda zp: 1.0 / np.sqrt(Om * (1.0 + zp)**3 + (1.0 - Om))
        dist, _ = quad(integrand, 0.0, z)
        return (299792.458 / H0) * dist
    Dl = comoving_distance(zl) / (1.0 + zl)
    Ds = comoving_distance(zs) / (1.0 + zs)
    Dls = Ds - Dl * (1.0 + zl) / (1.0 + zs)
    return (299792.458**2 * Ds) / (4.0 * np.pi * 4.30091e-9 * Dl * Dls)

def kappa_NFW(d_mpc, M, rs, rhos, sigmac):
    kappa0 = rs * rhos / sigmac
    x = d_mpc / rs
    return 2.0 * kappa0 * Fg(x)

# 2. Main execution
if __name__ == "__main__":
    zl = 0.5
    zs = 2.0
    sigmac = get_sigmac(zl, zs)
    
    M_total = 1e14  # 100g total mass
    m_sub = 1e13   # 10g subhalo (10% mass fraction)
    
    # Radii from 1 to 500 kpc
    r_kpc = np.linspace(1.0, 500.0, 1000)
    r_mpc = r_kpc / 1000.0
    
    # NFW parameters
    rs_total, rhos_total = get_nfw_params(M_total, zl)
    rs_90, rhos_90 = get_nfw_params(M_total - m_sub, zl)
    
    # 1. 90g Smooth Host (The baseline suggested by user)
    k_smooth_90 = kappa_NFW(r_mpc, M_total - m_sub, rs_90, rhos_90, sigmac)
    
    # 2. Option B host (exactly 90g NFW)
    k_B_host = kappa_NFW(r_mpc, M_total - m_sub, rs_90, rhos_90, sigmac)
    
    # 3. Option 2 host (amplitude-scaled: 0.9 * 100g NFW)
    k_smooth_100 = kappa_NFW(r_mpc, M_total, rs_total, rhos_total, sigmac)
    k_opt2_host = 0.9 * k_smooth_100
    
    # Plotting
    plt.figure(figsize=(9, 6.5))
    
    # The 90g Smooth reference line
    plt.plot(r_kpc, k_smooth_90, color='black', linestyle='--', linewidth=3.0,
             label=r'90g Smooth Host reference ($\kappa_{\rm host}(90\rm g)$)')
    
    # Option B host
    plt.plot(r_kpc, k_B_host, color='#0891b2', linestyle='-', linewidth=2.0, alpha=0.8,
             label=r'Option B Host ($\kappa_{\rm host}(90\rm g)$) - Overlaps reference exactly')
             
    # Option 2 host
    plt.plot(r_kpc, k_opt2_host, color='#ea580c', linestyle='-', linewidth=2.5,
             label=r'Option 2 Host ($0.9 \times \kappa_{\rm host}(100\rm g)$)')
    
    plt.yscale('log')
    plt.title('Comparison against a 90g Smooth Baseline (No Subhalos shown)', 
              fontsize=13, fontweight='bold', pad=15, color='#111827')
    plt.xlabel('Distance from Host Center $r$ [kpc]', fontsize=12, labelpad=10)
    plt.ylabel(r'Convergence $\kappa(r)$', fontsize=12, labelpad=10)
    
    plt.grid(True, which="both", ls="-", color='#e5e7eb', alpha=0.8)
    plt.legend(frameon=True, facecolor='#f9fafb', edgecolor='#e5e7eb', fontsize=10)
    
    plt.xlim(1.0, 500.0)
    plt.ylim(1e-4, 1.0)
    
    plt.tight_layout()
    
    os.makedirs('plots', exist_ok=True)
    plot_path = 'plots/comparison_with_90g.png'
    plt.savefig(plot_path, dpi=300, facecolor='#ffffff')
    print(f"Saved plot to {plot_path}")
