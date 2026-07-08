import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

rows = np.load('plots/screen_rows.npy')          # (z, M, barNH, cH3, d3)
z, M, barNH, cH3, d3 = rows.T

# model totals (from the run)
K2model, K3model, dK2c, dK3c = 7.306e-4, 7.421e-5, 2.850e-5, 1.327e-6

fig = plt.figure(figsize=(14.5, 10.5))
gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# ---- helper grids over host mass and lens redshift ---------------------------
zu = np.unique(z)
Mu = np.unique(M)
zi = {v: i for i, v in enumerate(zu)}
Mi = {v: i for i, v in enumerate(Mu)}

def contribution_grid(values):
    grid = np.full((len(Mu), len(zu)), np.nan)
    for zz, mm, vv in zip(z, M, values):
        if vv > 0:
            grid[Mi[mm], zi[zz]] = vv
    return grid

host_grid = contribution_grid(cH3)
clump_grid = contribution_grid(d3)
total_grid = contribution_grid(cH3 + d3)
host_norm = LogNorm(vmin=np.nanmax(host_grid)*1e-4, vmax=np.nanmax(host_grid))

# ---- panel 1: halo-only source map ------------------------------------------
im1 = ax1.pcolormesh(zu, Mu, host_grid, norm=host_norm, cmap='magma',
                     shading='nearest')
ax1.set_yscale('log')
ax1.set_xlabel('lens redshift $z_l$')
ax1.set_ylabel(r'host mass $M\ [M_\odot]$')
ax1.set_title(r'Halo-only $\langle\kappa^3\rangle$ source'
              '\n(no subhalos; host contribution per bin)')
cb1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cb1.set_label(r'$d\langle\kappa^3\rangle_{\rm host}$ per bin')

# ---- panel 2: halo + subhalos on the exact same scale ------------------------
im2 = ax2.pcolormesh(zu, Mu, total_grid, norm=host_norm, cmap='magma',
                     shading='nearest')
ax2.set_yscale('log')
ax2.set_xlabel('lens redshift $z_l$')
ax2.set_ylabel(r'host mass $M\ [M_\odot]$')
ax2.set_title(r'Halo + subhalo $\langle\kappa^3\rangle$ source'
              '\n(same color scale; total boost is small)')
cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cb2.set_label(r'$d\langle\kappa^3\rangle_{\rm host+sub}$ per bin')

# ---- panel 3: absolute subhalo correction ------------------------------------
im3 = ax3.pcolormesh(zu, Mu, clump_grid,
                     norm=LogNorm(vmin=np.nanmax(clump_grid)*1e-4,
                                  vmax=np.nanmax(clump_grid)),
                     cmap='inferno', shading='nearest')
ax3.set_yscale('log')
ax3.set_xlabel('lens redshift $z_l$')
ax3.set_ylabel(r'host mass $M\ [M_\odot]$')
ax3.set_title(r'Absolute subhalo correction'
              '\n(this is what gets added to the host panel)')
cb3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
cb3.set_label(r'$\Delta d\langle\kappa^3\rangle_{\rm sub}$ per bin')

# ---- panel 4: fractional subhalo boost ---------------------------------------
ratio_grid = np.full_like(host_grid, np.nan)
ok_ratio = np.isfinite(host_grid) & np.isfinite(clump_grid) & (host_grid > 0)
ratio_grid[ok_ratio] = clump_grid[ok_ratio] / host_grid[ok_ratio]
ratio_pct = 100 * ratio_grid
im4 = ax4.pcolormesh(zu, Mu, ratio_pct, vmin=0,
                     vmax=np.nanpercentile(ratio_pct, 99),
                     cmap='viridis', shading='nearest')
ax4.set_yscale('log')
ax4.set_xlabel('lens redshift $z_l$')
ax4.set_ylabel(r'host mass $M\ [M_\odot]$')
ax4.set_title('Fractional subhalo boost per bin\n'
              r'global $\langle\kappa^3\rangle$ boost = %.1f%%'
              % (100*dK3c/K3model))
cb4 = fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
cb4.set_label(r'$\Delta d\langle\kappa^3\rangle_{\rm sub}/'
              r'd\langle\kappa^3\rangle_{\rm host}$ [%]')

fig.suptitle(r'$\S4$ gate ($z_s=1$): substructure adds +3.9% to $\langle\kappa^2\rangle$, '
             r'+1.8% to $\langle\kappa^3\rangle$ $\Rightarrow$ dimensionless skewness 3.76$\to$3.61',
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig('plots/subhalo_screen.png', dpi=140, bbox_inches='tight')
print('wrote plots/subhalo_screen.png')

try:
    scan = np.load('plots/screen_zs_scan.npz')
except FileNotFoundError:
    scan = None

if scan is not None:
    zs = scan['zs']
    K2 = scan['K2_model']
    K3 = scan['K3_model']
    d2 = scan['dK2c']
    d3_scan = scan['dK3c']
    skew_model = scan['skew_model']
    skew_clumps = scan['skew_clumps']

    fig2, (bx1, bx2, bx3) = plt.subplots(1, 3, figsize=(14.5, 4.4))
    bx1.plot(zs, 100*d2/K2, marker='o', lw=2.2, label=r'$\Delta\langle\kappa^2\rangle_c / \langle\kappa^2\rangle$')
    bx1.plot(zs, 100*d3_scan/K3, marker='s', lw=2.2, label=r'$\Delta\langle\kappa^3\rangle_c / \langle\kappa^3\rangle$')
    bx1.set_xlabel(r'source redshift $z_s$')
    bx1.set_ylabel('subhalo boost [%]')
    bx1.set_title('Fractional subhalo correction')
    bx1.grid(alpha=0.3)
    bx1.legend(fontsize=9)

    bx2.plot(zs, K2, marker='o', lw=2.2, label=r'host $\langle\kappa^2\rangle$')
    bx2.plot(zs, K3, marker='s', lw=2.2, label=r'host $\langle\kappa^3\rangle$')
    bx2.plot(zs, d2, marker='o', lw=1.8, ls='--', label=r'sub $\Delta\langle\kappa^2\rangle$')
    bx2.plot(zs, d3_scan, marker='s', lw=1.8, ls='--', label=r'sub $\Delta\langle\kappa^3\rangle$')
    bx2.set_yscale('log')
    bx2.set_xlabel(r'source redshift $z_s$')
    bx2.set_ylabel('moment amplitude')
    bx2.set_title('Moment growth with source redshift')
    bx2.grid(alpha=0.3, which='both')
    bx2.legend(fontsize=8)

    bx3.plot(zs, skew_model, marker='o', lw=2.2, label='host only')
    bx3.plot(zs, skew_clumps, marker='s', lw=2.2, label='host + subhalos')
    bx3.set_xlabel(r'source redshift $z_s$')
    bx3.set_ylabel(r'$\langle\kappa^3\rangle / \langle\kappa^2\rangle^{3/2}$')
    bx3.set_title('Dimensionless skewness')
    bx3.grid(alpha=0.3)
    bx3.legend(fontsize=9)

    fig2.suptitle('Source-redshift scan: subhalos remain a small fractional correction',
                  fontsize=12)
    fig2.tight_layout(rect=[0, 0, 1, 0.92])
    fig2.savefig('plots/subhalo_screen_zs_scan.png', dpi=140, bbox_inches='tight')
    print('wrote plots/subhalo_screen_zs_scan.png')
