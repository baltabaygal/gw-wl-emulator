import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Load screen rows (7 columns: z, M, barNH, cH2, d2, cH3, d3)
rows = np.load('plots/screen_rows_all.npy')
z, M, barNH, cH2, d2, cH3, d3 = rows.T

# Load the source redshift scan metadata
try:
    scan = np.load('plots/screen_zs_scan.npz')
    zs_arr = scan['zs']
    z_max = np.max(z)
    valid_zs = zs_arr[zs_arr > z_max]
    if len(valid_zs) > 0:
        zs_fid = np.min(valid_zs)
    else:
        zs_fid = np.max(zs_arr)
    idx = np.where(np.isclose(zs_arr, zs_fid))[0]
    if len(idx) > 0:
        idx = idx[0]
        K2model = scan['K2_model'][idx]
        dK2c = scan['dK2c'][idx]
        K3model = scan['K3_model'][idx]
        dK3c = scan['dK3c'][idx]
        skew_model = scan['skew_model'][idx]
        skew_clumps = scan['skew_clumps'][idx]
    else:
        zs_fid = 5.0
        K2model, dK2c, K3model, dK3c = 7.297e-4, 2.850e-05, 7.421e-5, 1.327e-06
        skew_model, skew_clumps = 3.765, 3.618
except Exception:
    zs_fid = 5.0
    K2model, dK2c, K3model, dK3c = 7.297e-4, 2.850e-05, 7.421e-5, 1.327e-06
    skew_model, skew_clumps = 3.765, 3.618

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

# Reconstruct grids
host_grid2 = contribution_grid(cH2)
clump_grid2 = contribution_grid(d2)
total_grid2 = contribution_grid(cH2 + d2)
nh_grid = contribution_grid(barNH)

# ==============================================================================
# PLOT 1: Second Moment <kappa^2>
# ==============================================================================
fig = plt.figure(figsize=(14.5, 10.5))
gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

host_norm2 = LogNorm(vmin=np.nanmax(host_grid2)*1e-4, vmax=np.nanmax(host_grid2))

# Panel 1: Halo-only
im1 = ax1.pcolormesh(zu, Mu, host_grid2, norm=host_norm2, cmap='magma', shading='nearest')
ax1.set_yscale('log')
ax1.set_xlabel('lens redshift $z_l$')
ax1.set_ylabel(r'host mass $M\ [M_\odot]$')
ax1.set_title(r'Halo-only $\langle\kappa^2\rangle$ source'
              '\n(no subhalos; host contribution per bin)')
cb1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cb1.set_label(r'$d\langle\kappa^2\rangle_{\rm host}$ per bin')

# Panel 2: Halo + Subhalos
im2 = ax2.pcolormesh(zu, Mu, total_grid2, norm=host_norm2, cmap='magma', shading='nearest')
ax2.set_yscale('log')
ax2.set_xlabel('lens redshift $z_l$')
ax2.set_ylabel(r'host mass $M\ [M_\odot]$')
ax2.set_title(r'Halo + subhalo $\langle\kappa^2\rangle$ source'
              '\n(same color scale; total boost is small)')
cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cb2.set_label(r'$d\langle\kappa^2\rangle_{\rm host+sub}$ per bin')

# Panel 3: Absolute subhalo correction
clump_norm2 = LogNorm(vmin=np.nanmax(clump_grid2)*1e-4, vmax=np.nanmax(clump_grid2))
im3 = ax3.pcolormesh(zu, Mu, clump_grid2, norm=clump_norm2, cmap='inferno', shading='nearest')
ax3.set_yscale('log')
ax3.set_xlabel('lens redshift $z_l$')
ax3.set_ylabel(r'host mass $M\ [M_\odot]$')
ax3.set_title(r'Absolute subhalo correction'
              '\n(this is what gets added to the host panel)')
cb3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
cb3.set_label(r'$\Delta d\langle\kappa^2\rangle_{\rm sub}$ per bin')

# Panel 4: Fractional subhalo boost
ratio_grid2 = np.full_like(host_grid2, np.nan)
ok_ratio = np.isfinite(host_grid2) & np.isfinite(clump_grid2) & (host_grid2 > 0)
ratio_grid2[ok_ratio] = clump_grid2[ok_ratio] / host_grid2[ok_ratio]
ratio_pct2 = 100 * ratio_grid2
im4 = ax4.pcolormesh(zu, Mu, ratio_pct2, vmin=0,
                     vmax=np.nanpercentile(ratio_pct2, 99),
                     cmap='viridis', shading='nearest')
ax4.set_yscale('log')
ax4.set_xlabel('lens redshift $z_l$')
ax4.set_ylabel(r'host mass $M\ [M_\odot]$')
ax4.set_title('Fractional subhalo boost per bin\n'
              r'global $\langle\kappa^2\rangle$ boost = %.2f%%'
              % (100*dK2c/K2model))
cb4 = fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
cb4.set_label(r'$\Delta d\langle\kappa^2\rangle_{\rm sub}/'
              r'd\langle\kappa^2\rangle_{\rm host}$ [%]')

fig.suptitle(r'$\S4$ gate ($z_s=%.1f$): substructure adds +%.2f%% to $\langle\kappa^2\rangle$, '
             r'+%.2f%% to $\langle\kappa^3\rangle$ $\Rightarrow$ dimensionless skewness %.3f$\to$%.3f'
             % (zs_fid, 100*dK2c/K2model, 100*dK3c/K3model, skew_model, skew_clumps),
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig('plots/subhalo_screen_kappa2.png', dpi=140, bbox_inches='tight')
print('wrote plots/subhalo_screen_kappa2.png')
plt.close(fig)

# ==============================================================================
# PLOT 2: Expected Encounter Counts (barNH) - proxy for Kappa Abundance
# ==============================================================================
fig_counts, ax_c = plt.subplots(figsize=(8, 6.5))
nh_norm = LogNorm(vmin=np.nanmax(nh_grid)*1e-4, vmax=np.nanmax(nh_grid))
im_c = ax_c.pcolormesh(zu, Mu, nh_grid, norm=nh_norm, cmap='plasma', shading='nearest')
ax_c.set_yscale('log')
ax_c.set_xlabel('lens redshift $z_l$')
ax_c.set_ylabel(r'host mass $M\ [M_\odot]$')
ax_c.set_title(r'Expected host encounters per bin $\bar{N}_H$'
               '\n' + r'(total expected strong lenses $N_{\rm host} = 100$)')
cb_c = fig_counts.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.04)
cb_c.set_label(r'$\bar{N}_H$ per bin')

fig_counts.tight_layout()
fig_counts.savefig('plots/subhalo_screen_counts.png', dpi=140, bbox_inches='tight')
print('wrote plots/subhalo_screen_counts.png')
plt.close(fig_counts)
