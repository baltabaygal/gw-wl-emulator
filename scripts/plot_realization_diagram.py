#!/usr/bin/env python3
"""
Diagram of one weak-lensing realization — Vaskonen MC geometry.

~100 resolved halos drawn from a realistic HMF-weighted distribution,
plus filaments and subhalos.  Three panels:
  Main: side-view with all ~100 halos, filaments, subhalos, LoS, lens planes
  Inset (upper-right): zoom into one massive host
  Inset (lower-right): cumulative κ buildup
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe
from scipy.ndimage import gaussian_filter
from scipy.integrate import quad
import os

np.random.seed(42)

# ── cosmology ─────────────────────────────────────────────────────────
Om, h0, sigma8 = 0.315, 0.674, 0.811
rhoc = 277.394 * h0**2
zs = 1.0

def dc_approx(z):
    f = lambda zz: 1.0 / np.sqrt(Om * (1 + zz)**3 + (1 - Om))
    val, _ = quad(f, 0, z)
    return val * 2997.9 / h0

dc_source = dc_approx(zs)

def z_to_x(z):
    return dc_approx(z) / dc_source

# ── generate ~100 halos from a realistic distribution ─────────────────
# The HMF is steeply falling: most resolved halos are near the threshold
# mass (~10^12–10^13 at z~0.5).  We mock this with a Schechter-like draw.
# Redshift distribution ∝ dV/dz ∝ dc(z)^2 / H(z), peaked at z~0.4–0.6.

N_halos_target = 100

# redshift: draw from dV/dz ∝ dc^2 / sqrt(Om(1+z)^3 + OmL)
z_candidates = np.linspace(0.02, zs - 0.02, 5000)
dV = np.array([dc_approx(z)**2 / np.sqrt(Om*(1+z)**3 + (1-Om)) for z in z_candidates])
dV /= dV.sum()
z_halos = np.random.choice(z_candidates, size=N_halos_target, p=dV)

# mass: Schechter-like dn/dlnM ∝ M^(-0.9) * exp(-M/M*)  with M* ~ 10^14
# Most halos cluster near 10^12–10^13
log_M_min, log_M_max = 11.5, 15.5
M_star = 1e14
log_M_draw = np.random.uniform(log_M_min, log_M_max, size=N_halos_target * 20)
M_draw = 10**log_M_draw
weight = M_draw**(-0.9) * np.exp(-M_draw / M_star)
weight /= weight.sum()
M_halos = np.random.choice(M_draw, size=N_halos_target, p=weight, replace=False)

# impact parameter: uniform in area → r = rmax * sqrt(U)
# rmax scales with mass; in normalised diagram coords
def rmax_diagram(M):
    """Visual rmax in diagram y-coords (exaggerated)."""
    return 0.03 + 0.35 * (M / 1e15)**0.35

r_halos = np.array([rmax_diagram(M) * np.sqrt(np.random.uniform()) for M in M_halos])
phi_halos = np.random.uniform(0, 2*np.pi, N_halos_target)
y_halos = r_halos * np.sign(np.random.randn(N_halos_target))  # ± side of LoS

# visual radius scales with mass
def vis_radius(M):
    return 0.008 + 0.10 * (M / 1e15)**0.33

# ── filaments (~30, also from mass function, lower mass end) ──────────
N_fil = 30
z_fil = np.random.choice(z_candidates, size=N_fil, p=dV)
log_M_fil = np.random.uniform(12.5, 14.5, N_fil)
M_fil = 10**log_M_fil
y_fil = np.random.uniform(-0.9, 0.9, N_fil)
angle_fil = np.random.uniform(0, 180, N_fil)

# ── subhalos: inside the 5 most massive hosts ─────────────────────────
top5_idx = np.argsort(M_halos)[-5:]

class SubHalo:
    def __init__(self, dx, dy, m, host_M):
        self.dx, self.dy, self.m, self.host_M = dx, dy, m, host_M

subhalos_by_host = {}
for idx in top5_idx:
    M_host = M_halos[idx]
    R_host = vis_radius(M_host)
    n_sub = np.random.poisson(6)
    subs = []
    for _ in range(n_sub):
        ang = np.random.uniform(0, 2*np.pi)
        rfrac = np.random.beta(3, 1.5) * 0.9  # anti-biased
        dx = rfrac * R_host * np.cos(ang)
        dy = rfrac * R_host * np.sin(ang)
        mfrac = 10**np.random.uniform(-3, -1)
        subs.append(SubHalo(dx, dy, M_host * mfrac, M_host))
    subhalos_by_host[idx] = subs


# ── colour palette ────────────────────────────────────────────────────
BG       = "#080C18"
FG       = "#C8D2E6"
ACCENT1  = "#4DB8FF"
ACCENT1B = "#1A6FCC"
ACCENT2  = "#FF5E7D"
ACCENT3  = "#FFCC33"
ACCENT4  = "#8BF0C5"
LOS_COL  = "#30F0D0"
GRID_COL = "#1C2545"

# ══════════════════════════════════════════════════════════════════════
#                           FIGURE
# ══════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(22, 9), facecolor=BG)

ax = fig.add_axes([0.05, 0.10, 0.58, 0.82], facecolor=BG)
ax_host = fig.add_axes([0.66, 0.45, 0.32, 0.50], facecolor='#0A0F22')
ax_kappa = fig.add_axes([0.66, 0.10, 0.32, 0.28], facecolor='#0A0F22')

y_lim = 1.0

# ═══════════════════ MAIN PANEL ═══════════════════════════════════════

# ── background κ field ────────────────────────────────────────────────
noise = np.random.randn(120, 300) * 0.5
kf = gaussian_filter(noise, sigma=[8, 5])
kf = (kf - kf.min()) / (kf.max() - kf.min() + 1e-12)
kappa_cm = LinearSegmentedColormap.from_list("kf",
    ["#080C18", "#0E1630", "#162845", "#1A3868", "#2A58A0"], N=256)
ax.imshow(kf, extent=[-0.04, 1.04, -y_lim, y_lim],
          aspect='auto', cmap=kappa_cm, alpha=0.3, origin='lower',
          interpolation='bilinear', zorder=0)

# ── redshift planes ──────────────────────────────────────────────────
for zp in np.arange(0.1, 1.0, 0.1):
    xp = z_to_x(zp)
    ax.axvline(xp, color=GRID_COL, lw=0.6, ls='--', alpha=0.5, zorder=1)
    ax.text(xp, y_lim - 0.02, f'z={zp:.1f}', fontsize=7,
            color='#3A4A70', ha='center', va='top', rotation=90,
            fontfamily='monospace')

# ── line of sight with glow ──────────────────────────────────────────
for w, a in [(12, 0.015), (6, 0.04), (2.5, 0.10), (1.0, 0.45)]:
    ax.plot([0, 1], [0, 0], color=LOS_COL, lw=w, alpha=a, zorder=5,
            solid_capstyle='round')

# ── observer & source ────────────────────────────────────────────────
ax.scatter([0], [0], s=300, color='#FFFFFF', marker='D', zorder=25,
           edgecolors=LOS_COL, linewidths=2.5)
ax.text(0, -0.12, 'Observer', fontsize=14, color=FG, ha='center',
        fontweight='bold', path_effects=[pe.withStroke(linewidth=3, foreground=BG)])

ax.scatter([1], [0], s=500, color=ACCENT3, marker='*', zorder=25,
           edgecolors='#FFE899', linewidths=1.5)
ax.text(1, -0.12, f'GW source\nz = {zs}', fontsize=14, color=ACCENT3,
        ha='center', fontweight='bold',
        path_effects=[pe.withStroke(linewidth=3, foreground=BG)])

# ── filaments (aspect-corrected) ──────────────────────────────────────
_ax_w_in = 0.58 * 22.0; _ax_h_in = 0.82 * 9.0
_asp = (_ax_w_in / 1.08) / (_ax_h_in / 2.0)   # ~3.2
_th = np.linspace(0, 2*np.pi, 80)
for i in range(N_fil):
    fx = z_to_x(z_fil[i])
    fy = y_fil[i]
    rs_vis = 0.05 + 0.12 * (M_fil[i] / 1e14)**(1./3.)
    L_vis = 0.10 + 0.25 * (M_fil[i] / 1e14)**(1./3.)
    a_d = L_vis / 2.0; b_d = rs_vis
    ang = np.radians(angle_fil[i])
    ca, sa = np.cos(ang), np.sin(ang)
    for (sf, af) in [(1.0, 0.05), (0.55, 0.15)]:
        ex = a_d * sf * np.cos(_th); ey = b_d * sf * np.sin(_th)
        rx = (ex * ca - ey * sa) / _asp; ry = ex * sa + ey * ca
        ax.fill(fx + rx, fy + ry, color=ACCENT2, alpha=af, zorder=3)

# ── ALL ~100 halos ────────────────────────────────────────────────────
x_halo_arr = np.array([z_to_x(z) for z in z_halos])

# aspect ratio correction for the main axes (data coords)
# x spans 0→1 over 0.58*22 = 12.76 inches, y spans -1→1 over 0.82*9 = 7.38 inches
# so x_data / y_data pixel ratio = (1.0/12.76) / (2.0/7.38) ≈ 0.289
aspect_corr = (2.0 / (0.82 * 9)) / (1.08 / (0.58 * 22))

for i in range(N_halos_target):
    xh = x_halo_arr[i]
    yh = y_halos[i]
    R = vis_radius(M_halos[i])
    logM = np.log10(M_halos[i])

    # NFW glow — use Ellipse to correct for aspect ratio
    Rx = R / aspect_corr  # x-extent in data coords

    # number of glow rings scales with mass (big halos get more detail)
    if logM > 14.0:
        rings = [(1.3, 0.04), (1.0, 0.08), (0.65, 0.15), (0.35, 0.28), (0.12, 0.45)]
    elif logM > 13.0:
        rings = [(1.2, 0.06), (0.7, 0.14), (0.25, 0.32)]
    else:
        rings = [(1.0, 0.10), (0.4, 0.28)]

    for frac, alp in rings:
        c = mpatches.Ellipse((xh, yh), Rx * frac * 2, R * frac * 2,
                             color=ACCENT1, alpha=alp, lw=0, zorder=6)
        ax.add_patch(c)

    # κ_thr contour for massive halos
    if logM > 13.0:
        c_edge = mpatches.Ellipse((xh, yh), Rx * 1.3 * 2, R * 1.3 * 2,
                                  fill=False, edgecolor=ACCENT1B, lw=0.5,
                                  ls=':', alpha=0.3, zorder=7)
        ax.add_patch(c_edge)

    # mass label only for the biggest ones (avoid clutter)
    if logM > 14.0:
        ax.text(xh, yh + R * 1.4, f'$10^{{{logM:.1f}}}$', fontsize=6.5,
                color=ACCENT1, ha='center', va='bottom', alpha=0.8,
                path_effects=[pe.withStroke(linewidth=2, foreground=BG)])

    # impact parameter line only for large halos
    if logM > 13.5:
        ax.plot([xh, xh], [0, yh], color=ACCENT1, lw=0.3, ls='--', alpha=0.15, zorder=4)

    # subhalos
    if i in subhalos_by_host:
        for sh in subhalos_by_host[i]:
            sx = xh + sh.dx / aspect_corr
            sy = yh + sh.dy
            sz = max(3, 20 * (sh.m / sh.host_M)**0.25)
            ax.scatter([sx], [sy], s=sz, color=ACCENT3, alpha=0.85,
                       zorder=12, edgecolors='none')

# ── convergence arrows for top-3 ──────────────────────────────────────
for idx in top5_idx[:3]:
    xh = x_halo_arr[idx]
    yh = y_halos[idx]
    ax.annotate('', xy=(xh, yh * 0.1),
                xytext=(xh, yh * 0.5),
                arrowprops=dict(arrowstyle='->', color=ACCENT4, lw=0.8, alpha=0.25,
                                connectionstyle='arc3,rad=0.2'), zorder=8)

# ── κ_W annotation ───────────────────────────────────────────────────
ax.annotate('Gaussian $\\kappa_W$\n(unresolved, weak)',
            xy=(0.5, -0.03), xytext=(0.5, -y_lim + 0.12),
            fontsize=8, color='#5A7AB0', ha='center',
            arrowprops=dict(arrowstyle='->', color='#3A5580', lw=0.8, alpha=0.35),
            path_effects=[pe.withStroke(linewidth=2, foreground=BG)], zorder=15)

# ── halo count annotation ────────────────────────────────────────────
ax.text(0.99, 0.99, f'{N_halos_target} resolved halos\n(drawn from HMF)',
        transform=ax.transAxes, fontsize=9, va='top', ha='right',
        color=ACCENT1, alpha=0.8,
        bbox=dict(boxstyle='round,pad=0.3', fc='#0C1225', ec=ACCENT1B, alpha=0.7),
        path_effects=[pe.withStroke(linewidth=1, foreground=BG)], zorder=30)

# ── info box ──────────────────────────────────────────────────────────
info = (
    f"Source: $z_s = {zs}$  ·  $d_c \\approx {dc_source:.0f}$ Mpc\n"
    f"$\\kappa_{{\\rm thr}}$ splits strong / weak\n"
    f"Mass grid: $10^7 - 10^{{17}} M_\\odot$\n"
    f"$\\Omega_m = {Om}$, $h = {h0}$, $\\sigma_8 = {sigma8}$\n"
    f"NFW + pseudo-ε, CYL filaments\n"
    f"JvdB14 anti-biased subhalos"
)
ax.text(0.01, 0.99, info, transform=ax.transAxes, fontsize=8,
        va='top', color=FG, fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', fc='#0C1225', ec='#253060', alpha=0.92),
        zorder=30)

# ── axes ──────────────────────────────────────────────────────────────
ax.set_xlim(-0.04, 1.04)
ax.set_ylim(-y_lim, y_lim)
ax.set_xlabel('Normalised comoving distance along LoS', fontsize=12, color=FG, labelpad=8)
ax.set_ylabel('Transverse offset  [exaggerated]', fontsize=12, color=FG, labelpad=8)
ax.set_title('One Weak-Lensing Realization  —  Vaskonen MC Geometry',
             fontsize=17, color='#FFFFFF', fontweight='bold', pad=15,
             path_effects=[pe.withStroke(linewidth=4, foreground=BG)])
ax.tick_params(colors=FG, labelsize=9)
for sp in ax.spines.values(): sp.set_color('#253060')

# ── legend ────────────────────────────────────────────────────────────
handles = [
    Line2D([0],[0], marker='D', color=BG, markerfacecolor='#FFF',
           markeredgecolor=LOS_COL, ms=8, ls='None', label='Observer'),
    Line2D([0],[0], marker='*', color=BG, markerfacecolor=ACCENT3,
           ms=12, ls='None', label='GW Source'),
    Line2D([0],[0], color=LOS_COL, lw=1.5, label='Line of Sight'),
    Line2D([0],[0], marker='o', color=BG, markerfacecolor=ACCENT1,
           ms=10, ls='None', label=f'NFW halo ({N_halos_target} resolved)'),
    Line2D([0],[0], marker='o', color=BG, markerfacecolor=ACCENT2,
           ms=8, ls='None', label=f'Filament ({N_fil} CYL)'),
    Line2D([0],[0], marker='o', color=BG, markerfacecolor=ACCENT3,
           ms=5.5, ls='None', label='Subhalo clump'),
    mpatches.Patch(fc='#162040', ec='none', alpha=0.5, label='Weak $\\kappa$ field'),
    Line2D([0],[0], color='#3A4A70', lw=0.8, ls='--', label='Lens plane ($\\Delta z=0.1$)'),
]
leg = ax.legend(handles=handles, loc='lower left', fontsize=8,
                facecolor='#0C1225', edgecolor='#253060', labelcolor=FG,
                framealpha=0.92, borderpad=0.7)
leg.set_zorder(30)


# ═══════════════════ INSET: HOST ZOOM ════════════════════════════════
ax_host.set_aspect('equal')
zoom_idx = top5_idx[-1]  # most massive
zoom_M = M_halos[zoom_idx]

# NFW 2D kappa
nr = 400
xg = np.linspace(-1.4, 1.4, nr)
YG, XG = np.meshgrid(xg, xg)
RG = np.sqrt(XG**2 + YG**2)
rs_n = 0.22
x_nfw = RG / rs_n + 1e-8
kappa_2d = np.where(x_nfw < 1,
    1.0 / (x_nfw**2 - 1 + 1e-6) * (1 - np.arctanh(np.sqrt(np.clip(1-x_nfw,0,None)/(1+x_nfw+1e-8))) / np.sqrt(np.clip(1-x_nfw**2,1e-12,None))),
    np.where(x_nfw > 1,
        1.0 / (x_nfw**2 - 1 + 1e-6) * (1 - np.arctan(np.sqrt(np.clip(x_nfw-1,0,None)/(x_nfw+1+1e-8))) / np.sqrt(np.clip(x_nfw**2-1,1e-12,None))),
        1.0/3.0))
kappa_2d = np.clip(kappa_2d, 0, np.nanpercentile(kappa_2d[np.isfinite(kappa_2d) & (kappa_2d>0)], 99.5))
kappa_2d = np.nan_to_num(kappa_2d, 0)
kappa_2d /= (kappa_2d.max() + 1e-12)

nfw_cm = LinearSegmentedColormap.from_list("nfw",
    ["#0A0F22","#0F1A40","#1A3878","#2A60B0","#4DB8FF","#80D0FF","#C8EEFF"], N=256)
ax_host.imshow(kappa_2d, extent=[-1.4,1.4,-1.4,1.4], origin='lower',
               cmap=nfw_cm, alpha=0.75, interpolation='bilinear', zorder=1)

# r200 and rs
for (rr, lab, col, ls) in [(1.0,'$r_{200}$',ACCENT1,'--'),(rs_n,'$r_s$','#6090C0',':')]:
    ax_host.add_patch(plt.Circle((0,0), rr, fill=False, ec=col, lw=1.0, ls=ls, alpha=0.55, zorder=10))
    ax_host.text(rr*0.71+0.04, rr*0.71+0.04, lab, fontsize=10, color=col, alpha=0.7,
                 path_effects=[pe.withStroke(linewidth=2, foreground='#0A0F22')])

# subhalos
if zoom_idx in subhalos_by_host:
    R_host = vis_radius(zoom_M)
    for sh in subhalos_by_host[zoom_idx]:
        sx = sh.dx / R_host
        sy = sh.dy / R_host
        sz = max(15, 60*(sh.m/sh.host_M)**0.25)
        ax_host.scatter([sx],[sy], s=sz, color=ACCENT3, alpha=0.9, zorder=15,
                        edgecolors='#FFE8A0', linewidths=0.4)
        ax_host.add_patch(plt.Circle((sx,sy), max(0.02, 0.08*(sh.m/sh.host_M)**0.2),
                                      color=ACCENT3, alpha=0.12, lw=0, zorder=14))

# LoS ray
impact = -0.35
ax_host.plot([-1.4,1.4],[impact,impact], color=LOS_COL, lw=1.0, alpha=0.5, zorder=8)
ax_host.scatter([0],[impact], s=50, marker='+', color=LOS_COL, lw=2, alpha=0.8, zorder=12)
ax_host.text(-1.3, impact+0.07, 'LoS ray', fontsize=8, color=LOS_COL, alpha=0.6,
             path_effects=[pe.withStroke(linewidth=2, foreground='#0A0F22')])
ax_host.annotate('', xy=(0.06,impact), xytext=(0.06,0),
                 arrowprops=dict(arrowstyle='<->', color='#FFF', lw=0.8, alpha=0.5), zorder=11)
ax_host.text(0.12, impact/2, '$r$', fontsize=12, color='#FFF', alpha=0.65,
             path_effects=[pe.withStroke(linewidth=2, foreground='#0A0F22')])
ax_host.scatter([0],[0], s=70, marker='x', color='#FFF', lw=1.5, alpha=0.7, zorder=20)

ax_host.text(0.03, 0.97,
    f'Host interior\n'
    f'$M = 10^{{{np.log10(zoom_M):.1f}}}\\,M_\\odot$\n'
    f'Anti-biased radial profile\n'
    f'Dynamic floor $m_{{\\rm res}}(r)$\n'
    f'$\\Delta\\kappa = \\kappa_{{\\rm clump}} - m \\cdot g(r)$',
    transform=ax_host.transAxes, fontsize=7.5, va='top', color=FG, fontfamily='monospace',
    bbox=dict(boxstyle='round,pad=0.4', fc='#0C1225', ec='#253060', alpha=0.9), zorder=30)

ax_host.set_xlim(-1.4,1.4); ax_host.set_ylim(-1.4,1.4)
ax_host.set_xlabel('$x / r_{200}$', fontsize=10, color=FG, labelpad=4)
ax_host.set_ylabel('$y / r_{200}$', fontsize=10, color=FG, labelpad=4)
ax_host.tick_params(colors=FG, labelsize=8)
for sp in ax_host.spines.values(): sp.set_color('#3A5080'); sp.set_linewidth(1.5)

# zoom connector
R_zoom = vis_radius(zoom_M)
Rx_zoom = R_zoom / aspect_corr
xz = x_halo_arr[zoom_idx]
yz = y_halos[zoom_idx]
rect = mpatches.FancyBboxPatch((xz - Rx_zoom*1.5, yz - R_zoom*1.5),
                                Rx_zoom*3, R_zoom*3,
                                boxstyle="round,pad=0.003", fill=False,
                                edgecolor=ACCENT1, lw=1.5, alpha=0.5, zorder=20)
ax.add_patch(rect)
ax.text(xz, yz - R_zoom*1.7, '↗ zoom', fontsize=7, color=ACCENT1, ha='center',
        alpha=0.6, path_effects=[pe.withStroke(linewidth=2, foreground=BG)])


# ═══════════════════ KAPPA BUILDUP ════════════════════════════════════
z_arr = np.linspace(0.001, zs, 600)
x_arr = np.array([z_to_x(z) for z in z_arr])
dkw = 0.0015 * np.random.randn(len(x_arr)) * np.sqrt(np.diff(np.concatenate([[0], x_arr])))
kappa_cum = np.cumsum(dkw)

# sort halos by x position for orderly kicks
sort_order = np.argsort(x_halo_arr)
halo_kick_pts = []
for i in sort_order:
    idx_in_arr = np.searchsorted(x_arr, x_halo_arr[i])
    if idx_in_arr < len(kappa_cum):
        kick = max(0.0005, 0.008 * (M_halos[i]/1e14)**0.5 * np.random.uniform(0.5, 1.5))
        kappa_cum[idx_in_arr:] += kick
        halo_kick_pts.append((x_arr[idx_in_arr], kappa_cum[idx_in_arr], M_halos[i]))

ax_kappa.fill_between(x_arr, 0, kappa_cum, color=ACCENT1, alpha=0.12, zorder=2)
ax_kappa.plot(x_arr, kappa_cum, color=ACCENT1, lw=1.0, alpha=0.8, zorder=3)

# highlight the biggest kicks
for xk, yk, Mk in halo_kick_pts:
    if Mk > 5e13:
        ax_kappa.scatter([xk], [yk], s=15, color=ACCENT3, zorder=10, edgecolors='none')

ax_kappa.text(0.97, 0.97, f'{N_halos_target} step contributions\nΣ(strong) + Gaussian(weak)',
              transform=ax_kappa.transAxes, fontsize=7, va='top', ha='right',
              color=FG, alpha=0.7, fontfamily='monospace',
              bbox=dict(boxstyle='round,pad=0.3', fc='#0C1225', ec='#253060', alpha=0.8))

ax_kappa.set_xlabel('Normalised $d_c$', fontsize=9, color=FG, labelpad=4)
ax_kappa.set_ylabel('$\\kappa_{\\rm cum}$', fontsize=10, color=FG, labelpad=4)
ax_kappa.set_title('Convergence buildup along LoS', fontsize=10, color=FG, pad=6)
ax_kappa.tick_params(colors=FG, labelsize=7)
ax_kappa.set_xlim(0, 1)
for sp in ax_kappa.spines.values(): sp.set_color('#253060')

# ── bottom note ───────────────────────────────────────────────────────
fig.text(0.34, 0.02,
    'Transverse separations heavily exaggerated  (real impact ~ kpc).  '
    f'{N_halos_target} halos drawn from HMF; most are low-mass (~$10^{{12}}$).  '
    'Subhalos shown inside the 5 most massive hosts.',
    fontsize=7.5, color='#4A5A80', ha='center', fontstyle='italic',
    path_effects=[pe.withStroke(linewidth=2, foreground=BG)])

# ── save ──────────────────────────────────────────────────────────────
outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plots', 'realization_diagram.png')
os.makedirs(os.path.dirname(outpath), exist_ok=True)
fig.savefig(outpath, dpi=200, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print(f"Saved → {os.path.abspath(outpath)}")
