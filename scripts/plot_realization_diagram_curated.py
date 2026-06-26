#!/usr/bin/env python3
"""
Diagram of one weak-lensing realization (curated / exaggerated version).

~8 hand-picked halos for maximum visual clarity. Companion to the
full 100-halo version in plot_realization_diagram.py.
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

BG       = "#080C18"
FG       = "#C8D2E6"
ACCENT1  = "#4DB8FF"
ACCENT1B = "#1A6FCC"
ACCENT2  = "#FF5E7D"
ACCENT3  = "#FFCC33"
ACCENT4  = "#8BF0C5"
LOS_COL  = "#30F0D0"
GRID_COL = "#1C2545"

class Halo:
    def __init__(self, z, M, y, R):
        self.z, self.M = z, M
        self.x = z_to_x(z)
        self.y, self.R = y, R
        self.subs = []

class SubHalo:
    def __init__(self, dx, dy, m, host_M):
        self.dx, self.dy, self.m = dx, dy, m
        self.host_M = host_M

class Filament:
    def __init__(self, z, y, w, L, angle):
        self.x = z_to_x(z)
        self.y, self.w, self.L, self.angle = y, w, L, angle

halos = [
    Halo(0.08, 5e14,   0.55,  0.12),
    Halo(0.18, 1e14,  -0.40,  0.07),
    Halo(0.32, 3e14,   0.30,  0.10),
    Halo(0.45, 8e13,  -0.25,  0.06),
    Halo(0.58, 2e14,   0.48,  0.09),
    Halo(0.70, 1e14,  -0.55,  0.07),
    Halo(0.80, 5e13,   0.65,  0.05),
    Halo(0.92, 4e13,  -0.35,  0.04),
]

filaments = [
    Filament(0.14, -0.60,  0.18, 0.35,   5),
    Filament(0.40,  0.65,  0.20, 0.40, 150),
    Filament(0.62, -0.70,  0.15, 0.30,  85),
    Filament(0.85,  0.42,  0.12, 0.25,  45),
]

for h in [halos[0], halos[2], halos[4]]:
    n = np.random.poisson(7)
    for _ in range(n):
        angle = np.random.uniform(0, 2*np.pi)
        rfrac = np.random.beta(3, 1.5) * 0.9
        dx = rfrac * h.R * np.cos(angle)
        dy = rfrac * h.R * np.sin(angle)
        mfrac = 10**np.random.uniform(-3, -1)
        h.subs.append(SubHalo(dx, dy, h.M * mfrac, h.M))

fig = plt.figure(figsize=(22, 9), facecolor=BG)
ax = fig.add_axes([0.05, 0.10, 0.58, 0.82], facecolor=BG)
ax_host = fig.add_axes([0.66, 0.45, 0.32, 0.50], facecolor='#0A0F22')
ax_kappa = fig.add_axes([0.66, 0.10, 0.32, 0.28], facecolor='#0A0F22')
y_lim = 1.0

# background
noise = np.random.randn(120, 300) * 0.5
kf = gaussian_filter(noise, sigma=[8, 5])
kf = (kf - kf.min()) / (kf.max() - kf.min() + 1e-12)
kappa_cm = LinearSegmentedColormap.from_list("kf",
    ["#080C18", "#0E1630", "#162845", "#1A3868", "#2A58A0"], N=256)
ax.imshow(kf, extent=[-0.04, 1.04, -y_lim, y_lim],
          aspect='auto', cmap=kappa_cm, alpha=0.3, origin='lower',
          interpolation='bilinear', zorder=0)

for zp in np.arange(0.1, 1.0, 0.1):
    xp = z_to_x(zp)
    ax.axvline(xp, color=GRID_COL, lw=0.6, ls='--', alpha=0.5, zorder=1)
    ax.text(xp, y_lim - 0.02, f'z={zp:.1f}', fontsize=7,
            color='#3A4A70', ha='center', va='top', rotation=90, fontfamily='monospace')

for w, a in [(12, 0.015), (6, 0.04), (2.5, 0.10), (1.0, 0.45)]:
    ax.plot([0, 1], [0, 0], color=LOS_COL, lw=w, alpha=a, zorder=5, solid_capstyle='round')

ax.scatter([0], [0], s=300, color='#FFFFFF', marker='D', zorder=25,
           edgecolors=LOS_COL, linewidths=2.5)
ax.text(0, -0.14, 'Observer', fontsize=14, color=FG, ha='center',
        fontweight='bold', path_effects=[pe.withStroke(linewidth=3, foreground=BG)])
ax.scatter([1], [0], s=500, color=ACCENT3, marker='*', zorder=25,
           edgecolors='#FFE899', linewidths=1.5)
ax.text(1, -0.14, f'GW source\nz = {zs}', fontsize=14, color=ACCENT3,
        ha='center', fontweight='bold',
        path_effects=[pe.withStroke(linewidth=3, foreground=BG)])

# ── compute display aspect ratio for correct filament angles ──────────
# The panel x spans 0.58 of fig_width=22", data x spans 1.08
# The panel y spans 0.82 of fig_height=9", data y spans 2.0
_ax_w_inches = 0.58 * 22.0   # 12.76"
_ax_h_inches = 0.82 * 9.0    # 7.38"
_x_range = 1.08              # -0.04 to 1.04
_y_range = 2.0               # -1 to 1
_px_per_xunit = _ax_w_inches / _x_range  # ~11.8 "/unit
_px_per_yunit = _ax_h_inches / _y_range  # ~3.69 "/unit
_aspect_ratio = _px_per_xunit / _px_per_yunit   # ~3.2

for fil in filaments:
    ang_rad = np.radians(fil.angle)
    # half-lengths of major/minor axes in DATA units
    a_data = fil.L / 2.0    # semi-major
    b_data = fil.w           # semi-minor (w already half-width)
    # Generate ellipse vertices in data coords, accounting for aspect
    theta = np.linspace(0, 2*np.pi, 120)
    # In display space, the ellipse is: (a*cos(t), b*sin(t)) rotated by angle
    # Convert display-space offsets to data-space:
    #   dx_data = dx_display / aspect_ratio,  dy_data = dy_display
    cos_a, sin_a = np.cos(ang_rad), np.sin(ang_rad)
    for (scale, af) in [(1.0, 0.08), (0.55, 0.20)]:
        ex = a_data * scale * np.cos(theta)
        ey = b_data * scale * np.sin(theta)
        # rotate in display-proportional space, then scale x back to data
        rx = (ex * cos_a - ey * sin_a) / _aspect_ratio
        ry = (ex * sin_a + ey * cos_a)
        ax.fill(fil.x + rx, fil.y + ry, color=ACCENT2, alpha=af, zorder=3)
    # dotted outline
    ex = a_data * np.cos(theta)
    ey = b_data * np.sin(theta)
    rx = (ex * cos_a - ey * sin_a) / _aspect_ratio
    ry = (ex * sin_a + ey * cos_a)
    ax.plot(fil.x + rx, fil.y + ry, color=ACCENT2, lw=0.7, ls=':', alpha=0.35, zorder=4)
    # orientation axis line through center
    ax_len = a_data * 0.7
    dx_ax = ax_len * cos_a / _aspect_ratio
    dy_ax = ax_len * sin_a
    ax.plot([fil.x - dx_ax, fil.x + dx_ax],
            [fil.y - dy_ax, fil.y + dy_ax],
            color=ACCENT2, lw=1.2, alpha=0.45, zorder=4)
    ax.text(fil.x + dx_ax * 1.15, fil.y + dy_ax * 1.15, 'fil', fontsize=6,
            color=ACCENT2, fontstyle='italic', alpha=0.55, ha='center', va='center',
            path_effects=[pe.withStroke(linewidth=2, foreground=BG)])

for halo in halos:
    x, y, R = halo.x, halo.y, halo.R
    for frac, alp in [(1.4, 0.03), (1.0, 0.07), (0.65, 0.14), (0.35, 0.26), (0.12, 0.45)]:
        c = mpatches.Ellipse((x, y), R * frac * 2 * 0.07, R * frac * 2,
                             color=ACCENT1, alpha=alp, lw=0, zorder=6)
        ax.add_patch(c)
    c_edge = mpatches.Ellipse((x, y), R * 1.4 * 2 * 0.07, R * 1.4 * 2,
                              fill=False, edgecolor=ACCENT1B, lw=0.8, ls=':',
                              alpha=0.4, zorder=7)
    ax.add_patch(c_edge)
    logM = np.log10(halo.M)
    ax.text(x, y + R * 1.5, f'$10^{{{logM:.1f}}}$', fontsize=7.5,
            color=ACCENT1, ha='center', va='bottom', alpha=0.85,
            path_effects=[pe.withStroke(linewidth=2, foreground=BG)])
    ax.plot([x, x], [0, y], color=ACCENT1, lw=0.4, ls='--', alpha=0.2, zorder=4)
    for sh in halo.subs:
        sx = x + sh.dx * 0.07
        sy = y + sh.dy
        sz = max(4, 30 * (sh.m / halo.M)**0.25)
        ax.scatter([sx], [sy], s=sz, color=ACCENT3, alpha=0.85, zorder=12, edgecolors='none')

for halo in [halos[0], halos[2], halos[4]]:
    ax.annotate('', xy=(halo.x, halo.y * 0.12),
                xytext=(halo.x, halo.y * 0.55),
                arrowprops=dict(arrowstyle='->', color=ACCENT4, lw=1.0, alpha=0.3,
                                connectionstyle='arc3,rad=0.2'), zorder=8)

ax.annotate('Gaussian $\\kappa_W$\n(unresolved, weak)',
            xy=(0.5, -0.03), xytext=(0.5, -y_lim + 0.12),
            fontsize=8, color='#5A7AB0', ha='center',
            arrowprops=dict(arrowstyle='->', color='#3A5580', lw=0.8, alpha=0.35),
            path_effects=[pe.withStroke(linewidth=2, foreground=BG)], zorder=15)

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
        bbox=dict(boxstyle='round,pad=0.5', fc='#0C1225', ec='#253060', alpha=0.92), zorder=30)

ax.set_xlim(-0.04, 1.04); ax.set_ylim(-y_lim, y_lim)
ax.set_xlabel('Normalised comoving distance along LoS', fontsize=12, color=FG, labelpad=8)
ax.set_ylabel('Transverse offset  [exaggerated]', fontsize=12, color=FG, labelpad=8)
ax.set_title('One Weak-Lensing Realization  —  Vaskonen MC Geometry  (curated)',
             fontsize=17, color='#FFFFFF', fontweight='bold', pad=15,
             path_effects=[pe.withStroke(linewidth=4, foreground=BG)])
ax.tick_params(colors=FG, labelsize=9)
for sp in ax.spines.values(): sp.set_color('#253060')

handles = [
    Line2D([0],[0], marker='D', color=BG, markerfacecolor='#FFF',
           markeredgecolor=LOS_COL, ms=8, ls='None', label='Observer'),
    Line2D([0],[0], marker='*', color=BG, markerfacecolor=ACCENT3,
           ms=12, ls='None', label='GW Source'),
    Line2D([0],[0], color=LOS_COL, lw=1.5, label='Line of Sight'),
    Line2D([0],[0], marker='o', color=BG, markerfacecolor=ACCENT1,
           ms=10, ls='None', label='NFW halo (resolved)'),
    Line2D([0],[0], marker='o', color=BG, markerfacecolor=ACCENT2,
           ms=8, ls='None', label='Filament (CYL)'),
    Line2D([0],[0], marker='o', color=BG, markerfacecolor=ACCENT3,
           ms=5.5, ls='None', label='Subhalo clump'),
    mpatches.Patch(fc='#162040', ec='none', alpha=0.5, label='Weak $\\kappa$ field'),
    Line2D([0],[0], color='#3A4A70', lw=0.8, ls='--', label='Lens plane ($\\Delta z=0.1$)'),
]
leg = ax.legend(handles=handles, loc='lower left', fontsize=8.5,
                facecolor='#0C1225', edgecolor='#253060', labelcolor=FG,
                framealpha=0.92, borderpad=0.7)
leg.set_zorder(30)

# ═══════════════════ INSET: HOST ZOOM ════════════════════════════════
ax_host.set_aspect('equal')
host = halos[0]

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

for (rr, lab, col, ls) in [(1.0,'$r_{200}$',ACCENT1,'--'),(rs_n,'$r_s$','#6090C0',':')]:
    ax_host.add_patch(plt.Circle((0,0), rr, fill=False, ec=col, lw=1.0, ls=ls, alpha=0.55, zorder=10))
    ax_host.text(rr*0.71+0.04, rr*0.71+0.04, lab, fontsize=10, color=col, alpha=0.7,
                 path_effects=[pe.withStroke(linewidth=2, foreground='#0A0F22')])

hz_R = host.R
for sh in host.subs:
    sx = sh.dx / hz_R; sy = sh.dy / hz_R
    sz = max(15, 60*(sh.m/sh.host_M)**0.25)
    ax_host.scatter([sx],[sy], s=sz, color=ACCENT3, alpha=0.9, zorder=15,
                    edgecolors='#FFE8A0', linewidths=0.4)
    ax_host.add_patch(plt.Circle((sx,sy), max(0.02, 0.08*(sh.m/sh.host_M)**0.2),
                                  color=ACCENT3, alpha=0.12, lw=0, zorder=14))

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
    f'Host interior\n$M = 10^{{{np.log10(host.M):.1f}}}\\,M_\\odot$\n'
    f'Anti-biased radial profile\nDynamic floor $m_{{\\rm res}}(r)$\n'
    f'$\\Delta\\kappa = \\kappa_{{\\rm clump}} - m \\cdot g(r)$',
    transform=ax_host.transAxes, fontsize=7.5, va='top', color=FG, fontfamily='monospace',
    bbox=dict(boxstyle='round,pad=0.4', fc='#0C1225', ec='#253060', alpha=0.9), zorder=30)

ax_host.set_xlim(-1.4,1.4); ax_host.set_ylim(-1.4,1.4)
ax_host.set_xlabel('$x / r_{200}$', fontsize=10, color=FG, labelpad=4)
ax_host.set_ylabel('$y / r_{200}$', fontsize=10, color=FG, labelpad=4)
ax_host.tick_params(colors=FG, labelsize=8)
for sp in ax_host.spines.values(): sp.set_color('#3A5080'); sp.set_linewidth(1.5)

hR = host.R
rect = mpatches.FancyBboxPatch((host.x - hR*0.07*1.5, host.y - hR*1.5),
                                hR*0.07*3, hR*3,
                                boxstyle="round,pad=0.005", fill=False,
                                edgecolor=ACCENT1, lw=1.5, alpha=0.5, zorder=20)
ax.add_patch(rect)

# ═══════════════════ KAPPA BUILDUP ════════════════════════════════════
z_arr = np.linspace(0.001, zs, 600)
x_arr = np.array([z_to_x(z) for z in z_arr])
dkw = 0.003 * np.random.randn(len(x_arr)) * np.sqrt(np.diff(np.concatenate([[0], x_arr])))
kappa_cum = np.cumsum(dkw)

for halo in halos:
    idx = np.searchsorted(x_arr, halo.x)
    if idx < len(kappa_cum):
        kick = max(0.002, 0.015 * (halo.M/1e14)**0.45 * np.random.uniform(0.6, 1.4))
        kappa_cum[idx:] += kick

ax_kappa.fill_between(x_arr, 0, kappa_cum, color=ACCENT1, alpha=0.12, zorder=2)
ax_kappa.plot(x_arr, kappa_cum, color=ACCENT1, lw=1.2, alpha=0.8, zorder=3)

for halo in [halos[0], halos[2], halos[4]]:
    idx = np.searchsorted(x_arr, halo.x)
    if idx < len(kappa_cum):
        ax_kappa.scatter([x_arr[idx]], [kappa_cum[idx]], s=25, color=ACCENT3, zorder=10, edgecolors='none')

ax_kappa.set_xlabel('Normalised $d_c$', fontsize=9, color=FG, labelpad=4)
ax_kappa.set_ylabel('$\\kappa_{\\rm cum}$', fontsize=10, color=FG, labelpad=4)
ax_kappa.set_title('Convergence buildup along LoS', fontsize=10, color=FG, pad=6)
ax_kappa.tick_params(colors=FG, labelsize=7)
ax_kappa.set_xlim(0, 1)
for sp in ax_kappa.spines.values(): sp.set_color('#253060')

fig.text(0.34, 0.02,
    'Transverse separations heavily exaggerated  (real impact ~ kpc).  '
    'Curated version: 8 representative halos shown for visual clarity.',
    fontsize=7.5, color='#4A5A80', ha='center', fontstyle='italic',
    path_effects=[pe.withStroke(linewidth=2, foreground=BG)])

outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plots', 'realization_diagram_curated.png')
os.makedirs(os.path.dirname(outpath), exist_ok=True)
fig.savefig(outpath, dpi=200, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print(f"Saved → {os.path.abspath(outpath)}")
