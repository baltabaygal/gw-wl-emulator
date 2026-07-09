#!/usr/bin/env python3
"""Comprehensive asymptotic analysis of sigma_full(z_s) (halo-only Campbell).

Reads campbell_P_of_z.txt (P(z), chi(z), cumulative moments M2,M3,M4 to z=30) and
campbell_sigma_dense.txt (exact moment-formula sigma on a dense z_s grid).

Panels:
  (a) sigma(z_s), 0.02-30: exact; DERIVED z^{3/2} low-z asymptote (no free params);
      sigma_inf saturation; the broken-power-law fit extrapolated to show its failure.
  (b) local slope dln(sigma)/dln(z_s): exact vs BPL; 3/2 at low z, -> 0 at high z.
  (c) P(z): where the variance is generated.
  (d) high-z_s closed form: sigma^2 vs 1/chi_s is EXACTLY a parabola once P's support
      is exhausted (frozen moments) -- shown against the exact curve.

Run from playground/. Writes plots/campbell_asymptotics.{png,pdf}.
"""
import numpy as np
import matplotlib.pyplot as plt

INK, MUTED, BASE, GRID = "#0b0b0b", "#898781", "#c3c2b7", "#e1e0d9"
BLUE, RED, GREEN, ORANGE = "#2a78d6", "#e34948", "#2e9e62", "#e28425"
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "axes.edgecolor": BASE, "font.size": 10})

z, chi, P, M2, M3, M4 = np.loadtxt("campbell_P_of_z.txt", unpack=True)
zsd, sd = np.loadtxt("campbell_sigma_dense.txt", unpack=True)

# horizon distance and sigma_inf (frozen moments at z=30)
CHIHOR = 1.401955e7  # kpc, from campbell_moments stdout
m2, m3, m4 = M2[-1], M3[-1], M4[-1]
siginf = np.sqrt(m2 - 2*m3/CHIHOR + m4/CHIHOR**2)

# derived low-z asymptote: sigma = sqrt(P(0)/30) * chi'(0) * z^{3/2}
A_pred = 0.044312

def bpl(zz, A=0.0431, z0=1.834, q=1.207, p=1.407):
    return A * zz**1.5 / (1 + (zz/z0)**q)**(p/q)

fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.6))
(a, b), (c, d) = axes

# (a) sigma(zs) with asymptotes
a.plot(zsd, sd, "-", color=BLUE, lw=2.0, zorder=5, label=r"exact (moment formula)")
a.plot(zsd, A_pred*zsd**1.5, ":", color=INK, lw=1.4, zorder=3,
       label=r"derived: $\sqrt{P(0)/30}\,\frac{c}{H_0} z_s^{3/2}$ (no free params)")
a.axhline(siginf, color=GREEN, ls="--", lw=1.4, zorder=2,
          label=rf"$\sigma_\infty = {siginf:.3f}$ (frozen moments, $\chi_s\to\chi_{{\rm hor}}$)")
a.plot(zsd, bpl(zsd), "-.", color=RED, lw=1.4, zorder=4,
       label="broken-power-law fit (fitted on 0.2–10)")
a.axvspan(10, 30, color=GRID, alpha=0.45, zorder=1)
a.set_xscale("log"); a.set_yscale("log")
a.set_xlabel(r"$z_s$"); a.set_ylabel(r"$\sigma_\kappa$")
a.set_ylim(2e-4, 0.3)
a.legend(frameon=False, fontsize=8.2, loc="lower right")
a.set_title(r"(a) exact $\sigma(z_s)$ and both asymptotes")

# (b) local slope
lz, ls_ = np.log(zsd), np.log(sd)
slope = np.gradient(ls_, lz)
slope_bpl = np.gradient(np.log(bpl(zsd)), lz)
b.plot(zsd, slope, "-", color=BLUE, lw=2.0, label="exact")
b.plot(zsd, slope_bpl, "-.", color=RED, lw=1.4, label="BPL fit")
b.axhline(1.5, color=INK, ls=":", lw=1.2); b.axhline(0.0, color=GREEN, ls="--", lw=1.2)
b.text(0.025, 1.52, "3/2 (derived)", fontsize=8.5)
b.text(12, 0.03, r"0 (saturation)", fontsize=8.5, color=GREEN)
b.axvspan(10, 30, color=GRID, alpha=0.45, zorder=1)
b.set_xscale("log")
b.set_xlabel(r"$z_s$"); b.set_ylabel(r"$d\ln\sigma / d\ln z_s$")
b.legend(frameon=False, fontsize=8.5)
b.set_title("(b) local power-law slope")

# (c) P(z)
c.plot(z, P, "-", color=ORANGE, lw=1.8)
c.set_xscale("log"); c.set_yscale("log")
c.set_xlabel(r"$z$"); c.set_ylabel(r"$P(z)$  [kpc$^{-2}$ per unit $z$]")
c.text(0.004, P.max()*0.25, "monotonic: flat to $z\\approx0.5$,\nthen HMF-driven decline\n"
       "(3 decades down by $z=20$)", fontsize=8.5)
c.set_title(r"(c) the variance source density $P(z)$" "\n"
            r"$\sigma^2(z_s)=\int_0^{z_s} P(z)\,[\chi(1-\chi/\chi_s)]^2 dz$")

# (d) sigma^2 vs 1/chi_s: exact parabola at high z_s (zoom on z_s >= 2)
mhi = zsd >= 2.0
u = 1.0/np.interp(zsd[mhi], z, chi)
d.plot(u*1e7, sd[mhi]**2, "-", color=BLUE, lw=2.2, label=r"exact $\sigma^2(z_s)$")
uu = np.linspace(1.0/CHIHOR, u.max(), 200)
d.plot(uu*1e7, m2 - 2*m3*uu + m4*uu**2, "--", color=GREEN, lw=1.6,
       label=r"frozen-moment parabola $M_2 - 2M_3 u + M_4 u^2$")
d.plot([1e7/CHIHOR], [siginf**2], "*", color=GREEN, ms=14, zorder=6,
       label=r"$u = 1/\chi_{\rm hor}$:  $\sigma_\infty^2$")
for zz in (2, 5, 10, 20, 30):
    uz = 1e7/np.interp(zz, z, chi)
    d.annotate(rf"$z_s={zz}$", xy=(uz, np.interp(zz, zsd, sd)**2),
               xytext=(uz + 0.04, np.interp(zz, zsd, sd)**2 + 0.0008), fontsize=8, color=MUTED)
d.set_xlabel(r"$u = 1/\chi_s$  [$10^{-7}$ kpc$^{-1}$]")
d.set_ylabel(r"$\sigma^2$")
d.legend(frameon=False, fontsize=8.5, loc="upper right")
d.set_title(r"(d) high-$z_s$ closed form: $\sigma^2$ is a parabola in $1/\chi_s$")

for ax in axes.flat:
    ax.grid(color=GRID, lw=.5, which="both"); ax.set_axisbelow(True)
plt.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"../plots/campbell_asymptotics.{ext}", dpi=200)
print("wrote plots/campbell_asymptotics.{png,pdf}")

# quantitative report
print(f"\nsigma_inf = {siginf:.4f};  sigma(10) = {np.interp(10, zsd, sd):.4f} "
      f"({np.interp(10, zsd, sd)/siginf*100:.0f}% of saturation)")
for zz in (0.05, 0.1, 0.2):
    print(f"low-z check z={zz}: exact/derived-asymptote = "
          f"{np.interp(zz, zsd, sd)/(A_pred*zz**1.5):.4f}")
for zz in (10, 15, 20, 30):
    e = np.interp(zz, zsd, sd)
    print(f"BPL extrapolation error at z={zz}: {(bpl(zz)/e - 1)*100:+.1f}%")
EOF = None
