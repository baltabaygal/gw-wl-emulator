import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = np.loadtxt("tmp/kthr_driver/kthr_vs_zs.dat")
zs, kthr = d[:,0], d[:,1]
fig, ax = plt.subplots(figsize=(7.2,4.6))
ax.loglog(zs, kthr, "o-", color="#1f4fd8", lw=2, ms=5,
          label=r"$\kappa_{\rm thr}$ from $\langle N\rangle=100$ rule (real code)")
ax.axhline(1e-3, ls="--", color="#d62728", lw=1.8,
           label=r"flat default $\kappa_{\rm thr}=10^{-3}$")
# annotate anchors + the implied <N> under the flat rule
for x,y,t in [(1.0,1.278e-4,r"$z_s{=}1:\ 1.28\times10^{-4}$"),
              (10.0,1.373e-3,r"$z_s{=}10:\ 1.37\times10^{-3}$")]:
    ax.annotate(t, (x,y), textcoords="offset points", xytext=(8,-14), fontsize=9)
ax.scatter([1,10],[1.278e-4,1.373e-3], s=70, facecolors="none", edgecolors="k", zorder=5)
ax.set_xlabel(r"source redshift  $z_s$")
ax.set_ylabel(r"explicit-halo threshold  $\kappa_{\rm thr}$")
ax.set_title(r"$\kappa_{\rm thr}(z_s)$ that keeps $\langle N\rangle=100$ strong halos "
             "(vs the flat 1e-3 default)")
ax.grid(True, which="both", alpha=0.25); ax.legend(frameon=False, loc="upper left")
# second axis note: <N> under flat rule
ax.text(0.13, 1.15e-3, r"under flat $10^{-3}$:  $\langle N\rangle\!\approx$0.4 / 11 / 146"
        "\n"r"at $z_s$=0.2 / 1 / 10", fontsize=8.5, color="#d62728", va="top")
plt.tight_layout(); plt.savefig("plots/kappathr_vs_zs.png", dpi=140)
print("saved plots/kappathr_vs_zs.png")
print(f"scaling: kthr(10)/kthr(0.1) = {kthr[-1]/kthr[0]:.0f}x over z_s 0.1->10")
