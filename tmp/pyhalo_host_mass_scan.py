"""
Host-mass scan for pyHalo bound remnants vs this work's evolved SHMF.

The x-axis is ln(m/M_host). The y-axis is dN/dln(m/M_host) inside the same
pyHalo strong-lensing aperture, not divided by area.

Run:
  .venv_pyhalo/bin/python tmp/pyhalo_host_mass_scan.py --nreal 100
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import gamma as Gamma
from scipy.special import gammaincc
from pyHalo.preset_models import preset_model_from_name


parser = argparse.ArgumentParser()
parser.add_argument("--mhosts", default="1e12,3e12,1e13,3e13,1e14")
parser.add_argument("--zl", type=float, default=0.5)
parser.add_argument("--zs", type=float, default=2.0)
parser.add_argument("--mlo", type=float, default=1e7)
parser.add_argument("--nreal", type=int, default=100)
parser.add_argument("--suffix", default="")
args = parser.parse_args()

MHOSTS = [float(x) for x in args.mhosts.split(",")]
ZL, ZS, MLO, NREAL = args.zl, args.zs, args.mlo, args.nreal
suffix = f"_{args.suffix}" if args.suffix else f"_n{NREAL}"
CDM = preset_model_from_name("CDM")

ALPHA, BETA, OMEGA = -0.82, 50.0, 4.0
PSI_RES = 1e-4

# Cosmology/JvdB helpers copied from tmp/psub_check.py, matching the current pipeline.
Om, sigma8, h, ns, Ob = 0.315, 0.811, 0.674, 0.965, 0.0493
zeq = 3402.0
OmR = Om / (1 + zeq)
OmL = 1 - Om - OmR
rho_c0 = 277.394 * h**2
rho_m0 = Om * rho_c0


def Az(z):
    return Om * (1 + z) ** 3 + OmR * (1 + z) ** 4 + OmL


def OmegaMz(z):
    return Om * (1 + z) ** 3 / Az(z)


def OmegaLz(z):
    return OmL / Az(z)


def Dg(z):
    omz, olz = OmegaMz(z), OmegaLz(z)
    return 2.5 * omz / (omz ** (4 / 7) - olz + (1 + omz / 2) * (1 + olz / 70)) / (1 + z) / 0.7869370293916


def deltac(z):
    return (3 / 5) * (3 * np.pi / 2) ** (2 / 3) / Dg(z)


def T_EH98(k):
    th = 2.728 / 2.7
    Omh2, Obh2 = Om * h * h, Ob * h * h
    sound = 44.5 * np.log(9.83 / Omh2) / np.sqrt(1 + 10 * Obh2**0.75)
    alpha_g = 1 - 0.328 * np.log(431 * Omh2) * Ob / Om + 0.38 * np.log(22.3 * Omh2) * (Ob / Om) ** 2
    gamma_eff = Om * h * (alpha_g + (1 - alpha_g) / (1 + (0.43 * k * sound * h) ** 4))
    q = k / h * th**2 / gamma_eff
    L0 = np.log(2 * np.e + 1.8 * q)
    C0 = 14.2 + 731 / (1 + 62.5 * q)
    return L0 / (L0 + C0 * q * q)


def sigma2_r(radius_mpc_over_h):
    radius = radius_mpc_over_h / h

    def integrand(logk):
        k = np.exp(logk)
        window = 3 * (np.sin(k * radius) - k * radius * np.cos(k * radius)) / (k * radius) ** 3
        return k**3 * (k**ns * T_EH98(k) ** 2) * window**2 / (2 * np.pi**2)

    return quad(integrand, np.log(1e-4), np.log(1e3), limit=200)[0]


SIGMA_NORM = sigma8**2 / sigma2_r(8.0)
LOG_M_GRID = np.linspace(np.log(1e5), np.log(1e17), 200)
SIGMA_GRID = np.array([
    np.sqrt(SIGMA_NORM * sigma2_r((3 * np.exp(logm) / (4 * np.pi * rho_m0)) ** (1 / 3) / 1000 * h))
    for logm in LOG_M_GRID
])


def sigma_m(mass):
    return np.interp(np.log(mass), LOG_M_GRID, SIGMA_GRID)


def z_form(mass, z0, frac=0.5):
    alpha_f = 0.815 * np.exp(-2 * frac**3) / frac**0.707
    w_f = np.sqrt(2 * np.log(alpha_f + 1))
    rhs = deltac(z0) + w_f * np.sqrt(sigma_m(frac * mass) ** 2 - sigma_m(mass) ** 2)
    return brentq(lambda zf: deltac(zf) - rhs, z0, 30.0)


def n_tau(mass, z0):
    zf = z_form(mass, z0)

    def integrand(z):
        delta = OmegaMz(z) - 1
        dvir = 18 * np.pi**2 + 82 * delta - 39 * delta**2
        return 6.006 * np.sqrt(dvir / 178) / (1 + z)

    return quad(integrand, z0, zf)[0]


def gamma_norm(fs):
    s = (1 + ALPHA) / OMEGA
    denom = Gamma(s) * (gammaincc(s, BETA * PSI_RES**OMEGA) - gammaincc(s, BETA))
    return OMEGA * BETA**s / denom * fs


def gnorm_of(mass, z0):
    nt = n_tau(mass, z0)
    fs = 0.3563 / nt**0.6 - 0.075
    return gamma_norm(fs), fs, nt, z_form(mass, z0)


def cons14(mass, z):
    b = -0.101 + 0.026 * z
    a = 0.520 + (0.905 - 0.520) * np.exp(-0.617 * z**1.21)
    return 10 ** (a + b * np.log10(mass / (1e12 / h)))


def r200_and_c(mass, z):
    rhoz = Az(z) * rho_c0
    c = cons14(mass, z)
    r200 = (3 * mass / (4 * np.pi * 200 * rhoz)) ** (1 / 3)
    return r200, c


def plot_positive(ax, x, y, *args, **kwargs):
    y = np.asarray(y)
    ok = np.isfinite(y) & (y > 0)
    return ax.plot(np.asarray(x)[ok], y[ok], *args, **kwargs)


def sig_n_factory(c_h, r200):
    xs3 = np.linspace(1e-4, 1, 3000)
    nu3 = (1 / np.sqrt((xs3 / 0.54) ** (-2.5) + 1)) / (1 + c_h * xs3) ** 2

    def sig_n(Rk):
        zmax = np.sqrt(max(r200**2 - Rk**2, 0))
        if zmax <= 0:
            return 0.0
        zz = np.linspace(0, zmax, 1500)
        return 2 * np.trapezoid(np.interp(np.sqrt(Rk**2 + zz**2) / r200, xs3, nu3), zz)

    return sig_n


def aperture_fraction(R_ap, c_h, r200):
    sig_n = sig_n_factory(c_h, r200)
    r_split = min(40.0, 0.2 * r200)
    rg = np.concatenate([np.linspace(0.2, r_split, 250), np.linspace(r_split + 0.5, r200, 350)])
    sn = np.array([sig_n(r) for r in rg])
    total = np.trapezoid(sn * 2 * np.pi * rg, rg)
    selected = rg < R_ap
    return np.trapezoid(sn[selected] * 2 * np.pi * rg[selected], rg[selected]) / total


def run_host(MHOST):
    m_inf, m_bnd = [], []
    kpc_per_asec = None
    mhi = MHOST
    for _ in range(NREAL):
        real = CDM(
            z_lens=ZL,
            z_source=ZS,
            log_m_host=np.log10(MHOST),
            log_mlow=np.log10(MLO),
            log_mhigh=np.log10(mhi),
            LOS_normalization=0.0,
            cone_opening_angle_arcsec=12.0,
            two_halo_contribution=False,
        )
        for h in real.halos:
            m_inf.append(h.mass)
            try:
                m_bnd.append(h.bound_mass)
            except Exception:
                m_bnd.append(np.nan)
            if kpc_per_asec is None:
                kpc_per_asec = h.lens_cosmo.cosmo.kpc_proper_per_asec(ZL)

    m_inf = np.asarray(m_inf)
    m_bnd = np.asarray(m_bnd)
    r_ap = 6.0 * kpc_per_asec
    a_ap = np.pi * r_ap**2

    psi_lo = MLO / MHOST
    bins_psi = np.logspace(np.log10(psi_lo), 0.0, 20)
    lnw = np.diff(np.log(bins_psi))
    ctr_psi = np.sqrt(bins_psi[1:] * bins_psi[:-1])
    bins_m = bins_psi * MHOST

    h_inf = np.histogram(m_inf, bins=bins_m)[0] / NREAL / lnw
    mb = m_bnd[np.isfinite(m_bnd) & (m_bnd > 0)]
    h_bnd = np.histogram(mb, bins=bins_m)[0] / NREAL / lnw

    r200, c_h = r200_and_c(MHOST, ZL)
    f_ap = aperture_fraction(r_ap, c_h, r200)
    gamma, fs, nt, zf = gnorm_of(MHOST, ZL)
    ours = gamma * ctr_psi**ALPHA * np.exp(-BETA * ctr_psi**OMEGA) * f_ap

    return {
        "MHOST": MHOST,
        "mhi": mhi,
        "R_ap": r_ap,
        "A_ap": a_ap,
        "f_ap": f_ap,
        "psi": ctr_psi,
        "lnpsi": np.log(ctr_psi),
        "pyhalo_infall": h_inf,
        "pyhalo_bound": h_bnd,
        "ours": ours,
        "n_per_real": len(m_inf) / NREAL,
        "strip_median": np.nanmedian(m_bnd / m_inf),
        "gamma": gamma,
        "fs": fs,
        "ntau": nt,
        "zform": zf,
        "r200": r200,
        "c_host": c_h,
    }


rows = []
for mhost in MHOSTS:
    print(f"running Mhost={mhost:.2e} with {NREAL} realizations")
    row = run_host(mhost)
    rows.append(row)
    print(
        f"  N/real={row['n_per_real']:.0f}, median bound/infall={row['strip_median']:.4f}, "
        f"gamma={row['gamma']:.4f}, fs={row['fs']:.3f}, f_ap={row['f_ap']:.4f}"
    )

colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(rows)))
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)

for color, row in zip(colors, rows):
    label = rf"$M={row['MHOST']:.0e}M_\odot$"
    plot_positive(axes[0], row["lnpsi"], row["pyhalo_bound"], "s-", color=color, ms=3.8, label=label)
    plot_positive(axes[0], row["lnpsi"], row["ours"], "-", color=color, lw=2)

    ratio = np.divide(row["ours"], row["pyhalo_bound"], out=np.full_like(row["ours"], np.nan), where=row["pyhalo_bound"] > 0)
    plot_positive(axes[1], row["lnpsi"], ratio, "o-", color=color, ms=3.8, label=label)

axes[0].set_yscale("log")
axes[0].set_xlabel(r"$\ln(m/M_{\rm host})$")
axes[0].set_ylabel(r"$dN/d\ln(m/M_{\rm host})$ in aperture")
axes[0].set_title("Bound SHMF: pyHalo markers, this work lines")
axes[0].grid(alpha=0.3, which="both")
axes[0].legend(fontsize=8)

axes[1].axhline(1.0, color="0.35", lw=1)
axes[1].set_yscale("log")
axes[1].set_xlabel(r"$\ln(m/M_{\rm host})$")
axes[1].set_ylabel("ours / pyHalo bound")
axes[1].set_title("Amplitude ratio")
axes[1].grid(alpha=0.3, which="both")
axes[1].legend(fontsize=8)

fig.tight_layout()
png = f"/Users/baltabay/Desktop/gw-wl-emulator/plots/figures/pyhalo_host_mass_scan{suffix}.png"
fig.savefig(png, dpi=200)

npz = f"/Users/baltabay/Desktop/gw-wl-emulator/tmp/pyhalo_host_mass_scan{suffix}.npz"
np.savez(
    npz,
    MHOSTS=np.array([r["MHOST"] for r in rows]),
    R_ap=np.array([r["R_ap"] for r in rows]),
    f_ap=np.array([r["f_ap"] for r in rows]),
    n_per_real=np.array([r["n_per_real"] for r in rows]),
    strip_median=np.array([r["strip_median"] for r in rows]),
    gamma=np.array([r["gamma"] for r in rows]),
    fs=np.array([r["fs"] for r in rows]),
    ntau=np.array([r["ntau"] for r in rows]),
    zform=np.array([r["zform"] for r in rows]),
    r200=np.array([r["r200"] for r in rows]),
    c_host=np.array([r["c_host"] for r in rows]),
    lnpsi=np.array([r["lnpsi"] for r in rows], dtype=object),
    psi=np.array([r["psi"] for r in rows], dtype=object),
    pyhalo_infall=np.array([r["pyhalo_infall"] for r in rows], dtype=object),
    pyhalo_bound=np.array([r["pyhalo_bound"] for r in rows], dtype=object),
    ours=np.array([r["ours"] for r in rows], dtype=object),
)

print(f"saved {png}")
print(f"saved {npz}")
