"""
Visualize only the subhalos that the production dynamic split would instantiate.

This is a companion to scripts/subhalo_demo.py.  The older plot shows the full
Jiang-van den Bosch population above a fixed SHMF resolution.  This one mirrors the
production gate:

    kappa_thr,sub = subhalo_factor * kappa_thr,host
    psi_lo = max(m_floor, m_reach(host_distance)) / M_host

and then draws only clumps with psi_lo <= m/M <= 0.1, using the same power-law sampling
used in cpp/subhalo.cpp over the resolved interval.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import gamma as Gamma
from scipy.special import gammaincc


ROOT = Path("/Users/baltabay/Desktop/gw-wl-emulator")

# Code-matched cosmology constants.
OM = 0.315
SIGMA8 = 0.811
H = 0.674
NS = 0.965
OB = 0.0493
ZEQ = 3402.0
OMR = OM / (1.0 + ZEQ)
OML = 1.0 - OM - OMR
H0 = 0.000102247 * H
CH = 306.535
RHO_C0 = 277.394 * H**2
RHO_M0 = OM * RHO_C0
H0_INV_GYR = 9.778 / H

ALPHA = -0.82
BETA = 50.0
OMEGA = 4.0
PSI_RES = 1.0e-4
PSI_MAX = 0.1


def az(z: float) -> float:
    return OM * (1.0 + z) ** 3 + OMR * (1.0 + z) ** 4 + OML


def ez(z: float) -> float:
    return np.sqrt(az(z))


def omega_m_z(z: float) -> float:
    return OM * (1.0 + z) ** 3 / az(z)


def dvir(z: float) -> float:
    d = omega_m_z(z) - 1.0
    return 18.0 * np.pi**2 + 82.0 * d - 39.0 * d**2


def dc(z: float) -> float:
    return quad(lambda zp: CH / (H0 * ez(zp)), 0.0, z)[0]


def dl(z: float) -> float:
    return (1.0 + z) * dc(z)


def sigma_crit(zs: float, zl: float) -> float:
    ds_a = dl(zs) / (1.0 + zs) ** 2
    dl_a = dl(zl) / (1.0 + zl) ** 2
    dls_a = ds_a - dl_a * (1.0 + zl) / (1.0 + zs)
    return 2.08871e16 * ds_a / (4.0 * np.pi * dl_a * dls_a)


def transfer_eh98(k: float) -> float:
    theta = 2.728 / 2.7
    omh2 = OM * H * H
    obh2 = OB * H * H
    sound = 44.5 * np.log(9.83 / omh2) / np.sqrt(1.0 + 10.0 * obh2**0.75)
    alpha_g = 1.0 - 0.328 * np.log(431.0 * omh2) * OB / OM
    alpha_g += 0.38 * np.log(22.3 * omh2) * (OB / OM) ** 2
    gamma_eff = OM * H * (alpha_g + (1.0 - alpha_g) / (1.0 + (0.43 * k * sound * H) ** 4))
    q = k / H * theta**2 / gamma_eff
    l0 = np.log(2.0 * np.e + 1.8 * q)
    c0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
    return l0 / (l0 + c0 * q * q)


def sigma2_r(radius_mpc_over_h: float) -> float:
    radius_mpc = radius_mpc_over_h / H

    def integrand(log_k: float) -> float:
        k = np.exp(log_k)
        x = k * radius_mpc
        window = 3.0 * (np.sin(x) - x * np.cos(x)) / x**3
        pk = k**NS * transfer_eh98(k) ** 2
        return k**3 * pk / (2.0 * np.pi**2) * window**2

    return quad(integrand, np.log(1.0e-4), np.log(1.0e3), limit=200)[0]


SIGMA_NORM = SIGMA8**2 / sigma2_r(8.0)


def sigma_m(mass: float) -> float:
    radius_kpc = (3.0 * mass / (4.0 * np.pi * RHO_M0)) ** (1.0 / 3.0)
    return np.sqrt(SIGMA_NORM * sigma2_r(radius_kpc / 1000.0 * H))


def growth(z: float) -> float:
    numerator = ez(z) * quad(lambda zp: (1.0 + zp) / ez(zp) ** 3, z, np.inf)[0]
    denominator = ez(0.0) * quad(lambda zp: (1.0 + zp) / ez(zp) ** 3, 0, np.inf)[0]
    return numerator / denominator


def z_form(mass: float, z0: float, frac: float = 0.5) -> float:
    dc0 = 1.686 / growth(z0)
    af = 0.815 * np.exp(-2.0 * frac**3) / frac**0.707
    wf = np.sqrt(2.0 * np.log(af + 1.0))
    rhs = dc0 + wf * np.sqrt(sigma_m(frac * mass) ** 2 - sigma_m(mass) ** 2)
    return brentq(lambda zf: 1.686 / growth(zf) - rhs, z0, 30.0)


def n_tau(mass: float, z0: float) -> tuple[float, float]:
    zf = z_form(mass, z0)
    val = quad(
        lambda z: H0_INV_GYR
        / ((1.0 + z) * ez(z))
        / (1.628 / H * (dvir(z) / 178.0) ** (-0.5) * ez(z) ** (-1.0)),
        z0,
        zf,
    )[0]
    return val, zf


def gamma_norm(fs: float) -> float:
    s = (1.0 + ALPHA) / OMEGA
    denom = Gamma(s) * (gammaincc(s, BETA * PSI_RES**OMEGA) - gammaincc(s, BETA))
    return OMEGA * BETA**s / denom * fs


def dndlnpsi(psi: np.ndarray | float, gamma: float) -> np.ndarray | float:
    return gamma * np.asarray(psi) ** ALPHA * np.exp(-BETA * np.asarray(psi) ** OMEGA)


def conc(mass: float, z: float) -> float:
    a = 0.520 + (0.905 - 0.520) * np.exp(-0.617 * z**1.21)
    b = -0.101 + 0.026 * z
    return 10.0 ** (a + b * np.log10(mass / (1.0e12 / H)))


def nfw_params(mass: float, z: float) -> tuple[float, float, float, float]:
    c = conc(mass, z)
    r200 = (3.0 * mass / (4.0 * np.pi * 200.0 * az(z) * RHO_C0)) ** (1.0 / 3.0)
    rs = r200 / c
    h_c = np.log(1.0 + c) - c / (1.0 + c)
    rhos = 200.0 * az(z) * RHO_C0 * c**3 / (3.0 * h_c)
    return rs, rhos, c, r200


def fg_kappa(x: np.ndarray | float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    lo = x < 1.0 - 1.0e-7
    hi = x > 1.0 + 1.0e-7
    eq = ~(lo | hi)
    xl = x[lo]
    xh = x[hi]
    tl = np.arctanh(np.sqrt((1.0 - xl) / (1.0 + xl))) / np.sqrt(1.0 - xl**2)
    th = np.arctan(np.sqrt((xh - 1.0) / (1.0 + xh))) / np.sqrt(xh**2 - 1.0)
    out[lo] = (1.0 - 2.0 * tl) / (xl**2 - 1.0)
    out[hi] = (1.0 - 2.0 * th) / (xh**2 - 1.0)
    out[eq] = 1.0 / 3.0
    return out


def kappa_nfw(mass: float, zl: float, zs: float, radius: float) -> float:
    rs, rhos, _, _ = nfw_params(mass, zl)
    kappa0 = rs * rhos / sigma_crit(zs, zl)
    return float(2.0 * kappa0 * fg_kappa(np.array([max(radius / rs, 1.0e-12)]))[0])


def reach_radius(mass: float, zl: float, zs: float, kappa_thr: float) -> float:
    if kappa_nfw(mass, zl, zs, 1.0e-6) <= kappa_thr:
        return 0.0
    root = brentq(
        lambda log_r: kappa_nfw(mass, zl, zs, np.exp(log_r)) - kappa_thr,
        np.log(1.0e-6),
        np.log(1.0e6),
    )
    return float(np.exp(root))


def dynamic_mass_floor(
    host_distance: float,
    zl: float,
    zs: float,
    kappa_thr_sub: float,
    m_floor: float,
    m_max: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    masses = np.logspace(np.log10(m_floor), np.log10(m_max), 240)
    reaches = np.array([reach_radius(m, zl, zs, kappa_thr_sub) for m in masses])
    idx = int(np.searchsorted(reaches, host_distance, side="left"))
    if idx >= len(masses):
        return np.inf, masses, reaches
    return float(max(m_floor, masses[idx])), masses, reaches


def sample_biased_radii(count: int, r200: float, c: float, rng: np.random.Generator) -> np.ndarray:
    if count == 0:
        return np.array([])
    x_grid = np.linspace(0.0, 1.0, 5000)
    x = np.maximum(x_grid, 1.0e-9)
    bias = 1.0 / np.sqrt((x / 0.54) ** (-2.5) + 1.0)
    weight = x**2 / (1.0 + c * x) ** 2 * bias
    cdf = np.cumsum(0.5 * (weight[1:] + weight[:-1]) * np.diff(x_grid))
    cdf = np.concatenate([[0.0], cdf])
    cdf /= cdf[-1]
    return np.interp(rng.uniform(0.0, 1.0, count), cdf, x_grid) * r200


def sample_realization(
    seed: int,
    host_mass: float,
    z_lens: float,
    psi_lo: float,
    gamma: float,
    r200: float,
    c_host: float,
) -> dict:
    rng = np.random.default_rng(seed)
    if psi_lo >= PSI_MAX:
        n_mean = 0.0
    else:
        n_mean = gamma / ALPHA * (PSI_MAX**ALPHA - psi_lo**ALPHA)
    count = int(rng.poisson(n_mean))
    if count == 0:
        return {"seed": seed, "mass": np.array([]), "x": np.array([]), "y": np.array([]), "n_mean": n_mean}

    pa_lo = psi_lo**ALPHA
    pa_hi = PSI_MAX**ALPHA
    psi = (pa_lo + rng.uniform(0.0, 1.0, count) * (pa_hi - pa_lo)) ** (1.0 / ALPHA)
    mass = psi * host_mass
    r3d = sample_biased_radii(count, r200, c_host, rng)
    costh = rng.uniform(-1.0, 1.0, count)
    phi = rng.uniform(0.0, 2.0 * np.pi, count)
    r2d = r3d * np.sqrt(1.0 - costh**2)
    return {
        "seed": seed,
        "mass": mass,
        "x": r2d * np.cos(phi),
        "y": r2d * np.sin(phi),
        "n_mean": n_mean,
    }


def nfw_sigma_shape(x: np.ndarray) -> np.ndarray:
    x = np.maximum(x, 1.0e-5)
    out = np.empty_like(x)
    lo = x < 1.0 - 1.0e-6
    hi = x > 1.0 + 1.0e-6
    eq = ~(lo | hi)
    xl = x[lo]
    xh = x[hi]
    out[lo] = (1.0 - 2.0 / np.sqrt(1.0 - xl**2) * np.arctanh(np.sqrt((1.0 - xl) / (1.0 + xl)))) / (
        xl**2 - 1.0
    )
    out[hi] = (1.0 - 2.0 / np.sqrt(xh**2 - 1.0) * np.arctan(np.sqrt((xh - 1.0) / (xh + 1.0)))) / (
        xh**2 - 1.0
    )
    out[eq] = 1.0 / 3.0
    return out


def projected_nfw_sigma(radius: np.ndarray, rs: float, rhos: float) -> np.ndarray:
    return 2.0 * rhos * rs * nfw_sigma_shape(radius / rs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-mass", type=float, default=1.0e14)
    parser.add_argument("--z-lens", type=float, default=0.5)
    parser.add_argument("--z-source", type=float, default=1.0)
    parser.add_argument("--host-distance", type=float, default=1000.0)
    parser.add_argument("--subhalo-factor", type=float, default=1.0e-4)
    parser.add_argument("--kappa-thr-host", type=float, default=1.276e-4)
    parser.add_argument("--m-floor", type=float, default=1.0e7)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--out", type=Path, default=ROOT / "plots" / "subhalo_resolved_demo.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rs_host, rhos_host, c_host, r200_host = nfw_params(args.host_mass, args.z_lens)
    nt, zf = n_tau(args.host_mass, args.z_lens)
    fs = 0.3563 / nt**0.6 - 0.075
    gamma = gamma_norm(fs)

    m_gate, reach_masses, reaches = dynamic_mass_floor(
        args.host_distance,
        args.z_lens,
        args.z_source,
        args.subhalo_factor * args.kappa_thr_host,
        args.m_floor,
        PSI_MAX * args.host_mass,
    )
    psi_lo = m_gate / args.host_mass
    if not np.isfinite(psi_lo) or psi_lo >= PSI_MAX:
        raise RuntimeError("No resolved subhalos for this host distance and threshold.")
    fs_res = gamma * (PSI_MAX ** (1.0 + ALPHA) - psi_lo ** (1.0 + ALPHA)) / (1.0 + ALPHA)

    reals = [
        sample_realization(seed, args.host_mass, args.z_lens, psi_lo, gamma, r200_host, c_host)
        for seed in args.seeds
    ]

    print(f"host M={args.host_mass:.3e} Msun z_lens={args.z_lens:g} z_source={args.z_source:g}")
    print(f"r200={r200_host:.1f} kpc c={c_host:.2f} rs={rs_host:.1f} kpc")
    print(f"fs={fs:.3f} fs_resolved={fs_res:.3f} Ntau={nt:.2f} zform={zf:.2f} gamma={gamma:.4f}")
    print(f"subhalo_factor={args.subhalo_factor:g}; kappa_thr_sub={args.subhalo_factor * args.kappa_thr_host:.3e}")
    print(f"host distance r={args.host_distance:.1f} kpc -> m_gate={m_gate:.3e} Msun psi_lo={psi_lo:.3e}")
    print("resolved counts:", [len(r["mass"]) for r in reals], f"mean model={reals[0]['n_mean']:.1f}")

    fig = plt.figure(figsize=(14, 12.8))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.0, 1.0, 1.05], hspace=0.42, wspace=0.34)
    extent = 1.35 * max(r200_host, args.host_distance)
    grid = np.linspace(-extent, extent, 360)
    xg, yg = np.meshgrid(grid, grid)
    host_sigma = projected_nfw_sigma(np.sqrt(xg**2 + yg**2), rs_host, rhos_host)

    for i, real in enumerate(reals[:3]):
        ax = fig.add_subplot(gs[0, 2 * i : 2 * i + 2])
        ax.imshow(
            host_sigma,
            extent=[-extent, extent, -extent, extent],
            origin="lower",
            cmap="bone_r",
            norm=LogNorm(vmin=np.percentile(host_sigma, 3), vmax=np.percentile(host_sigma, 99.8)),
        )
        if len(real["mass"]):
            sizes = 6.0 + 42.0 * (real["mass"] / max(real["mass"])) ** (1.0 / 3.0)
            sc = ax.scatter(
                real["x"],
                real["y"],
                s=sizes,
                c=np.log10(real["mass"]),
                cmap="autumn",
                edgecolor="black",
                linewidth=0.25,
                alpha=0.86,
                zorder=3,
            )
        ax.scatter([args.host_distance], [0.0], marker="+", s=150, lw=2.0, color="#00a6d6", zorder=5)
        ax.add_patch(plt.Circle((0.0, 0.0), r200_host, fill=False, ec="#00a6d6", ls="--", lw=1.2))
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_aspect("equal")
        ax.set_title(f"seed {real['seed']}: resolved N={len(real['mass'])}", fontsize=10)
        ax.set_xlabel("host-centered x [kpc]")
        if i == 0:
            ax.set_ylabel("host-centered y [kpc]")

    fig.text(
        0.5,
        0.965,
        "Resolved production subhalos only: "
        f"M={args.host_mass:.1e} Msun, z_l={args.z_lens:g}, z_s={args.z_source:g}, "
        f"r_ray={args.host_distance:.0f} kpc, factor={args.subhalo_factor:g}, "
        f"m_gate={m_gate:.2e} Msun",
        ha="center",
        fontsize=13,
    )

    ax = fig.add_subplot(gs[1, :])
    logpsi = np.linspace(np.log10(args.m_floor / args.host_mass), np.log10(PSI_MAX), 420)
    psi = 10.0**logpsi
    y = np.log10(np.log(10.0) * dndlnpsi(psi, gamma))
    ax.plot(logpsi, y, color="0.25", lw=2.0, label="JvdB14 SHMF")
    ax.axvspan(logpsi.min(), np.log10(psi_lo), color="0.88", label="not instantiated by dynamic gate")
    ax.axvline(np.log10(psi_lo), color="#dc2626", lw=2.0, label=fr"dynamic floor $\psi_{{lo}}={psi_lo:.1e}$")
    ax.axvline(np.log10(PSI_RES), color="0.45", ls="--", lw=1.4, label=fr"JvdB $\psi_{{res}}={PSI_RES:g}$")
    bins = np.linspace(np.log10(psi_lo), np.log10(PSI_MAX), 24)
    widths = np.diff(bins)
    centers = 0.5 * (bins[:-1] + bins[1:])
    all_psi = np.concatenate([r["mass"] / args.host_mass for r in reals if len(r["mass"])])
    counts, _ = np.histogram(np.log10(all_psi), bins=bins)
    ok = counts > 0
    ax.errorbar(
        centers[ok],
        np.log10((counts[ok] / len(reals)) / widths[ok]),
        yerr=0.434 / np.sqrt(counts[ok]),
        fmt="s",
        ms=4,
        capsize=2,
        color="#2563eb",
        label=f"{len(reals)}-seed mean of instantiated clumps",
    )
    ax.set_xlabel(r"$\log_{10}(m/M)$")
    ax.set_ylabel(r"$\log_{10}[dN/d\log_{10}(m/M)]$")
    ax.set_title("Mass function with the production dynamic floor")
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)

    real0 = reals[0]
    ax = fig.add_subplot(gs[2, :3])
    ray_distance = np.sqrt((real0["x"] - args.host_distance) ** 2 + real0["y"] ** 2)
    if len(real0["mass"]):
        sc = ax.scatter(
            ray_distance,
            real0["mass"],
            c=np.log10(real0["mass"]),
            s=10.0 + 50.0 * (real0["mass"] / max(real0["mass"])) ** (1.0 / 3.0),
            cmap="autumn",
            edgecolor="black",
            linewidth=0.2,
            alpha=0.85,
        )
    ax.axvline(args.host_distance, color="0.4", ls="--", lw=1.2, label="host-center distance")
    ax.set_yscale("log")
    ax.set_xlabel("actual ray-to-clump distance d [kpc]")
    ax.set_ylabel("resolved clump mass [Msun]")
    ax.set_title(f"Seed {real0['seed']}: created clumps by actual distance to ray")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)

    ax = fig.add_subplot(gs[2, 3:])
    if len(real0["mass"]):
        kappa = np.array(
            [kappa_nfw(m, args.z_lens, args.z_source, max(d, 1.0e-9)) for m, d in zip(real0["mass"], ray_distance)]
        )
        ax.hist(np.log10(np.maximum(kappa, 1.0e-30)), bins=34, color="#f97316", alpha=0.82)
        ax.axvline(np.log10(args.subhalo_factor * args.kappa_thr_host), color="#dc2626", lw=2.0)
    ax.set_xlabel(r"$\log_{10}\kappa_c(d)$ at the ray")
    ax.set_ylabel("number of instantiated clumps")
    ax.set_title("Their actual convergence contributions after placement")
    ax.grid(alpha=0.3)

    cax = fig.add_axes([0.91, 0.73, 0.012, 0.16])
    fig.colorbar(sc, cax=cax, label=r"$\log_{10} m$")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
