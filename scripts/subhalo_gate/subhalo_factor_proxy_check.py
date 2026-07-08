"""
Quantify the mismatch between the production r-based subhalo gate and the actual
ray-to-clump distance d that sets a clump's lensing importance.

For a fixed host mass, lens redshift, source redshift, and ray-host distance r_ray,
we draw brute-force subhalo realizations above m_floor using the same SHMF and
anti-biased radial profile as the production code. For each drawn clump we compare:

1. the production proxy: resolve if reach(m) >= r_ray
2. the geometric truth for that realized clump: resolve if reach(m) >= d_ray_clump

The key diagnostic is how much kappa^2 at the ray sits in clumps that the proxy
would miss but the d-based gate would keep. Large values mean a small
subhalo_factor is compensating for proxy bias rather than physics.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import gamma as Gamma
from scipy.special import gammaincc


ROOT = Path("/Users/baltabay/Desktop/gw-wl-emulator")

# Code-matched cosmology/constants copied from scripts/subhalo_resolved_demo.py.
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
PSI_MAX = 1.0


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
    r_lo, r_hi = 1.0e-6, 1.0e6
    if kappa_nfw(mass, zl, zs, r_lo) <= kappa_thr:
        return 0.0
    # if even the far bracket edge is still above threshold (only happens for the
    # SHMF-suppressed near-host-mass tail where the clump weight ~exp(-50)~0), the
    # reach exceeds the bracket; return the edge rather than failing brentq.
    if kappa_nfw(mass, zl, zs, r_hi) >= kappa_thr:
        return float(r_hi)
    root = brentq(
        lambda log_r: kappa_nfw(mass, zl, zs, np.exp(log_r)) - kappa_thr,
        np.log(r_lo),
        np.log(r_hi),
    )
    return float(np.exp(root))


def build_reach_interpolator(
    m_min: float,
    m_max: float,
    zl: float,
    zs: float,
    kappa_thr: float,
    n_grid: int = 320,
):
    masses = np.logspace(np.log10(m_min), np.log10(m_max), n_grid)
    reach = np.array([reach_radius(m, zl, zs, kappa_thr) for m in masses], dtype=float)
    log_m = np.log(masses)

    def interp(query_mass: np.ndarray) -> np.ndarray:
        q = np.asarray(query_mass, dtype=float)
        return np.interp(np.log(q), log_m, reach, left=reach[0], right=reach[-1])

    return interp


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


def sample_brute_realization(
    seed: int,
    host_mass: float,
    z_lens: float,
    m_floor: float,
    gamma: float,
    r200: float,
    c_host: float,
) -> dict[str, np.ndarray | int | float]:
    rng = np.random.default_rng(seed)
    psi_lo = m_floor / host_mass
    if psi_lo >= PSI_MAX:
        return {"seed": seed, "mass": np.array([]), "x": np.array([]), "y": np.array([]), "n_mean": 0.0}

    n_mean = gamma / ALPHA * (PSI_MAX**ALPHA - psi_lo**ALPHA)
    count = int(rng.poisson(n_mean))
    if count == 0:
        return {"seed": seed, "mass": np.array([]), "x": np.array([]), "y": np.array([]), "n_mean": n_mean}

    pa_lo = psi_lo**ALPHA
    pa_hi = PSI_MAX**ALPHA
    accepted_mass = []
    while len(accepted_mass) < count:
        u = rng.uniform(0.0, 1.0, count - len(accepted_mass))
        psi = (pa_lo + u * (pa_hi - pa_lo)) ** (1.0 / ALPHA)
        keep = rng.uniform(0.0, 1.0, len(psi)) <= np.exp(-BETA * psi**OMEGA)
        accepted_mass.extend((psi[keep] * host_mass).tolist())
    mass = np.asarray(accepted_mass[:count], dtype=float)

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


def summarize(
    host_mass: float,
    z_lens: float,
    z_source: float,
    r_ray: float,
    subhalo_factor: float,
    kappa_thr_host: float,
    m_floor: float,
    seeds: list[int],
) -> tuple[dict[str, float], list[dict[str, float]]]:
    rs_host, rhos_host, c_host, r200_host = nfw_params(host_mass, z_lens)
    nt, zf = n_tau(host_mass, z_lens)
    fs = 0.3563 / nt**0.6 - 0.075
    gamma = gamma_norm(fs)
    kappa_thr_sub = subhalo_factor * kappa_thr_host
    reach_interp = build_reach_interpolator(
        m_min=m_floor,
        m_max=PSI_MAX * host_mass,
        zl=z_lens,
        zs=z_source,
        kappa_thr=kappa_thr_sub,
    )

    seed_rows: list[dict[str, float]] = []
    for seed in seeds:
        real = sample_brute_realization(seed, host_mass, z_lens, m_floor, gamma, r200_host, c_host)
        mass = np.asarray(real["mass"], dtype=float)
        if len(mass) == 0:
            seed_rows.append({"seed": float(seed), "n_total": 0.0})
            continue

        x = np.asarray(real["x"], dtype=float)
        y = np.asarray(real["y"], dtype=float)
        d = np.sqrt((x - r_ray) ** 2 + y**2)
        reach = reach_interp(mass)
        kappa = np.array([kappa_nfw(m, z_lens, z_source, max(di, 1.0e-9)) for m, di in zip(mass, d)], dtype=float)
        k2 = kappa**2

        proxy_keep = reach >= r_ray
        truth_keep = reach >= d
        missed = (~proxy_keep) & truth_keep
        overkept = proxy_keep & (~truth_keep)

        row = {
            "seed": float(seed),
            "n_total": float(len(mass)),
            "n_proxy": float(np.count_nonzero(proxy_keep)),
            "n_truth": float(np.count_nonzero(truth_keep)),
            "n_missed": float(np.count_nonzero(missed)),
            "n_overkept": float(np.count_nonzero(overkept)),
            "k2_total": float(k2.sum()),
            "k2_proxy": float(k2[proxy_keep].sum()),
            "k2_truth": float(k2[truth_keep].sum()),
            "k2_missed": float(k2[missed].sum()),
            "k2_overkept": float(k2[overkept].sum()),
            "mass_total": float(mass.sum()),
            "mass_missed": float(mass[missed].sum()),
            "r200": float(r200_host),
            "rs": float(rs_host),
            "c_host": float(c_host),
            "gamma": float(gamma),
            "fs": float(fs),
            "zf": float(zf),
        }
        seed_rows.append(row)

    valid = [row for row in seed_rows if row.get("n_total", 0.0) > 0.0]
    summary: dict[str, float] = {
        "host_mass": host_mass,
        "z_lens": z_lens,
        "z_source": z_source,
        "r_ray": r_ray,
        "subhalo_factor": subhalo_factor,
        "kappa_thr_host": kappa_thr_host,
        "kappa_thr_sub": kappa_thr_sub,
        "m_floor": m_floor,
        "r200": r200_host,
        "rs": rs_host,
        "c_host": c_host,
        "gamma": gamma,
        "fs": fs,
        "zf": zf,
    }
    if valid:
        for key in ("n_total", "n_proxy", "n_truth", "n_missed", "n_overkept",
                    "k2_total", "k2_proxy", "k2_truth", "k2_missed", "k2_overkept",
                    "mass_total", "mass_missed"):
            summary[f"{key}_mean"] = float(np.mean([row[key] for row in valid]))
        summary["missed_count_frac"] = summary["n_missed_mean"] / summary["n_total_mean"]
        summary["missed_mass_frac"] = summary["mass_missed_mean"] / summary["mass_total_mean"]
        summary["missed_k2_frac_total"] = summary["k2_missed_mean"] / summary["k2_total_mean"]
        summary["missed_k2_frac_truth"] = summary["k2_missed_mean"] / max(summary["k2_truth_mean"], 1.0e-300)
        summary["proxy_to_truth_k2_ratio"] = summary["k2_proxy_mean"] / max(summary["k2_truth_mean"], 1.0e-300)
    return summary, seed_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-mass", type=float, default=1.0e13)
    parser.add_argument("--z-lens", type=float, default=0.5)
    parser.add_argument("--z-source", type=float, default=1.0)
    parser.add_argument("--r-ray", type=float, default=1000.0)
    parser.add_argument("--subhalo-factor", type=float, default=1.0e-5)
    parser.add_argument("--kappa-thr-host", type=float, default=1.276e-4)
    parser.add_argument("--m-floor", type=float, default=1.0e7)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, rows = summarize(
        host_mass=args.host_mass,
        z_lens=args.z_lens,
        z_source=args.z_source,
        r_ray=args.r_ray,
        subhalo_factor=args.subhalo_factor,
        kappa_thr_host=args.kappa_thr_host,
        m_floor=args.m_floor,
        seeds=args.seeds,
    )

    print(
        f"host M={summary['host_mass']:.3e} Msun  z_l={summary['z_lens']:g}  "
        f"z_s={summary['z_source']:g}  r_ray={summary['r_ray']:.1f} kpc"
    )
    print(
        f"factor={summary['subhalo_factor']:.3e}  "
        f"kappa_thr_sub={summary['kappa_thr_sub']:.3e}  "
        f"m_floor={summary['m_floor']:.3e}"
    )
    print(
        f"host geometry: r200={summary['r200']:.1f} kpc  rs={summary['rs']:.1f} kpc  "
        f"c={summary['c_host']:.2f}  fs={summary['fs']:.3f}  zf={summary['zf']:.2f}"
    )
    if "n_total_mean" not in summary:
        print("No brute-force clumps were drawn for these parameters.")
        return

    print("\nmean over seeds:")
    print(
        f"  clumps: total={summary['n_total_mean']:.1f}  "
        f"proxy_keep={summary['n_proxy_mean']:.1f}  truth_keep={summary['n_truth_mean']:.1f}"
    )
    print(
        f"  proxy misses {summary['n_missed_mean']:.1f} clumps "
        f"({100.0 * summary['missed_count_frac']:.2f}% of brute sample)"
    )
    print(
        f"  missed mass fraction = {100.0 * summary['missed_mass_frac']:.2f}%"
    )
    print(
        f"  missed kappa^2 fraction of all brute clumps = {100.0 * summary['missed_k2_frac_total']:.2f}%"
    )
    print(
        f"  missed kappa^2 fraction of truth-kept clumps = {100.0 * summary['missed_k2_frac_truth']:.2f}%"
    )
    print(
        f"  proxy/truth retained kappa^2 = {summary['proxy_to_truth_k2_ratio']:.3f}"
    )

    print("\nper-seed:")
    for row in rows:
        if row.get("n_total", 0.0) == 0.0:
            print(f"  seed {int(row['seed'])}: no clumps")
            continue
        frac = 100.0 * row["k2_missed"] / max(row["k2_truth"], 1.0e-300)
        print(
            f"  seed {int(row['seed'])}: n_total={int(row['n_total'])} "
            f"n_missed={int(row['n_missed'])}  missed kappa^2 / truth = {frac:.2f}%"
        )


if __name__ == "__main__":
    main()
