import json
import math
import numpy as np
import matplotlib.pyplot as plt

from nu_distribution_analysis import (
    PI,
    deltac,
    d_growth,
    tm_cdm,
    sigma_cdm,
    st_f_nu,
    dlnf_dlnnu,
)


def hz(z, omega_m, omega_r, omega_l, h0):
    return h0 * np.sqrt(omega_m * (1 + z) ** 3 + omega_r * (1 + z) ** 4 + omega_l)


def dc_integral(z, omega_m, omega_r, omega_l, h0, n=4000):
    if z <= 0:
        return 0.0
    zz = np.geomspace(1e-6, z, n)
    invh = 1.0 / hz(zz, omega_m, omega_r, omega_l, h0)
    return float(np.trapezoid(306.535 * invh, zz))


def sigma_crit(zs, zl, omega_m, omega_r, omega_l, h0):
    ds_a = dc_integral(zs, omega_m, omega_r, omega_l, h0) / (1 + zs)
    dl_a = dc_integral(zl, omega_m, omega_r, omega_l, h0) / (1 + zl)
    dls_a = ds_a - dl_a * (1 + zl) / (1 + zs)
    return 2.08871e16 * ds_a / (4.0 * PI * dl_a * dls_a)


def concentration_1402(z, m, h):
    a = 0.520 + (0.905 - 0.520) * np.exp(-0.617 * z**1.21)
    b = -0.101 + 0.026 * z
    return 10 ** (a + b * np.log10(m / (1e12 / h)))


def fnfw(x):
    if x > 1:
        return (1 - 2 * np.arctan(np.sqrt((x - 1) / (1 + x))) / np.sqrt(x * x - 1)) / (x * x - 1)
    if x < 1:
        return (1 - 2 * np.arctanh(np.sqrt((1 - x) / (1 + x))) / np.sqrt(1 - x * x)) / (x * x - 1)
    return 1.0 / 3.0


def kappa_nfw(kappa0, x):
    return 2.0 * kappa0 * fnfw(x)


def r_lens_nfw(m, zl, zs, rhoc, h, omega_m, omega_r, omega_l, h0, kappa_thr):
    c = concentration_1402(zl, m, h)
    r200 = (3.0 * m / (4.0 * PI * 200.0 * rhoc)) ** (1.0 / 3.0)
    rs = r200 / c
    rhos = 200.0 * rhoc * c**3 * (1 + c) / (3.0 * ((1 + c) * np.log(1 + c) - c))

    sigmac = sigma_crit(zs, zl, omega_m, omega_r, omega_l, h0)
    k0 = rs * rhos / sigmac

    x1 = 1.0e-6
    x2 = 1.0e6
    if kappa_nfw(k0, x1) <= kappa_thr:
        return 0.0

    for _ in range(80):
        xm = np.exp((np.log(x1) + np.log(x2)) / 2.0)
        if kappa_nfw(k0, xm) > kappa_thr:
            x1 = xm
        else:
            x2 = xm
    return np.exp((np.log(x1) + np.log(x2)) / 2.0) * rs


def build_distributions(z=1.0, zs=2.0, kappa_thr=0.01):
    # Simulator cosmology defaults
    omega_m = 0.315
    omega_b = 0.0493
    sigma8 = 0.811
    h = 0.674
    t0 = 2.7255
    ns = 0.965
    zeq = 3402.0

    omega_r = omega_m / (1.0 + zeq)
    omega_l = 1.0 - omega_m - omega_r
    omega_c = omega_m - omega_b
    h0 = 0.000102247 * h
    rhoc = 277.394 * h**2
    rho_m0 = omega_m * rhoc
    m8 = 4.0 * PI / 3.0 * (8000.0 / h) ** 3.0 * rho_m0

    m_grid = np.logspace(8, 16, 180)
    lnm = np.log(m_grid)
    nk = 1000
    tm_func = lambda k: tm_cdm(k, omega_m, omega_b, omega_c, h, t0, zeq, lambda zz: hz(zz, omega_m, omega_r, omega_l, h0))

    sigma_m8, _ = sigma_cdm(m8, 1.0, nk, rho_m0, h0, ns, tm_func)
    delta_h8 = sigma8 / sigma_m8

    sig = np.zeros_like(m_grid)
    dsig_dm = np.zeros_like(m_grid)
    for i, m in enumerate(m_grid):
        sig[i], dsig_dm[i] = sigma_cdm(m, delta_h8, nk, rho_m0, h0, ns, tm_func)

    dc = deltac(z, omega_m, omega_r, omega_l)
    nu = (dc**2) / (sig**2)
    fnu = st_f_nu(nu)
    dlnf = dlnf_dlnnu(nu)

    s = sig**2
    pfc = fnu / s
    dn_dlnm = -rho_m0 * pfc * 2.0 * sig * dsig_dm

    # Old proxy
    sigma_old = m_grid ** (2.0 / 3.0)
    w_old = dn_dlnm * sigma_old
    pm_old = w_old / np.trapezoid(w_old, lnm)

    # Thresholded NFW
    r_lens = np.array([r_lens_nfw(m, z, zs, rhoc, h, omega_m, omega_r, omega_l, h0, kappa_thr) for m in m_grid])
    sigma_new = PI * r_lens**2
    w_new = dn_dlnm * sigma_new
    if np.all(w_new <= 0):
        raise RuntimeError("Threshold produced zero cross section over entire mass range")
    pm_new = w_new / np.trapezoid(w_new, lnm)

    # M_min
    valid = np.where(sigma_new > 0)[0]
    m_min = float(m_grid[valid[0]]) if valid.size else np.nan

    # nu summary (new model)
    mean_nu = float(np.trapezoid(pm_new * nu, lnm))
    cdf_new = np.concatenate(([0.0], np.cumsum((pm_new[1:] + pm_new[:-1]) / 2.0 * np.diff(lnm))))
    cdf_new /= cdf_new[-1]
    nu_median = float(np.interp(0.5, cdf_new, nu))
    nu_mode = float(nu[np.argmax(pm_new)])
    avg_dlnf = float(np.trapezoid(pm_new * dlnf, lnm))
    gamma_pred = float(-2.0 * avg_dlnf)

    # P(nu) curves for comparison
    lnnu = np.log(nu)
    bins = np.linspace(lnnu.min(), lnnu.max(), 70)
    hist_old, edges = np.histogram(lnnu, bins=bins, weights=w_old, density=False)
    hist_new, _ = np.histogram(lnnu, bins=bins, weights=w_new, density=False)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dlnnu = np.diff(edges)
    pnu_old = hist_old / (hist_old.sum() * dlnnu)
    pnu_new = hist_new / (hist_new.sum() * dlnnu)
    nu_centers = np.exp(centers)

    return {
        "m_grid": m_grid,
        "nu": nu,
        "dn_dlnm": dn_dlnm,
        "pm_old": pm_old,
        "pm_new": pm_new,
        "nu_centers": nu_centers,
        "pnu_old": pnu_old,
        "pnu_new": pnu_new,
        "r_lens": r_lens,
        "sigma_lens_new": sigma_new,
        "sigma_lens_old": sigma_old,
        "m_min": m_min,
        "mean_nu": mean_nu,
        "nu_median": nu_median,
        "nu_mode": nu_mode,
        "gamma_pred": gamma_pred,
        "z": z,
        "zs": zs,
        "kappa_thr": kappa_thr,
    }


def main():
    z = 1.0
    zs = 2.0
    kappa_thr = 0.01
    out = build_distributions(z=z, zs=zs, kappa_thr=kappa_thr)

    # Plot old vs new P(nu)
    plt.figure(figsize=(8, 5))
    plt.plot(out["nu_centers"], out["pnu_old"], label="Old proxy: sigma ~ M^(2/3)", lw=1.8)
    plt.plot(out["nu_centers"], out["pnu_new"], label="Thresholded NFW", lw=1.8)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("nu")
    plt.ylabel("P(nu) (per dlnnu)")
    plt.title(f"P(nu) comparison at z={z}, z_s={zs}, kappa_thr={kappa_thr}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("nu_old_vs_thresholded_pnu_z1.png", dpi=170)

    # Plot r_lens(M) and sigma_lens(M) vs M^(2/3) proxy (normalized)
    plt.figure(figsize=(8, 5))
    plt.plot(out["m_grid"], out["r_lens"], lw=1.8)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("M [Msun]")
    plt.ylabel("r_lens")
    plt.title(f"Thresholded NFW lensing radius at z={z}, kappa_thr={kappa_thr}")
    plt.tight_layout()
    plt.savefig("nfw_rlens_vs_mass_z1.png", dpi=170)

    plt.figure(figsize=(8, 5))
    snew = out["sigma_lens_new"]
    sold = out["sigma_lens_old"]
    # normalize both curves for shape comparison
    snew_n = snew / np.max(snew)
    sold_n = sold / np.max(sold)
    plt.plot(out["m_grid"], sold_n, lw=1.8, label="Proxy M^(2/3) (normalized)")
    plt.plot(out["m_grid"], snew_n, lw=1.8, label="Thresholded NFW area (normalized)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("M [Msun]")
    plt.ylabel("Relative cross section")
    plt.title(f"Cross-section shape comparison at z={z}, kappa_thr={kappa_thr}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("sigma_lens_shape_compare_z1.png", dpi=170)

    summary = {
        "z": out["z"],
        "zs_assumed": out["zs"],
        "kappa_thr": out["kappa_thr"],
        "M_min": out["m_min"],
        "mean_nu": out["mean_nu"],
        "nu_median": out["nu_median"],
        "nu_mode": out["nu_mode"],
        "gamma_pred": out["gamma_pred"],
    }
    with open("nu_thresholded_summary_z1.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Threshold sweep
    thr_values = [0.005, 0.01, 0.02, 0.05]
    sweep = []
    for thr in thr_values:
        o = build_distributions(z=z, zs=zs, kappa_thr=thr)
        sweep.append(
            {
                "kappa_thr": thr,
                "mean_nu": o["mean_nu"],
                "nu_median": o["nu_median"],
                "nu_mode": o["nu_mode"],
                "gamma_pred": o["gamma_pred"],
                "M_min": o["m_min"],
            }
        )
    with open("nu_threshold_sweep_z1.json", "w") as f:
        json.dump({"z": z, "zs_assumed": zs, "rows": sweep}, f, indent=2)

    # Sweep plot gamma and mean_nu vs threshold
    t = np.array([r["kappa_thr"] for r in sweep])
    g = np.array([r["gamma_pred"] for r in sweep])
    mn = np.array([r["mean_nu"] for r in sweep])

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(t, g, "o-", lw=1.8, label="gamma_pred")
    ax1.set_xscale("log")
    ax1.set_xlabel("kappa_thr")
    ax1.set_ylabel("gamma_pred")
    ax2 = ax1.twinx()
    ax2.plot(t, mn, "s--", lw=1.4, color="tab:orange", label="mean_nu")
    ax2.set_ylabel("mean_nu")
    ax1.set_title("Threshold sensitivity at z=1")
    fig.tight_layout()
    fig.savefig("nu_threshold_sensitivity_z1.png", dpi=170)

    print(json.dumps(summary, indent=2))
    print("SWEEP", json.dumps(sweep, indent=2))
    print(
        "WROTE nu_old_vs_thresholded_pnu_z1.png nfw_rlens_vs_mass_z1.png "
        "sigma_lens_shape_compare_z1.png nu_thresholded_summary_z1.json "
        "nu_threshold_sweep_z1.json nu_threshold_sensitivity_z1.png"
    )


if __name__ == "__main__":
    main()
