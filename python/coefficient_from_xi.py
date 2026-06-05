import json
import numpy as np
import matplotlib.pyplot as plt

from nu_thresholded_nfw_analysis import (
    build_distributions,
    concentration_1402,
    sigma_crit,
    r_lens_nfw,
    fnfw,
    hz,
    PI,
)


def g_nfw(x):
    if x > 1:
        return 2 * np.arctan(np.sqrt((x - 1) / (1 + x))) / np.sqrt(x * x - 1) + np.log(x / 2)
    if x < 1:
        return 2 * np.arctanh(np.sqrt((1 - x) / (1 + x))) / np.sqrt(1 - x * x) + np.log(x / 2)
    return 1 + np.log(0.5)


def kappa_nfw(kappa0, x):
    return 2.0 * kappa0 * fnfw(x)


def gamma_nfw(kappa0, x):
    return 2.0 * kappa0 * (2.0 * g_nfw(x) / (x * x) - fnfw(x))


def interp_gamma_from_sweep(kappa_thr):
    with open("nu_threshold_sweep_z1.json") as f:
        rows = json.load(f)["rows"]
    k = np.array([r["kappa_thr"] for r in rows], float)
    g = np.array([r["gamma_pred"] for r in rows], float)
    o = np.argsort(k)
    k = k[o]
    g = g[o]
    return float(np.interp(np.log(np.clip(kappa_thr, k.min(), k.max())), np.log(k), g))


def representative_halo():
    d = build_distributions(z=1.0, zs=2.0, kappa_thr=0.01)
    nu = d["nu"]
    m = d["m_grid"]
    i = int(np.argmin(np.abs(nu - 3.0)))
    return float(m[i]), float(nu[i])


def compute_xi_profile_for_threshold(m_halo, z_l, z_s, kappa_thr):
    # Cosmology defaults (same as simulator setup used in previous scripts)
    omega_m = 0.315
    omega_b = 0.0493
    h = 0.674
    zeq = 3402.0
    omega_r = omega_m / (1.0 + zeq)
    omega_l = 1.0 - omega_m - omega_r
    omega_c = omega_m - omega_b
    h0 = 0.000102247 * h
    rhoc = 277.394 * h**2

    c = concentration_1402(z_l, m_halo, h)
    r200 = (3.0 * m_halo / (4.0 * PI * 200.0 * rhoc)) ** (1.0 / 3.0)
    r_s = r200 / c
    rho_s = 200.0 * rhoc * c**3 * (1 + c) / (3.0 * ((1 + c) * np.log(1 + c) - c))
    sigmac = sigma_crit(z_s, z_l, omega_m, omega_r, omega_l, h0)
    k0 = r_s * rho_s / sigmac

    r_lens = r_lens_nfw(m_halo, z_l, z_s, rhoc, h, omega_m, omega_r, omega_l, h0, kappa_thr)
    if r_lens <= 0:
        return None

    # Area integration on linear radius with log-spaced sampling for center stability.
    b = np.geomspace(r_lens * 1e-6, r_lens, 3000)
    x = b / r_s
    kap = np.array([kappa_nfw(k0, xx) for xx in x])
    gam = np.array([gamma_nfw(k0, xx) for xx in x])
    denom = (1.0 - kap) ** 2 - gam**2

    # Keep only weak-lensing-valid annuli (denom > 0).
    good = denom > 0
    xi = np.full_like(b, np.nan, dtype=float)
    xi[good] = -np.log(denom[good])

    # Area-weighted average jump over available weak-lensing area.
    w = 2.0 * PI * b
    num = np.trapezoid(np.nan_to_num(xi, nan=0.0) * w, b)
    den = np.trapezoid(w * good.astype(float), b)
    xi_mean = float(num / den) if den > 0 else np.nan

    return {
        "b": b,
        "xi": xi,
        "r_lens": float(r_lens),
        "r_s": float(r_s),
        "weak_area_fraction": float(den / (PI * r_lens**2)),
        "xi_mean": xi_mean,
    }


def main():
    m_halo, nu_halo = representative_halo()
    z_l = 1.0
    z_s = 2.0
    kmax_obs = 20.420352248334  # global common-band value used previously

    thresholds = [0.005, 0.01, 0.02, 0.05]
    rows = []
    xi_plot = None
    for thr in thresholds:
        prof = compute_xi_profile_for_threshold(m_halo, z_l, z_s, thr)
        if prof is None or not np.isfinite(prof["xi_mean"]) or prof["xi_mean"] <= 0:
            rows.append(
                {
                    "kappa_thr": thr,
                    "xi_mean": None,
                    "C": None,
                    "kappa_eff_from_C_over_kmax": None,
                    "gamma_pred_updated": None,
                }
            )
            continue

        # xi_mean ~ alpha * kappa_thr  => C = 1/alpha = kappa_thr / xi_mean
        alpha = prof["xi_mean"] / thr
        C = 1.0 / alpha
        kappa_eff = C / kmax_obs
        gamma_upd = interp_gamma_from_sweep(kappa_eff)
        rows.append(
            {
                "kappa_thr": thr,
                "xi_mean": float(prof["xi_mean"]),
                "C": float(C),
                "kappa_eff_from_C_over_kmax": float(kappa_eff),
                "gamma_pred_updated": float(gamma_upd),
                "r_lens": prof["r_lens"],
                "weak_area_fraction": prof["weak_area_fraction"],
            }
        )
        if abs(thr - 0.01) < 1e-12:
            xi_plot = prof

    # Representative xi(b) plot at kappa_thr=0.01
    if xi_plot is not None:
        bb = xi_plot["b"] / xi_plot["r_lens"]
        plt.figure(figsize=(7.2, 4.8))
        plt.plot(bb, xi_plot["xi"], lw=1.6)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("b / r_lens")
        plt.ylabel("xi(b) = ln(mu)")
        plt.title(f"Representative xi(b), nu~{nu_halo:.2f}, kappa_thr=0.01")
        plt.tight_layout()
        plt.savefig("xi_profile_representative_nu3.png", dpi=170)

    out = {
        "representative_halo": {"M": m_halo, "nu": nu_halo, "z_l": z_l, "z_s": z_s},
        "kmax_obs_used": kmax_obs,
        "rows": rows,
    }
    with open("coefficient_from_xi_summary.json", "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))
    print("WROTE coefficient_from_xi_summary.json xi_profile_representative_nu3.png")


if __name__ == "__main__":
    main()
