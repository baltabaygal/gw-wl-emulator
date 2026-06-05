import json
import numpy as np
import matplotlib.pyplot as plt

from nu_thresholded_nfw_analysis import (
    build_distributions,
    concentration_1402,
    sigma_crit,
    r_lens_nfw,
    fnfw,
    PI,
)


KMAX = 20.420352248334
ZL = 1.0
ZS = 2.0
STARTS = [0.001, 0.003, 0.005, 0.010, 0.020, 0.050]
MAX_ITERS = 10
TOL = 1e-4


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


def xi_mean_for_mass(m_halo, kappa_thr, n_b=500):
    omega_m = 0.315
    omega_b = 0.0493
    h = 0.674
    zeq = 3402.0
    omega_r = omega_m / (1.0 + zeq)
    omega_l = 1.0 - omega_m - omega_r
    h0 = 0.000102247 * h
    rhoc = 277.394 * h**2

    c = concentration_1402(ZL, m_halo, h)
    r200 = (3.0 * m_halo / (4.0 * PI * 200.0 * rhoc)) ** (1.0 / 3.0)
    r_s = r200 / c
    rho_s = 200.0 * rhoc * c**3 * (1 + c) / (3.0 * ((1 + c) * np.log(1 + c) - c))
    sigmac = sigma_crit(ZS, ZL, omega_m, omega_r, omega_l, h0)
    k0 = r_s * rho_s / sigmac

    r_lens = r_lens_nfw(m_halo, ZL, ZS, rhoc, h, omega_m, omega_r, omega_l, h0, kappa_thr)
    if r_lens <= 0:
        return np.nan, 0.0

    b = np.geomspace(r_lens * 1e-6, r_lens, n_b)
    x = b / r_s
    kap = np.array([kappa_nfw(k0, xx) for xx in x])
    gam = np.array([gamma_nfw(k0, xx) for xx in x])
    denom = (1.0 - kap) ** 2 - gam**2
    good = denom > 0
    if not np.any(good):
        return np.nan, r_lens
    xi = np.zeros_like(b)
    xi[good] = -np.log(denom[good])
    w = 2.0 * PI * b
    num = np.trapezoid(xi * w * good.astype(float), b)
    den = np.trapezoid(w * good.astype(float), b)
    if den <= 0:
        return np.nan, r_lens
    return float(num / den), r_lens


def step(kappa_n):
    d = build_distributions(z=ZL, zs=ZS, kappa_thr=kappa_n)
    m = d["m_grid"]
    pm = d["pm_new"]  # per dlnM normalized
    nu = d["nu"]
    lnm = np.log(m)

    # Representative-halo C around nu~3
    irep = int(np.argmin(np.abs(nu - 3.0)))
    xi_rep, _ = xi_mean_for_mass(float(m[irep]), kappa_n, n_b=1200)
    c_single = float(kappa_n / xi_rep) if np.isfinite(xi_rep) and xi_rep > 0 else np.nan

    # Distribution-averaged C over masses that matter most to pm (fast + informative).
    idx = np.where(pm > 1e-5 * np.max(pm))[0]
    if idx.size == 0:
        idx = np.arange(m.size)
    # Thin for speed.
    idx = idx[::2]
    m_sel = m[idx]
    pm_sel = pm[idx]
    lnm_sel = lnm[idx]

    c_vals = []
    for ms in m_sel:
        xi_m, _ = xi_mean_for_mass(float(ms), kappa_n, n_b=320)
        if np.isfinite(xi_m) and xi_m > 0:
            c_vals.append(kappa_n / xi_m)
        else:
            c_vals.append(np.nan)
    c_vals = np.array(c_vals, dtype=float)

    good = np.isfinite(c_vals)
    if np.count_nonzero(good) < 3:
        c_dist = np.nan
    else:
        w = pm_sel[good]
        # weights are per dlnM density; integrate in lnm then normalize
        wn = w / np.trapezoid(w, lnm_sel[good])
        c_dist = float(np.trapezoid(wn * c_vals[good], lnm_sel[good]))

    kappa_next = float(c_dist / KMAX) if np.isfinite(c_dist) else np.nan
    gamma_pred = interp_gamma_from_sweep(kappa_next) if np.isfinite(kappa_next) else np.nan

    return {
        "kappa": float(kappa_n),
        "kappa_next": kappa_next,
        "nu_mean": float(d["mean_nu"]),
        "nu_median": float(d["nu_median"]),
        "nu_mode": float(d["nu_mode"]),
        "C_single": c_single,
        "C_dist": c_dist,
        "gamma_pred": gamma_pred,
    }


def run_chain(kappa0):
    hist = []
    k = float(kappa0)
    for _ in range(MAX_ITERS):
        rec = step(k)
        hist.append(rec)
        if not np.isfinite(rec["kappa_next"]):
            break
        if abs(rec["kappa_next"] - rec["kappa"]) < TOL:
            k = rec["kappa_next"]
            break
        k = rec["kappa_next"]
    return hist


def main():
    all_hist = {}
    final_table = []
    for k0 in STARTS:
        h = run_chain(k0)
        all_hist[str(k0)] = h
        kf = h[-1]["kappa_next"] if h else np.nan
        final_table.append({"start_kappa": k0, "final_kappa": kf, "iters": len(h)})

    # Aggregate converged diagnostics from chains that have finite last step.
    finals = [v[-1] for v in all_hist.values() if len(v) and np.isfinite(v[-1]["kappa_next"])]
    kstars = np.array([f["kappa_next"] for f in finals], dtype=float)
    nustars = np.array([f["nu_mean"] for f in finals], dtype=float)
    cstars = np.array([f["C_dist"] for f in finals], dtype=float)
    gstars = np.array([f["gamma_pred"] for f in finals], dtype=float)

    summary = {
        "kmax_used": KMAX,
        "start_values": STARTS,
        "fixed_point_table": final_table,
        "kappa_star_median": float(np.median(kstars)) if kstars.size else None,
        "nu_star_mean_median": float(np.median(nustars)) if nustars.size else None,
        "C_star_median": float(np.median(cstars)) if cstars.size else None,
        "gamma_pred_star_median": float(np.median(gstars)) if gstars.size else None,
    }

    with open("kappa_fixed_point_summary.json", "w") as f:
        json.dump({"summary": summary, "chains": all_hist}, f, indent=2)

    # Plot trajectories
    plt.figure(figsize=(8, 5))
    for k0, h in all_hist.items():
        ks = [r["kappa"] for r in h]
        kn = [r["kappa_next"] for r in h]
        # plot k_n sequence including final k_{n+1}
        seq = ks + ([kn[-1]] if len(kn) else [])
        it = np.arange(len(seq))
        plt.plot(it, seq, marker="o", lw=1.5, label=f"start {k0}")
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("kappa_n")
    plt.title("Fixed-point trajectories for kappa update map")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig("kappa_fixed_point_trajectories.png", dpi=170)

    print(json.dumps(summary, indent=2))
    print("WROTE kappa_fixed_point_summary.json kappa_fixed_point_trajectories.png")


if __name__ == "__main__":
    main()
