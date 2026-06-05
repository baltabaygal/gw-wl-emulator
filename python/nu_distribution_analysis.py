import json
import math
import numpy as np
import matplotlib.pyplot as plt


PI = np.pi


def d_growth(z, omega_m, omega_r, omega_l):
    a = omega_m * (1 + z) ** 3 + omega_r * (1 + z) ** 4 + omega_l
    omz = omega_m * (1 + z) ** 3 / a
    olz = omega_l / a
    denom = (omz ** (4.0 / 7.0) - olz + (1 + omz / 2.0) * (1 + olz / 70.0))
    return (5.0 / 2.0) * omz / denom / (1 + z) / 0.7869370293916


def deltac(z, omega_m, omega_r, omega_l):
    deltac0 = 3.0 / 5.0 * (3.0 * PI / 2.0) ** (2.0 / 3.0)
    return deltac0 / d_growth(z, omega_m, omega_r, omega_l)


def tm_cdm(k, omega_m, omega_b, omega_c, h, t0, zeq, hz):
    keq = 0.00326227 * hz(zeq) / (1.0 + zeq)
    omh2 = omega_m * h * h
    obh2 = omega_b * h * h
    oc_om = omega_c / omega_m
    ob_om = omega_b / omega_m

    ksilk = 0.0016 * obh2 ** 0.52 * omh2 ** 0.73 * (1 + (10.4 * omh2) ** -0.95)
    a1 = (46.9 * omh2) ** 0.67 * (1 + (32.1 * omh2) ** -0.532)
    a2 = (12.0 * omh2) ** 0.424 * (1 + (45.0 * omh2) ** -0.582)
    b1 = 0.944 / (1 + (458 * omh2) ** -0.708)
    b2 = (0.395 * omh2) ** -0.026
    acnum = a1 ** (-ob_om) * a2 ** (-(ob_om ** 3.0))
    bcnum = 1.0 / (1 + b1 * (oc_om ** b2 - 1))
    s2 = 44.5 * 1000 * np.log(9.83 / omh2) / np.sqrt(1 + 10 * obh2 ** (3.0 / 4.0))
    b3 = 0.313 * omh2 ** -0.419 * (1 + 0.607 * omh2 ** 0.674)
    b4 = 0.238 * omh2 ** 0.223
    zd = 1291 * omh2 ** 0.251 / (1 + 0.659 * omh2 ** 0.828) * (1 + b3 * obh2 ** b4)
    rd = 31.5 * obh2 * (t0 / 2.7) ** -4.0 / (zd / 1000)

    y = (1 + zeq) / (1 + zd)
    g2 = y * (-6 * np.sqrt(1 + y) + (2.0 + 3.0 * y) * np.log((np.sqrt(1 + y) + 1) / (np.sqrt(1 + y) - 1)))
    ab = 2.07 * keq * s2 * (1 + rd) ** (-3.0 / 4.0) * g2
    bb = 0.5 + ob_om + (3.0 - 2.0 * ob_om) * np.sqrt(1 + (17.2 * omh2) ** 2.0)
    bnode = 8.41 * omh2 ** 0.435

    qk = k / (13.41 * keq)
    fk = 1.0 / (1 + (k * s2 / 5.4) ** 4.0)
    c1_1 = 14.2 + 386.0 / (1 + 69.9 * qk ** 1.08)
    c1_a = 14.2 / acnum + 386.0 / (1 + 69.9 * qk ** 1.08)
    to1_1 = np.log(np.e + 1.8 * bcnum * qk) / (np.log(np.e + 1.8 * bcnum * qk) + c1_1 * qk ** 2.0)
    to1_a = np.log(np.e + 1.8 * bcnum * qk) / (np.log(np.e + 1.8 * bcnum * qk) + c1_a * qk ** 2.0)
    tc = fk * to1_1 + (1 - fk) * to1_a

    s3 = s2 / (1 + (bnode / (k * s2)) ** 3.0) ** (1.0 / 3.0)
    j0 = np.sin(k * s3) / (k * s3)
    tb = (to1_1 / (1 + (k * s2 / 5.2) ** 2.0) + ab / (1 + (bb / (k * s2)) ** 3.0) * np.exp(-(k / ksilk) ** 1.4)) * j0
    return ob_om * tb + oc_om * tc


def ws(x):
    c = 1.0 / 0.43
    b = 6.0
    return 1.0 / (1.0 + (x / c) ** b)


def dws(x):
    c = 1.0 / 0.43
    b = 6.0
    xc = x / c
    return -b * xc ** b / (x * (1.0 + xc ** b) ** 2.0)


def deltak(k, delta_h, h0, ns, tm):
    return np.sqrt((306.535 * k / h0) ** (3.0 + ns) * (delta_h * tm) ** 2)


def sigma_cdm(M, delta_h, nk, rho_m0, h0, ns, tm_func):
    rm = (3.0 * M / (4.0 * PI * rho_m0)) ** (1.0 / 3.0)
    drm = rm / (3.0 * M)
    kmax = 1000.0 / rm
    kmin = 1.0e-6 * kmax
    dlogk = (np.log(kmax) - np.log(kmin)) / (nk - 1)

    sigma2 = 0.0
    dsigma2 = 0.0
    k2 = kmin
    for _ in range(nk):
        k1 = k2
        k2 = np.exp(np.log(k2) + dlogk)
        d1 = deltak(k1, delta_h, h0, ns, tm_func(k1))
        d2 = deltak(k2, delta_h, h0, ns, tm_func(k2))
        w1 = ws(k1 * rm)
        w2 = ws(k2 * rm)
        sigma2 += (k2 - k1) * ((w1 * d1) ** 2 / k1 + (w2 * d2) ** 2 / k2) / 2.0
        dsigma2 += (k2 - k1) * (
            2.0 * k2 * drm * dws(k2 * rm) * w2 * d2 ** 2 / k2
            + 2.0 * k1 * drm * dws(k1 * rm) * w1 * d1 ** 2 / k1
        ) / 2.0

    sigma = np.sqrt(sigma2)
    dsigma = dsigma2 / (2.0 * sigma)
    return sigma, dsigma


def st_f_nu(nu, p=0.3, q=0.8):
    a = 1.0 / (1.0 + 2.0 ** (-p) * math.gamma(0.5 - p) / np.sqrt(PI))
    return a * (1.0 + (q * nu) ** (-p)) * np.sqrt(q * nu / (2.0 * PI)) * np.exp(-q * nu / 2.0)


def dlnf_dlnnu(nu, p=0.3, q=0.8):
    x = (q * nu) ** (-p)
    return -p * x / (1.0 + x) + 0.5 - 0.5 * q * nu


def main():
    z = 1.0
    mmin = 1e8
    mmax = 1e16
    nm = 180
    nk = 1000

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

    hz = lambda zz: h0 * np.sqrt(omega_m * (1 + zz) ** 3.0 + omega_r * (1 + zz) ** 4.0 + omega_l)
    tm_func = lambda k: tm_cdm(k, omega_m, omega_b, omega_c, h, t0, zeq, hz)

    # Normalize deltaH8 exactly as in simulator.
    sigma_m8, _ = sigma_cdm(m8, 1.0, nk, rho_m0, h0, ns, tm_func)
    delta_h8 = sigma8 / sigma_m8

    m_grid = np.logspace(np.log10(mmin), np.log10(mmax), nm)
    sig = np.zeros_like(m_grid)
    dsig_dm = np.zeros_like(m_grid)
    for i, m in enumerate(m_grid):
        sig[i], dsig_dm[i] = sigma_cdm(m, delta_h8, nk, rho_m0, h0, ns, tm_func)

    dc = deltac(z, omega_m, omega_r, omega_l)
    nu = (dc**2) / (sig**2)
    fnu = st_f_nu(nu)
    dlnf = dlnf_dlnnu(nu)

    # dn/dlnM from simulator expression: -rhoM0 * pFC * 2*sigma*dsigma/dM, with pFC = f(nu)/S
    s = sig**2
    pfc = fnu / s
    dn_dlnm = -rho_m0 * pfc * 2.0 * sig * dsig_dm

    # First-pass lensing cross-section proxy.
    sigma_lens = m_grid ** (2.0 / 3.0)
    w = dn_dlnm * sigma_lens
    lnm = np.log(m_grid)

    # Normalize over dlnM.
    norm = np.trapezoid(w, lnm)
    pm = w / norm

    # nu stats from M-weighted distribution
    mean_nu = float(np.trapezoid(pm * nu, lnm))
    cdf = np.concatenate(([0.0], np.cumsum((pm[1:] + pm[:-1]) / 2.0 * np.diff(lnm))))
    cdf = cdf / cdf[-1]
    nu_median = float(np.interp(0.5, cdf, nu))
    nu_mode = float(nu[np.argmax(pm)])

    # Weighted average slope and gamma prediction
    avg_dlnf = float(np.trapezoid(pm * dlnf, lnm))
    gamma_pred = float(-2.0 * avg_dlnf)

    # Build P(nu) by weighted histogram in log-nu.
    lnnu = np.log(nu)
    bins = np.linspace(lnnu.min(), lnnu.max(), 70)
    hist, edges = np.histogram(lnnu, bins=bins, weights=w, density=False)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dlnnu = np.diff(edges)
    p_lnnu = hist / (hist.sum() * dlnnu)
    nu_centers = np.exp(centers)

    # Plots
    plt.figure(figsize=(8, 5))
    plt.plot(m_grid, pm, lw=1.8)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("M [Msun]")
    plt.ylabel("P(M) (per dlnM)")
    plt.title("Cross-section weighted halo mass distribution at z=1")
    plt.tight_layout()
    plt.savefig("nu_weighted_PM_z1.png", dpi=170)

    plt.figure(figsize=(8, 5))
    plt.plot(nu_centers, p_lnnu, lw=1.8)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("nu")
    plt.ylabel("P(nu) (per dlnnu)")
    plt.title("Cross-section weighted nu distribution at z=1")
    plt.tight_layout()
    plt.savefig("nu_weighted_Pnu_z1.png", dpi=170)

    out = {
        "z": z,
        "mass_range_msun": [mmin, mmax],
        "mean_nu": mean_nu,
        "nu_median": nu_median,
        "nu_mode": nu_mode,
        "avg_dlnf_dlnnu": avg_dlnf,
        "gamma_pred": gamma_pred,
    }
    with open("nu_distribution_summary_z1.json", "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))
    print("WROTE nu_weighted_PM_z1.png nu_weighted_Pnu_z1.png nu_distribution_summary_z1.json")


if __name__ == "__main__":
    main()
