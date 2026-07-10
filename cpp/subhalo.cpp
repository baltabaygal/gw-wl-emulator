#include "cosmology.h"
#include "lensing.h"      // declarations of kappa0NFW, kappagammaNFWeps
#include "subhalo.h"
#include <algorithm>
#include <cmath>
#include <gsl/gsl_sf_gamma.h>
#include <limits>
#include <thread>

double Sigmacf(cosmology &C, double zs, double zl);   // defined in lensing.cpp

namespace {

inline double linfast(double y1, double y2, double x1, double x2, double x) {
    return y1 + (x - x1) / (x2 - x1) * (y2 - y1);
}

inline void interpolateNFWMass(cosmology &C, int jz, double m, double log_m, double log_Mmin, double inv_dlogM, double &rs, double &rhos) {
    const int NM = C.NM;
    if (m <= C.Mlist[0]) {
        rs = C.NFWlist[jz][0][0];
        rhos = C.NFWlist[jz][0][1];
    } else if (m >= C.Mlist[NM - 1]) {
        rs = C.NFWlist[jz][NM - 1][0];
        rhos = C.NFWlist[jz][NM - 1][1];
    } else {
        double val = (log_m - log_Mmin) * inv_dlogM;
        int jm = static_cast<int>(val) + 1;
        const double m1 = C.Mlist[jm - 1];
        const double m2 = C.Mlist[jm];
        rs = linfast(C.NFWlist[jz][jm - 1][0], C.NFWlist[jz][jm][0], m1, m2, m);
        rhos = linfast(C.NFWlist[jz][jm - 1][1], C.NFWlist[jz][jm][1], m1, m2, m);
    }
}

struct ClumpAccum {
    double kappa = 0.0;
    double gamma1 = 0.0;
    double gamma2 = 0.0;
};

inline double safeNFWGammaCore(double x, const std::array<double, 2> &Fg) {
    if (x < 1.0e-4) return 0.5;
    return 2.0 * Fg[1] / (x * x) - Fg[0];
}

ClumpAccum evalClumpRange(cosmology &C, int jz, double M, double Sigmac,
                          double rcos, double rsin, double r200,
                          const vector<double> &xcdf, int Nu,
                          double alpha, double beta, double omega,
                          double psi_min, double psi_max,
                          double log_Mmin, double inv_dlogM, double d_cut,
                          int begin, int end, rgen &mt) {
    ClumpAccum acc;
    const double pa_min = pow(psi_min, alpha);
    const double pa_max = pow(psi_max, alpha);
    const double inv_alpha = 1.0 / alpha;
    const double log_M = std::log(M);
    for (int k = begin; k < end; k++) {
        double u = randomreal(0.0, 1.0, mt);
        double psi = pow(pa_min + u * (pa_max - pa_min), inv_alpha);
        // exact SHMF: thin the power-law proposal by the exponential cutoff.
        // Poisson(N_powerlaw) + thinning == Poisson(N_exact)  (Poisson thinning theorem).
        if (randomreal(0.0, 1.0, mt) > exp(-beta * pow(psi, omega))) continue;
        double m = psi * M;
        double log_m = std::log(m);

        double ur = randomreal(0.0, 1.0, mt);
        double tt = ur * (Nu - 1);
        int i = (int)tt;
        if (i >= Nu - 1) i = Nu - 2;
        double x = xcdf[i] + (tt - i) * (xcdf[i + 1] - xcdf[i]);
        double r3d = x * r200;
        double cth = randomreal(-1.0, 1.0, mt);
        double psaz = randomreal(0.0, 2.0 * PI, mt);
        double R2d = r3d * sqrt(1.0 - cth * cth);

        double dx = rcos - R2d * cos(psaz);
        double dy = rsin - R2d * sin(psaz);
        double d = sqrt(dx * dx + dy * dy);
        if (d > d_cut) continue;

        double inv_d = (d > 1e-30) ? 1.0 / d : 0.0;
        // single-angle gamma projection, matching the original Vaskonen host convention
        // (kept by decision 2026-07-02; see tmp/shear_convention_check.py for the spin-2 form)
        double cos_phid = dx * inv_d;
        double sin_phid = dy * inv_d;

        double rs_c, rhos_c;
        interpolateNFWMass(C, jz, m, log_m, log_Mmin, inv_dlogM, rs_c, rhos_c);
        double kappa0_c = kappa0NFW(rs_c, rhos_c, Sigmac);
        double x_c = std::max(d / rs_c, 1.0e-12);
        auto Fg = FgNFW(x_c);
        double kappa_c = 2.0 * kappa0_c * Fg[0];
        double gamma_c = 2.0 * kappa0_c * safeNFWGammaCore(x_c, Fg);

        acc.kappa += kappa_c;
        acc.gamma1 += cos_phid * gamma_c;
        acc.gamma2 += sin_phid * gamma_c;
    }
    return acc;
}

} // namespace

// Build all (z,M)-grid subhalo tables. Cheap; called once per run.
void Subhalo::precompute(cosmology &C, double zs, double kappathr, double kappathr_host) {
    int Nz = C.Nz, NM = C.NM;
    log_Mmin = std::log(C.Mlist[0]);
    double log_Mmax = std::log(C.Mlist[NM - 1]);
    inv_dlogM = (NM - 1) / (log_Mmax - log_Mmin);
    fsub.assign(Nz, vector<double>(NM, 0.0));
    Nsub.assign(Nz, vector<double>(NM, 0.0));
    rs_sm.assign(Nz, vector<double>(NM, 0.0));
    rhos_sm.assign(Nz, vector<double>(NM, 0.0));
    r200h.assign(Nz, vector<double>(NM, 0.0));
    chost.assign(Nz, vector<double>(NM, 0.0));
    gnorm.assign(Nz, vector<double>(NM, 0.0));
    invRad.assign(Nz, vector<vector<double>>(NM));
    muW.assign(Nz, vector<vector<double>>(NM));
    sW.assign(Nz, vector<vector<double>>(NM));
    lyW.assign(Nz, vector<std::array<double,2>>(NM, {0.0, 1.0}));
    fsb.assign(Nz, vector<double>(NM, 0.0));

    // clump-reach table: r_thr[jz][jm] = distance at which a clump of mass Mlist[jm]
    // reaches kappa_thr (monotonic in mass).  m_res(r) = inverse of this, looked up
    // per host in addClumps.  Independent of host mass; built per lens redshift.
    r_thr.assign(Nz, vector<double>(NM, 0.0));
    for (int jz = 0; jz < Nz; jz++) {
        double zl = C.zlist[jz];
        if (zl >= zs) continue;
        for (int jm = 0; jm < NM; jm++)
            r_thr[jz][jm] = rmaxfNFW(C, zs, zl, C.Mlist[jm], kappathr);
    }

    const double s = (1.0 + alpha) / omega;
    // formation-redshift weight (Giocoli+2007, a_f = 0.815 e^{-2f^3}/f^0.707, f=0.5):  w_f = sqrt(2 ln(a_f+1))
    const double af = 0.815 * exp(-0.25) / pow(0.5, 0.707);
    const double wf = sqrt(2.0 * log(af + 1.0));
    // SHMF normalization denominator (Gamma upper-incomplete), JvdB14 eq. 23
    const double gden = gsl_sf_gamma_inc(s, beta * pow(psi_res, omega)) - gsl_sf_gamma_inc(s, beta);

    for (int jz = 0; jz < Nz; jz++) {
        double z = C.zlist[jz];
        double dcz = C.deltac(z);
        for (int jM = 0; jM < NM; jM++) {
            double M = C.Mlist[jM];
            if (M <= C.Mmin) continue;                 // no clump above the grid floor fits

            double sigM = interpolate(M, C.sigmalist);
            double sigH = interpolate(0.5 * M, C.sigmalist);
            double dsig2 = sigH * sigH - sigM * sigM;
            if (dsig2 <= 0.0) continue;

            // formation redshift: solve deltac(zf) = deltac(z) + wf sqrt(dsig2)
            double rhs = dcz + wf * sqrt(dsig2);
            if (C.deltac(30.0) < rhs) continue;
            double zlo = z, zhi = 30.0;
            for (int it = 0; it < 60; it++) {
                double zm = 0.5 * (zlo + zhi);
                if (C.deltac(zm) < rhs) zlo = zm; else zhi = zm;
            }
            double zf = 0.5 * (zlo + zhi);

            // dynamical age N_tau = int_z^zf 6.006 (Dvir/178)^1/2 /(1+z') dz'  (Hz cancels)
            int Nstep = 200; double Ntau = 0.0, dz = (zf - z) / Nstep;
            for (int i = 0; i < Nstep; i++) {
                double zz = z + (i + 0.5) * dz;
                double d = C.OmegaMz(zz) - 1.0;
                double Dvir = 18.0 * PI * PI + 82.0 * d - 39.0 * d * d;
                Ntau += 6.006 * sqrt(Dvir / 178.0) / (1.0 + zz) * dz;
            }
            if (Ntau <= 0.0) continue;

            double fs = 0.3563 / pow(Ntau, 0.6) - 0.075;   // JvdB14 eq. 26 (all orders)
            if (fs <= 0.0 || fs >= 0.95) continue;
            double gam = omega * pow(beta, s) / gden * fs; // eq. 23
            gnorm[jz][jM] = gam;                           // used by addClumps (dynamic floor)

            double psi_min = C.Mmin / M;
            if (psi_min >= psi_max) continue;
            // mean PROPOSAL count above floor (pure power law; addClumps thins by the
            // exponential cutoff, so the realized count is smaller — profiling only)
            double Nm = (gam / alpha) * (pow(psi_max, alpha) - pow(psi_min, alpha));
            if (Nm <= 0.0) continue;

            fsub[jz][jM] = fs;
            Nsub[jz][jM] = Nm;

            // reduced smooth host at (1-fs) M, and full-host geometry for the radial scale
            vector<double> NFWsm = interpolate2(z, (1.0 - fs) * M, C.zlist, C.Mlist, C.NFWlist);
            rs_sm[jz][jM] = NFWsm[0]; rhos_sm[jz][jM] = NFWsm[1];
            vector<double> NFWf = interpolate2(z, M, C.zlist, C.Mlist, C.NFWlist);
            double c = NFWf[2];
            chost[jz][jM] = c; r200h[jz][jM] = NFWf[0] * c;

            // inverse radial CDF x(u) for the anti-biased profile
            //   dN/dshell ~ x^2/(1+c x)^2 * B(x),  B(x)=1/sqrt((x/0.54)^-2.5 + 1)
            const int Nx = 4000; vector<double> xs(Nx), cdf(Nx);
            double acc = 0.0;
            for (int i = 0; i < Nx; i++) {
                xs[i] = static_cast<double>(i) / (Nx - 1);
                double x = xs[i];
                double B = (x > 0.0) ? 1.0 / sqrt(pow(x / 0.54, -2.5) + 1.0) : 0.0;
                double w = x * x / pow(1.0 + c * x, 2.0) * B;
                if (i > 0) {
                    double xp = xs[i - 1];
                    double Bp = (xp > 0.0) ? 1.0 / sqrt(pow(xp / 0.54, -2.5) + 1.0) : 0.0;
                    double wp = xp * xp / pow(1.0 + c * xp, 2.0) * Bp;
                    acc += 0.5 * (w + wp) * (x - xp);
                }
                cdf[i] = acc;
            }
            vector<double> xu(Nu);
            for (int k = 0; k < Nu; k++) {
                double u = acc * (double)k / (Nu - 1);    // un-normalized target
                int lo = 0, hi = Nx - 1;
                while (hi - lo > 1) { int mid = (lo + hi) / 2; if (cdf[mid] < u) lo = mid; else hi = mid; }
                double t = (u - cdf[lo]) / (cdf[hi] - cdf[lo] + 1e-300);
                xu[k] = xs[lo] + t * (xs[hi] - xs[lo]);
            }
            invRad[jz][jM] = xu;

            if (kappathr_host > 0.0) {
                buildWsubBin(C, zs, jz, jM, kappathr_host);
            }
        }
    }
    built = true;
}

// Wsub tables for one (jz, jM) bin: Campbell mean/std of the UNRESOLVED clumps
// (psi < psi_lo(y), the exact addClumps floor) at ray impact parameter y, integrated
// over the true clump-ray distance d with the projected anti-biased radial profile:
//   mu(y)   = int dlnpsi dN/dlnpsi [1 - keep] int d^2x sigma2D(x) kappa_c(d)
//   s2(y)   = same with kappa_c^2
// plus the full bound fraction fsb for the model-3 host reduction. Quadrature grids
// are coarse (few-% on a term whose variance share is ~1%); the validated reference
// is playground/analytic/wsub_partition_proof.py.
void Subhalo::buildWsubBin(cosmology &C, double zs, int jz, int jM, double kappathr_host) {
    const double zl = C.zlist[jz], M = C.Mlist[jM];
    const double g = gnorm[jz][jM];
    if (g <= 0.0) return;
    const double rmaxH = rmaxfNFW(C, zs, zl, M, kappathr_host);
    if (rmaxH <= 0.0) return;                      // host never explicitly encountered
    const double Sigmac = Sigmacf(C, zs, zl);
    const double r200 = r200h[jz][jM];
    const double c = chost[jz][jM];

    const double psi_min = m_floor / M;            // same lower bound as brute sampling
    if (psi_min >= psi_max) return;

    // full bound fraction over [psi_min, psi_max] (exp-cutoff SHMF, host reduction)
    const double s_m = (1.0 + alpha) / omega;
    double fs_b = g / (omega * pow(beta, s_m)) *
        (gsl_sf_gamma_inc(s_m, beta * pow(psi_min, omega)) -
         gsl_sf_gamma_inc(s_m, beta * pow(psi_max, omega)));
    fsb[jz][jM] = std::max(0.0, std::min(0.95, fs_b));

    // projected anti-biased surface density sigma2D(R), cell-centered uniform Rhat grid
    const int NR = 128, NUa = 96;
    vector<double> p2(NR, 0.0);
    auto p3 = [&](double x) {
        if (x <= 0.0 || x > 1.0) return 0.0;
        double B = 1.0 / sqrt(pow(x / 0.54, -2.5) + 1.0);
        return x * x / pow(1.0 + c * x, 2.0) * B;
    };
    double p2norm = 0.0;
    for (int i = 0; i < NR; i++) {
        double Rh = (i + 0.5) / NR;
        double umax = sqrt(std::max(1.0 - Rh * Rh, 0.0));
        double du = umax / (NUa - 1), acc2 = 0.0;
        for (int k = 0; k < NUa; k++) {
            double u = k * du;
            double w = (k == 0 || k == NUa - 1) ? 0.5 : 1.0;
            acc2 += w * p3(sqrt(Rh * Rh + u * u)) * Rh / (Rh * Rh + u * u);
        }
        p2[i] = acc2 * du;
        p2norm += p2[i] / NR;
    }
    if (p2norm <= 0.0) return;
    // sigma2D at physical separation s [kpc]: p2(s/r200) / (2 pi (s/r200) r200^2)
    auto sig2d = [&](double s) {
        double Rh = s / r200;
        if (Rh >= 1.0) return 0.0;
        double t = Rh * NR - 0.5;
        int i = (int)std::floor(t);
        double p;
        if (i < 0) p = p2[0] * (Rh * NR / 0.5);   // p2 ~ Rhat near 0
        else if (i >= NR - 1) p = p2[NR - 1];
        else p = p2[i] + (t - i) * (p2[i + 1] - p2[i]);
        p /= p2norm;
        return p / (2.0 * PI * std::max(Rh, 1e-12) * r200 * r200);
    };

    // grids
    const int Ny = NyW, Nd = 48, Nth = 32, Nm = 32;
    const double y0 = 0.05, y1 = std::max(rmaxH, 2.0 * y0);
    const double ly0 = log(y0), dly = (log(y1) - ly0) / (Ny - 1);
    lyW[jz][jM] = {ly0, dly};
    const double d0 = 0.05, d1 = rmaxH + r200;
    const double dld = (log(d1) - log(d0)) / (Nd - 1);
    vector<double> dg(Nd);
    for (int id = 0; id < Nd; id++) dg[id] = exp(log(d0) + id * dld);

    // fd[iy][id] = [int_0^{2pi} sigma2D(s) dtheta] * d^2 * dlnd   (log-d quadrature)
    vector<double> fd(Ny * Nd);
    const double dth = PI / (Nth - 1);
    for (int iy = 0; iy < Ny; iy++) {
        double y = exp(ly0 + iy * dly);
        for (int id = 0; id < Nd; id++) {
            double d = dg[id], ang = 0.0;
            for (int it = 0; it < Nth; it++) {
                double w = (it == 0 || it == Nth - 1) ? 0.5 : 1.0;
                double s2 = y * y + d * d - 2.0 * y * d * cos(it * dth);
                ang += w * sig2d(sqrt(std::max(s2, 0.0)));
            }
            double wd = (id == 0 || id == Nd - 1) ? 0.5 : 1.0;
            fd[iy * Nd + id] = 2.0 * ang * dth * d * d * dld * wd;
        }
    }

    // clump kernels and J-integrals on the log-psi grid
    const double lp0 = log(psi_min), dlp = (log(psi_max) - lp0) / (Nm - 1);
    vector<double> J1(Nm * Ny), J2(Nm * Ny), wm(Nm), lpg(Nm);
    vector<double> k1(Nd);
    for (int im = 0; im < Nm; im++) {
        double lp = lp0 + im * dlp;
        lpg[im] = lp;
        double psi = exp(lp), m = psi * M;
        wm[im] = g * pow(psi, alpha) * exp(-beta * pow(psi, omega));   // dN/dlnpsi
        double rs_c, rhos_c;
        interpolateNFWMass(C, jz, m, std::log(m), log_Mmin, inv_dlogM, rs_c, rhos_c);
        double kappa0 = kappa0NFW(rs_c, rhos_c, Sigmac);
        for (int id = 0; id < Nd; id++)
            k1[id] = 2.0 * kappa0 * FgNFW(std::max(dg[id] / rs_c, 1.0e-12))[0];
        for (int iy = 0; iy < Ny; iy++) {
            double a1 = 0.0, a2 = 0.0;
            const double *f = &fd[iy * Nd];
            for (int id = 0; id < Nd; id++) {
                a1 += f[id] * k1[id];
                a2 += f[id] * k1[id] * k1[id];
            }
            J1[im * Ny + iy] = a1;
            J2[im * Ny + iy] = a2;
        }
    }

    // per-y mass integral over the UNRESOLVED band [psi_min, psi_lo(y)), with the
    // exact addClumps floor rule and a partial last trapezoid cell at the cut
    muW[jz][jM].assign(Ny, 0.0);
    sW[jz][jM].assign(Ny, 0.0);
    const vector<double> &rth = r_thr[jz];
    for (int iy = 0; iy < Ny; iy++) {
        double y = exp(ly0 + iy * dly);
        int jlo = static_cast<int>(std::lower_bound(rth.begin(), rth.end(), y) - rth.begin());
        double psi_lo = (jlo >= C.NM) ? psi_max : std::max(C.Mlist[jlo], C.Mmin) / M;
        psi_lo = std::min(psi_lo, psi_max);
        if (psi_lo <= psi_min) continue;           // everything resolved at this y
        double lc = log(psi_lo);
        double mu = 0.0, s2 = 0.0;
        for (int im = 0; im < Nm - 1; im++) {
            double la = lpg[im], lb = lpg[im + 1];
            double F1a = wm[im] * J1[im * Ny + iy],     F1b = wm[im + 1] * J1[(im + 1) * Ny + iy];
            double F2a = wm[im] * J2[im * Ny + iy],     F2b = wm[im + 1] * J2[(im + 1) * Ny + iy];
            if (lc >= lb) {                        // full cell
                mu += 0.5 * (F1a + F1b) * dlp;
                s2 += 0.5 * (F2a + F2b) * dlp;
            } else if (lc > la) {                  // partial cell up to the cut
                double t = (lc - la) / dlp;
                double F1c = F1a + t * (F1b - F1a);
                double F2c = F2a + t * (F2b - F2a);
                mu += 0.5 * (F1a + F1c) * (lc - la);
                s2 += 0.5 * (F2a + F2c) * (lc - la);
                break;
            } else break;
        }
        muW[jz][jM][iy] = mu;
        sW[jz][jM][iy] = sqrt(std::max(s2, 0.0));
    }
}

// model-3 lookup: linear interpolation on the uniform log-y grid, clamped at both ends.
void Subhalo::wsubTerm(int jz, int jM, double r, double &mu, double &sigma) const {
    mu = 0.0; sigma = 0.0;
    const vector<double> &m_ = muW[jz][jM];
    if (m_.empty()) return;
    const vector<double> &s_ = sW[jz][jM];
    const double ly0 = lyW[jz][jM][0], dly = lyW[jz][jM][1];
    double t = (std::log(std::max(r, 1.0e-12)) - ly0) / dly;
    if (t <= 0.0) { mu = m_.front(); sigma = s_.front(); return; }
    if (t >= (int)m_.size() - 1) { mu = m_.back(); sigma = s_.back(); return; }
    int i = (int)t;
    double f = t - i;
    mu = m_[i] + f * (m_[i + 1] - m_[i]);
    sigma = s_[i] + f * (s_[i + 1] - s_[i]);
}

// Add one host's discrete subhalos to (kappa, gamma1, gamma2).
// Only clumps above the dynamic floor are resolved. The floor is the inverse of the
// monotone clump-reach r_thr(m), keyed to the host-center distance r; lensing.cpp uses
// the same floor when reducing the smooth host mass.
// Option B (default): host is already reduced to (1-f_s_res)*M in lensing.cpp; bare clumps added here.
// Option A (legacy):  host stays at full M; each clump subtracts m*gslope to remove displaced mass.
int Subhalo::addClumps(cosmology &C, int jz, int jM, double zl, double M, double Sigmac,
                        double r, double phi, rgen &mt,
                        double &kappa, double &gamma1, double &gamma2,
                        int subhalo_model, bool subhalo_brute,
                        int threads, int parallel_threshold) {
    double g = gnorm[jz][jM];
    if (g <= 0.0) return 0;

    double psi_lo = 0.0;
    if (subhalo_brute) {
        psi_lo = m_floor / M;
    } else {
        // dynamic floor: smallest clump whose reach r_thr(m) >= r, keyed to the HOST-CENTER
        // distance (MUST match lensing.cpp's f_s_res). This under-resolves clumps that land
        // closer to the ray than r; a worst-case max(0, r - r200) criterion is NOT usable
        // instead — NFW kappa diverges on-axis, so it degenerates to brute force for every
        // ray inside r200. The bias is absorbed empirically by lowering subhalo_factor
        // (rescales kappa_thr for clumps) until the kappa-variance converges to brute:
        // see scripts/subhalo_factor_convergence.py.
        const vector<double> &rth = r_thr[jz];
        int jlo = static_cast<int>(std::lower_bound(rth.begin(), rth.end(), r) - rth.begin());
        if (jlo >= C.NM) return 0;                       // nothing reaches kappa_thr at distance r
        psi_lo = std::max(C.Mlist[jlo], C.Mmin) / M;
    }
    if (psi_lo >= psi_max) return 0;                 // no resolved clumps in this host

    // resolved count for this sightline (exp ~ 1 over clump range -> power law)
    const double pa_lo = pow(psi_lo, alpha);
    const double pa_hi = pow(psi_max, alpha);
    double Nres = (g / alpha) * (pa_hi - pa_lo);
    if (Nres <= 0.0) return 0;
    std::poisson_distribution<int> pois(Nres);
    int Nc = pois(mt);
    if (Nc <= 0) return 0;

    // removal slope: d kappa_host(r)/dM at this impact (full host stays smooth)
    const double logM = std::log(M);
    double gslope = 0.0;
    if (subhalo_model == 0) {
        const double Me = 1.02 * M, logMe = std::log(Me);
        double rsH, rhosH, rsHe, rhosHe;
        interpolateNFWMass(C, jz, M,  logM,  log_Mmin, inv_dlogM, rsH,  rhosH);
        interpolateNFWMass(C, jz, Me, logMe, log_Mmin, inv_dlogM, rsHe, rhosHe);
        double kH  = 2.0 * kappa0NFW(rsH,  rhosH,  Sigmac) * FgNFW(r / rsH )[0];
        double kHe = 2.0 * kappa0NFW(rsHe, rhosHe, Sigmac) * FgNFW(r / rsHe)[0];
        gslope = (kHe - kH) / (Me - M);           // = d kappa_host / dM
    }

    const vector<double> &xcdf = invRad[jz][jM];
    const double r200 = r200h[jz][jM];
    const double rcos = r * cos(phi), rsin = r * sin(phi);
    const double inv_alpha = 1.0 / alpha;

    if (subhalo_model != 0 && threads > 1 && Nc >= parallel_threshold) {
        int nthreads = std::min(threads, Nc);
        vector<ClumpAccum> accs(nthreads);
        vector<rgen> mts;
        mts.reserve(nthreads);
        for (int t = 0; t < nthreads; t++) mts.emplace_back(mt());

        vector<std::thread> workers;
        workers.reserve(nthreads);
        const double d_cut = std::numeric_limits<double>::infinity();
        for (int t = 0; t < nthreads; t++) {
            int begin = (Nc * t) / nthreads;
            int end = (Nc * (t + 1)) / nthreads;
            workers.emplace_back([&, t, begin, end]() {
                accs[t] = evalClumpRange(C, jz, M, Sigmac, rcos, rsin, r200, xcdf, Nu,
                                         alpha, beta, omega, psi_lo, psi_max, log_Mmin,
                                         inv_dlogM, d_cut, begin, end, mts[t]);
            });
        }
        for (auto &worker : workers) worker.join();
        for (const auto &acc : accs) {
            kappa += acc.kappa;
            gamma1 += acc.gamma1;
            gamma2 += acc.gamma2;
        }
        return Nc;
    }

    for (int k = 0; k < Nc; k++) {
        // clump mass: power-law inverse-CDF proposal over [psi_lo, psi_max], thinned by the
        // SHMF exponential cutoff (Poisson thinning => exact dN/dpsi incl. exp(-beta psi^omega))
        double u = randomreal(0.0, 1.0, mt);
        double psi = pow(pa_lo + u * (pa_hi - pa_lo), inv_alpha);
        if (randomreal(0.0, 1.0, mt) > exp(-beta * pow(psi, omega))) continue;
        double m = psi * M;
        double log_m = std::log(m);

        // clump 3D host-centric radius from the anti-biased profile, projected to 2D
        double ur = randomreal(0.0, 1.0, mt);
        double tt = ur * (Nu - 1);
        int i = (int)tt; if (i >= Nu - 1) i = Nu - 2;
        double x = xcdf[i] + (tt - i) * (xcdf[i + 1] - xcdf[i]);
        double r3d = x * r200;
        double cth = randomreal(-1.0, 1.0, mt);
        double psaz = randomreal(0.0, 2.0 * PI, mt);
        double R2d = r3d * sqrt(1.0 - cth * cth);

        // ray -> clump separation in the lens plane
        double dx = rcos - R2d * cos(psaz);
        double dy = rsin - R2d * sin(psaz);
        double d = sqrt(dx * dx + dy * dy);
        double inv_d = (d > 1e-30) ? 1.0 / d : 0.0;
        double cos_phid = dx * inv_d, sin_phid = dy * inv_d;

        // circular NFW clump kappa/gamma, minus the displaced smooth mass (m * gslope)
        double rs_c, rhos_c;
        interpolateNFWMass(C, jz, m, log_m, log_Mmin, inv_dlogM, rs_c, rhos_c);
        double kappa0_c = kappa0NFW(rs_c, rhos_c, Sigmac);
        double x_c = std::max(d / rs_c, 1.0e-12);
        auto Fg = FgNFW(x_c);
        double kappa_c = 2.0 * kappa0_c * Fg[0];
        double gamma_c = 2.0 * kappa0_c * safeNFWGammaCore(x_c, Fg);

        if (subhalo_model == 0) {
            kappa += kappa_c - m * gslope;
        } else {
            kappa += kappa_c;
        }
        gamma1 += cos_phid * gamma_c;
        gamma2 += sin_phid * gamma_c;
    }
    return Nc;
}
