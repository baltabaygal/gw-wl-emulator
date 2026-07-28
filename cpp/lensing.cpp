#include "cosmology.h"
#include "lensing.h"
// #include <functional>
#include <functional>  // For std::function
#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_bessel.h>

double Sigmacf(cosmology &C, double zs, double zl) {
    // angular diameter distances
    double DsA = C.DL(zs)/pow(1+zs,2.0);
    double DlA = C.DL(zl)/pow(1+zl,2.0);
    double DlsA = DsA - DlA*(1+zl)/(1+zs);
    
    // Sigma_c
    return 2.08871e16*DsA/(4.0*PI*DlA*DlsA);
}


/* ---------------------------------------------------------------------------------------------------------------------------------------------- */
/*                                                              NFW halos                                                                         */
/* ---------------------------------------------------------------------------------------------------------------------------------------------- */

// from astro-ph/0608153
array<double,2> FgNFW(double x) {
    if (x > 1) {
        double t = atan(sqrt((x-1)/(1+x))) / sqrt(x*x - 1);
        return {(1 - 2*t) / (x*x - 1), 2*t + log(x/2)};
    }
    if (x < 1) {
        double t = atanh(sqrt((1-x)/(1+x))) / sqrt(1 - x*x);
        return {(1 - 2*t) / (x*x - 1), 2*t + log(x/2)};
    }
    return {1.0/3.0, 1 + log(0.5)};
}

static inline double safeNFWGammaCore(double x, const array<double, 2> &Fg) {
    if (x < 1.0e-4) return 0.5;
    return 2.0 * Fg[1] / (x * x) - Fg[0];
}

double kappa0NFW(double rs, double rhos, double Sigmac) {
    return rs*rhos/Sigmac;
}

// fix ellipticity using the fit of astro-ph/0508497
double epsilonNFW(cosmology &C, double z, double M) {
    double s = 0.54*pow(M/exp(interpolate(z, C.logMcharlist)), -0.05);
    return max(0.0, (1.0-s)/(1.0+s));
}

// kappa and gamma for pseudo elliptical NFW halos
array<double,2> kappagammaNFWeps(double epsilon, double kappa0, double x, double phi) {
    double a1eps = 1.0-epsilon;
    double a2eps = 1.0+epsilon;
    double x1eps = sqrt(a1eps)*cos(phi)*x;
    double x2eps = sqrt(a2eps)*sin(phi)*x;
    double xeps = max(sqrt(pow(x1eps,2.0) + pow(x2eps,2.0)), 1.0e-12);
    double phieps = atan2(x2eps, x1eps);
    
    auto Fg = FgNFW(xeps);
    double kappaeps0 = 2.0*kappa0*Fg[0];
    double gammaeps0 = 2.0*kappa0*safeNFWGammaCore(xeps, Fg);

    double kappaeps = kappaeps0 + epsilon*cos(2.0*phieps)*gammaeps0;
    double gammaeps2 = pow(gammaeps0,2.0) + 2.0*epsilon*cos(2.0*phieps)*gammaeps0*kappaeps0 + pow(epsilon,2.0)*(pow(kappaeps0,2.0) - pow(cos(2.0*phieps)*gammaeps0,2.0));
    double gammaeps = sqrt(max(0.0, gammaeps2));
    
    return {kappaeps, gammaeps};
}
array<double,2> kappagammaNFW(cosmology &C, double zs, double zl, double r, double M, double phi, double epsilon) {
    double Sigmac = Sigmacf(C, zs, zl);
    
    // NFW scale radius and density
    vector<double> NFWparams = interpolate2(zl, M, C.zlist, C.Mlist, C.NFWlist);
    double rs = NFWparams[0];
    double rhos = NFWparams[1];
    
    double kappa0 = kappa0NFW(rs, rhos, Sigmac);
    
    return kappagammaNFWeps(epsilon, kappa0, r/rs, phi);
}

// maximal r so that kappa_NFW > kappa_thr
double rmaxfNFW(cosmology &C, double zs, double zl, double M, double kappathr) {
    double rmax;
    double logr1 = log(1.0e-6), logr2 = log(1.0e6);
    if (kappagammaNFW(C, zs, zl, exp(logr1), M, 0.0, 0.0)[0] > kappathr) {
        while (logr2-logr1 > 0.02) {
            rmax = exp((logr2+logr1)/2.0);
            if (kappagammaNFW(C, zs, zl, rmax, M, 0.0, 0.0)[0] > kappathr) {
                logr1 = log(rmax);
            } else {
                logr2 = log(rmax);
            }
        }
        rmax = exp((logr2+logr1)/2.0);
    } else {
        rmax = 0.0;
    }
    return rmax;
}

// number of halos with kappa_NFW > kappa_thr in each z and M bin
vector<vector<vector<double> > > deltaNhfNFW(cosmology &C, double zs, double kappathr) {
    vector<vector<vector<double> > > dNh(C.Nz, vector<vector<double> > (C.NM, vector<double> (3, 0.0)));
    double zl, dz, M, Mb, dlnM, dndlnM, rmax, sigma, sigmab;
    for (int jz = 1; jz < C.Nz; jz++) {
        zl = C.zlist[jz];
        dz = zl - C.zlist[jz-1];
        if (zl < zs) {
            for (int jM = 1; jM < C.NM; jM++) {
                M = C.Mlist[jM];
                sigma = C.sigmalist[jM][1];
                
                dlnM = log(M) - log(C.Mlist[jM-1]);
                dndlnM = C.HMFlist[jz][jM][0];
                rmax = rmaxfNFW(C, zs, zl, M, kappathr);
                dNh[jz][jM][0] = CLIGHT*PI*pow((1.0+zl)*rmax,2.0)/C.Hz(zl)*dndlnM*dlnM*dz;
                
                Mb = 2.0*PI*pow(rmax,2.0)*(C.dc(zl)-C.dc(zl-dz))*C.rhoM0;
                sigmab = interpolate(Mb, C.sigmalist);
                
                // see Baumann (5.129)
                dNh[jz][jM][1] = C.Dg(zl)*sigmab*C.halobias(zl, sigma);
                
                dNh[jz][jM][2] = rmax;
            }
        }
    }
    return dNh;
}

// total number of halos with kappa_NFW > kappa_thr
double NhfNFW(cosmology &C, double zs, double kappathr) {
    double Nh = 0.0;
    double zl, dz, M, dlnM, dndlnM, rmax;
    for (int jz = 1; jz < C.Nz; jz++) {
        zl = C.zlist[jz];
        dz = zl - C.zlist[jz-1];
        if (zl < zs) {
            for (int jM = 1; jM < C.NM; jM++) {
                M = C.Mlist[jM];
                dlnM = log(M) - log(C.Mlist[jM-1]);
                dndlnM = C.HMFlist[jz][jM][0];
                rmax = rmaxfNFW(C, zs, zl, M, kappathr);
                Nh += CLIGHT*PI*pow((1.0+zl)*rmax,2.0)/C.Hz(zl)*dndlnM*dlnM*dz;
            }
        }
    }
    return Nh;
}

// variance of kappa from weak lenses
double sigmakappaW(cosmology &C, double zs, double kappathr, double eps_floor) {
    double kappa2 = 0.0;
    double zl, dz, M, dlnM, dndlnM, r, kappar;
    int Nr = 100;
    double dlnr = 0.01;
    double Edlnr = exp(dlnr);
    for (int jz = 1; jz < C.Nz; jz++) {
        zl = C.zlist[jz];
        dz = zl - C.zlist[jz-1];
        if (zl < zs) {
            for (int jM = 1; jM < C.NM; jM++) {
                M = C.Mlist[jM];
                dlnM = log(M) - log(C.Mlist[jM-1]);
                dndlnM = C.HMFlist[jz][jM][0];
                                
                r = rmaxfNFW(C, zs, zl, M, kappathr);
                if (r == 0.0) {
                    r = 1.0e-6;
                }

                double SigmacW = Sigmacf(C, zs, zl);
                vector<double> NFWpW = interpolate2(zl, M, C.zlist, C.Mlist, C.NFWlist);
                double rsW = NFWpW[0];
                double kappa0W = kappa0NFW(rsW, NFWpW[1], SigmacW);

                kappar = kappathr;
                while (kappar > eps_floor*kappathr) {
                    kappar = kappagammaNFWeps(0.0, kappa0W, r/rsW, 0.0)[0];
                    // Campbell's theorem for Poisson-distributed halo counts: Var = int n kappa^2,
                    // with log-annulus area element d(pi r^2) = 2 pi r^2 dlnr
                    // (see docs/sigmakappaw_measure_note.md)
                    kappa2 += CLIGHT*2.0*PI*pow((1.0+zl)*r,2.0)/C.Hz(zl)*dndlnM*pow(kappar,2.0)*dlnr*dlnM*dz;
                    r = r*Edlnr;
                }
            }
        }
    }
    return sqrt(kappa2);
}


/* ---------------------------------------------------------------------------------------------------------------------------------------------- */
/*                                                           Cylindrical filaments                                                                */
/* ---------------------------------------------------------------------------------------------------------------------------------------------- */

// axis in the lens plane
double kappa0CYL(double rs, double rhos, double Sigmac) {
    return rs*rhos/Sigmac;
}

// uniform density inside the cylinder
double kappaCYL1(double r, double rs, double kappa0) {
    if (r > rs) {
        return 0.0;
    }
    return kappa0*2.0*sqrt(1.0-pow(r/rs,2.0));
}
double gammaCYL1(double r, double rs, double kappa0) {
    return kappaCYL1(r, rs, kappa0);
}

// 1/(1+(r/r_s)^2) profile inside the cylinder
double kappaCYL2(double r, double rs, double kappa0) {
    if (r > rs) {
        return 0.0;
    }
    return kappa0*2.0*PI/sqrt(1.0+pow(r/rs,2.0));
}
double gammaCYL2(double r, double rs, double kappa0) {
    return kappa0*4.0*PI*(sqrt(1.0+pow(r/rs,2.0)) - 1.0)*pow(rs/r,2.0) - kappaCYL2(r, rs, kappa0);
}

array<double,2> kappagammaCYL(cosmology &C, double zs, double zl, double r, double M, double phi) {
    double Sigmac = Sigmacf(C, zs, zl);
    
    // cylinder radius and length, M = mass inside radius r_s (see astro-ph/0406665)
    double rs = 1000.0*pow(M/1.0e14, 1.0/3.0);
    double L = 20000.0*pow(M/1.0e14, 1.0/3.0);
    
    // average density inside radius r_s = 10*rho_c, rough approximation of rho_s density when it is not in the lens plane
    double rhos = 14.4*C.rhoc*L/max(2.0*rs, abs(cos(phi))*L);
    
    double kappa0 = kappa0CYL(rs, rhos, Sigmac);
    
    return {kappaCYL2(r, rs, kappa0), gammaCYL2(r, rs, kappa0)};
}

// maximal r so that kappa_CYL > kappa_thr
double rmaxfCYL(cosmology &C, double zs, double zl, double M, double kappathr) {
    double rmax;
    double logr1 = log(1.0e-6), logr2 = log(1.0e6);
    if (kappagammaCYL(C, zs, zl, exp(logr1), M, 0.0)[0] > kappathr) {
        while (logr2-logr1 > 0.02) {
            rmax = exp((logr2+logr1)/2.0);
            if (kappagammaCYL(C, zs, zl, rmax, M, 0.0)[0] > kappathr) {
                logr1 = log(rmax);
            } else {
                logr2 = log(rmax);
            }
        }
        rmax = exp((logr2+logr1)/2.0);
    } else {
        rmax = 0.0;
    }
    return rmax;
}

// number of halos with kappa_NFW > kappa_thr in each z and M bin
vector<vector<vector<double> > > deltaNhfCYL(cosmology &C, double zs, double kappathr) {
    vector<vector<vector<double> > > dNh(C.Nz, vector<vector<double> > (C.NM, vector<double> (3, 0.0)));
    double zl, dz, M, Mb, dlnM, dndlnM, rmax, sigma, sigmab;
    for (int jz = 1; jz < C.Nz; jz++) {
        zl = C.zlist[jz];
        dz = zl - C.zlist[jz-1];
        if (zl < zs) {
            for (int jM = 1; jM < C.NM; jM++) {
                M = C.Mlist[jM];
                sigma = C.sigmalist[jM][1];
                
                dlnM = log(M) - log(C.Mlist[jM-1]);
                dndlnM = C.FMFlist[jz][jM];
                rmax = rmaxfCYL(C, zs, zl, M, kappathr);
                dNh[jz][jM][0] = CLIGHT*PI*pow((1.0+zl)*rmax,2.0)/C.Hz(zl)*dndlnM*dlnM*dz;
                
                Mb = 2.0*PI*pow(rmax,2.0)*(C.dc(zl)-C.dc(zl-dz))*C.rhoM0;
                sigmab = interpolate(Mb, C.sigmalist);
                
                // see Baumann (5.129)
                dNh[jz][jM][1] = C.Dg(zl)*sigmab*C.halobias(zl, sigma);
                
                dNh[jz][jM][2] = rmax;
            }
        }
    }
    return dNh;
}

// number of filaments with kappa_CYL > kappa_thr
double NhfCYL(cosmology &C, double zs, double kappathr) {
    double Nh = 0.0;
    double zl, dz, M, dlnM, dndlnM, rmax;
    for (int jz = 1; jz < C.Nz; jz++) {
        zl = C.zlist[jz];
        dz = zl - C.zlist[jz-1];
        if (zl < zs) {
            for (int jM = 1; jM < C.NM; jM++) {
                M = C.Mlist[jM];
                dlnM = log(M) - log(C.Mlist[jM-1]);
                dndlnM = C.FMFlist[jz][jM];
                rmax = rmaxfCYL(C, zs, zl, M, kappathr);
                Nh += CLIGHT*PI*pow((1.0+zl)*rmax,2.0)/C.Hz(zl)*dndlnM*dlnM*dz;
            }
        }
    }
    return Nh;
}


/* ---------------------------------------------------------------------------------------------------------------------------------------------- */
/*                                       Correlated 1D environment field for the bias layer (bias_model = 1)                                       */
/* ---------------------------------------------------------------------------------------------------------------------------------------------- */
//
// Replaces the legacy iid per-(jz,jM) log-normal count modulation with segment
// averages of ONE Gaussian field delta_1D(chi) along the LOS (2026-07-16;
// design: docs/bias_field_design_note.md, prototype:
// playground/bias_field/field1d_prototype.ipynb).
//
//   P_1D(kpar; Rperp) = (1/2pi) int dkperp kperp P0(sqrt(kpar^2+kperp^2))
//                       * Wdisk(kperp Rperp)^2            (KP91 eq. 3.8)
//   modes k_n = 2 pi n / L, L = 1.05 chi(z_s), n = 1..N_max = L/Rperp
//   (floored at 4); per-mode field variance 2 P_1D(k_n)/L.
//
// The per-z-shell SEGMENT AVERAGES of the periodic truncated mode sum form a
// Gaussian vector with covariance
//   Cov_ij = sum_n (2/L) P_1D(k_n) cos(k_n (c_i - c_j)) snc_i(k_n) snc_j(k_n),
//   snc_i(k) = sin(k L_i / 2)/(k L_i / 2),
// which is realized exactly (equal in law to drawing the modes) by a Cholesky
// factor: dbar = chol * g, g ~ N(0,1)^n. The mode sum is evaluated EXACTLY for
// all N_max (a log-k quadrature was tried and rejected: it cannot resolve the
// ~1e4 sinc oscillations and erred at the 2.5% level on sigma — see
// playground/bias_field/validate_field_covariance.py). Trig recursion keeps
// the worst case (N_max ~ 1e6 at Rperp ~ 8 kpc) at a few seconds, one-off.
// P0(k) is the GROWTH-FREE linear power (cosmology::Pk0) — the same
// normalization as the sigmalist tables; growth enters per cell via
// b(M,z_l) Dg(z_l), exactly as in the legacy layer (Baumann 5.129 convention).

// Per-cell sub-threshold Campbell moments for the weak arm (bias_weak):
//   m[jz][jM] = int nbar kappa,  v[jz][jM] = int nbar kappa^2
// over the sub-threshold annuli (r from rmax outward to the eps floor), with
// the SAME integrand, log-annulus measure, stepping, and floor semantics as
// sigmakappaW — so that sum_{jz,jM} v == sigmakappaW^2 (asserted at build
// time; the only difference vs sigmakappaW is per-cell bookkeeping, i.e.
// summation associativity).
static void weakMomentsNFW(cosmology &C, double zs, double kappathr, double eps_floor,
                           vector<vector<double> > &m, vector<vector<double> > &v) {
    m.assign(C.Nz, vector<double>(C.NM, 0.0));
    v.assign(C.Nz, vector<double>(C.NM, 0.0));
    double zl, dz, M, dlnM, dndlnM, r, kappar;
    double dlnr = 0.01;
    double Edlnr = exp(dlnr);
    for (int jz = 1; jz < C.Nz; jz++) {
        zl = C.zlist[jz];
        dz = zl - C.zlist[jz-1];
        if (zl < zs) {
            for (int jM = 1; jM < C.NM; jM++) {
                M = C.Mlist[jM];
                dlnM = log(M) - log(C.Mlist[jM-1]);
                dndlnM = C.HMFlist[jz][jM][0];

                r = rmaxfNFW(C, zs, zl, M, kappathr);
                if (r == 0.0) {
                    r = 1.0e-6;
                }

                double SigmacW = Sigmacf(C, zs, zl);
                vector<double> NFWpW = interpolate2(zl, M, C.zlist, C.Mlist, C.NFWlist);
                double rsW = NFWpW[0];
                double kappa0W = kappa0NFW(rsW, NFWpW[1], SigmacW);

                kappar = kappathr;
                while (kappar > eps_floor*kappathr) {
                    kappar = kappagammaNFWeps(0.0, kappa0W, r/rsW, 0.0)[0];
                    double pref = CLIGHT*2.0*PI*pow((1.0+zl)*r,2.0)/C.Hz(zl)*dndlnM*dlnr*dlnM*dz;
                    m[jz][jM] += pref*kappar;
                    v[jz][jM] += pref*pow(kappar,2.0);
                    r = r*Edlnr;
                }
            }
        }
    }
}

// Smoothing window W~(x)^2 for the ISOTROPIC bias-field windows (bias_window
// 1 = spherical top-hat, 2 = Gaussian), x = |k| R. Window 0 (transverse disk)
// acts on k_perp alone and stays precomputed on the k_perp grid in build().
static inline double biasWindow2(double x, int window) {
    if (window == 2) {                                 // Gaussian, exp(-x^2/2)
        double W = exp(-0.5*x*x);
        return W*W;
    }
    // spherical top-hat, 3 (sin x - x cos x)/x^3; the closed form loses
    // ~3e-16/x^2 to cancellation, so use W = 1 - x^2/10 + x^4/280 below 1e-2
    // (next term x^6/15120 < 1e-16 there).
    double W;
    if (x < 1.0e-2) {
        double x2 = x*x;
        W = 1.0 - x2/10.0*(1.0 - x2/28.0);
    } else {
        W = 3.0*(sin(x) - x*cos(x))/(x*x*x);
    }
    return W*W;
}

struct BiasField1D {
    int n = 0;                        // number of z-shells (jz = 1 .. n, zlist[jz] < zs)
    double Rperp = 0.0, L = 0.0;
    int window = 0;                   // cfg.bias_window (0 disk, 1 top-hat, 2 Gaussian)
    long Nmax = 0;                    // mode count L/Rperp (>= 4)
    std::vector<double> sig2;         // Cov_ii per shell (z=0 field, segment-averaged)
    std::vector<double> chol;         // lower-triangular Cholesky of Cov, row-major n*n

    // ---- weak (sub-threshold) arm tables, built only when bias_weak (see
    // lensing.h): per shell i, on a uniform delta grid over +-DGRID_SIG
    // sigma_i, log tables of T_i(delta) = sum_M m_iM lambda_iM(delta) and
    // V_i(delta) = sum_M v_iM lambda_iM(delta); S_i = T_i - msum_i. Both are
    // positive exponential mixtures, near-linear in log — linear interp on
    // the log tables stays accurate even at b Dg sigma ~ O(1). Outside the
    // grid delta is clamped (Gaussian field: P(|d| > 6 sigma) ~ 2e-9).
    static constexpr int NGRID_W = 193;
    static constexpr double DGRID_SIG = 6.0;
    bool has_weak = false;
    std::vector<double> msum;         // per shell: sum_M m_iM
    std::vector<double> lnT, lnV;     // row-major n*NGRID_W

    // shell index for a given jz (cells at jz have chi in [dc(z_{jz-1}), dc(z_jz)])
    inline int shell(int jz) const { return (jz >= 1 && jz <= n) ? jz - 1 : -1; }

    // weak-arm lookup: conditional mean shift S and Campbell variance V of
    // shell i at field value delta (linear interp of the log tables)
    inline void weakSV(int i, double delta, double &S, double &V) const {
        double si = sqrt(sig2[i]);
        double half = DGRID_SIG*si;
        double x = std::min(std::max(delta, -half), half);
        double t = (x + half)/(2.0*half)*(NGRID_W - 1);
        int g = std::min(static_cast<int>(t), NGRID_W - 2);
        double f = t - g;
        const double *rT = &lnT[static_cast<size_t>(i)*NGRID_W];
        const double *rV = &lnV[static_cast<size_t>(i)*NGRID_W];
        S = exp((1.0 - f)*rT[g] + f*rT[g+1]) - msum[i];
        V = exp((1.0 - f)*rV[g] + f*rV[g+1]);
    }

    // Build the weak-arm tables. skappaW = sigmakappaW(...) with the SAME
    // kappathr/eps_floor — used for the sum_v == sigma_W^2 consistency gate.
    void buildWeak(cosmology &C, double zs, double kappathr, double eps_floor,
                   double skappaW) {
        if (n == 0) return;
        vector<vector<double> > mc, vc;
        weakMomentsNFW(C, zs, kappathr, eps_floor, mc, vc);
        double sv = 0.0;
        for (int jz = 0; jz < C.Nz; jz++)
            for (int jM = 0; jM < C.NM; jM++) sv += vc[jz][jM];
        if (fabs(sv - skappaW*skappaW) > 1.0e-9*skappaW*skappaW) {
            throw std::runtime_error("bias_weak: sum of per-cell v moments != sigma_W^2 "
                                     "(weakMomentsNFW drifted from sigmakappaW)");
        }
        msum.assign(n, 0.0);
        lnT.assign(static_cast<size_t>(n)*NGRID_W, 0.0);
        lnV.assign(static_cast<size_t>(n)*NGRID_W, 0.0);
        for (int i = 0; i < n; i++) {
            int jz = i + 1;
            double zl = C.zlist[jz];
            double Dgz = C.Dg(zl);
            double si = sqrt(sig2[i]);
            for (int g = 0; g < NGRID_W; g++) {
                double delta = (-DGRID_SIG + 2.0*DGRID_SIG*g/(NGRID_W - 1))*si;
                double T = 0.0, V = 0.0;
                for (int jM = 1; jM < C.NM; jM++) {
                    double mm = mc[jz][jM];
                    if (mm <= 0.0 && vc[jz][jM] <= 0.0) continue;
                    double a = Dgz*C.halobias(zl, C.sigmalist[jM][1]);
                    double lam = exp(a*delta - 0.5*a*a*sig2[i]);
                    T += mm*lam;
                    V += vc[jz][jM]*lam;
                }
                lnT[static_cast<size_t>(i)*NGRID_W + g] = log(std::max(T, 1.0e-300));
                lnV[static_cast<size_t>(i)*NGRID_W + g] = log(std::max(V, 1.0e-300));
            }
            for (int jM = 1; jM < C.NM; jM++) msum[i] += mc[jz][jM];
        }
        has_weak = true;
    }

    // win defaults to the legacy disk window so the out-of-tree probes that
    // include lensing.cpp (playground/*.cpp) keep compiling and keep their
    // pre-2026-07-20 behavior; production always passes cfg.bias_window.
    void build(cosmology &C, double zs, double Rp, int win = 0) {
        Rperp = Rp;
        window = win;
        // ---- shells
        std::vector<double> clo, chi_;
        for (int jz = 1; jz < C.Nz && C.zlist[jz] < zs; jz++) {
            clo.push_back(C.dc(C.zlist[jz-1]));
            chi_.push_back(C.dc(C.zlist[jz]));
        }
        n = static_cast<int>(clo.size());
        if (n == 0) return;

        L = 1.05*C.dc(zs);
        Nmax = std::max(4L, static_cast<long>(std::floor(L/Rperp)));
        if (Nmax > 5000000L) {
            throw std::invalid_argument("bias_Rperp too small: N_max = L/Rperp > 5e6 modes");
        }
        const double kmin = 2.0*PI/L;
        const double kmax = 2.0*PI*static_cast<double>(Nmax)/L;

        // ---- P_1D table (log-log interpolated; window 0 = disk via GSL J1 on
        // k_perp, precomputed here; windows 1/2 act on |k| and are evaluated
        // inside the k_par loop below). The k_perp grid is shared: the disk
        // window is the slowest-decaying of the three (W~^2 ~ x^-3 vs x^-4
        // top-hat and Gaussian), so its accepted range/resolution bounds them
        // (verified, data/results/bias_window/report.md).
        const int nkperp = 2048, nktab = 600;
        const double Rw = std::max(Rperp, 10.0);          // window floor, as in the prototype
        const double kperp_lo = 1.0e-9, kperp_hi = 60.0/Rw;
        const double dlnkp = log(kperp_hi/kperp_lo)/(nkperp - 1);
        std::vector<double> kperp(nkperp), W2(nkperp);
        for (int i = 0; i < nkperp; i++) {
            kperp[i] = kperp_lo*exp(dlnkp*i);
            double x = kperp[i]*Rperp;
            double Wd = (x < 1.0e-6) ? 1.0 : 2.0*gsl_sf_bessel_J1(x)/x;
            W2[i] = Wd*Wd;
        }
        std::vector<double> lktab(nktab), lPtab(nktab);
        const double lk0 = log(0.5*kmin), lk1 = log(kmax);
        for (int t = 0; t < nktab; t++) {
            double lk = lk0 + (lk1 - lk0)*t/(nktab - 1);
            double kpar = exp(lk);
            // trapezoid in ln kperp of kperp^2 P0 W^2 (notebook P1D, verbatim).
            // Only the window weight differs between the three shapes: the disk
            // reads the precomputed W2[i] (k_perp), the isotropic windows are
            // evaluated at |k| = kk, which is why they cannot be tabulated.
            double s = 0.0, fprev = 0.0;
            for (int i = 0; i < nkperp; i++) {
                double kk = sqrt(kpar*kpar + kperp[i]*kperp[i]);
                double w2 = (window == 0) ? W2[i] : biasWindow2(kk*Rperp, window);
                double f = kperp[i]*kperp[i]*C.Pk0(kk)*w2;
                if (i > 0) s += 0.5*(f + fprev)*dlnkp;
                fprev = f;
            }
            lktab[t] = lk;
            lPtab[t] = log(std::max(s/(2.0*PI), 1.0e-300));
        }
        auto P1D = [&](double k) {
            double lk = log(k);
            if (lk <= lktab.front()) return exp(lPtab.front());
            if (lk >= lktab.back())  return exp(lPtab.back());
            int t = static_cast<int>((lk - lktab.front())/(lktab[1] - lktab[0]));
            t = std::min(t, nktab - 2);
            double f = (lk - lktab[t])/(lktab[t+1] - lktab[t]);
            return exp((1.0 - f)*lPtab[t] + f*lPtab[t+1]);
        };

        // ---- covariance: EXACT mode sum, k_q = 2 pi q / L, q = 1..Nmax.
        // Phases at the shell EDGES advance by a fixed rotation per mode
        // (uniform k grid), so no per-mode sincos is needed; resynced every
        // 4096 modes against drift. sinc terms come from the edge phases:
        //   cos(k c_i) snc_i = [sin(k b_i) - sin(k a_i)] / (k Lh_i)
        //   sin(k c_i) snc_i = [cos(k a_i) - cos(k b_i)] / (k Lh_i)
        std::vector<double> a(n), b(n), Lh(n);
        for (int i = 0; i < n; i++) {
            a[i]  = clo[i];
            b[i]  = chi_[i];
            Lh[i] = chi_[i] - clo[i];
        }
        std::vector<double> Cov(static_cast<size_t>(n)*n, 0.0);
        std::vector<double> ca(n), sa(n), cb(n), sb(n);       // phases at edges
        std::vector<double> ra_c(n), ra_s(n), rb_c(n), rb_s(n); // per-mode rotations
        const double dk = 2.0*PI/L;
        for (int i = 0; i < n; i++) {
            ra_c[i] = cos(dk*a[i]); ra_s[i] = sin(dk*a[i]);
            rb_c[i] = cos(dk*b[i]); rb_s[i] = sin(dk*b[i]);
        }
        std::vector<double> cq(n), sq(n);
        for (long q = 1; q <= Nmax; q++) {
            double kq = dk*static_cast<double>(q);
            if (q == 1 || (q & 4095) == 0) {                  // init / resync
                for (int i = 0; i < n; i++) {
                    ca[i] = cos(kq*a[i]); sa[i] = sin(kq*a[i]);
                    cb[i] = cos(kq*b[i]); sb[i] = sin(kq*b[i]);
                }
            } else {                                          // rotate by dk
                for (int i = 0; i < n; i++) {
                    double c0 = ca[i], s0 = sa[i];
                    ca[i] = c0*ra_c[i] - s0*ra_s[i];
                    sa[i] = s0*ra_c[i] + c0*ra_s[i];
                    c0 = cb[i]; s0 = sb[i];
                    cb[i] = c0*rb_c[i] - s0*rb_s[i];
                    sb[i] = s0*rb_c[i] + c0*rb_s[i];
                }
            }
            double rw = sqrt(2.0*P1D(kq)/L);                  // per-mode field std
            for (int i = 0; i < n; i++) {
                double inv = 1.0/(kq*Lh[i]);
                cq[i] = rw*(sb[i] - sa[i])*inv;               // cos(k c) snc
                sq[i] = rw*(ca[i] - cb[i])*inv;               // sin(k c) snc
            }
            for (int i = 0; i < n; i++) {
                double *row = &Cov[static_cast<size_t>(i)*n];
                double ci = cq[i], si = sq[i];
                for (int jj = 0; jj <= i; jj++)
                    row[jj] += ci*cq[jj] + si*sq[jj];
            }
        }
        for (int i = 0; i < n; i++)
            for (int jj = i + 1; jj < n; jj++)
                Cov[static_cast<size_t>(i)*n + jj] = Cov[static_cast<size_t>(jj)*n + i];

        sig2.assign(n, 0.0);
        double trace = 0.0;
        for (int i = 0; i < n; i++) {
            sig2[i] = Cov[static_cast<size_t>(i)*n + i];
            trace += sig2[i];
        }

        // ---- Cholesky (PSD by construction; tiny relative jitter for roundoff)
        const double jitter = 1.0e-12*trace/n;
        for (int i = 0; i < n; i++) Cov[static_cast<size_t>(i)*n + i] += jitter;
        chol.assign(static_cast<size_t>(n)*n, 0.0);
        for (int i = 0; i < n; i++) {
            for (int jj = 0; jj <= i; jj++) {
                double s = Cov[static_cast<size_t>(i)*n + jj];
                for (int kk = 0; kk < jj; kk++)
                    s -= chol[static_cast<size_t>(i)*n + kk]*chol[static_cast<size_t>(jj)*n + kk];
                if (i == jj) {
                    chol[static_cast<size_t>(i)*n + i] = sqrt(std::max(s, 0.0));
                } else {
                    double d = chol[static_cast<size_t>(jj)*n + jj];
                    chol[static_cast<size_t>(i)*n + jj] = (d > 0.0) ? s/d : 0.0;
                }
            }
        }
    }
};


/* ---------------------------------------------------------------------------------------------------------------------------------------------- */
/*                                                    PDF of amplifications                                                                       */
/* ---------------------------------------------------------------------------------------------------------------------------------------------- */

// find threshold kappa
double findkappathr(int N, function<double(double)> Nf) {
    double kappa1 = 1.0e-12, kappa2 = 1.0;
    double kappathr = pow(10.0, (log10(kappa1) + log10(kappa2))/2.0);
    while (log10(kappa2) - log10(kappa1) > 0.01) {
        if (Nf(kappathr) > N) {
            kappa1 = kappathr;
        } else {
            kappa2 = kappathr;
        }
        kappathr = pow(10.0, (log10(kappa1) + log10(kappa2))/2.0);
    }
    return kappathr;
}

// sample lnmu from the PDF of amplifications

vector<lensing::RealizationRaw> lensing::sample_lnmu_raw(cosmology &C, double zs, rgen &mt, const LensingConfig &cfg) {
    // guard: only subhalo_model 0 (legacy gslope) and 1 (reduced-host, default) are defined.
    // Any other value would add full clumps in addClumps WITHOUT reducing the host in
    // add_host, silently injecting ~f_s*M spurious mass. Reject rather than mis-simulate.
    // model 2 = DIAGNOSTIC ONLY: bare clumps, no host reduction / no gslope (spurious mean mass,
    // but Var(kappa)-Var(kappa_nosub) is the clean clump shot-noise). Used to isolate whether the
    // subhalo_factor sensitivity is clump shot-noise vs the mass-subtraction bookkeeping.
    // model 4 (2026-07-23, docs/subhalo/mass_conserving_carve_note.md §10): the simplified
    // production model requested by the supervisor -- every subhalo is sampled individually
    // down to the absolute floor psi_min = m_floor/M (= M_min/M), the host is carved to
    // M - Sum_i m_i (exact per-realization mass conservation), and there is NO unresolved
    // term (no M_u, no kappa_u/Wsub, no dynamic floor, no subhalo_factor). Equivalent to
    // model 1 + subhalo_brute + subhalo_carve, exposed as a single named model so callers
    // need not know the flag combination. Carve is intrinsic (guarded below).
    // model 5 (2026-07-27): model 4 plus a per-clump convergence threshold kappa_thr,sub.
    // The population is unchanged -- every subhalo still exists down to psi_min = m_floor/M
    // and there is still no unresolved/Gaussian stand-in -- but only the clumps whose kappa
    // at the ray clears the threshold are rendered, sampled from the restricted Poisson
    // intensity so the ~1e6 rejects per ray are never instantiated. This is the same
    // resolution rule the host halos already obey, applied self-consistently to subhalos.
    if (cfg.subhalo && cfg.subhalo_model != 0 && cfg.subhalo_model != 1 && cfg.subhalo_model != 2
        && cfg.subhalo_model != 3 && cfg.subhalo_model != 4 && cfg.subhalo_model != 5) {
        throw std::invalid_argument("subhalo_model must be 0 (legacy), 1 (reduced-host resolved-only), "
                                    "2 (diagnostic bare), 3 (reduced-host + Wsub), 4 (brute-to-floor + carve), "
                                    "or 5 (brute-to-floor + carve + kappa_thr,sub)");
    }
    // model 3 replaces the unresolved clumps by the analytic mu+Gaussian term keyed to the
    // dynamic floor; brute mode resolves everything, so combining them double-counts.
    if (cfg.subhalo && cfg.subhalo_model == 3 && cfg.subhalo_brute) {
        throw std::invalid_argument("subhalo_brute is incompatible with subhalo_model 3 (use model 1 or 2 for brute references)");
    }
    // model 4 carves the host by the realized substructure mass; without the carve it would
    // add brute clumps on top of a full-mass host (spurious ~f_s*M). Carve is not optional.
    if (cfg.subhalo && cfg.subhalo_model == 4 && !cfg.subhalo_carve) {
        throw std::invalid_argument("subhalo_model 4 requires subhalo_carve = true (the carve is intrinsic to the model)");
    }
    // model 5 inherits the carve for the same reason: dropped clumps keep their mass in the
    // smooth host, which only works if the host is built at M - Sum_i m_i(retained).
    if (cfg.subhalo && cfg.subhalo_model == 5 && !cfg.subhalo_carve) {
        throw std::invalid_argument("subhalo_model 5 requires subhalo_carve = true (the carve is intrinsic to the model)");
    }
    if (cfg.subhalo && cfg.subhalo_model == 5 && cfg.subhalo_brute) {
        throw std::invalid_argument("subhalo_brute is meaningless for subhalo_model 5 (the model is brute by construction, thresholded on kappa)");
    }
    if (cfg.subhalo && cfg.subhalo_model == 5 && cfg.subhalo_kappathr <= 0.0
        && !(cfg.subhalo_kappathr_factor > 0.0)) {
        throw std::invalid_argument("subhalo_model 5 needs subhalo_kappathr > 0 or subhalo_kappathr_factor > 0");
    }
    // The virial convention rescales psi and the radial extent inside the clump sampler.
    // Models 0-3 additionally reduce the host through the incomplete-Gamma f_s_res / Wsub
    // tables, which are still written in M_200 units, so mixing them would be inconsistent.
    // Gate to the production (5) + reference (4) path.
    if (cfg.subhalo && cfg.subhalo_virial
        && !(cfg.subhalo_model == 4 || cfg.subhalo_model == 5)) {
        throw std::invalid_argument("subhalo_virial requires subhalo_model 4 or 5 "
                                    "(models 0-3 reduce the host with M_200-referred tables)");
    }
    if (cfg.bias_model != 0 && cfg.bias_model != 1) {
        throw std::invalid_argument("bias_model must be 0 (legacy iid cell bias) or 1 (correlated 1D field)");
    }
    if (cfg.bias_model == 1 && cfg.bias_Rperp <= 0.0) {
        throw std::invalid_argument("bias_model = 1 requires bias_Rperp > 0 (comoving kpc)");
    }
    if (cfg.bias_window < 0 || cfg.bias_window > 2) {
        throw std::invalid_argument("bias_window must be 0 (transverse disk), 1 (spherical "
                                    "top-hat) or 2 (Gaussian)");
    }
    if (cfg.bias_window != 0 && cfg.bias_model != 1) {
        throw std::invalid_argument("bias_window != 0 requires bias_model = 1 (the window "
                                    "shapes the correlated field's power spectrum)");
    }
    if (cfg.bias_weak && cfg.bias_model != 1) {
        throw std::invalid_argument("bias_weak requires bias_model = 1 (the correlated field "
                                    "supplies the conditioning); it is a no-op when bias = 0");
    }
    using Clock = std::chrono::steady_clock;
    auto elapsed_seconds = [](Clock::time_point start) {
        return std::chrono::duration<double>(Clock::now() - start).count();
    };
    auto total_start = Clock::now();
    LensingProfile *profile = cfg.profile;
    if (profile != nullptr) {
        profile->reset(C.NM);
        profile->Nreal = cfg.Nreal;
        profile->zs = zs;
        for (int jM = 0; jM < C.NM; jM++) {
            profile->M_host[jM] = C.Mlist[jM];
        }
    }
    
    // fix threshold kappa
    function<double(double)> NfNFW = [&C, zs](double kappa) {
        return NhfNFW(C, zs, kappa);
    };
    double kappathr_default = (cfg.kappathr_flat > 0.0)
        ? cfg.kappathr_flat
        : findkappathr(cfg.Nhalos, NfNFW);
    double kappathrH = (cfg.custom_kappathr > 0.0) ? cfg.custom_kappathr : kappathr_default;

    if (cfg.write > 0) {
        cout << kappathrH << endl;
    }

    // distribution of kappa_NFW < kappa_thr. The integration floor is held at its
    // ABSOLUTE default value kappa_min = 0.001*kappathr_default, so sweeping
    // custom_kappathr does not drag the floor with it (which would discard real
    // background variance at large thresholds). With custom_kappathr unset this
    // reduces to eps_floor = 0.001 exactly. With the default (legacy <N>=Nhalos)
    // rule kappathr_default is the per-z_s threshold, so the floor tracks it; with
    // a flat threshold it is z_s-independent (kappa_min = 0.001*kappathr_flat).
    double skappaW = sigmakappaW(C, zs, kappathrH, 0.001*kappathr_default/kappathrH);
    normal_distribution<double> PkappaW(0.0, skappaW);
    if (skappaW < 0.0) {
        cout << "Error: negative standard deviation." << endl;
    }
    
    // bias_weak (+ field + bias on): the weak background is drawn CONDITIONALLY
    // on the field, which is only available after the field block below — the
    // initial fill is skipped and done there instead. All other paths keep the
    // legacy fill here (bit-identical stream for bias_model = 0).
    const bool weak_conditional = cfg.bias_weak && cfg.bias_model == 1 && cfg.bias != 0;
    vector<RealizationRaw> raw(cfg.Nreal);
    for (int j = 0; j < cfg.Nreal; j++) {
        raw[j].kappa = weak_conditional ? 0.0 : PkappaW(mt);
        raw[j].gamma1 = 0.0;
        raw[j].gamma2 = 0.0;
        raw[j].kappa_nosub = raw[j].kappa;   // paired baseline starts from the weak part
        raw[j].kappa_weak = raw[j].kappa;
    }
    
    vector<vector<vector<double> > > dNH = deltaNhfNFW(C, zs, kappathrH);
    vector<vector<vector<double> > > dNF = deltaNhfCYL(C, zs, kappathrH);

    // subhalo substructure tables (rebuilt per run; zs-independent physics)
    if (cfg.subhalo) {
        auto precompute_start = Clock::now();
        subhalo_.m_floor = cfg.m_floor;
        subhalo_.psi_min_fixed = cfg.psi_min_fixed;
        subhalo_.virial = cfg.subhalo_virial;   // must be set BEFORE precompute()
        // model 3 additionally builds the Wsub (unresolved mu/sigma) tables, which need
        // the HOST threshold for the encounter-disc radius rmax per bin.
        // model 5 reuses r_thr as the clump REACH D(m), so it must be built at
        // kappa_thr,sub rather than at subhalo_factor * kappathrH.
        double clump_kappathr = cfg.subhalo_factor * kappathrH;
        if (cfg.subhalo_model == 5) {
            clump_kappathr = (cfg.subhalo_kappathr > 0.0)
                               ? cfg.subhalo_kappathr
                               : cfg.subhalo_kappathr_factor * kappathrH;
        }
        subhalo_.precompute(C, zs, clump_kappathr,
                            (cfg.subhalo_model == 3) ? kappathrH : 0.0,
                            cfg.subhalo_model == 5);
        if (profile != nullptr) {
            profile->subhalo_precompute_seconds = elapsed_seconds(precompute_start);
        }
    }
    if (cfg.write > 0) {
        writeToFile(C.zlist, C.Mlist, dNH, C.outdir/"dNH.dat");
        writeToFile(C.zlist, C.Mlist, dNF, C.outdir/"dNF.dat");
    }

    array<double,2> kappagamma;
    normal_distribution<double> pG(0.0, 1.0);

    // bias_model = 1: draw the correlated per-shell environment field for ALL
    // realizations up front (the sampling loops below are cell-major, so shell
    // jz needs every realization's field value when its cells are processed).
    // This is the ONLY model-1 RNG consumption outside the shared code path;
    // model 0 draws nothing here and is bit-identical to the legacy stream.
    // bias == 0 (nobias control arms) skips the build entirely — lambda is
    // forced to 1 in the loop, so the field would be dead weight (and its
    // RNG draws would needlessly shift the halo stream vs bias_model = 0).
    // Memory: Nreal * n_shells floats (e.g. 400k * 99 = 158 MB at the default
    // Nreal — use smaller batches per process for production scans).
    BiasField1D bfield;
    std::vector<float> bfvals;
    if (cfg.bias_model == 1 && cfg.bias != 0) {
        bfield.build(C, zs, cfg.bias_Rperp, cfg.bias_window);
        if (bfield.n > 0) {
            const int nsh = bfield.n;
            bfvals.resize(static_cast<size_t>(cfg.Nreal)*nsh);
            std::vector<double> g(nsh);
            for (int j = 0; j < cfg.Nreal; j++) {
                for (int i = 0; i < nsh; i++) g[i] = pG(mt);
                for (int i = 0; i < nsh; i++) {
                    double s = 0.0;
                    const double *row = &bfield.chol[static_cast<size_t>(i)*nsh];
                    for (int kk = 0; kk <= i; kk++) s += row[kk]*g[kk];
                    bfvals[static_cast<size_t>(j)*nsh + i] = static_cast<float>(s);
                }
            }
        }
    }

    // bias_weak: conditional weak background, drawn from the SAME realized
    // field values as the count modulation (Cox split of Campbell's theorem;
    // see lensing.h). Consumes one normal per realization, at this fixed
    // stream point. Degenerate n == 0 (z_s below the first grid shell) falls
    // back to the legacy unconditional draw.
    if (weak_conditional) {
        if (bfield.n > 0) {
            bfield.buildWeak(C, zs, kappathrH, 0.001*kappathr_default/kappathrH, skappaW);
            const int nsh = bfield.n;
            for (int j = 0; j < cfg.Nreal; j++) {
                double sumS = 0.0, sumV = 0.0, Sw, Vw;
                const float *fj = &bfvals[static_cast<size_t>(j)*nsh];
                for (int i = 0; i < nsh; i++) {
                    bfield.weakSV(i, static_cast<double>(fj[i]), Sw, Vw);
                    sumS += Sw;
                    sumV += Vw;
                }
                double kw = sumS + sqrt(std::max(sumV, 0.0))*pG(mt);
                raw[j].kappa = kw;
                raw[j].kappa_nosub = kw;
                raw[j].kappa_weak = kw;
            }
        } else {
            for (int j = 0; j < cfg.Nreal; j++) {
                raw[j].kappa = PkappaW(mt);
                raw[j].kappa_nosub = raw[j].kappa;
                raw[j].kappa_weak = raw[j].kappa;
            }
        }
    }
    poisson_distribution<int> PN;
    double zl, M, rmaxH, rmaxF, r, phi, phiH, phiF, epsilon = 0.0, barNH, barNF, sigma, deltab, lambda;
    int NH, NF;
    int NtotH = 0, NtotF = 0;;
    for (int jz = 0; jz < C.Nz; jz++) {
        zl = C.zlist[jz];
        if (zl < zs) {
            for (int jM = 0; jM < C.NM; jM++) {
                M = C.Mlist[jM];
                
                // generate realizations
                barNH = dNH[jz][jM][0];
                sigma = dNH[jz][jM][1];
                rmaxH = dNH[jz][jM][2];
                
                barNF = dNF[jz][jM][0];
                rmaxF = dNF[jz][jM][2];
                
                double Sigmac = (barNH > 0.0 || barNF > 0.0) ? Sigmacf(C, zs, zl) : 0.0;
                double rsH = 1.0, kappa0H = 0.0;
                if (barNH > 0.0) {
                    vector<double> NFWp = interpolate2(zl, M, C.zlist, C.Mlist, C.NFWlist);
                    rsH = NFWp[0];
                    kappa0H = kappa0NFW(rsH, NFWp[1], Sigmac);
                    if (cfg.ell > 0) epsilon = epsilonNFW(C, zl, M);
                }
                double rsF = 0.0, LF = 0.0, kappa0baseF = 0.0;
                if (cfg.fil > 0 && barNF > 0.0) {
                    rsF = 1000.0*pow(M/1.0e14, 1.0/3.0);
                    LF = 20000.0*pow(M/1.0e14, 1.0/3.0);
                    kappa0baseF = rsF * 14.4*C.rhoc*LF / Sigmac;
                }

                // bias_model = 1: cell amplitude b(M,z_l) Dg(z_l) (growth-free field)
                // and this jz's shell index; the compensation uses the shell's own
                // segment variance so <lambda> = 1 exactly. The same lambda multiplies
                // the filament counts, as in the legacy layer.
                int bsh = -1;
                double bDg = 0.0, bcomp = 0.0;
                // Filament clustering amplitude (cfg.fil_bias): same field, but the
                // filament counts ride b_fil(M,z) = filbias (PBS of pFCfil) instead
                // of the halo bias. Only the correlated field (bias_model = 1) carries
                // this; bDgF stays 0 otherwise and lambdaF falls back to the halo
                // lambda in the loop below (so fil_bias is a no-op in the legacy layer).
                double bDgF = 0.0, bcompF = 0.0;
                if (cfg.bias_model == 1) {
                    bsh = bfield.shell(jz);
                    if (bsh >= 0) {
                        bDg = C.Dg(zl)*C.halobias(zl, C.sigmalist[jM][1]);
                        bcomp = 0.5*bDg*bDg*bfield.sig2[bsh];
                        if (cfg.fil_bias) {
                            bDgF = C.Dg(zl)*C.filbias(zl, C.sigmalist[jM][1]);
                            bcompF = 0.5*bDgF*bDgF*bfield.sig2[bsh];
                        }
                    }
                }

                auto add_host = [&](int j) {
                    // --- Mass-conserving host carve (scheme A, 2026-07-22; docs/subhalo/
                    // mass_conserving_carve_note.md). Draw the resolved clumps FIRST, then
                    // build the smooth host at M - Sum_i m_i - M_u(r) so the total halo mass
                    // is M exactly in every realization. The host build draws no random
                    // numbers, so this reorder leaves the RNG stream identical to the
                    // deterministic (non-carve) path. Applies to model 3 (production) and the
                    // brute reference (model 1/2 + subhalo_brute; M_u = 0 there). Carve-off
                    // and models 0 / 2-non-brute fall through to the original path below.
                    if (cfg.subhalo && cfg.subhalo_carve &&
                        (cfg.subhalo_model == 3 || cfg.subhalo_model == 4 || cfg.subhalo_model == 5 ||
                         (cfg.subhalo_brute && (cfg.subhalo_model == 1 || cfg.subhalo_model == 2)))) {

                        // 1. resolved clumps first: accumulate their kappa/gamma into raw[j]
                        //    and their realized total mass Sum_i m_i (Msum). For model 5 the
                        //    restricted sampler renders only the clumps above kappa_thr,sub and
                        //    Msum is the RETAINED mass -- the rest stays in the smooth host,
                        //    which is exactly what makes the threshold mass-conserving.
                        auto subhalo_start = Clock::now();
                        double Msum = 0.0;
                        int Nc = (cfg.subhalo_model == 5)
                            ? subhalo_.addClumpsRestricted(C, jz, jM, M, Sigmac, r, phi, mt,
                                                           raw[j].kappa, raw[j].gamma1, raw[j].gamma2,
                                                           &Msum)
                            : subhalo_.addClumps(C, jz, jM, zl, M, Sigmac, r, phi, mt,
                                                 raw[j].kappa, raw[j].gamma1, raw[j].gamma2,
                                                 cfg.subhalo_model, cfg.subhalo_brute,
                                                 cfg.subhalo_threads, cfg.subhalo_parallel_threshold,
                                                 &Msum);
                        if (profile != nullptr) {
                            profile->subhalo_seconds[jM] += elapsed_seconds(subhalo_start);
                            profile->subhalo_calls[jM]++;
                            profile->subhalo_clumps[jM] += static_cast<uint64_t>(Nc);
                        }

                        // 2. mean unresolved mass at this ray (0 for brute: all clumps explicit)
                        double M_u = (cfg.subhalo_model == 3)
                                       ? subhalo_.unresolvedMass(C, jz, jM, r, M) : 0.0;

                        // 3. carved smooth host at M - Sum m_i - M_u, with a negative-mass guard
                        auto smooth_start_c = Clock::now();
                        // Virial mode: Msum is drawn against the M_vir budget, but the smooth
                        // host is parameterized by its M_200-equivalent grid mass. Convert so the
                        // host keeps the same FRACTIONAL mass (1-f_s) in both apertures; the
                        // remainder is the substructure in the r_200..r_vir shell, which was
                        // never part of the M_200 budget. vr == 1 in legacy mode (bitwise).
                        const double vr = subhalo_.virialRatio(jz, jM);
                        double M_host_eff = M - Msum / vr - M_u;
                        if (M_host_eff < C.Mmin) {          // ~9-sigma SHMF excursion; never seen in 40k
                            M_host_eff = C.Mmin;
                            if (profile != nullptr) profile->subhalo_carve_negatives++;
                        }
                        double rsH_c = rsH, kappa0H_c = kappa0H, epsilon_c = epsilon;
                        {
                            vector<double> NFWp = interpolate2(zl, M_host_eff, C.zlist, C.Mlist, C.NFWlist);
                            rsH_c = NFWp[0];
                            kappa0H_c = kappa0NFW(rsH_c, NFWp[1], Sigmac);
                            if (cfg.ell > 0) epsilon_c = epsilonNFW(C, zl, M_host_eff);
                        }
                        auto kg_host = kappagammaNFWeps(epsilon_c, kappa0H_c, r / rsH_c, phiH);
                        raw[j].kappa += kg_host[0];
                        raw[j].gamma1 += cos(phi) * kg_host[1];
                        raw[j].gamma2 += sin(phi) * kg_host[1];

                        // full unperturbed host at M_total tracks kappa_nosub (as in models 1/3)
                        auto kg_full = kappagammaNFWeps(epsilon, kappa0H, r / rsH, phiH);
                        raw[j].kappa_nosub += kg_full[0];

                        if (profile != nullptr) {
                            profile->smooth_host_seconds[jM] += elapsed_seconds(smooth_start_c);
                            profile->host_events[jM]++;
                            if (subhalo_.fsub[jz][jM] > 0.0) {
                                profile->fsub_weighted_sum[jM] += subhalo_.fsub[jz][jM];
                                profile->nsub_mean_weighted_sum[jM] += subhalo_.Nsub[jz][jM];
                            }
                        }

                        // 4. unresolved-clump term (model 3): mean profile + Gaussian fluctuation
                        //    (kappa only, like the field-level sigma_W). Same RNG draw (pG) and
                        //    position in the stream as the non-carve model-3 path.
                        if (cfg.subhalo_model == 3) {
                            double muU, sU;
                            subhalo_.wsubTerm(jz, jM, r, muU, sU);
                            raw[j].kappa += muU + sU * pG(mt);
                        }

                        NtotH++;
                        return;
                    }

                    auto smooth_start = Clock::now();
                    double rsH_eff = rsH;
                    double kappa0H_eff = kappa0H;
                    double epsilon_eff = epsilon;

                    if (cfg.subhalo && cfg.subhalo_model == 1) {
                        double g = subhalo_.gnorm[jz][jM];
                        double M_host_eff = M;
                        if (g > 0.0) {
                            double psi_lo = 0.0;
                            if (cfg.subhalo_brute) {
                                psi_lo = cfg.m_floor / M;
                            } else {
                                // dynamic floor keyed to host-center distance r; MUST match Subhalo::addClumps
                                // (r-vs-d bias is absorbed by tuning subhalo_factor to convergence)
                                const auto &rth = subhalo_.r_thr[jz];
                                int jlo = static_cast<int>(std::lower_bound(rth.begin(), rth.end(), r) - rth.begin());
                                if (jlo < C.NM) {
                                    psi_lo = std::max(C.Mlist[jlo], C.Mmin) / M;
                                } else {
                                    psi_lo = subhalo_.psi_max;
                                }
                            }
                            if (psi_lo < subhalo_.psi_max) {
                                // resolved mass fraction with the SHMF exponential cutoff included,
                                // f_s_res = g/(omega beta^s) [Gamma(s, beta psi_lo^omega) - Gamma(s, beta psi_max^omega)],
                                // s=(1+alpha)/omega — MUST equal the expectation of addClumps' thinned sampler.
                                const double s_m = (1.0 + subhalo_.alpha) / subhalo_.omega;
                                double f_s_res = g / (subhalo_.omega * pow(subhalo_.beta, s_m)) *
                                    (gsl_sf_gamma_inc(s_m, subhalo_.beta * pow(psi_lo, subhalo_.omega)) -
                                     gsl_sf_gamma_inc(s_m, subhalo_.beta * pow(subhalo_.psi_max, subhalo_.omega)));
                                f_s_res = std::max(0.0, std::min(0.95, f_s_res));
                                M_host_eff = (1.0 - f_s_res) * M;
                            }
                        }
                        vector<double> NFWp = interpolate2(zl, M_host_eff, C.zlist, C.Mlist, C.NFWlist);
                        rsH_eff = NFWp[0];
                        kappa0H_eff = kappa0NFW(rsH_eff, NFWp[1], Sigmac);
                        if (cfg.ell > 0) epsilon_eff = epsilonNFW(C, zl, M_host_eff);
                    } else if (cfg.subhalo && cfg.subhalo_model == 3) {
                        // Wsub scheme: host reduced by the FULL bound fraction (y-independent);
                        // the unresolved clumps' mean comes back via muW(y) below, so the
                        // y-dependent incomplete-Gamma bookkeeping of model 1 is not needed.
                        double f_b = subhalo_.fsb[jz][jM];
                        if (f_b > 0.0) {
                            double M_host_eff = (1.0 - f_b) * M;
                            vector<double> NFWp = interpolate2(zl, M_host_eff, C.zlist, C.Mlist, C.NFWlist);
                            rsH_eff = NFWp[0];
                            kappa0H_eff = kappa0NFW(rsH_eff, NFWp[1], Sigmac);
                            if (cfg.ell > 0) epsilon_eff = epsilonNFW(C, zl, M_host_eff);
                        }
                    }

                    kappagamma = kappagammaNFWeps(epsilon_eff, kappa0H_eff, r/rsH_eff, phiH);
                    raw[j].kappa += kappagamma[0];
                    // single-angle gamma projection, matching the original Vaskonen code (kept by
                    // decision 2026-07-02; the spin-2 form would be -gamma_t (cos 2phi, sin 2phi) —
                    // see tmp/shear_convention_check.py; difference is ~0.5% on <gamma^2>)
                    raw[j].gamma1 += cos(phi)*kappagamma[1];
                    raw[j].gamma2 += sin(phi)*kappagamma[1];

                    // kappa_nosub tracks the full unperturbed host (NFW at M_total)
                    if (cfg.subhalo && (cfg.subhalo_model == 1 || cfg.subhalo_model == 3)) {
                        auto kg_full = kappagammaNFWeps(epsilon, kappa0H, r/rsH, phiH);
                        raw[j].kappa_nosub += kg_full[0];
                    } else {
                        raw[j].kappa_nosub += kappagamma[0];
                    }

                    if (profile != nullptr) {
                        profile->smooth_host_seconds[jM] += elapsed_seconds(smooth_start);
                        profile->host_events[jM]++;
                        if (cfg.subhalo && subhalo_.fsub[jz][jM] > 0.0) {
                            profile->fsub_weighted_sum[jM] += subhalo_.fsub[jz][jM];
                            profile->nsub_mean_weighted_sum[jM] += subhalo_.Nsub[jz][jM];
                        }
                    }

                    if (cfg.subhalo) {
                        auto subhalo_start = Clock::now();
                        int Nc = subhalo_.addClumps(C, jz, jM, zl, M, Sigmac, r, phi, mt,
                                                     raw[j].kappa, raw[j].gamma1, raw[j].gamma2,
                                                     cfg.subhalo_model, cfg.subhalo_brute,
                                                     cfg.subhalo_threads, cfg.subhalo_parallel_threshold);
                        if (cfg.subhalo_model == 3) {
                            // unresolved-clump term: mean profile + Gaussian fluctuation,
                            // exact per-encounter mean/variance at any subhalo_factor
                            // (docs/subhalo/wsub_gaussian_term_derivation.md); kappa only,
                            // like the field-level sigma_W.
                            double muU, sU;
                            subhalo_.wsubTerm(jz, jM, r, muU, sU);
                            raw[j].kappa += muU + sU * pG(mt);
                        }
                        if (profile != nullptr) {
                            profile->subhalo_seconds[jM] += elapsed_seconds(subhalo_start);
                            profile->subhalo_calls[jM]++;
                            profile->subhalo_clumps[jM] += static_cast<uint64_t>(Nc);
                        }
                    }

                    NtotH++;
                };

                for (int j = 0; j < cfg.Nreal; j++) {

                    // bias
                    if (cfg.bias_model == 1) {
                        // correlated 1D field: this shell's segment average, cell bias
                        // amplitude bDg; exact <lambda> = 1 bookkeeping via bcomp
                        lambda = (bsh >= 0)
                            ? exp(bDg*bfvals[static_cast<size_t>(j)*bfield.n + bsh] - bcomp)
                            : 1.0;
                    } else {
                        // legacy: iid per-cell draw (bit-identical default path)
                        deltab = sigma*pG(mt);
                        lambda = exp(deltab - pow(sigma,2.0)/2.0); // log-normal
                    }

                    if (cfg.bias == 0) {
                        lambda = 1.0;
                    }
                    // Filament count modulation. Same realized field, but the filament
                    // amplitude uses filbias (cfg.fil_bias, bias_model = 1). Defaults to
                    // the halo lambda, so fil_bias = 0 (and the whole legacy layer) is
                    // bitwise unchanged; no new RNG draw (bfvals[j][bsh] already realized).
                    double lambdaF = lambda;
                    if (cfg.fil_bias && cfg.bias != 0 && cfg.bias_model == 1 && bsh >= 0) {
                        lambdaF = exp(bDgF*bfvals[static_cast<size_t>(j)*bfield.n + bsh] - bcompF);
                    }
                    // look at this.
                    // generate halos
                    if (lambda*barNH < 0.2) { // if lambda is small, compare to a random number U(0,1) (faster)
                        if (lambda*barNH > randomreal(0.0, 1.0, mt)) {
                            r = sqrt(randomreal(0.0,1.0,mt))*rmaxH; // distance from the line-of-sight
                            phi = randomreal(0.0,2*PI,mt); // polar angle of r vector
                            phiH = randomreal(0.0,2*PI,mt); // orientation of the halo ellipticity
                            add_host(j);
                        }
                    } else { // for larger lambda, generate number of halos from Poisson distribution (slower)
                        PN = poisson_distribution<int>(lambda*barNH);
                        NH = PN(mt);
                        if (NH > 0) {
                            for (int jH = 0; jH < NH; jH++) {
                                r = sqrt(randomreal(0.0,1.0,mt))*rmaxH; // distance from the line-of-sight
                                phi = randomreal(0.0,2*PI,mt); // polar angle of r vector
                                phiH = randomreal(0.0,2*PI,mt); // orientation of the halo ellipticity
                                add_host(j);
                            }
                        }
                    }
                    
                    if (cfg.fil > 0) {
                        // generate filaments
                        if (lambdaF*barNF < 0.2) { // if lambda is small, compare to a random number U(0,1) (faster)
                            if (lambdaF*barNF > randomreal(0.0, 1.0, mt)) {
                                r = sqrt(randomreal(0.0,1.0,mt))*rmaxF; // distance from the line-of-sight
                                phi = randomreal(0.0,2*PI,mt); // polar angle of r vector
                                phiF = randomreal(0.0,2*PI,mt); // orientation of the filament
                                double kappa0F = kappa0baseF / max(2.0*rsF, abs(cos(phiF))*LF);
                                kappagamma = {kappaCYL2(r, rsF, kappa0F), gammaCYL2(r, rsF, kappa0F)};
                                
                                raw[j].kappa += kappagamma[0];
                                raw[j].kappa_nosub += kappagamma[0];
                                raw[j].gamma1 += cos(phi)*kappagamma[1];
                                raw[j].gamma2 += sin(phi)*kappagamma[1];

                                NtotF++;
                            }
                        } else { // for larger lambda, generate number of halos from Poisson distribution (slower)
                            PN = poisson_distribution<int>(lambdaF*barNF);
                            NF = PN(mt);
                            if (NF > 0) {
                                for (int jF = 0; jF < NF; jF++) {
                                    r = sqrt(randomreal(0.0,1.0,mt))*rmaxF; // distance from the line-of-sight
                                    phi = randomreal(0.0,2*PI,mt); // polar angle of r vector
                                    phiF = randomreal(0.0,2*PI,mt); // orientation of the filament
                                    double kappa0F = kappa0baseF / max(2.0*rsF, abs(cos(phiF))*LF);
                                    kappagamma = {kappaCYL2(r, rsF, kappa0F), gammaCYL2(r, rsF, kappa0F)};
                                    
                                    raw[j].kappa += kappagamma[0];
                                    raw[j].kappa_nosub += kappagamma[0];
                                    raw[j].gamma1 += cos(phi)*kappagamma[1];
                                    raw[j].gamma2 += sin(phi)*kappagamma[1];

                                    NtotF++;
                                }
                            }
                        }
                    }
                    
                }
            }
        }
    }
    if (profile != nullptr) {
        profile->total_seconds = elapsed_seconds(total_start);
    }
    return raw;
}


vector<double> lensing::sample_lnmu(cosmology &C, double zs, rgen &mt, const LensingConfig &cfg) {
    
    auto raw = sample_lnmu_raw(C, zs, mt, cfg);

    // Flux-conservation anchor <kappa> = 0 (see LensingConfig::kappa_anchor).
    double meankappa = 0.0;
    if (cfg.kappa_anchor == 2) {
        // external/analytic anchor: exactly independent realizations
        meankappa = cfg.kappa_anchor_value;
    } else if (cfg.kappa_anchor == 1) {
        // robust anchor: exclude kappa > cut rays (outside weak-lensing
        // validity) so a single monster ray cannot shift the whole batch
        double sum = 0.0;
        size_t nk = 0;
        for (auto &r : raw) {
            if (r.kappa <= cfg.kappa_anchor_cut) { sum += r.kappa; nk++; }
        }
        if (nk > 0) {
            meankappa = sum / nk;
        } else {  // pathological: every ray above cut -> legacy fallback
            for (auto &r : raw) meankappa += r.kappa;
            meankappa /= raw.size();
        }
    } else {
        // legacy (default): empirical mean over ALL rays -- bit-identical to
        // the pre-2026-07-13 behavior; one kappa>>1 ray shifts the batch by
        // -2*kappa/n (batch-anchor bug, kept as default pending decision)
        for (auto &r : raw)
            meankappa += r.kappa;
        meankappa /= raw.size();
    }
    
    vector<double> lnmulist;
    lnmulist.reserve(raw.size());

    InvalidSampleStats stats;
    stats.total_samples = raw.size();
    double detA_sum = 0.0;
    bool detA_seen = false;
    constexpr double huge_threshold = 1.0e12;
    constexpr double detA_near_zero_threshold = 1.0e-6;

    double kappaj, gammaj, muj, detA, logmu;
    for (auto &r : raw) {
        kappaj = r.kappa - meankappa;
        gammaj = sqrt(r.gamma1*r.gamma1 + r.gamma2*r.gamma2);
        detA = (1.0 - kappaj) * (1.0 - kappaj) - (gammaj * gammaj);

        if (!detA_seen) {
            stats.detA_min = detA;
            stats.detA_max = detA;
            detA_seen = true;
        } else {
            stats.detA_min = std::min(stats.detA_min, detA);
            stats.detA_max = std::max(stats.detA_max, detA);
        }
        detA_sum += detA;
        if (std::abs(detA) < detA_near_zero_threshold) {
            stats.detA_near_zero_count++;
        }

        bool invalid = false;
        if (std::isnan(kappaj)) {
            stats.nan_kappa++;
            invalid = true;
        }
        if (std::isnan(gammaj)) {
            stats.nan_gamma++;
            invalid = true;
        }
        if (detA <= 0.0) {
            stats.negative_detA++;
            invalid = true;
            if (cfg.strict_weak_lensing) {
                stats.strict_weak_lensing_rejects++;
            }
        }

        if (detA == 0.0) {
            muj = std::numeric_limits<double>::infinity();
        } else {
            muj = 1.0 / detA;
        }

        if (!std::isfinite(muj)) {
            stats.nonfinite_mu++;
            invalid = true;
        }
        if (muj <= 0.0) {
            stats.negative_mu++;
            invalid = true;
        }
        if (std::isfinite(muj) && std::abs(muj) > huge_threshold) {
            stats.overflow_mu++;
            invalid = true;
        }

        if (!invalid) {
            logmu = log(muj);
            if (!std::isfinite(logmu)) {
                stats.invalid_logmu++;
                invalid = true;
            } else {
                lnmulist.push_back(logmu);
                stats.valid_samples++;
            }
        } else {
            // Keep invalid_logmu classification explicit when log(mu) is undefined.
            if (muj <= 0.0 || !std::isfinite(muj)) {
                stats.invalid_logmu++;
            }
        }
    }

    stats.invalid_samples = stats.total_samples - stats.valid_samples;
    stats.detA_mean = (stats.total_samples > 0) ? (detA_sum / static_cast<double>(stats.total_samples)) : 0.0;
    if (!detA_seen) {
        stats.detA_min = 0.0;
        stats.detA_max = 0.0;
    }
    last_invalid_stats_ = stats;
    
    return lnmulist;
}


// probability distribution of lnmu, {lnmu, dP/dlnmu}
vector<vector<double> > lensing::Plnmuf(cosmology &C, double zs, rgen &mt,
                                       int fil, int bias, int ell, int write) {

    LensingConfig cfg;
    cfg.Nreal = this->Nreal;
    cfg.Nhalos = this->Nhalos;
    cfg.Nbins = this->Nbins;
    cfg.fil = fil;
    cfg.bias = bias;
    cfg.ell = ell;
    cfg.write = write;

    vector<double> lnmulist = sample_lnmu(C, zs, mt, cfg);
    vector<vector<double> > Plnmu = binSample(lnmulist, cfg.Nbins);
    
    int jmin = 0, jmax = Plnmu.size()-1;
    for (int j = 0; j < Plnmu.size()-4; j++) {
        if (Plnmu[j][1] > 0 && Plnmu[j+1][1] > 0 && Plnmu[j+2][1] > 0 && Plnmu[j+3][1] > 0 && Plnmu[j+4][1] > 0) {
            jmin = j;
            j = Plnmu.size();
        }
    }
    for (int j = max(4,jmin); j < Plnmu.size(); j++) {
        if (Plnmu[j-4][1] == 0 && Plnmu[j-3][1] == 0 && Plnmu[j-2][1] == 0 && Plnmu[j-1][1] == 0 && Plnmu[j][1] == 0) {
            jmax = j;
            j = Plnmu.size();
        }
    }
    //cout << jmin << "   " << jmax << endl;
    Plnmu.erase(Plnmu.begin() + jmax, Plnmu.end());
    Plnmu.erase(Plnmu.begin(), Plnmu.begin() + jmin);
    
    // convert from image plane to source plane (P_S ~ P_I/mu) and normalize
    double dlnmu = Plnmu[1][0] - Plnmu[0][0];
    double norm = 0.0;
    for (int j = 0; j < Plnmu.size(); j++) {
        Plnmu[j][1] *= exp(-Plnmu[j][0]);
        norm += Plnmu[j][1]*dlnmu;
    }
    for (int j = 0; j < Plnmu.size(); j++) {
        Plnmu[j][1] *= 1.0/norm;
    }
    return Plnmu;
}

// loglikelihood of the Hubble digram data
double lensing::loglikelihood(cosmology &C, double DLthr, vector<vector<double> > &data, vector<double> &par, int lens, int dm, rgen &mt) {
    
    // initialize cosmology
    C.OmegaM = par[0];
    C.sigma8 = par[1];
    C.h = par[2];
    C.initialize(dm, pow(10.0, par[3]));
    
    double z, DL0, DL, sigmaDL, Y, dY, Pdet;
    
    // compute loglikelihood
    double logL = 0.0;
    if (lens == 0) { // model without lensing
        for (int j = 0; j < data.size(); j++) {
            z = data[j][0];
            DL0 = C.DL(z);
            DL = data[j][1];
            sigmaDL = data[j][2];
            
            // compute P_det
            dY = sigmaDL/10.0;
            Y = min(DL0, DLthr) - 3.0*sigmaDL;
            Pdet = 0.0;
            while (Y <= min(DLthr, DL0 + 3.0*sigmaDL)) {
                Pdet += dY*NPDF(Y, DL0, sigmaDL);
                Y += dY;
            }
            logL += logNPDF(DL, DL0, sigmaDL) - log(Pdet);
        }
    }
    else { // model with lensing
        
        // compute the lensing distributions
        vector<vector<vector<double> > > Plnmuz(C.Zlist.size());
        for (int jz = 0; jz < C.Zlist.size(); jz++) {
            z = C.Zlist[jz];
            Plnmuz[jz] = Plnmuf(C, z, mt, 1, 1, 1, 0);
        }
        
        int jz;
        vector<vector<double> > Plnmu;
        double L, dlnmu;
        for (int j = 0; j < data.size(); j++) {
            z = data[j][0];
            
            DL = data[j][1];
            sigmaDL = data[j][2];
            
            // find the closest z at which P(lnmu) is computed
            jz = lower_bound(C.Zlist.begin(), C.Zlist.end(), z) - C.Zlist.begin();
            if ((jz > 0 && C.Zlist[jz]-z > z-C.Zlist[jz-1]) || jz >= C.Zlist.size()) {
                jz--;
            }
            Plnmu = Plnmuz[jz];
            dlnmu = Plnmu[1][0] - Plnmu[0][0];
            
            // integrate over lnmu
            dY = sigmaDL/10.0;
            Pdet = 0.0, L = 0.0;
            for (int i = 0; i < Plnmu.size(); i++) {
                DL0 = C.DL(z)/exp(Plnmu[i][0]/2.0);
                
                L += dlnmu*Plnmu[i][1]*NPDF(DL, DL0, sigmaDL);
                
                // integrate over Y
                Y = min(DL0, DLthr) - 3.0*sigmaDL;
                while (Y <= min(DLthr, DL0 + 3.0*sigmaDL)) {
                    Pdet += dlnmu*Plnmu[i][1]*dY*NPDF(Y, DL0, sigmaDL);
                    Y += dY;
                }
            }
            
            logL += log(L/Pdet);
        }
    }
    
    return logL;
}

// MCMC inference of the Hubble diagram data
void lensing::Hubble_diagram_fit(cosmology &C, double DLthr, vector<vector<double> > &data, vector<double> &initial, vector<double> &steps , vector<vector<double> > &priors, int N, int Nburnin, int lens, int dm, rgen &mt, fs::path filename) {
    
    // loglikelihood
    function<double(vector<double>&)> logpdf = [this, &C, DLthr, &data, lens, dm, &mt](vector<double> &par) {
        return loglikelihood(C, DLthr, data, par, lens, dm, mt);
    };
    
    // no cut
    function<double(vector<double>&)> cut = [](vector<double> &par) {
        return 1.0;
    };
    
    MCMC_sampling(N, Nburnin, logpdf, initial, steps, priors, cut, mt, 1, 0, filename);
}
