// Consistency probe for the radial measure in sigmakappaW (lensing.cpp:157-193).
//
// Every ingredient is taken from the production code itself:
//   - halo counting:   dN = 306.535*PI*((1+z)r)^2/Hz * dndlnM*dlnM*dz   (lensing.cpp:121, 149)
//   - halo positions:  uniform in area, r = sqrt(U)*rmax                (lensing.cpp:533)
//   - kappa kernel:    kappagammaNFWeps with kappa0NFW/Sigmacf/NFWlist  (as in sigmakappaW)
//   - weak-band domain: from rmax (kappa = kappa_thr) out to the radius
//     where the loop's own 0.001*kappa_thr stop condition triggers
//
// Outputs, for the fiducial cosmology at given zs and kappa_thr from findkappathr(100):
//   A. expected number of weak-band halos per sightline:
//        (a1) sigmakappaW's loop measure  sum PI*r^2*dlnr        [replicated verbatim]
//        (a2) the code's own disk formula PI*(rstop^2 - rmax^2)  [same domain]
//      If the loop measure were correct these would agree; ratio a1/a2 is ~0.5.
//   B. self-check: the replicated loop reproduces sigmakappaW() to float precision.
//   C. brute-force Monte Carlo: sample the weak-band halos explicitly (counts from
//      the disk formula, positions uniform in area) and measure Var(sum kappa).
//      Compare with sigmakappaW()^2 and with the Campbell integral 2*k2 (i.e. the
//      loop's kappa^2 accumulator with the 2*pi*r^2*dlnr measure, no k1^2/Nh term).
//
// Build (from repo root; needs build/libgwcore.a from `make build`):
//   clang++ -std=c++17 -O2 -Icpp -I/opt/homebrew/include playground/sigmakappaw_probe.cpp \
//     -Lbuild -lgwcore -L/opt/homebrew/lib -lgsl -lgslcblas -o build/sigmakappaw_probe
// Run:  build/sigmakappaw_probe [zs=1.0] [Nreal=2000]

#include "cosmology.h"  // pulls in basics.h (which has no include guard)
#include "lensing.h"

#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

// in lensing.cpp but not declared in lensing.h
double Sigmacf(cosmology &C, double zs, double zl);
double NhfNFW(cosmology &C, double zs, double kappathr);
double findkappathr(int N, std::function<double(double)> Nf);

namespace {

struct Cell {
    double rmax;    // inner edge of the weak band (kappa = kappa_thr), or 1e-6
    double rstop;   // outer edge (loop's 0.001*kappa_thr stop)
    double rs;      // NFW scale radius
    double kappa0;  // NFW kappa normalization
    double lambda;  // expected halo count in the band, disk formula
};

cosmology make_planck_cosmology() {
    cosmology C;
    C.OmegaM = 0.315;
    C.OmegaB = 0.0493;
    C.zeq = 3402.0;
    C.sigma8 = 0.811;
    C.h = 0.674;
    C.T0 = 2.7255;
    C.ns = 0.965;
    C.Mmin = 1.0e7;
    C.Mmax = 1.0e17;
    C.NM = 100;
    C.zmin = 0.01;
    C.zmax = 10.01;
    C.Nz = 100;
    C.outdir = "dataL";
    C.initialize(0);
    return C;
}

}  // namespace

int main(int argc, char **argv) {
    const double zs = (argc > 1) ? std::atof(argv[1]) : 1.0;
    const int Nreal = (argc > 2) ? std::atoi(argv[2]) : 2000;

    cosmology C = make_planck_cosmology();
    std::function<double(double)> NfNFW = [&C, zs](double kappa) { return NhfNFW(C, zs, kappa); };
    const double kappathr = findkappathr(100, NfNFW);
    std::cout << "zs = " << zs << "   kappa_thr(N=100) = " << kappathr << "\n";

    // ---- Part A + B: replicate the sigmakappaW loop verbatim, and in parallel
    //      accumulate the disk-formula count over the same domain.
    const double dlnr = 0.01;
    const double Edlnr = std::exp(dlnr);
    double Nh_loop = 0.0, k1_loop = 0.0, k2_loop = 0.0;  // his measure PI*r^2*dlnr
    double Nh_disk = 0.0;                                // his disk formula, same domain
    std::vector<Cell> cells;

    for (int jz = 1; jz < C.Nz; jz++) {
        const double zl = C.zlist[jz];
        const double dz = zl - C.zlist[jz - 1];
        if (zl >= zs) continue;
        for (int jM = 1; jM < C.NM; jM++) {
            const double M = C.Mlist[jM];
            const double dlnM = std::log(M) - std::log(C.Mlist[jM - 1]);
            const double dndlnM = C.HMFlist[jz][jM][0];

            double r = rmaxfNFW(C, zs, zl, M, kappathr);
            if (r == 0.0) r = 1.0e-6;
            const double rmax = r;

            const double Sigmac = Sigmacf(C, zs, zl);
            std::vector<double> NFWp = interpolate2(zl, M, C.zlist, C.Mlist, C.NFWlist);
            const double rs = NFWp[0];
            const double kappa0 = kappa0NFW(rs, NFWp[1], Sigmac);

            const double pref = 306.535 * PI / C.Hz(zl) * dndlnM * dlnM * dz;
            double kappar = kappathr;
            while (kappar > 0.001 * kappathr) {  // verbatim sigmakappaW loop
                kappar = kappagammaNFWeps(0.0, kappa0, r / rs, 0.0)[0];
                Nh_loop += pref * std::pow((1.0 + zl) * r, 2.0) * dlnr;
                k1_loop += pref * std::pow((1.0 + zl) * r, 2.0) * kappar * dlnr;
                k2_loop += pref * std::pow((1.0 + zl) * r, 2.0) * kappar * kappar * dlnr;
                r = r * Edlnr;
            }
            const double rstop = r;

            // the code's own counting formula (lensing.cpp:121,149) on the same band
            const double lambda =
                pref * (std::pow((1.0 + zl) * rstop, 2.0) - std::pow((1.0 + zl) * rmax, 2.0));
            Nh_disk += lambda;
            if (lambda > 0.0) cells.push_back({rmax, rstop, rs, kappa0, lambda});
        }
    }

    const double sigma_replicated = std::sqrt(k2_loop - k1_loop * k1_loop / Nh_loop);
    const double sigma_code = sigmakappaW(C, zs, kappathr);

    std::cout.precision(6);
    std::cout << "\n[B] self-check: replicated loop sigma = " << sigma_replicated
              << "   sigmakappaW() = " << sigma_code
              << "   (must agree)\n";
    std::cout << "\n[A] weak-band halo count per sightline, same domain, same prefactor:\n"
              << "    loop measure  sum PI*r^2*dlnr      : " << Nh_loop << "\n"
              << "    disk formula  PI*(rstop^2-rmax^2)  : " << Nh_disk << "\n"
              << "    ratio loop/disk = " << Nh_loop / Nh_disk << "   (0.5 = missing factor 2)\n";

    // ---- Part C: brute-force MC of the weak kappa sum.
    std::mt19937_64 mt(20260708);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    double s1 = 0.0, s2 = 0.0;
    double count_accum = 0.0;
    for (int j = 0; j < Nreal; j++) {
        double kap = 0.0;
        for (const Cell &c : cells) {
            std::poisson_distribution<int> pois(c.lambda);
            const int N = pois(mt);
            count_accum += N;
            for (int i = 0; i < N; i++) {
                // uniform in area within the band, the code's own convention (lensing.cpp:533)
                const double rr =
                    std::sqrt(c.rmax * c.rmax + U(mt) * (c.rstop * c.rstop - c.rmax * c.rmax));
                kap += kappagammaNFWeps(0.0, c.kappa0, rr / c.rs, 0.0)[0];
            }
        }
        s1 += kap;
        s2 += kap * kap;
    }
    const double mean_mc = s1 / Nreal;
    const double var_mc = s2 / Nreal - mean_mc * mean_mc;
    const double var_err = var_mc * std::sqrt(2.0 / (Nreal - 1.0));

    std::cout << "\n[C] brute-force MC, " << Nreal << " realizations, mean weak-band halos/realization = "
              << count_accum / Nreal << "\n"
              << "    Var(sum kappa) MC          : " << var_mc << " +- " << var_err << "\n"
              << "    sigmakappaW()^2            : " << sigma_code * sigma_code
              << "   -> MC/code = " << var_mc / (sigma_code * sigma_code) << "\n"
              << "    Campbell 2*k2 (2pi, no k1) : " << 2.0 * k2_loop
              << "   -> MC/Campbell = " << var_mc / (2.0 * k2_loop) << "\n"
              << "    (mean of sum kappa MC = " << mean_mc << ", Campbell mean 2*k1 = "
              << 2.0 * k1_loop << ")\n";
    return 0;
}
