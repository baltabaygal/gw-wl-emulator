// MC-measured variance of the EXPLICIT (strong, kappa > kappa_thr) halo sum vs
// kappa_thr, to complete the sigmaW_vs_kthr picture: by band additivity of the
// Campbell variance, Var_explicit(kt) + Var_background(kt; fixed kappa_min)
// should equal the flat full variance (the plateau of sigmaW_vs_kthr.cpp).
//
// Sampling follows the production conventions exactly (same as
// sigmakappaw_poisson_vs_fixed.cpp, but for the disk r < rmax instead of the
// weak band): per (z,M) cell the expected count is the disk formula
// lambda = 306.535*PI/Hz * dndlnM*dlnM*dz * ((1+zl)*rmax)^2, counts are Poisson,
// positions uniform in area, kappa from kappagammaNFWeps.
//
// The explicit sum is heavy-tailed (rare high-kappa encounters carry most of the
// variance), so (a) the realization count is scaled adaptively so every threshold
// sees enough halo events, (b) counts are drawn as ONE total Poisson(lambda_tot)
// per realization with halos assigned to cells by the mixture lambda_c/lambda_tot
// (exact, by Poisson superposition; far faster than per-cell draws when most
// lambda_c << 1), and (c) the error on Var is the honest heavy-tail estimate
// Var(s^2) = (m4 - m2^2)/N from the measured fourth moment, not the Gaussian 2/N.
//
// Build (repo root, after `make build`):
//   c++ -std=c++17 -O2 -Icpp -I/opt/homebrew/include \
//     playground/sigma_explicit_vs_kthr.cpp -Lbuild -lgwcore \
//     -L/opt/homebrew/lib -lgsl -lgslcblas -o build/sigma_explicit_vs_kthr
// Run:  build/sigma_explicit_vs_kthr [zs=1.0] [seed=20260708]
// Writes playground/sigma_explicit_vs_kthr_zs<zs>.txt
//   (columns: kappa_thr  n_expected  sigma_explicit_MC  relerr(Var)  Nreal)

#include "cosmology.h"
#include "lensing.h"

#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

double Sigmacf(cosmology &C, double zs, double zl);
double NhfNFW(cosmology &C, double zs, double kappathr);
double findkappathr(int N, std::function<double(double)> Nf);

namespace {
struct Cell {
    double rmax2;   // disk edge squared (kappa = kappa_thr there)
    double rs;
    double kappa0;
    double lambda;  // expected count in the disk
};
}  // namespace

int main(int argc, char **argv) {
    const double zs = (argc > 1) ? std::atof(argv[1]) : 1.0;
    const uint64_t seed = (argc > 2) ? std::strtoull(argv[2], nullptr, 10) : 20260708ULL;

    cosmology C;
    C.OmegaM = 0.315; C.OmegaB = 0.0493; C.zeq = 3402.0; C.sigma8 = 0.811;
    C.h = 0.674; C.T0 = 2.7255; C.ns = 0.965;
    C.Mmin = 1.0e7; C.Mmax = 1.0e17; C.NM = 100;
    C.zmin = 0.01; C.zmax = 10.01; C.Nz = 100;
    C.outdir = "dataL";
    C.initialize(0);

    // same kappa_thr grid as sigmaW_vs_kthr.cpp
    std::vector<double> kthr;
    for (int i = 0; i <= 32; i++) kthr.push_back(std::pow(10.0, -5.0 + 8.0 * i / 32.0));

    char fn[256];
    std::snprintf(fn, sizeof(fn), "playground/sigma_explicit_vs_kthr_zs%g.txt", zs);
    FILE *fp = std::fopen(fn, "w");
    std::fprintf(fp, "# kappa_thr   n_expected   sigma_explicit_MC   relerr(Var)   Nreal\n");

    std::mt19937_64 mt(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);

    std::printf("%-12s %-12s %-18s %-12s %-10s\n",
                "kappa_thr", "n_expected", "sigma_explicit_MC", "relerr(Var)", "Nreal");
    for (double kt : kthr) {
        // build the strong-disk cells for this threshold
        std::vector<Cell> cells;
        double n_tot = 0.0;
        for (int jz = 1; jz < C.Nz; jz++) {
            const double zl = C.zlist[jz];
            const double dz = zl - C.zlist[jz - 1];
            if (zl >= zs) continue;
            for (int jM = 1; jM < C.NM; jM++) {
                const double M = C.Mlist[jM];
                const double dlnM = std::log(M) - std::log(C.Mlist[jM - 1]);
                const double dndlnM = C.HMFlist[jz][jM][0];

                const double rmax = rmaxfNFW(C, zs, zl, M, kt);
                if (rmax <= 0.0) continue;  // halo never reaches kappa_thr

                const double Sigmac = Sigmacf(C, zs, zl);
                std::vector<double> NFWp = interpolate2(zl, M, C.zlist, C.Mlist, C.NFWlist);
                const double rs = NFWp[0];
                const double kappa0 = kappa0NFW(rs, NFWp[1], Sigmac);

                const double zf2 = (1.0 + zl) * (1.0 + zl);
                const double lambda =
                    306.535 * PI / C.Hz(zl) * dndlnM * dlnM * dz * zf2 * rmax * rmax;
                if (lambda > 0.0) cells.push_back({rmax * rmax, rs, kappa0, lambda});
                n_tot += lambda;
            }
        }

        if (n_tot <= 0.0 || cells.empty()) {
            std::fprintf(fp, "%.6e %.6e %.8e %.4f %ld\n", kt, n_tot, 0.0, 0.0, 0L);
            std::printf("%-12.3e %-12.4e %-18.6e %-12s %-10s\n", kt, n_tot, 0.0, "-", "-");
            std::fflush(fp);
            continue;
        }

        // adaptive realization count: aim for ~3000 halo events, floor 5e4, cap 5e6
        long Nreal = std::lround(3000.0 / n_tot);
        if (Nreal < 50000) Nreal = 50000;
        if (Nreal > 5000000) Nreal = 5000000;

        // Poisson MC of the explicit sum. Total count ~ Poisson(n_tot), each halo
        // assigned to a cell with prob lambda_c/n_tot (exact by superposition).
        std::vector<double> cum(cells.size());
        {
            double acc = 0.0;
            for (size_t ic = 0; ic < cells.size(); ic++) { acc += cells[ic].lambda; cum[ic] = acc; }
        }
        std::poisson_distribution<long> Ptot(n_tot);
        double s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0;  // raw moments of the per-realization sum
        for (long j = 0; j < Nreal; j++) {
            const long N = Ptot(mt);
            double kap = 0.0;
            for (long i = 0; i < N; i++) {
                const double u = U(mt) * cum.back();
                const size_t ic = std::lower_bound(cum.begin(), cum.end(), u) - cum.begin();
                const Cell &c = cells[ic];
                const double rr2 = U(mt) * c.rmax2;  // uniform in area
                kap += kappagammaNFWeps(0.0, c.kappa0, std::sqrt(rr2) / c.rs, 0.0)[0];
            }
            s1 += kap; s2 += kap * kap; s3 += kap * kap * kap; s4 += kap * kap * kap * kap;
        }
        const double m1 = s1 / Nreal;
        const double var = s2 / Nreal - m1 * m1;
        const double sig = std::sqrt(std::max(var, 0.0));
        // central fourth moment and the heavy-tail error on the variance estimate
        const double m4c = s4 / Nreal - 4.0 * m1 * s3 / Nreal + 6.0 * m1 * m1 * s2 / Nreal
                           - 3.0 * m1 * m1 * m1 * m1;
        const double relerr = (var > 0.0)
            ? std::sqrt(std::max(m4c - var * var, 0.0) / Nreal) / var : 0.0;
        std::fprintf(fp, "%.6e %.6e %.8e %.4f %ld\n", kt, n_tot, sig, relerr, Nreal);
        std::printf("%-12.3e %-12.4e %-18.6e %-12.4f %-10ld\n", kt, n_tot, sig, relerr, Nreal);
        std::fflush(stdout);
        std::fflush(fp);
    }
    std::fclose(fp);
    std::printf("\nwrote %s\n", fn);
    return 0;
}
