// Analytic reduction of the full Campbell variance sigma_full(zs).
//
// sigmakappaW's integrand is n(M,z) * kappa^2 over the impact parameter r. Because
// the NFW kappa profile is self-similar, kappa(r) = kappa0(M,z) * K(r/rs) with a
// universal shape K(x) (= 2 Fg(x) in the code's convention, kappa0 = rs*rhos/Sigma_c),
// the r-integral factorizes into a PURE NUMBER:
//
//   int_0^inf kappa^2 2 pi r dr = 2 pi kappa0^2 rs^2 * C2,   C2 = int_0^inf K(x)^2 x dx
//
// so the full (all-halo) Campbell variance collapses to a 1D redshift integral:
//
//   sigma^2(zs) = 2 pi C2 * int_0^zs dz 306.535/Hz(z) (1+z)^2
//                     int dlnM dn/dlnM(M,z) [rhos(M,z) rs(M,z)^2 / Sigma_c(z,zs)]^2
//
// This probe computes C2 once (fine log grid) and the reduced integral on the same
// zs grid as sigma_full_vs_zs.txt, for comparison against the brute plateau values.
//
// Build (repo root, after `make build`):
//   c++ -std=c++17 -O2 -Icpp -I/opt/homebrew/include \
//     playground/campbell_reduced.cpp -Lbuild -lgwcore \
//     -L/opt/homebrew/lib -lgsl -lgslcblas -o build/campbell_reduced
// Run:  build/campbell_reduced
// Writes playground/campbell_reduced_vs_zs.txt (columns: zs  sigma_reduced)

#include "cosmology.h"
#include "lensing.h"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

double Sigmacf(cosmology &C, double zs, double zl);

int main() {
    cosmology C;
    C.OmegaM = 0.315; C.OmegaB = 0.0493; C.zeq = 3402.0; C.sigma8 = 0.811;
    C.h = 0.674; C.T0 = 2.7255; C.ns = 0.965;
    C.Mmin = 1.0e7; C.Mmax = 1.0e17; C.NM = 100;
    C.zmin = 0.01; C.zmax = 10.01; C.Nz = 100;
    C.outdir = "dataL";
    C.initialize(0);

    // ---- C2 = int_0^inf K(x)^2 x dx with K(x) = kappa(x)/kappa0 (kappa0 = 1 here)
    double C2 = 0.0;
    const double dlnx = 1.0e-3;
    for (double lnx = std::log(1.0e-8); lnx < std::log(1.0e8); lnx += dlnx) {
        const double x = std::exp(lnx);
        const double K = kappagammaNFWeps(0.0, 1.0, x, 0.0)[0];
        C2 += K * K * x * x * dlnx;  // K^2 x dx = K^2 x^2 dlnx
    }
    std::printf("C2 = int K^2 x dx = %.8f   (K = 2*Fg, i.e. C2 = 4*int Fg^2 x dx)\n", C2);

    // ---- reduced 1D integral on the sigma_full_vs_zs grid
    std::vector<double> zss;
    for (int i = 0; i <= 24; i++)
        zss.push_back(0.2 * std::pow(10.0 / 0.2, i / 24.0));

    FILE *fp = std::fopen("playground/campbell_reduced_vs_zs.txt", "w");
    std::fprintf(fp, "# C2 = %.8f\n# zs   sigma_reduced\n", C2);
    std::printf("%-8s %-14s\n", "zs", "sigma_reduced");
    for (double zs : zss) {
        double var = 0.0;
        for (int jz = 1; jz < C.Nz; jz++) {
            const double zl = C.zlist[jz];
            const double dz = zl - C.zlist[jz - 1];
            if (zl >= zs) continue;
            const double Sigmac = Sigmacf(C, zs, zl);
            const double path = 306.535 / C.Hz(zl) * (1.0 + zl) * (1.0 + zl) * dz;
            double massint = 0.0;
            for (int jM = 1; jM < C.NM; jM++) {
                const double dlnM = std::log(C.Mlist[jM]) - std::log(C.Mlist[jM - 1]);
                const double dndlnM = C.HMFlist[jz][jM][0];
                const double rs = C.NFWlist[jz][jM][0];
                const double rhos = C.NFWlist[jz][jM][1];
                const double src = rhos * rs * rs / Sigmac;  // = kappa0 * rs
                massint += dndlnM * src * src * dlnM;
            }
            var += 2.0 * PI * C2 * path * massint;
        }
        std::fprintf(fp, "%.6f %.8e\n", zs, std::sqrt(var));
        std::printf("%-8.3f %-14.6e\n", zs, std::sqrt(var));
    }
    std::fclose(fp);
    std::printf("\nwrote playground/campbell_reduced_vs_zs.txt\n");
    return 0;
}
