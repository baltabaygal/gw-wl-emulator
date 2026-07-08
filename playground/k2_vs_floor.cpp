// Convergence of the weak-lens variance K2 with respect to the convergence-floor
// fraction eps_floor (sigmakappaW's outward radial cutoff: the loop integrates each
// halo outward until its kappa < eps_floor * kappa_thr).
//
// sigmakappaW is a deterministic integral, so we just call it for a grid of eps_floor
// and record K2 = sigmakappaW(...)^2. If K2 plateaus as eps_floor -> 0, the integral
// is converged and the (previously hard-coded) 0.001 default is justified.
//
// Build (repo root; needs build/libgwcore.a from `make build`, rebuilt after the
// sigmakappaW eps_floor change):
//   c++ -std=c++17 -O2 -Icpp -I/opt/homebrew/include \
//     playground/k2_vs_floor.cpp -Lbuild -lgwcore \
//     -L/opt/homebrew/lib -lgsl -lgslcblas -o build/k2_vs_floor
// Run:  build/k2_vs_floor [zs=1.0]
// Writes playground/k2_vs_floor_zs<zs>.txt  (columns: eps_floor  K2)

#include "cosmology.h"
#include "lensing.h"

#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <vector>

double NhfNFW(cosmology &C, double zs, double kappathr);
double findkappathr(int N, std::function<double(double)> Nf);

int main(int argc, char **argv) {
    const double zs = (argc > 1) ? std::atof(argv[1]) : 1.0;

    cosmology C;
    C.OmegaM = 0.315; C.OmegaB = 0.0493; C.zeq = 3402.0; C.sigma8 = 0.811;
    C.h = 0.674; C.T0 = 2.7255; C.ns = 0.965;
    C.Mmin = 1.0e7; C.Mmax = 1.0e17; C.NM = 100;
    C.zmin = 0.01; C.zmax = 10.01; C.Nz = 100;
    C.outdir = "dataL";
    C.initialize(0);

    std::function<double(double)> NfNFW = [&C, zs](double k) { return NhfNFW(C, zs, k); };
    const double kappathr = findkappathr(100, NfNFW);
    std::cout << "zs = " << zs << "   kappa_thr(N=100) = " << kappathr << "\n";

    // log-spaced eps_floor from 1e-1 down to 1e-7
    std::vector<double> eps;
    for (int i = 0; i <= 30; i++) eps.push_back(std::pow(10.0, -1.0 - 6.0 * i / 30.0));

    char fn[256];
    std::snprintf(fn, sizeof(fn), "playground/k2_vs_floor_zs%g.txt", zs);
    FILE *fp = std::fopen(fn, "w");
    std::fprintf(fp, "# eps_floor   K2\n");

    double K2_ref = std::pow(sigmakappaW(C, zs, kappathr, 1.0e-7), 2.0);  // deepest floor = reference
    std::printf("%-12s %-14s %-10s\n", "eps_floor", "K2", "K2/K2(1e-7)");
    for (double e : eps) {
        const double K2 = std::pow(sigmakappaW(C, zs, kappathr, e), 2.0);
        std::fprintf(fp, "%.6e %.8e\n", e, K2);
        std::printf("%-12.2e %-14.6e %-10.5f\n", e, K2, K2 / K2_ref);
    }
    std::fclose(fp);
    std::printf("\nwrote %s\n", fn);
    std::printf("default eps_floor=0.001 gives K2/K2(1e-7) = %.5f\n",
                std::pow(sigmakappaW(C, zs, kappathr, 1.0e-3), 2.0) / K2_ref);
    return 0;
}
