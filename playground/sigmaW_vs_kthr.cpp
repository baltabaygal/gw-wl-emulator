// Does the analytic background sigma_W collapse at large kappa_thr because the
// floor is a FIXED FRACTION of kappa_thr (kappa_min = eps_floor*kappa_thr rises
// with the threshold, discarding real variance)?
//
// Sweep kappa_thr and compute sigmakappaW two ways:
//   (a) fixed fraction:  eps_floor = 1e-3                      (what the sweep plot used)
//   (b) fixed absolute:  kappa_min = 1e-3 * kappa_thr(N=100),  eps_floor = kappa_min/kappa_thr
// If (b) plateaus at the full sigma while (a) turns over, the collapse is the
// floor artifact and scaling eps_floor ~ 1/kappa_thr fixes it.
//
// Build (repo root, after `make build`):
//   c++ -std=c++17 -O2 -Icpp -I/opt/homebrew/include \
//     playground/sigmaW_vs_kthr.cpp -Lbuild -lgwcore \
//     -L/opt/homebrew/lib -lgsl -lgslcblas -o build/sigmaW_vs_kthr
// Run:  build/sigmaW_vs_kthr [zs=1.0]
// Writes playground/sigmaW_vs_kthr_zs<zs>.txt (columns: kappa_thr sigmaW_fixedeps sigmaW_fixedkmin)

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
    const double kthr_fid = findkappathr(100, NfNFW);   // production N=100 threshold
    const double kappa_min = 1.0e-3 * kthr_fid;         // production absolute floor
    std::cout << "zs = " << zs << "   kappa_thr(N=100) = " << kthr_fid
              << "   fixed kappa_min = " << kappa_min << "\n";

    // log grid kappa_thr = 1e-5 .. 1e3 (same range as the MC sweep plot)
    std::vector<double> kthr;
    for (int i = 0; i <= 32; i++) kthr.push_back(std::pow(10.0, -5.0 + 8.0 * i / 32.0));

    char fn[256];
    std::snprintf(fn, sizeof(fn), "playground/sigmaW_vs_kthr_zs%g.txt", zs);
    FILE *fp = std::fopen(fn, "w");
    std::fprintf(fp, "# kappa_thr   sigmaW(eps=1e-3 fixed)   sigmaW(kappa_min=%.3e fixed)\n", kappa_min);

    std::printf("%-12s %-22s %-22s\n", "kappa_thr", "sigW_fixed_eps(1e-3)", "sigW_fixed_kmin");
    for (double kt : kthr) {
        const double sA = sigmakappaW(C, zs, kt, 1.0e-3);
        const double sB = sigmakappaW(C, zs, kt, kappa_min / kt);
        std::fprintf(fp, "%.6e %.8e %.8e\n", kt, sA, sB);
        std::printf("%-12.3e %-22.6e %-22.6e\n", kt, sA, sB);
        std::fflush(stdout);
    }
    std::fclose(fp);
    std::printf("\nwrote %s\n", fn);
    return 0;
}
