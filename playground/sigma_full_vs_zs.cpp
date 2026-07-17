// Dense tabulation of the full (halo-only, unclustered) convergence standard
// deviation sigma_full(zs) — the analytic Campbell variance of ALL halos:
// sigma_full = plateau of sigmakappaW(zs, kappa_thr -> large, floor fixed at
// kappa_min = 1e-3 * kappa_thr(N=100, zs)), matching the partition-study plateaus.
//
// Build (repo root, after `make build`):
//   c++ -std=c++17 -O2 -Icpp -I/opt/homebrew/include \
//     playground/sigma_full_vs_zs.cpp -Lbuild -lgwcore \
//     -L/opt/homebrew/lib -lgsl -lgslcblas -o build/sigma_full_vs_zs
// Run:  build/sigma_full_vs_zs
// Writes playground/sigma_full_vs_zs.txt (columns: zs  kappa_thr_fid  sigma_full)

#include "cosmology.h"
#include "lensing.h"

#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <vector>

double NhfNFW(cosmology &C, double zs, double kappathr);
double findkappathr(int N, std::function<double(double)> Nf);

int main() {
    cosmology C;
    C.OmegaM = 0.315; C.OmegaB = 0.0493; C.zeq = 3402.0; C.sigma8 = 0.811;
    C.h = 0.674; C.T0 = 2.7255; C.ns = 0.965;
    C.Mmin = 1.0e7; C.Mmax = 1.0e17; C.NM = 100;
    C.zmin = 0.01; C.zmax = 10.01; C.Nz = 100;
    C.outdir = "dataL";
    C.initialize(0);

    // log grid zs = 0.2 .. 10 (25 points)
    std::vector<double> zss;
    for (int i = 0; i <= 24; i++)
        zss.push_back(0.2 * std::pow(10.0 / 0.2, i / 24.0));

    FILE *fp = std::fopen("playground/sigma_full_vs_zs.txt", "w");
    std::fprintf(fp, "# zs   kappa_thr_fid(N=100)   sigma_full\n");
    std::printf("%-8s %-14s %-12s\n", "zs", "kthr_fid", "sigma_full");
    for (double zs : zss) {
        std::function<double(double)> NfNFW = [&C, zs](double k) { return NhfNFW(C, zs, k); };
        const double kfid = findkappathr(100, NfNFW);
        const double kmin = 1.0e-3 * kfid;
        const double kbig = 1.0e3;  // threshold far above any halo's central kappa
        const double sfull = sigmakappaW(C, zs, kbig, kmin / kbig);
        std::fprintf(fp, "%.6f %.6e %.8e\n", zs, kfid, sfull);
        std::printf("%-8.3f %-14.4e %-12.6f\n", zs, kfid, sfull);
        std::fflush(stdout);
        std::fflush(fp);
    }
    std::fclose(fp);
    std::printf("\nwrote playground/sigma_full_vs_zs.txt\n");
    return 0;
}
