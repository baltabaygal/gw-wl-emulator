// Host-population weight probe (2026-07-27): dump the per-(z_l, M) expected host
// count that NhfNFW integrates, so the analytic kappa_thr,sub sweep can be weighted
// by the REAL production host population instead of a re-derived Python HMF.
//
// NhfNFW (cpp/lensing.cpp:138) accumulates
//   dNh(jz,jM) = c*pi*((1+zl)*rmax)^2 / H(zl) * dndlnM * dlnM * dz
// with rmax = rmaxfNFW(C, zs, zl, M, kappathr). This probe prints exactly that
// summand per grid cell, plus the running total (which must reproduce NhfNFW).
//
// Build (repo root; same pattern as engine_checkpoint_probe):
//   clang++ -std=c++17 -O2 -I/opt/homebrew/include -L/opt/homebrew/lib \
//     playground/host_weight_probe.cpp cpp/basics.cpp cpp/cosmology.cpp \
//     cpp/lensing.cpp cpp/subhalo.cpp -lgsl -lgslcblas \
//     -o playground/host_weight_probe
// Run (repo root; uses the dataL/ cosmology cache):
//   ./playground/host_weight_probe <zs> [kappathr]
//   kappathr <= 0  ->  use the legacy fixed-<N>=100 threshold for this zs
//
// Output: "# " header lines, then "W zl M dNh" rows (kpc/Msun engine units).

#include "../cpp/cosmology.h"
#include "../cpp/lensing.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>

double NhfNFW(cosmology &C, double zs, double kappathr);            // lensing.cpp
double rmaxfNFW(cosmology &C, double zs, double zl, double M, double kappathr);
double findkappathr(int N, std::function<double(double)> Nf);       // lensing.cpp

int main(int argc, char **argv) {
    const double zs = (argc > 1) ? atof(argv[1]) : 1.0;
    double kthr = (argc > 2) ? atof(argv[2]) : -1.0;

    cosmology C;
    C.OmegaM = 0.315; C.sigma8 = 0.811; C.h = 0.674;
    C.As = -1.0; C.OmegaB = 0.0493; C.zeq = 3402.0; C.T0 = 2.7255; C.ns = 0.965;
    C.Mmin = 1e7; C.Mmax = 1e17; C.NM = 100;
    C.zmin = 0.01; C.zmax = 10.01; C.Nz = 100;
    C.outdir = "dataL";
    C.initialize(0);

    if (kthr <= 0.0) {
        // legacy fixed-<N>=100 rule, exactly as lensing.cpp:728-733 builds it
        std::function<double(double)> NfNFW = [&C, zs](double kappa) {
            return NhfNFW(C, zs, kappa);
        };
        kthr = findkappathr(100, NfNFW);
    }

    printf("# zs %.6g\n", zs);
    printf("# kappathr %.10e\n", kthr);
    printf("# NhfNFW %.10e\n", NhfNFW(C, zs, kthr));
    printf("# Nz %d NM %d\n", C.Nz, C.NM);
    printf("# columns: W zl M dNh rmax_kpc\n");

    double total = 0.0;
    for (int jz = 1; jz < C.Nz; jz++) {
        const double zl = C.zlist[jz];
        const double dz = zl - C.zlist[jz - 1];
        if (zl >= zs) continue;
        for (int jM = 1; jM < C.NM; jM++) {
            const double M = C.Mlist[jM];
            const double dlnM = log(M) - log(C.Mlist[jM - 1]);
            const double dndlnM = C.HMFlist[jz][jM][0];
            const double rmax = rmaxfNFW(C, zs, zl, M, kthr);
            const double dNh = CLIGHT * PI * pow((1.0 + zl) * rmax, 2.0) / C.Hz(zl)
                               * dndlnM * dlnM * dz;
            total += dNh;
            printf("W %.10e %.10e %.10e %.10e\n", zl, M, dNh, rmax);
        }
    }
    printf("# total %.10e\n", total);
    return 0;
}
