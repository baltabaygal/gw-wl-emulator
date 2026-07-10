// Probe the Wsub tables: psi_lo(y), muW(y), sW(y) for one host bin at several
// subhalo_factor values. Build:
//   c++ -std=c++17 -O2 -Icpp -I/opt/homebrew/include tmp/wsub_table_probe.cpp \
//     build/libgwcore.a -L/opt/homebrew/lib -lgsl -lgslcblas -o build/wsub_table_probe
#include "cosmology.h"
#include "lensing.h"
#include "subhalo.h"
#include <cstdio>
#include <cmath>
#include <algorithm>

int main() {
    cosmology C;
    C.OmegaM = 0.315; C.sigma8 = 0.811; C.h = 0.674;
    C.OmegaB = 0.0493; C.zeq = 3402.0; C.T0 = 2.7255; C.ns = 0.965;
    C.Mmin = 1e7; C.Mmax = 1e17; C.NM = 100;
    C.zmin = 0.01; C.zmax = 10.01; C.Nz = 100; C.outdir = "dataL";
    C.initialize(0);

    double zs = 1.0, kthr_host = 1.0e-3;
    // pick a bin: zl ~ 0.5, M ~ 1e13
    int jz = 0, jM = 0;
    for (int j = 0; j < C.Nz; j++) if (C.zlist[j] < 0.5) jz = j;
    for (int j = 0; j < C.NM; j++) if (C.Mlist[j] < 1.0e13) jM = j;
    printf("bin: zl=%.3f M=%.3e\n", C.zlist[jz], C.Mlist[jM]);

    for (double f : {1e-5, 1e-3, 1e-1, 1.0}) {
        Subhalo S;
        S.m_floor = 1e7;
        S.precompute(C, zs, f * kthr_host, kthr_host);
        double M = C.Mlist[jM];
        const auto &rth = S.r_thr[jz];
        printf("factor=%g  fsb=%.4f  rth[0]=%.3g rth[50]=%.3g rth[last<M]=?\n",
               f, S.fsb[jz][jM], rth[0], rth[50]);
        double ly0 = S.lyW[jz][jM][0], dly = S.lyW[jz][jM][1];
        int Ny = (int)S.muW[jz][jM].size();
        if (Ny == 0) { printf("  (empty tables)\n"); continue; }
        for (int iy : {0, Ny/2, Ny-1}) {
            double y = std::exp(ly0 + iy * dly);
            int jlo = (int)(std::lower_bound(rth.begin(), rth.end(), y) - rth.begin());
            double psi_lo = (jlo >= C.NM) ? 1.0 : std::max(C.Mlist[jlo], C.Mmin) / M;
            printf("  y=%9.3f  jlo=%3d  psi_lo=%9.3g  muW=%10.4g  sW=%10.4g\n",
                   y, jlo, std::min(psi_lo, 1.0), S.muW[jz][jM][iy], S.sW[jz][jM][iy]);
        }
        // sorted check
        bool sorted = std::is_sorted(rth.begin(), rth.end());
        printf("  r_thr sorted: %s\n", sorted ? "yes" : "NO");
    }
    return 0;
}
