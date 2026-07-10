// Scan ALL bins for pathological Wsub table values.
#include "cosmology.h"
#include "lensing.h"
#include "subhalo.h"
#include <cstdio>
#include <cmath>
int main() {
    cosmology C;
    C.OmegaM = 0.315; C.sigma8 = 0.811; C.h = 0.674;
    C.OmegaB = 0.0493; C.zeq = 3402.0; C.T0 = 2.7255; C.ns = 0.965;
    C.Mmin = 1e7; C.Mmax = 1e17; C.NM = 100;
    C.zmin = 0.01; C.zmax = 10.01; C.Nz = 100; C.outdir = "dataL";
    C.initialize(0);
    double zs = 1.0, kthr_host = 1.0e-3;
    for (double f : {1e-3, 1.0}) {
        Subhalo S; S.m_floor = 1e7;
        S.precompute(C, zs, f * kthr_host, kthr_host);
        double smax = 0.0, mumax = 0.0; int bz=-1, bM=-1, bzm=-1, bMm=-1;
        for (int jz = 0; jz < C.Nz; jz++)
            for (int jM = 0; jM < C.NM; jM++)
                for (double v : S.sW[jz][jM]) if (v > smax) { smax = v; bz=jz; bM=jM; }
        for (int jz = 0; jz < C.Nz; jz++)
            for (int jM = 0; jM < C.NM; jM++)
                for (double v : S.muW[jz][jM]) if (v > mumax) { mumax = v; bzm=jz; bMm=jM; }
        printf("factor=%g: max sW=%.4g at (zl=%.3f, M=%.2e); max muW=%.4g at (zl=%.3f, M=%.2e)\n",
               f, smax, C.zlist[bz], C.Mlist[bM], mumax, C.zlist[bzm], C.Mlist[bMm]);
        // dump the worst bin's table
        double ly0 = S.lyW[bz][bM][0], dly = S.lyW[bz][bM][1];
        for (int iy = 0; iy < (int)S.sW[bz][bM].size(); iy += 8)
            printf("   y=%10.3f  muW=%10.4g  sW=%10.4g\n",
                   std::exp(ly0 + iy*dly), S.muW[bz][bM][iy], S.sW[bz][bM][iy]);
    }
    return 0;
}
