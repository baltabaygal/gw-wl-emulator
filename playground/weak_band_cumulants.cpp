// Campbell cumulants of the sub-threshold (weak) band vs kappa_thr, z_s fixed.
//
// k_n(kappa_thr) = int n(M,z) kappa^n dA dchi over the band annuli
// [rmax(kappa_thr) .. r(kappa_min)] — same walk/measure as production
// weakMomentsNFW. k2 = the shot variance (exactly compensated by the weak
// arm); k3, k4 = the Poisson skewness/kurtosis the Gaussianized weak arm
// DISCARDS. Normalizing by the kappa_thr = 1 row (the certified kappa<1
// core) gives the split-quality knob f_n(kappa_thr) = k_n(band)/k_n(core):
// the fraction of core non-Gaussianity handed to the Gaussian arm.
//
// Build (repo root):
//   c++ -std=c++17 -O2 -Icpp -I/opt/homebrew/include \
//     playground/weak_band_cumulants.cpp cpp/cosmology.cpp cpp/basics.cpp cpp/subhalo.cpp \
//     -L/opt/homebrew/lib -lgsl -lgslcblas -o build/weak_band_cumulants
// Run:  build/weak_band_cumulants [zs=1.0]
// Writes playground/weak_band_cumulants_zs<zs>.txt

#include "lensing.cpp"  // production internals: weakMomentsNFW conventions, rmaxfNFW, ...

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

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
    const double kthr_fid = findkappathr(100, NfNFW);
    const double kappa_min = 1.0e-3*kthr_fid;           // production absolute floor
    std::printf("zs = %g   kappa_thr(N=100) = %.6e   kappa_min = %.3e\n",
                zs, kthr_fid, kappa_min);

    // quarter-decade grid; the kt = 1 row is the kappa<1 core reference
    std::vector<double> kthr;
    for (int i = 0; i <= 28; i++) kthr.push_back(std::pow(10.0, -6.0 + 7.0*i/28.0));

    char fn[256];
    std::snprintf(fn, sizeof(fn), "playground/weak_band_cumulants_zs%g.txt", zs);
    FILE *fp = std::fopen(fn, "w");
    std::fprintf(fp, "# band Campbell cumulants k_n = int n kappa^n dA dchi over "
                 "[kappa_min, kappa_thr], zs=%g, kappa_min=%.6e\n", zs, kappa_min);
    std::fprintf(fp, "# kappa_thr  k1  k2  k3  k4\n");

    const double dlnr = 0.01;
    const double Edlnr = std::exp(dlnr);
    for (double kt : kthr) {
        const double eps_abs = kappa_min;
        double s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0;
        for (int jz = 1; jz < C.Nz; jz++) {
            const double zl = C.zlist[jz];
            if (zl >= zs) continue;
            const double dz = zl - C.zlist[jz-1];
            const double Sigmac = Sigmacf(C, zs, zl);
            for (int jM = 1; jM < C.NM; jM++) {
                const double M = C.Mlist[jM];
                const double dlnM = std::log(M) - std::log(C.Mlist[jM-1]);
                const double dndlnM = C.HMFlist[jz][jM][0];

                double r = rmaxfNFW(C, zs, zl, M, kt);
                if (r == 0.0) r = 1.0e-6;               // production convention

                std::vector<double> NFWp = interpolate2(zl, M, C.zlist, C.Mlist, C.NFWlist);
                const double rs = NFWp[0];
                const double kappa0 = kappa0NFW(rs, NFWp[1], Sigmac);

                double kappar = kt;
                while (kappar > eps_abs) {
                    kappar = kappagammaNFWeps(0.0, kappa0, r/rs, 0.0)[0];
                    const double pref = CLIGHT*2.0*PI*std::pow((1.0+zl)*r, 2.0)/C.Hz(zl)
                                        *dndlnM*dlnr*dlnM*dz;
                    const double w1 = pref*kappar;
                    s1 += w1;
                    s2 += w1*kappar;
                    s3 += w1*kappar*kappar;
                    s4 += w1*kappar*kappar*kappar;
                    r *= Edlnr;
                }
            }
        }
        std::fprintf(fp, "%.6e  %.6e %.6e %.6e %.6e\n", kt, s1, s2, s3, s4);
        std::printf("kt=%.3e  k2=%.4e  k3=%.4e  k4=%.4e\n", kt, s2, s3, s4);
        std::fflush(fp);
    }
    std::fclose(fp);
    std::printf("wrote %s\n", fn);
    return 0;
}
