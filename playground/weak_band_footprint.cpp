// Transverse footprint of the weak (sub-threshold) band vs kappa_thr, z_s fixed.
//
// For each kappa_thr, walks the same annuli as production's weakMomentsNFW
// (log annuli r -> r e^dlnr from rmax(M, z, kappa_thr) down to the ABSOLUTE
// floor kappa_min = 1e-3*kappa_thr(N=100)) and histograms the transverse
// radius r of every contribution under two weights:
//   clust : a_c * dm = Dg(z) b(M,z) * [nbar kappa dA dchi]  — what the
//           correlated-field (corr) term responds to (per-shell clustering
//           std = sum_M a m, footprint of that sum);
//   shot  : dv = nbar kappa^2 dA dchi                        — what the
//           Campbell shot term responds to.
// Writes weighted r-quantiles (q10/q50/q90) per kappa_thr. The point of the
// figure: the corr footprint is pinned at the floor-set outer scale (moves
// only logarithmically with kappa_thr), while the shot footprint tracks the
// explicit-arm boundary rmax ~ kappa_thr^{-1/2} — i.e. kappa_thr shapes the
// SPLIT geometry, not the scale of the clustering physics R_perp models.
//
// Build (repo root):
//   c++ -std=c++17 -O2 -Icpp -I/opt/homebrew/include \
//     playground/weak_band_footprint.cpp cpp/cosmology.cpp cpp/basics.cpp cpp/subhalo.cpp \
//     -L/opt/homebrew/lib -lgsl -lgslcblas -o build/weak_band_footprint
// Run:  build/weak_band_footprint [zs=1.0]
// Writes playground/weak_band_footprint_zs<zs>.txt

#include "lensing.cpp"  // production internals: weakMomentsNFW conventions, rmaxfNFW, ...

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

// ln r histogram: 1e-2 .. 1e7 kpc, 20 bins/decade
static const double LR_LO = std::log(1.0e-2);
static const double LR_HI = std::log(1.0e7);
static const int NBIN = 180;

static double quantile(const std::vector<double> &h, double q) {
    double tot = 0.0;
    for (double x : h) tot += x;
    if (tot <= 0.0) return 0.0;
    double acc = 0.0;
    for (int b = 0; b < NBIN; b++) {
        acc += h[b];
        if (acc >= q*tot) {
            // linear interp inside the bin
            const double frac = (h[b] > 0.0) ? (acc - q*tot)/h[b] : 0.0;
            const double lr = LR_LO + (LR_HI - LR_LO)*(b + 1.0 - frac)/NBIN;
            return std::exp(lr);
        }
    }
    return std::exp(LR_HI);
}

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

    // quarter-decade grid inside the model domain (band empty at kt <= kappa_min)
    std::vector<double> kthr;
    for (int i = 0; i <= 28; i++) kthr.push_back(std::pow(10.0, -6.0 + 7.0*i/28.0));

    char fn[256];
    std::snprintf(fn, sizeof(fn), "playground/weak_band_footprint_zs%g.txt", zs);
    FILE *fp = std::fopen(fn, "w");
    std::fprintf(fp, "# weak-band transverse footprint, zs=%g, kappa_min=%.6e (absolute)\n",
                 zs, kappa_min);
    std::fprintf(fp, "# kappa_thr  rC_q10  rC_q50  rC_q90  rS_q10  rS_q50  rS_q90  "
                 "sumClust  sumShot   [r in comoving kpc]\n");

    const double dlnr = 0.01;
    const double Edlnr = std::exp(dlnr);
    for (double kt : kthr) {
        const double eps_abs = kappa_min;               // absolute floor in kappa
        std::vector<double> hC(NBIN, 0.0), hS(NBIN, 0.0);
        for (int jz = 1; jz < C.Nz; jz++) {
            const double zl = C.zlist[jz];
            if (zl >= zs) continue;
            const double dz = zl - C.zlist[jz-1];
            const double Dgz = C.Dg(zl);
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
                const double a = Dgz*C.halobias(zl, C.sigmalist[jM][1]);

                double kappar = kt;
                while (kappar > eps_abs) {
                    kappar = kappagammaNFWeps(0.0, kappa0, r/rs, 0.0)[0];
                    const double pref = CLIGHT*2.0*PI*std::pow((1.0+zl)*r, 2.0)/C.Hz(zl)
                                        *dndlnM*dlnr*dlnM*dz;
                    const double lr = std::log(r);
                    int b = static_cast<int>((lr - LR_LO)/(LR_HI - LR_LO)*NBIN);
                    if (b < 0) b = 0;
                    if (b >= NBIN) b = NBIN - 1;
                    hC[b] += a*pref*kappar;
                    hS[b] += pref*kappar*kappar;
                    r *= Edlnr;
                }
            }
        }
        double sC = 0.0, sS = 0.0;
        for (int b = 0; b < NBIN; b++) { sC += hC[b]; sS += hS[b]; }
        std::fprintf(fp, "%.6e  %.6e %.6e %.6e  %.6e %.6e %.6e  %.6e %.6e\n",
                     kt, quantile(hC, 0.10), quantile(hC, 0.50), quantile(hC, 0.90),
                     quantile(hS, 0.10), quantile(hS, 0.50), quantile(hS, 0.90), sC, sS);
        std::printf("kt=%.3e  rC(q50)=%.4e  rS(q50)=%.4e kpc\n",
                    kt, quantile(hC, 0.50), quantile(hS, 0.50));
        std::fflush(fp);
    }
    std::fclose(fp);
    std::printf("wrote %s\n", fn);
    return 0;
}
