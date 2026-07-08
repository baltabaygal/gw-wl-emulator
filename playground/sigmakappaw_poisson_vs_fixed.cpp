// Does the -kappa1^2/Nh subtraction in sigmakappaW belong there?
//
// THE QUESTION
// ------------
// sigmakappaW returns sqrt(kappa2 - kappa1^2/Nh).  The subtraction is the
// fixed-count formula: if EXACTLY N halos are placed per sightline (N never
// fluctuates), then Var(sum kappa) = N*(<kappa^2> - <kappa>^2) = kappa2 - kappa1^2/Nh.
// But in this model the number of halos around a sightline is not fixed -- it is
// a Poisson draw (that is how the production code itself samples the resolved
// lenses, lensing.cpp:538-540: PN = poisson_distribution<int>(lambda*barNH)).
// For Poisson counts, Campbell's theorem gives Var(sum kappa) = integral n*kappa^2
// = kappa2, with NO subtraction: the count fluctuations put the <kappa>^2 piece back
// (law of total variance: Var = E[N]*Var(kappa) + Var(N)*<kappa>^2, and Var(N)=E[N]
// for Poisson recombines the two terms into E[N]*<kappa^2>).
//
// THE EXPERIMENT
// --------------
// Run the same brute-force draw of the weak-band halo field TWICE, identical in
// every respect except the count statistics:
//
//   (P) POISSON:  per (z,M) cell, N_c ~ Poisson(lambda_c)   [the model's own convention]
//   (F) FROZEN:   total N = round(sum_c lambda_c) EVERY realization; each halo's
//                 cell drawn with probability lambda_c/sum(lambda)  [the iid fixed-N
//                 model that the formula kappa2 - kappa1^2/Nh describes]
//
// Same lambda_c (the code's own disk-count formula), same uniform-in-area
// positions (the code's own convention, lensing.cpp:533), same NFW kappa kernel
// (kappagammaNFWeps, verbatim as in sigmakappaW).  Predictions, with K1, K2, n
// the CORRECT-measure (2*pi*r^2 dlnr) integrals over the same band:
//
//   Var_P = K2                    (Campbell, no subtraction)
//   Var_F = K2 - K1^2/n           (fixed-N formula)
//
// and the production code returns (K2 - K1^2/n)/2, i.e. the FROZEN-count answer,
// additionally halved by the measure bug.  If the Poisson MC lands on K2 and the
// frozen MC lands on K2 - K1^2/n, the subtraction is settled: it belongs to a
// universe with a frozen halo count, which is not this model.
//
// Build (repo root; needs build/libgwcore.a from `make build`):
//   clang++ -std=c++17 -O2 -Icpp -I/opt/homebrew/include \
//     playground/sigmakappaw_poisson_vs_fixed.cpp -Lbuild -lgwcore \
//     -L/opt/homebrew/lib -lgsl -lgslcblas -o build/sigmakappaw_poisson_vs_fixed
// Run:  build/sigmakappaw_poisson_vs_fixed [zs=1.0] [Nreal=20000] [seed=20260708]
// Writes per-realization sums to playground/sigmakappaw_sums_{poisson,frozen}_zs<zs>.txt

#include "cosmology.h"  // pulls in basics.h
#include "lensing.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

// in lensing.cpp but not declared in lensing.h
double Sigmacf(cosmology &C, double zs, double zl);
double NhfNFW(cosmology &C, double zs, double kappathr);
double findkappathr(int N, std::function<double(double)> Nf);

namespace {

struct Cell {
    double rmax2;   // inner band edge squared
    double rstop2;  // outer band edge squared
    double rs;      // NFW scale radius
    double kappa0;  // NFW kappa normalization
    double lambda;  // expected halo count in the band (disk formula)
    double zl;      // lens redshift (for the single-sightline diagnostic dump)
};

double var_of(const std::vector<double> &s) {
    double m = 0.0;
    for (double v : s) m += v;
    m /= s.size();
    double q = 0.0;
    for (double v : s) q += (v - m) * (v - m);
    return q / s.size();
}

}  // namespace

int main(int argc, char **argv) {
    const double zs = (argc > 1) ? std::atof(argv[1]) : 1.0;
    const int Nreal = (argc > 2) ? std::atoi(argv[2]) : 20000;
    const uint64_t seed = (argc > 3) ? std::strtoull(argv[3], nullptr, 10) : 20260708ULL;

    cosmology C;
    C.OmegaM = 0.315; C.OmegaB = 0.0493; C.zeq = 3402.0; C.sigma8 = 0.811;
    C.h = 0.674; C.T0 = 2.7255; C.ns = 0.965;
    C.Mmin = 1.0e7; C.Mmax = 1.0e17; C.NM = 100;
    C.zmin = 0.01; C.zmax = 10.01; C.Nz = 100;
    C.outdir = "dataL";
    C.initialize(0);

    std::function<double(double)> NfNFW = [&C, zs](double kappa) { return NhfNFW(C, zs, kappa); };
    const double kappathr = findkappathr(100, NfNFW);
    const double sigma_code = sigmakappaW(C, zs, kappathr);
    std::cout << "zs = " << zs << "   kappa_thr(N=100) = " << kappathr
              << "   sigmakappaW() = " << sigma_code << "\n";

    // ---- band edges + analytic K1, K2 (correct 2*pi*r^2 dlnr measure, midpoint
    //      rule on a 10x finer grid, over EXACTLY the band [rmax, rstop] that the
    //      production loop covers with its dlnr=0.01 / 0.001*kappa_thr stop rule)
    const double dlnr = 0.01;
    const double Edlnr = std::exp(dlnr);
    const double h = dlnr / 10.0;

    std::vector<Cell> cells;
    double n_tot = 0.0, K1 = 0.0, K2 = 0.0;

    for (int jz = 1; jz < C.Nz; jz++) {
        const double zl = C.zlist[jz];
        const double dz = zl - C.zlist[jz - 1];
        if (zl >= zs) continue;
        for (int jM = 1; jM < C.NM; jM++) {
            const double M = C.Mlist[jM];
            const double dlnM = std::log(M) - std::log(C.Mlist[jM - 1]);
            const double dndlnM = C.HMFlist[jz][jM][0];
            const double pref = 306.535 * PI / C.Hz(zl) * dndlnM * dlnM * dz;

            double r = rmaxfNFW(C, zs, zl, M, kappathr);
            if (r == 0.0) r = 1.0e-6;
            const double rmax = r;

            const double Sigmac = Sigmacf(C, zs, zl);
            std::vector<double> NFWp = interpolate2(zl, M, C.zlist, C.Mlist, C.NFWlist);
            const double rs = NFWp[0];
            const double kappa0 = kappa0NFW(rs, NFWp[1], Sigmac);

            // find rstop with the production loop's own stop rule (verbatim)
            int K = 0;
            double kappar = kappathr;
            while (kappar > 0.001 * kappathr) {
                kappar = kappagammaNFWeps(0.0, kappa0, r / rs, 0.0)[0];
                K++;
                r = r * Edlnr;
            }
            const double rstop = r;

            // expected count in the band: the code's own disk formula (lensing.cpp:121,149)
            const double zf2 = (1.0 + zl) * (1.0 + zl);
            const double lambda = pref * zf2 * (rstop * rstop - rmax * rmax);

            // K1, K2 on the same band, correct measure d(pi r^2) = 2 pi r^2 dlnr
            for (int i = 0; i < 10 * K; i++) {
                const double rm = rmax * std::exp((i + 0.5) * h);
                const double kap = kappagammaNFWeps(0.0, kappa0, rm / rs, 0.0)[0];
                const double w = 2.0 * pref * zf2 * rm * rm * h;
                K1 += w * kap;
                K2 += w * kap * kap;
            }

            n_tot += lambda;
            if (lambda > 0.0) cells.push_back({rmax * rmax, rstop * rstop, rs, kappa0, lambda, zl});
        }
    }

    const long Nfix = std::lround(n_tot);
    const double varP_pred = K2;                    // Campbell (Poisson counts)
    const double varF_pred = K2 - K1 * K1 / n_tot;  // fixed-count formula

    std::cout << "\nband totals:  n = " << n_tot << "   K1 = " << K1 << "   K2 = " << K2 << "\n"
              << "predictions:  Var_P (Campbell)  = " << varP_pred << "\n"
              << "              Var_F (fixed N)   = " << varF_pred
              << "   (subtraction = " << 100.0 * (1.0 - varF_pred / varP_pred) << "%)\n"
              << "              code sigma_W^2    = " << sigma_code * sigma_code
              << "   ( = Var_F/2: " << sigma_code * sigma_code / (varF_pred / 2.0) << " )\n";

    // ---- MC.  Both variants share lambda_c, positions, kernel; only the count
    //      statistics differ.
    std::mt19937_64 mt(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);

    auto draw_kappa = [&](const Cell &c) {
        const double rr2 = c.rmax2 + U(mt) * (c.rstop2 - c.rmax2);  // uniform in area
        return kappagammaNFWeps(0.0, c.kappa0, std::sqrt(rr2) / c.rs, 0.0)[0];
    };

    // (P) Poisson counts per cell — the model's own convention (lensing.cpp:538-540)
    // Realization 0 is recorded halo-by-halo (kappa, z_lens, impact parameter) so the
    // "brute force = stack many halos into one sightline" picture can be plotted. The
    // inlined draw here consumes the RNG identically to draw_kappa (one U per halo), so
    // sumsP is bit-for-bit what the plain draw_kappa loop produced.
    std::vector<double> sumsP(Nreal);
    std::vector<std::array<double, 3>> one_sightline;  // {kappa, z_lens, r_kpc} for realization 0
    {
        std::vector<std::poisson_distribution<long>> pois;
        pois.reserve(cells.size());
        for (const Cell &c : cells) pois.emplace_back(c.lambda);
        for (int j = 0; j < Nreal; j++) {
            double kap = 0.0;
            for (size_t ic = 0; ic < cells.size(); ic++) {
                const long N = pois[ic](mt);
                const Cell &c = cells[ic];
                for (long i = 0; i < N; i++) {
                    const double rr2 = c.rmax2 + U(mt) * (c.rstop2 - c.rmax2);  // uniform in area
                    const double kk = kappagammaNFWeps(0.0, c.kappa0, std::sqrt(rr2) / c.rs, 0.0)[0];
                    kap += kk;
                    if (j == 0) one_sightline.push_back({kk, c.zl, std::sqrt(rr2)});
                }
            }
            sumsP[j] = kap;
        }
    }
    std::cout << "\nPoisson MC done\n" << std::flush;

    // Dump the Poisson sums + print Var_P immediately: this is the physically
    // relevant half AND the arbiter of the measure (factor-of-2) question, so it
    // must survive even if the slower frozen-N loop below is killed by a timeout.
    {
        char fnP[256];
        std::snprintf(fnP, sizeof(fnP), "playground/sigmakappaw_sums_poisson_zs%g.txt", zs);
        FILE *fpP = std::fopen(fnP, "w");
        for (double v : sumsP) std::fprintf(fpP, "%.8e\n", v);
        std::fclose(fpP);
        // one sightline, halo by halo: columns = kappa  z_lens  r_kpc
        char fnO[256];
        std::snprintf(fnO, sizeof(fnO), "playground/sigmakappaw_one_sightline_zs%g.txt", zs);
        FILE *fpO = std::fopen(fnO, "w");
        for (const auto &h : one_sightline) std::fprintf(fpO, "%.8e %.5f %.6e\n", h[0], h[1], h[2]);
        std::fclose(fpO);
        std::printf("  one sightline (realization 0): %zu weak halos, Sum kappa = %.6e\n",
                    one_sightline.size(),
                    [&] { double s = 0; for (const auto &h : one_sightline) s += h[0]; return s; }());
        const double varP_now = var_of(sumsP);
        std::printf("  MC Poisson: Var = %.5e   Var/K2 = %.4f (+-%.4f)   mean = %.6e (K1 = %.6e)\n",
                    varP_now, varP_now / K2, std::sqrt(2.0 / (Nreal - 1.0)),
                    [&] { double m = 0; for (double v : sumsP) m += v; return m / Nreal; }(), K1);
        std::fflush(stdout);
    }

    // (F) frozen count: exactly Nfix halos every realization, cells picked with
    //     probability lambda_c / n_tot (iid mixture = the fixed-N formula's model)
    std::vector<double> sumsF(Nreal);
    {
        std::vector<double> cum(cells.size());
        double acc = 0.0;
        for (size_t ic = 0; ic < cells.size(); ic++) { acc += cells[ic].lambda; cum[ic] = acc; }
        for (int j = 0; j < Nreal; j++) {
            double kap = 0.0;
            for (long i = 0; i < Nfix; i++) {
                const double u = U(mt) * acc;
                const size_t ic = std::lower_bound(cum.begin(), cum.end(), u) - cum.begin();
                kap += draw_kappa(cells[ic]);
            }
            sumsF[j] = kap;
        }
    }
    std::cout << "frozen-N MC done (Nfix = " << Nfix << ")\n";

    const double varP = var_of(sumsP), varF = var_of(sumsF);
    const double relerr = std::sqrt(2.0 / (Nreal - 1.0));

    std::printf("\n%-34s %12s %12s %12s\n", "", "Var(sum kappa)", "/K2", "err");
    std::printf("%-34s %12.5e %12.4f %12.4f\n", "MC Poisson counts", varP, varP / K2, varP / K2 * relerr);
    std::printf("%-34s %12.5e %12.4f %12.4f\n", "MC frozen count", varF, varF / K2, varF / K2 * relerr);
    std::printf("%-34s %12.5e %12.4f\n", "pred Campbell K2", varP_pred, 1.0);
    std::printf("%-34s %12.5e %12.4f\n", "pred fixed-N K2-K1^2/n", varF_pred, varF_pred / K2);
    std::printf("%-34s %12.5e %12.4f\n", "code sigmakappaW^2", sigma_code * sigma_code,
                sigma_code * sigma_code / K2);
    std::printf("\nmeans: MC-P %.6e  MC-F %.6e  pred K1 %.6e\n",
                [&]{ double m=0; for(double v:sumsP) m+=v; return m/Nreal; }(),
                [&]{ double m=0; for(double v:sumsF) m+=v; return m/Nreal; }(), K1);

    // dump per-realization sums for plotting
    char fn[256];
    std::snprintf(fn, sizeof(fn), "playground/sigmakappaw_sums_poisson_zs%g.txt", zs);
    FILE *fp = std::fopen(fn, "w");
    for (double v : sumsP) std::fprintf(fp, "%.8e\n", v);
    std::fclose(fp);
    std::snprintf(fn, sizeof(fn), "playground/sigmakappaw_sums_frozen_zs%g.txt", zs);
    fp = std::fopen(fn, "w");
    for (double v : sumsF) std::fprintf(fp, "%.8e\n", v);
    std::fclose(fp);
    std::cout << "per-realization sums written to playground/sigmakappaw_sums_*_zs" << zs << ".txt\n";
    return 0;
}
