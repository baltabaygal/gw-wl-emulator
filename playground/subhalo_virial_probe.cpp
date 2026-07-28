// Mechanism probe for subhalo_virial (2026-07-28).
//
// The production A/B finds the virial convention AT THE SAMPLING FLOOR on P(lnmu),
// which is ambiguous on its own: it could mean "the fix works and simply does not
// matter for the PDF" or "the fix is silently inert". This probe settles that by
// measuring the POPULATION directly rather than the lensing observable, calling the
// production Subhalo::precompute + addClumps (model 4, brute to the floor) on one
// grid host under both conventions and comparing against the analytic prediction:
//
//   eta      = r_vir/r_200                     -> radial extent ratio
//   M_vir/M  = mu(eta c)/mu(c)                 -> psi mass-scale ratio
//   <Sum_i m_i>  should scale as M_vir/M       (f_s is a fraction of the psi scale)
//   max clump radius should scale as eta
//
// Build (repo root; GSL from homebrew):
//   clang++ -std=c++17 -O2 -I/opt/homebrew/include -L/opt/homebrew/lib \
//     playground/subhalo_virial_probe.cpp cpp/basics.cpp cpp/cosmology.cpp \
//     cpp/lensing.cpp cpp/subhalo.cpp -lgsl -lgslcblas \
//     -o playground/subhalo_virial_probe
//
// Run (from repo root; needs the dataL/ cosmology cache dir):
//   ./playground/subhalo_virial_probe [Nreal=4000] [zs=1.0] [seed=20260728]

#include "../cpp/cosmology.h"
#include "../cpp/lensing.h"
#include "../cpp/subhalo.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

double Sigmacf(cosmology &C, double zs, double zl);   // defined in lensing.cpp

static double nfwMu(double y) { return std::log(1.0 + y) - y / (1.0 + y); }

int main(int argc, char **argv) {
    int Nreal     = (argc > 1) ? std::atoi(argv[1]) : 4000;
    double zs     = (argc > 2) ? std::atof(argv[2]) : 1.0;
    uint64_t seed = (argc > 3) ? std::stoull(argv[3]) : 20260728ULL;

    cosmology C;
    C.OmegaM = 0.315; C.OmegaB = 0.0493; C.zeq = 3402.0;
    C.sigma8 = 0.811; C.h = 0.674; C.T0 = 2.7255; C.ns = 0.965;
    C.Mmin = 1.0e7; C.Mmax = 1.0e17; C.NM = 100;
    C.zmin = 0.01; C.zmax = 10.01; C.Nz = 100;
    C.outdir = "dataL";
    C.initialize(0);

    Subhalo legacy, virial;
    legacy.m_floor = 1.0e7; legacy.virial = false;
    virial.m_floor = 1.0e7; virial.virial = true;
    legacy.precompute(C, zs, 1.0e-2, 0.0);
    virial.precompute(C, zs, 1.0e-2, 0.0);

    std::cout.precision(6);
    std::cout << "Nreal=" << Nreal << " zs=" << zs << " seed=" << seed << "\n\n";
    std::cout << "  zl      M          eta_tab  eta_pred | Mvir/M tab  pred  |"
                 " <Msum>_v/<Msum>_l  pred | rmax_v/rmax_l  pred\n";

    const double targets_z[3] = {0.2, 0.5, 1.0};
    const double targets_M[3] = {1.0e13, 1.0e14, 1.0e15};

    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            int jz = 1, jM = 1;
            for (int j = 1; j < C.Nz; j++)
                if (std::fabs(std::log(C.zlist[j] / targets_z[a])) <
                    std::fabs(std::log(C.zlist[jz] / targets_z[a]))) jz = j;
            for (int j = 1; j < C.NM; j++)
                if (std::fabs(std::log(C.Mlist[j] / targets_M[b])) <
                    std::fabs(std::log(C.Mlist[jM] / targets_M[b]))) jM = j;
            const double zl = C.zlist[jz], M = C.Mlist[jM];
            if (zl >= zs) continue;
            if (legacy.gnorm[jz][jM] <= 0.0 || virial.gnorm[jz][jM] <= 0.0) continue;

            const double c = legacy.chost[jz][jM];
            const double eta_tab = virial.xmaxh[jz][jM];
            const double Mratio_tab = virial.Mpsih[jz][jM] / M;
            const double Mratio_pred = nfwMu(eta_tab * c) / nfwMu(c);

            // legacy tables must be untouched
            if (std::fabs(legacy.xmaxh[jz][jM] - 1.0) > 0 ||
                std::fabs(legacy.Mpsih[jz][jM] - M) > 0) {
                std::cout << "  *** legacy tables NOT at (1, M) -- bug\n";
            }

            const double Sigmac = Sigmacf(C, zs, zl);
            const double r200 = legacy.r200h[jz][jM];
            const double r = 0.3 * r200;

            double sumM_l = 0, sumM_v = 0;
            long   sumN_l = 0, sumN_v = 0;
            rgen mt_l(seed), mt_v(seed);
            for (int i = 0; i < Nreal; i++) {
                double k = 0, g1 = 0, g2 = 0, Ms = 0;
                sumN_l += legacy.addClumps(C, jz, jM, zl, M, Sigmac, r, 0.0, mt_l,
                                           k, g1, g2, 4, false, 1, 1 << 30, &Ms);
                sumM_l += Ms;
                k = g1 = g2 = Ms = 0;
                sumN_v += virial.addClumps(C, jz, jM, zl, M, Sigmac, r, 0.0, mt_v,
                                           k, g1, g2, 4, false, 1, 1 << 30, &Ms);
                sumM_v += Ms;
            }
            const double mMl = sumM_l / Nreal, mMv = sumM_v / Nreal;

            // outermost sampled 3D radius, straight from the inverse-CDF tables
            const double rmax_l = legacy.invRad[jz][jM].back();
            const double rmax_v = virial.invRad[jz][jM].back();

            std::cout << "  " << zl << "  " << M << "   " << eta_tab
                      << "   -      | " << Mratio_tab << "  " << Mratio_pred
                      << "  | " << (mMv / mMl) << "  " << Mratio_pred
                      << "  | " << (rmax_v / rmax_l) << "  " << eta_tab
                      << "   (Nc_v/Nc_l=" << (double)sumN_v / (double)sumN_l << ")\n";
        }
    }
    return 0;
}
