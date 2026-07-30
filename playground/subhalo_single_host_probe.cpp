// Single-host MC probe for the model-4 (brute-to-floor + carve) decomposition:
//   kappa_halo = kappa_NFW[M - sum_i m_i] + sum_i kappa_i
// Ground truth for Fig 4 (paper_prod/scripts/plot_fig_subhalo_sigma_decomposition.py):
// runs the PRODUCTION Subhalo::addClumps (subhalo_model = 4, psi_lo = m_floor/M) and
// the production host-carve on one (jz, jM) grid host, over a grid of ray impact
// parameters r, and writes the MC mean/std of the host, subhalo and total convergence
// components. The fig script's single-clump-pool Campbell moments are the analytic
// counterpart; the comparer (scratchpad/compare_fig4_probe.py) overlays both.
//
// Build (repo root; GSL from homebrew, same pattern as the halos build):
//   clang++ -std=c++17 -O2 -I/opt/homebrew/include -L/opt/homebrew/lib \
//     playground/subhalo_single_host_probe.cpp cpp/basics.cpp cpp/cosmology.cpp \
//     cpp/lensing.cpp cpp/subhalo.cpp -lgsl -lgslcblas \
//     -o playground/subhalo_single_host_probe
//
// Run (from repo root; needs the dataL/ cosmology cache dir like main_lensing):
//   ./playground/subhalo_single_host_probe [Nreal=20000] [Mtarget=1e13] [zl=0.5] \
//     [zs=1.0] [m_floor=1e7] [seed=20260723] [out=tmp/subhalo_single_host_probe.csv]

#include "../cpp/cosmology.h"
#include "../cpp/lensing.h"
#include "../cpp/subhalo.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

double Sigmacf(cosmology &C, double zs, double zl);   // defined in lensing.cpp

int main(int argc, char **argv) {
    int Nreal      = (argc > 1) ? std::atoi(argv[1]) : 20000;
    double Mtarget = (argc > 2) ? std::atof(argv[2]) : 1.0e13;
    double zl_t    = (argc > 3) ? std::atof(argv[3]) : 0.5;
    double zs      = (argc > 4) ? std::atof(argv[4]) : 1.0;
    double m_floor = (argc > 5) ? std::atof(argv[5]) : 1.0e7;
    uint64_t seed  = (argc > 6) ? std::stoull(argv[6]) : 20260723ULL;
    std::string out = (argc > 7) ? argv[7] : "tmp/subhalo_single_host_probe.csv";
    bool virial    = (argc > 8) ? (std::atoi(argv[8]) != 0) : false;

    cosmology C;
    C.OmegaM = 0.315; C.OmegaB = 0.0493; C.zeq = 3402.0;
    C.sigma8 = 0.811; C.h = 0.674; C.T0 = 2.7255; C.ns = 0.965;
    C.Mmin = 1.0e7; C.Mmax = 1.0e17; C.NM = 100;
    C.zmin = 0.01; C.zmax = 10.01; C.Nz = 100;
    C.outdir = "dataL";
    C.initialize(0);

    Subhalo sub;
    sub.m_floor = m_floor;
    sub.virial = virial;      // JvdB14 virial convention (ON in PRODUCTION_CONFIG)
    // kappathr feeds only the r_thr table, which the model-4 floor never consults;
    // pass the production-scale value so precompute follows the standard path.
    sub.precompute(C, zs, 1.0e-2, 0.0);

    // nearest grid host to (zl_t, Mtarget)
    int jz = 0, jM = 0;
    for (int j = 1; j < C.Nz; j++)
        if (std::fabs(std::log(C.zlist[j] / zl_t)) < std::fabs(std::log(C.zlist[jz] / zl_t))) jz = j;
    for (int j = 1; j < C.NM; j++)
        if (std::fabs(std::log(C.Mlist[j] / Mtarget)) < std::fabs(std::log(C.Mlist[jM] / Mtarget))) jM = j;
    const double zl = C.zlist[jz], M = C.Mlist[jM];
    if (sub.gnorm[jz][jM] <= 0.0) {
        std::cerr << "inactive subhalo bin jz=" << jz << " jM=" << jM << "\n";
        return 1;
    }

    const double Sigmac = Sigmacf(C, zs, zl);
    const double r200 = sub.r200h[jz][jM];
    const double psi_min = m_floor / M;

    std::cerr << "host: M=" << M << " zl=" << zl << " zs=" << zs
              << " r200=" << r200 << " c=" << sub.chost[jz][jM]
              << " fs(JvdB14)=" << sub.fsub[jz][jM]
              << " gnorm=" << sub.gnorm[jz][jM]
              << " psi_min=" << psi_min << " Sigmac=" << Sigmac
              << " Nreal=" << Nreal << "\n";

    rgen mt(seed);
    std::ofstream f(out);
    f.precision(10);
    f << "# M=" << M << " zl=" << zl << " zs=" << zs << " r200=" << r200
      << " c_host=" << sub.chost[jz][jM] << " fs_jvdb14=" << sub.fsub[jz][jM]
      << " gnorm=" << sub.gnorm[jz][jM] << " psi_min=" << psi_min
      << " Sigmac=" << Sigmac << " Nreal=" << Nreal << " seed=" << seed
      << " virial=" << (virial ? 1 : 0)
      << " xmax=" << sub.xmaxh[jz][jM] << " Mpsi=" << sub.Mpsih[jz][jM] << "\n";
    f << "x_r200,mean_host,mean_sub,mean_tot,sig_host,sig_sub,sig_tot,"
         "cov_hs,mean_Msum,sig_Msum,mean_Nc\n";

    const int Nx = 25;
    for (int ix = 0; ix < Nx; ix++) {
        const double x = 0.03 + (1.0 - 0.03) * ix / (Nx - 1);
        const double r = x * r200;
        double sh = 0, sh2 = 0, ss = 0, ss2 = 0, st = 0, st2 = 0, shs = 0;
        double sM = 0, sM2 = 0, sN = 0;
        for (int i = 0; i < Nreal; i++) {
            double ks = 0, g1 = 0, g2 = 0, Msum = 0;
            int Nc = sub.addClumps(C, jz, jM, zl, M, Sigmac, r, 0.0, mt,
                                   ks, g1, g2, /*subhalo_model=*/4,
                                   /*subhalo_brute=*/false, 1, 1 << 30, &Msum);
            double Mh = M - Msum;                      // model 4: M_u = 0
            if (Mh < C.Mmin) Mh = C.Mmin;              // production guard
            vector<double> NFWp = interpolate2(zl, Mh, C.zlist, C.Mlist, C.NFWlist);
            const double kh = 2.0 * kappa0NFW(NFWp[0], NFWp[1], Sigmac)
                                  * FgNFW(r / NFWp[0])[0];   // circular host (fig convention)
            const double kt = kh + ks;
            sh += kh; sh2 += kh * kh;
            ss += ks; ss2 += ks * ks;
            st += kt; st2 += kt * kt;
            shs += kh * ks;
            sM += Msum; sM2 += Msum * Msum; sN += Nc;
        }
        const double n = Nreal;
        const double mh = sh / n, ms = ss / n, mtot = st / n, mMs = sM / n;
        auto sd = [n](double s2, double m) { return std::sqrt(std::max(s2 / n - m * m, 0.0)); };
        f << x << "," << mh << "," << ms << "," << mtot << ","
          << sd(sh2, mh) << "," << sd(ss2, ms) << "," << sd(st2, mtot) << ","
          << (shs / n - mh * ms) << "," << mMs << "," << sd(sM2, mMs) << ","
          << sN / n << "\n";
        std::cerr << "  x=" << x << " mean_tot=" << mtot
                  << " sig_tot=" << sd(st2, mtot) << " <Nc>=" << sN / n << "\n";
    }
    std::cerr << "wrote " << out << "\n";
    return 0;
}
