// Zeroth-moment (pure counting) consistency check for sigmakappaW.
//
// THE QUESTION
// ------------
// For lenses of mass M in the redshift shell [z, z+dz], how many have their
// impact parameter r (distance of the lens from the line of sight, in the lens
// plane) inside the "weak band" [r_max, r_stop]?  This is a pure counting
// question -- no kappa weighting, no variance, no randomness -- and the code
// answers it in two different ways that must agree:
//
//  (a) Paper eq. (3):  dP_l/dr ∝ 2π(1+z)² r / H(z) · dn/dlnM.
//      Integrating over r:  N(r1<r<r2) = pref · π[((1+z)r2)² − ((1+z)r1)²],
//      with pref = 306.535/H(z) · dn/dlnM · dlnM · dz.
//      This "difference of disks" is EXACTLY the formula the code itself uses
//      to count resolved lenses:  NhfNFW / findkappathr (lensing.cpp:149)
//      counts N(r<r_max) = pref · π((1+z)r_max)², and that is what calibrates
//      <N> = 100.  So (a) is the convention the whole model is anchored to.
//
//  (b) The sigmakappaW radial loop (lensing.cpp:183-189) with the kappa
//      weights stripped out, i.e. its own "Nh" accumulator:
//      N = sum over the log grid  pref · π((1+z)r_i)² · dlnr.
//
// DEFINITIONS OF THE BAND EDGES (both are the code's own)
// -------------------------------------------------------
//  r_max  = rmaxfNFW(M, z; kappa_thr): the radius at which this lens's kappa
//           at the sightline equals kappa_thr.  Lenses closer than r_max are
//           "resolved" (drawn individually, eq. 3's theta(r_max - r)); lenses
//           beyond r_max belong to the weak Gaussian.  It is the INNER edge of
//           the weak band.  (If the lens never reaches kappa_thr, the code
//           sets r = 1e-6 and the band is effectively the whole disk.)
//  r_stop = where sigmakappaW's while-loop terminates: the first grid radius
//           at which the lens's kappa has fallen below 0.001*kappa_thr.  It is
//           the loop's numerical OUTER edge of the band (kappa ~ r^-2 there,
//           so r_stop ≈ sqrt(1000)·r_max ≈ 32 r_max).
//  We use the loop itself to find r_stop, so (a) and (b) are evaluated on the
//  IDENTICAL set of lenses; the kappa profile plays no other role.
//
// THE EXACT PREDICTION
// --------------------
// The grid radii are r_i = r_max·e^{i·dlnr}, i = 0..K-1, and r_stop = r_max·e^{K·dlnr}.
// The ratio (b)/(a) is then a pure number, independent of the cell, of r_max,
// of r_stop, of the prefactor, and of cosmology:
//
//   sum_i r_i² dlnr / (r_stop² − r_max²)
//     = dlnr · (e^{2K dlnr}−1)/(e^{2 dlnr}−1) / (e^{2K dlnr}−1)
//     = dlnr / (e^{2 dlnr} − 1)
//     = 0.01 / (e^{0.02} − 1) = 0.4950166...          -> 1/2 as dlnr -> 0.
//
// So if the loop measure were the correct annulus element, the ratio would be
// 1.  It is 0.495017 -- for every cell and every z_s -- which is the factor 2
// (times the trivial e^{0.02} discretization).
//
// Build (repo root):
//   clang++ -std=c++17 -O2 -Icpp -I/opt/homebrew/include \
//     playground/sigmakappaw_count_check.cpp -Lbuild -lgwcore \
//     -L/opt/homebrew/lib -lgsl -lgslcblas -o build/sigmakappaw_count_check
// Run:  build/sigmakappaw_count_check [zs=1.0]

#include "cosmology.h"  // pulls in basics.h
#include "lensing.h"

#include <cmath>
#include <functional>
#include <iostream>

// in lensing.cpp but not declared in lensing.h
double Sigmacf(cosmology &C, double zs, double zl);
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

    // the code's own threshold calibration (footnote 4 of the paper): kappa_thr
    // is fixed so that the resolved count -- formula (a) over the disk r<r_max,
    // summed over all cells -- equals 100.
    std::function<double(double)> NfNFW = [&C, zs](double kappa) { return NhfNFW(C, zs, kappa); };
    const double kappathr = findkappathr(100, NfNFW);
    std::cout << "zs = " << zs << "   kappa_thr = " << kappathr
              << "   check: NhfNFW(kappa_thr) = " << NhfNFW(C, zs, kappathr)
              << "  (calibrated to 100 with formula (a))\n\n";

    const double dlnr = 0.01;                 // sigmakappaW's grid step
    const double Edlnr = std::exp(dlnr);
    const double analytic = dlnr / (std::exp(2.0 * dlnr) - 1.0);

    double total_loop = 0.0, total_disk = 0.0;
    std::cout << "example cells (count_loop = (b), count_disk = (a)):\n"
              << "   z_l      M[Msun]    r_max[kpc]  r_stop[kpc] steps  count_loop  count_disk   ratio\n";

    for (int jz = 1; jz < C.Nz; jz++) {
        const double zl = C.zlist[jz];
        const double dz = zl - C.zlist[jz - 1];
        if (zl >= zs) continue;
        for (int jM = 1; jM < C.NM; jM++) {
            const double M = C.Mlist[jM];
            const double dlnM = std::log(M) - std::log(C.Mlist[jM - 1]);
            const double dndlnM = C.HMFlist[jz][jM][0];
            const double pref = 306.535 * PI / C.Hz(zl) * dndlnM * dlnM * dz;

            // inner band edge (same as sigmakappaW, incl. its r_max=0 -> 1e-6 fallback)
            double r = rmaxfNFW(C, zs, zl, M, kappathr);
            if (r == 0.0) r = 1.0e-6;
            const double rmax = r;

            // kappa profile, used ONLY to find where the loop stops (r_stop)
            const double Sigmac = Sigmacf(C, zs, zl);
            std::vector<double> NFWp = interpolate2(zl, M, C.zlist, C.Mlist, C.NFWlist);
            const double rs = NFWp[0];
            const double kappa0 = kappa0NFW(rs, NFWp[1], Sigmac);

            // (b): sigmakappaW's loop, verbatim, with the kappa^n weight deleted
            double count_loop = 0.0;
            int K = 0;
            double kappar = kappathr;
            while (kappar > 0.001 * kappathr) {
                kappar = kappagammaNFWeps(0.0, kappa0, r / rs, 0.0)[0];
                count_loop += pref * std::pow((1.0 + zl) * r, 2.0) * dlnr;
                K++;
                r = r * Edlnr;
            }
            const double rstop = r;

            // (a): eq. (3) integrated over the same band = difference of the
            //      code's own disk-count formula at r_stop and r_max
            const double count_disk =
                pref * (std::pow((1.0 + zl) * rstop, 2.0) - std::pow((1.0 + zl) * rmax, 2.0));

            total_loop += count_loop;
            total_disk += count_disk;

            // print a few representative cells
            const bool show =
                (std::abs(std::log10(M / 1.0e13)) < 0.06 && std::abs(zl - 0.5) < 0.02) ||
                (std::abs(std::log10(M / 1.0e10)) < 0.06 && std::abs(zl - 0.5) < 0.02) ||
                (std::abs(std::log10(M / 1.0e13)) < 0.06 && std::abs(zl - 0.2) < 0.01);
            if (show && count_disk > 0.0) {
                std::printf("  %6.3f  %9.2e  %10.3f  %10.1f  %4d  %10.4g  %10.4g   %.6f\n",
                            zl, M, rmax, rstop, K, count_loop, count_disk,
                            count_loop / count_disk);
            }
        }
    }

    std::cout << "\nTOTAL over all cells (this is sigmakappaW's own Nh accumulator vs eq. 3):\n"
              << "  (b) loop measure : " << total_loop << "\n"
              << "  (a) eq. (3)      : " << total_disk << "\n"
              << "  ratio (b)/(a)    = " << total_loop / total_disk << "\n"
              << "  exact prediction   dlnr/(e^{2 dlnr}-1) = " << analytic
              << "   -> 1/2 as dlnr -> 0\n";
    return 0;
}
