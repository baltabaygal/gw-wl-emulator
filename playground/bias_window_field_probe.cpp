// Dump the PRODUCTION BiasField1D covariance for a given (z_s, R_perp,
// bias_window) so the numpy replica can be checked against the C++ number for
// number.
//
// Why this exists: playground/bias_field/validate_field_covariance.py's layer-1
// gate compares two PYTHON integrals (verbatim replica vs independent-grid
// reference) — it proves the derivation, not that the shipped C++ evaluates it.
// For bias_window = 0 the pristine-vs-patched bitwise stream comparison covers
// the C++ side, but windows 1/2 are new code with no stream reference: a window
// applied to k_perp instead of |k| would still be deterministic, finite and
// non-inert, and would pass every other gate. This probe closes that gap.
//
// Includes cpp/lensing.cpp directly, so BiasField1D here IS production's.
//
// Build (repo root):
//   c++ -std=c++17 -O2 -Icpp -I/opt/homebrew/include \
//     playground/bias_window_field_probe.cpp cpp/cosmology.cpp cpp/basics.cpp cpp/subhalo.cpp \
//     -L/opt/homebrew/lib -lgsl -lgslcblas -o build/bias_window_field_probe
// Run:  build/bias_window_field_probe [zs=1.0] [Rperp=8441.0] [window=0]
// Writes to stdout: a header line, then n lines of sig2, then the n x n
// Cov = chol chol^T (row-major, one row per line, %.17g).

#include "lensing.cpp"  // production internals: BiasField1D

#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char **argv) {
    const double zs     = (argc > 1) ? std::atof(argv[1]) : 1.0;
    const double Rperp  = (argc > 2) ? std::atof(argv[2]) : 8441.0;
    const int    window = (argc > 3) ? std::atoi(argv[3]) : 0;

    cosmology C;
    C.OmegaM = 0.315; C.OmegaB = 0.0493; C.zeq = 3402.0; C.sigma8 = 0.811;
    C.h = 0.674; C.T0 = 2.7255; C.ns = 0.965;
    C.Mmin = 1.0e7; C.Mmax = 1.0e17; C.NM = 100;
    C.zmin = 0.01; C.zmax = 10.01; C.Nz = 100;
    C.outdir = "dataL";
    C.initialize(0);

    BiasField1D bf;
    bf.build(C, zs, Rperp, window);

    std::printf("# zs=%.17g Rperp=%.17g window=%d n=%d Nmax=%ld L=%.17g\n",
                zs, Rperp, window, bf.n, bf.Nmax, bf.L);
    for (int i = 0; i < bf.n; i++) std::printf("%.17g\n", bf.sig2[i]);
    // Cov is a local in build(); reconstruct it from the stored Cholesky
    // (equal to Cov + jitter*I, jitter = 1e-12*trace/n — 6 orders below the gate).
    const int n = bf.n;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k <= (i < j ? i : j); k++)
                s += bf.chol[static_cast<size_t>(i)*n + k]*bf.chol[static_cast<size_t>(j)*n + k];
            std::printf("%.17g%c", s, (j == n - 1) ? '\n' : ' ');
        }
    }
    return 0;
}
