#pragma once
#include <array>
#include <vector>
#include <random>

using std::vector;
typedef std::mt19937_64 rgen;   // matches basics.h

class cosmology;                 // forward declaration (subhalo.cpp includes cosmology.h)

// Subhalo / substructure module (JvdB14 evolved SHMF + anti-biased radial profile).
// All quantities precomputed on the cosmology (z,M) grid; clumps sampled in the loop.
class Subhalo {
public:
    // --- evolved SHMF parameters (Jiang & van den Bosch 2014, all orders) ---
    // psi_max = 1: full SHMF range as in JvdB14; the exp(-beta psi^omega) cutoff does the
    // suppression near psi->1 and is applied EXACTLY via Poisson thinning in addClumps and
    // via the incomplete-Gamma mass integral in lensing.cpp's f_s_res (changed 2026-07-04,
    // previously truncated at psi_max=0.1 where the cutoff was negligible).
    double alpha = -0.82, beta = 50.0, omega = 4.0, psi_res = 1.0e-4, psi_max = 1.0;
    double m_floor = 1.0e7;          // lensing-relevant lower clump mass [Msun]

    // built by precompute(); indexed [jz][jM]
    vector<vector<double>> fsub;     // resolved bound fraction f_s(M,z)
    vector<vector<double>> Nsub;     // mean clump count above m_floor
    vector<vector<double>> rs_sm;    // reduced-host NFW scale radius  (M_sm=(1-fs)M)
    vector<vector<double>> rhos_sm;  // reduced-host NFW characteristic density
    vector<vector<double>> r200h;    // host r200 (for the radial profile scaling)
    vector<vector<double>> chost;    // host concentration
    vector<vector<double>> gnorm;    // SHMF normalization gamma per (z,M) bin (eq. 23)
    vector<vector<double>> r_thr;    // [jz][jm] clump reach: rmaxfNFW(m,z) where kappa_clump=kappa_thr
    vector<vector<vector<double>>> invRad;  // inverse radial CDF x(u), Nu pts, per active bin
    int Nu = 128;
    bool built = false;
    double log_Mmin = 0.0;
    double inv_dlogM = 0.0;

    // --- Wsub unresolved-clump term (subhalo_model 3; 2026-07-09) -------------
    // Campbell mean/std of the clumps BELOW the dynamic floor psi_lo(y), integrated
    // over the true clump-ray distance (projected anti-biased profile), per active
    // (jz,jM) bin on a uniform log-y grid up to the host rmax. Built only when
    // precompute() gets kappathr_host > 0. The encounter then adds
    //   muW(y) + sW(y) * N(0,1)
    // to kappa, with the host reduced by the FULL bound fraction fsb (not the
    // y-dependent resolved fraction) — exact per-encounter mean and variance at any
    // subhalo_factor. Derivation: docs/subhalo/wsub_gaussian_term_derivation.md.
    int NyW = 48;
    vector<vector<vector<double>>> muW;   // [jz][jM][iy] unresolved mean kappa
    vector<vector<vector<double>>> sW;    // [jz][jM][iy] unresolved std of kappa
    vector<vector<std::array<double,2>>> lyW;  // [jz][jM] = {log y0, dlog y}
    vector<vector<double>> fsb;           // full bound mass fraction (model-3 host reduction)

    // build all (z,M)-grid tables; needs zs + kappathr (clump threshold) for the
    // clump-reach table r_thr; kappathr_host > 0 additionally builds the Wsub tables.
    void precompute(cosmology &C, double zs, double kappathr, double kappathr_host = 0.0);

    // model-3 lookup: unresolved-clump mean and std at host-center ray distance r.
    void wsubTerm(int jz, int jM, double r, double &mu, double &sigma) const;

    // internal: build the Wsub tables for one active (jz, jM) bin.
    void buildWsubBin(cosmology &C, double zs, int jz, int jM, double kappathr_host);

    // in-loop: add one host's subhalos to the running (kappa, gamma1, gamma2).
    // (jz,jM): host bin; zl,M: host redshift/mass; Sigmac: critical surface density;
    // (r,phi): line-of-sight host-centric impact parameter and azimuth.
    // Only clumps above the dynamic floor m_res(r) are drawn. In the default reduced-host
    // model the same resolved mass fraction is removed from the smooth host in lensing.cpp;
    // the legacy model instead subtracts m*dkappa_host/dM per clump.
    int addClumps(cosmology &C, int jz, int jM, double zl, double M, double Sigmac,
                   double r, double phi, rgen &mt,
                   double &kappa, double &gamma1, double &gamma2,
                   int subhalo_model = 0, bool subhalo_brute = false,
                   int threads = 1, int parallel_threshold = 200000);
};
