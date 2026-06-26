#pragma once
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
    double alpha = -0.82, beta = 50.0, omega = 4.0, psi_res = 1.0e-4, psi_max = 0.1;
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

    // build all (z,M)-grid tables; needs zs + kappathr for the clump-reach table r_thr.
    void precompute(cosmology &C, double zs, double kappathr);

    // in-loop: add one host's subhalos to the running (kappa, gamma1, gamma2).
    // (jz,jM): host bin; zl,M: host redshift/mass; Sigmac: critical surface density;
    // (r,phi): line-of-sight host-centric impact parameter and azimuth.
    // Only clumps above the dynamic floor m_res(r) (those that exceed kappa_thr at the
    // host's LoS distance) are drawn; mass conserved via per-clump removal m*dkappa_host/dM.
    int addClumps(cosmology &C, int jz, int jM, double zl, double M, double Sigmac,
                   double r, double phi, rgen &mt,
                   double &kappa, double &gamma1, double &gamma2,
                   int subhalo_model = 0, bool subhalo_brute = false,
                   int threads = 1, int parallel_threshold = 200000);
};
