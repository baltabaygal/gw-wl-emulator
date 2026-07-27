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
    // Diagnostic override (2026-07-23): > 0 replaces the absolute floor psi_min = m_floor/M
    // with a FIXED psi_min everywhere it is computed (buildWsubBin, unresolvedMass, addClumps
    // subhalo_brute path) -- e.g. psi_min_fixed = psi_res = 1e-4 tests the "no extrapolation
    // below the SHMF's own calibration point" choice, host-mass-independent by construction.
    // <= 0 (default) keeps the existing m_floor/M behavior, bitwise unchanged.
    double psi_min_fixed = -1.0;

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

    // --- model 5: restricted-intensity (kappa-thresholded) brute sampling (2026-07-27) ---
    // Model 4 instantiates every subhalo (~1e6 per ray at z_s=1, 45 ms/ray). Model 5 keeps
    // exactly the same population -- every subhalo still exists down to psi_min = m_floor/M,
    // no unresolved/Gaussian stand-in -- but only RENDERS those whose convergence at the ray
    // exceeds kappa_thr,sub, and samples that RESTRICTED Poisson intensity directly so the
    // rejects are never instantiated (draw-and-reject would save nothing).
    //
    // Exactness: the retention region is the disc d <= D(m) around the ray, where D(m) is the
    // clump reach r_thr built at kappa_thr,sub. Writing the projected clump number density as
    // Sigma_n(R) (normalized, so int Sigma_n dA = 1), the retained intensity per clump mass is
    // int_{disc} Sigma_n dA. We Poisson-thin against the global envelope Smax = max_R Sigma_n:
    //   propose with prob min(1, pi D^2 Smax), then
    //     small-target branch (pi D^2 Smax < 1): position uniform in the disc, accept with
    //       Sigma_n(R)/Smax  ->  E[accepted] = int_{disc} Sigma_n dA exactly;
    //     big-reach branch (capped at 1): position from the FULL radial profile as model 4,
    //       accept iff d <= D  ->  E[accepted] = P(d <= D) exactly.
    // Both branches are exact; the envelope only costs extra proposals, never accuracy. Because
    // Smax is global (not Sigma_n(y)), the whole proposal intensity is y-INDEPENDENT and
    // precomputes per (jz,jM). Dropped clumps keep their mass in the smooth host via the
    // mass-conserving carve, so mass_out accumulates RETAINED clump mass only.
    // Sizing + acceptance: data/results/subkappathr_population/report.md
    int NRp = 128;                               // projected-profile radial bins
    int Nq  = 96;                                // log-psi proposal grid points
    vector<vector<vector<double>>> p2p;          // [jz][jM][NRp] normalized projected profile
    vector<vector<double>> sig2dmax;             // [jz][jM] envelope max of Sigma_n [kpc^-2]
    vector<vector<vector<double>>> propCum;      // [jz][jM][Nq] cumulative proposal intensity
    vector<vector<std::array<double,2>>> lpq;    // [jz][jM] = {log psi_lo, dlog psi}
    bool restricted_built = false;

    // normalized projected clump number density at lens-plane separation s [kpc] from the
    // host centre; 0 outside r200. Integrates to 1 over the plane.
    double sigma2Dclump(int jz, int jM, double s) const;
    // clump reach D(m): log-interpolation of r_thr[jz] (which must have been built at
    // kappa_thr,sub for model 5) at clump mass m.
    double clumpReach(cosmology &C, int jz, double m) const;
    void buildRestrictedBin(cosmology &C, int jz, int jM);

    // build all (z,M)-grid tables; needs zs + kappathr (clump threshold) for the
    // clump-reach table r_thr; kappathr_host > 0 additionally builds the Wsub tables;
    // build_restricted = true additionally builds the model-5 tables (and then kappathr
    // MUST be kappa_thr,sub, since r_thr is reused as the clump reach D(m)).
    void precompute(cosmology &C, double zs, double kappathr, double kappathr_host = 0.0,
                    bool build_restricted = false);

    // model-3 lookup: unresolved-clump mean and std at host-center ray distance r.
    void wsubTerm(int jz, int jM, double r, double &mu, double &sigma) const;

    // mass-conserving carve (scheme A, 2026-07-22): mean UNRESOLVED bound mass at ray
    // distance r, M_u(r) = (f_s,b - f_s,res(r)) M -- the complement of the resolved band
    // that addClumps samples above the dynamic floor psi_lo(r). The carved host is built
    // at M - Sum_i m_i - M_u(r) so the total halo mass equals M in every realization.
    // Uses the same incomplete-Gamma resolved fraction as the model-1 host reduction and
    // the same r_thr dynamic floor as addClumps. Returns 0 if the bin is inactive.
    double unresolvedMass(cosmology &C, int jz, int jM, double r, double M) const;

    // internal: build the Wsub tables for one active (jz, jM) bin.
    void buildWsubBin(cosmology &C, double zs, int jz, int jM, double kappathr_host);

    // in-loop: add one host's subhalos to the running (kappa, gamma1, gamma2).
    // (jz,jM): host bin; zl,M: host redshift/mass; Sigmac: critical surface density;
    // (r,phi): line-of-sight host-centric impact parameter and azimuth.
    // Only clumps above the dynamic floor m_res(r) are drawn. In the default reduced-host
    // model the same resolved mass fraction is removed from the smooth host in lensing.cpp;
    // the legacy model instead subtracts m*dkappa_host/dM per clump.
    // mass_out (optional): if non-null, accumulates the realized total resolved-clump
    // mass Sum_i m_i (the mass carved from the host in the mass-conserving scheme A).
    int addClumps(cosmology &C, int jz, int jM, double zl, double M, double Sigmac,
                   double r, double phi, rgen &mt,
                   double &kappa, double &gamma1, double &gamma2,
                   int subhalo_model = 0, bool subhalo_brute = false,
                   int threads = 1, int parallel_threshold = 200000,
                   double *mass_out = nullptr);

    // model-5 in-loop sampler: same signature role as addClumps but draws only the
    // clumps that clear kappa_thr,sub. Returns the number RENDERED (not proposed).
    int addClumpsRestricted(cosmology &C, int jz, int jM, double M, double Sigmac,
                            double r, double phi, rgen &mt,
                            double &kappa, double &gamma1, double &gamma2,
                            double *mass_out = nullptr);
};
