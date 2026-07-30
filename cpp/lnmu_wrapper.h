#pragma once
#include <vector>
#include <cstdint>
#include "invalid_stats.h"

struct CosmologyParams {
    double OmegaM = 0.315;
    double sigma8 = 0.811;
    double h      = 0.674;

    // primordial amplitude: As > 0 normalizes P(k) directly (sigma8 ignored);
    // As <= 0 keeps the sigma8 normalization.
    double As     = -1.0;
    double kpivot = 5.0e-5;   // comoving kpc^-1 (= 0.05 Mpc^-1)

    // grid defaults
    double OmegaB = 0.0493;
    double zeq    = 3402.0;
    double T0     = 2.7255;
    double ns     = 0.965;

    double Mmin = 1e7;
    double Mmax = 1e17;
    int NM      = 100;

    double zmin = 0.01;
    double zmax = 10.01;
    int Nz      = 100;
};

struct SamplingParams {
    int Nreal   = 1000;
    uint64_t seed = 123;
    int fil     = 1;
    int bias    = 1;
    int ell     = 1;
    int Nhalos  = 100;
    bool strict_weak_lensing = false;
    bool subhalo = true;             // PAPER DEFAULT (2026-07-29): the draft's model includes subhalos
    double m_floor = 1.0e7;
    int subhalo_threads = 1;
    int subhalo_parallel_threshold = 200000;
    int subhalo_model = 5;   // PAPER DEFAULT (2026-07-29): brute population + per-clump kappa threshold
    bool subhalo_brute = false;
    double subhalo_factor = 1.0e-2;  // PDF-level brute acceptance, scripts/convergence/subhalo_factor_jsd.py (2026-07-12)
    bool subhalo_carve = true;       // mass-conserving realized-clump host carve (scheme A, 2026-07-22); false = legacy (1-f_s,b)M
    bool subhalo_virial = true;      // PAPER DEFAULT (2026-07-29): JvdB14 virial convention, psi = m/M_vir, profile to r_vir; model 4/5 only
    // model 5 only (2026-07-27): per-clump convergence threshold kappa_thr,sub.
    // > 0 absolute; <= 0 uses subhalo_kappathr_factor * host kappa_thr.
    double subhalo_kappathr = -1.0;
    double subhalo_kappathr_factor = 0.1;
    double psi_min_fixed = -1.0;    // diagnostic: > 0 fixes psi_min (overrides m_floor/M); e.g. 1e-4 = psi_res
    double kappathr_flat = -1.0;   // <= 0 = legacy <N>=Nhalos rule (default, reverted 2026-07-10); > 0 = flat explicit-halo threshold
    int kappa_anchor = 1;          // PAPER DEFAULT (2026-07-29): robust anchor. 0 = legacy batch mean,
    double kappa_anchor_cut = 1.0; // 1 robust (exclude kappa > cut), 2 external value below
    double kappa_anchor_value = 0.0; // used only when kappa_anchor == 2
    int bias_model = 1;            // PAPER DEFAULT (2026-07-29): correlated 1D field (draft sec. II.A); 0 = legacy iid cell bias
    double bias_Rperp = 20000.0;   // PAPER DEFAULT (2026-07-29): R_s = 20 Mpc, confirmed by Ville (bias_model = 1 only)
    bool bias_weak = true;         // PAPER DEFAULT (2026-07-29): sub-threshold background rides the field (requires bias_model = 1)
    int bias_window = 1;           // PAPER DEFAULT (2026-07-29): real-space spherical top-hat. 0 = legacy transverse disk;
                                   // 1 spherical top-hat, 2 Gaussian (both on |k|; require bias_model = 1)
    bool fil_bias = true;          // PAPER DEFAULT (2026-07-29): filaments use filbias (PBS of pFCfil, q=0.7); requires bias_model = 1
};

struct LnmuSampleDiagnostics {
    std::vector<double> lnmu;
    InvalidSampleStats invalid_stats;
};

std::vector<double> sample_lnmu(
    double z,
    const CosmologyParams& cosmo,
    const SamplingParams& sampling
);

LnmuSampleDiagnostics sample_lnmu_with_diagnostics(
    double z,
    const CosmologyParams& cosmo,
    const SamplingParams& sampling
);

struct LnmuStats {
    double mean = 0.0;
    double variance = 0.0;
    double skewness = 0.0;
    double mean_mu = 0.0;   // <mu> for flux conservation
};

LnmuStats compute_lnmu_stats(
    double z,
    const CosmologyParams& cosmo,
    const SamplingParams& sampling
);

LnmuStats compute_lnmu_stats_fast(
    double z,
    const CosmologyParams& cosmo,
    const SamplingParams& sampling
);
