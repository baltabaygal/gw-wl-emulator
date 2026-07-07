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
    bool subhalo = false;
    double m_floor = 1.0e7;
    int subhalo_threads = 1;
    int subhalo_parallel_threshold = 200000;
    int subhalo_model = 1;
    bool subhalo_brute = false;
    double subhalo_factor = 1.0e-5;  // cross-redshift plateau choice, scripts/subhalo_factor_redshift_check.py (2026-07-03)
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
