#pragma once
#include <vector>
#include <cstdint>

struct CosmologyParams {
    double OmegaM = 0.315;
    double sigma8 = 0.811;
    double h      = 0.674;

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
};

std::vector<double> sample_lnmu(
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
