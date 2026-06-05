#pragma once
#include <complex>
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
    int exact_poisson = 0;
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

struct RawLensingSamples {
    std::vector<double> kappa;
    std::vector<double> gamma1;
    std::vector<double> gamma2;
    double mean_kappa = 0.0;
};

struct RawLensingEvents {
    std::vector<int> realization;
    std::vector<int> is_filament;
    std::vector<double> z;
    std::vector<double> M;
    std::vector<double> b;
    std::vector<double> kappa;
    std::vector<double> gamma1;
    std::vector<double> gamma2;
};

struct TheoryKappaSupport {
    std::vector<double> kappa;
    std::vector<double> weight;
    double kappa_threshold = 0.0;
    double sigma_kappaW = 0.0;
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

RawLensingSamples sample_lensing_raw(
    double z,
    const CosmologyParams& cosmo,
    const SamplingParams& sampling
);

RawLensingEvents sample_lensing_events(
    double z,
    const CosmologyParams& cosmo,
    const SamplingParams& sampling
);

TheoryKappaSupport theory_kappa_support(
    double z,
    const CosmologyParams& cosmo,
    const SamplingParams& sampling,
    int n_u = 256
);

std::vector<std::complex<double>> theory_kappa_generator(
    double z,
    const CosmologyParams& cosmo,
    const SamplingParams& sampling,
    const std::vector<double>& q_values,
    int n_u = 256
);

std::vector<std::complex<double>> theory_kappa_generator_kspace(
    double z,
    const CosmologyParams& cosmo,
    const SamplingParams& sampling,
    const std::vector<double>& q_values,
    int n_u = 256,
    int n_kappa = 256,
    double cluster_power = 4.0
);
