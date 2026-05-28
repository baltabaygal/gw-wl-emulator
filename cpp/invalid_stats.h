#pragma once
#include <cstdint>

struct InvalidSampleStats {
    std::uint64_t total_samples = 0;
    std::uint64_t valid_samples = 0;
    std::uint64_t invalid_samples = 0;
    std::uint64_t negative_detA = 0;
    std::uint64_t nonfinite_mu = 0;
    std::uint64_t negative_mu = 0;
    std::uint64_t nan_kappa = 0;
    std::uint64_t nan_gamma = 0;
    std::uint64_t overflow_mu = 0;
    std::uint64_t invalid_logmu = 0;
    std::uint64_t strict_weak_lensing_rejects = 0;

    double detA_min = 0.0;
    double detA_max = 0.0;
    double detA_mean = 0.0;
    std::uint64_t detA_near_zero_count = 0;
};
