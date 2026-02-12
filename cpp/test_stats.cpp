#include "lnmu_wrapper.h"
#include <iostream>

int main() {

    CosmologyParams cosmo;
    SamplingParams sampling;

    sampling.Nreal = 1000;
    sampling.seed = 123;

    auto stats = compute_lnmu_stats(1.0, cosmo, sampling);

    std::cout << "mean lnmu = " << stats.mean << "\n";
    std::cout << "var lnmu  = " << stats.variance << "\n";
    std::cout << "skew      = " << stats.skewness << "\n";
    std::cout << "mean mu   = " << stats.mean_mu << "\n";

    return 0;
}
