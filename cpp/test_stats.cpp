#include "lnmu_wrapper.h"
#include <iostream>
#include <iomanip>
#include <cmath>

int main() {

    CosmologyParams cosmo;
    SamplingParams sampling;

    sampling.Nreal = 1000;
    sampling.seed = 123;

    std::cout << "Comparing compute_lnmu_stats vs compute_lnmu_stats_fast...\n\n";

    auto stats_slow = compute_lnmu_stats(1.5, cosmo, sampling);
    auto stats_fast = compute_lnmu_stats_fast(1.5, cosmo, sampling);

    std::cout << std::fixed << std::setprecision(8);
    
    std::cout << "                Slow         Fast         Diff\n";
    std::cout << "mean lnmu: " << stats_slow.mean << "  " << stats_fast.mean 
              << "  " << std::abs(stats_slow.mean - stats_fast.mean) << "\n";
    std::cout << "var lnmu:  " << stats_slow.variance << "  " << stats_fast.variance 
              << "  " << std::abs(stats_slow.variance - stats_fast.variance) << "\n";
    std::cout << "skewness:  " << stats_slow.skewness << "  " << stats_fast.skewness 
              << "  " << std::abs(stats_slow.skewness - stats_fast.skewness) << "\n";
    std::cout << "mean mu:   " << stats_slow.mean_mu << "  " << stats_fast.mean_mu 
              << "  " << std::abs(stats_slow.mean_mu - stats_fast.mean_mu) << "\n";

    // Check agreement
    double tol = 1e-10;
    bool agree = (std::abs(stats_slow.mean - stats_fast.mean) < tol &&
                  std::abs(stats_slow.variance - stats_fast.variance) < tol &&
                  std::abs(stats_slow.skewness - stats_fast.skewness) < tol &&
                  std::abs(stats_slow.mean_mu - stats_fast.mean_mu) < tol);
    
    std::cout << "\nMatch: " << (agree ? "✓ YES" : "✗ NO") << "\n";

    return agree ? 0 : 1;
}

