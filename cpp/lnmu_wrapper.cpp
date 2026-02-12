#include "lnmu_wrapper.h"
#include "cosmology.h"
#include "lensing.h"
#include <cmath>


struct OnlineMoments {
    size_t n = 0;
    double mean = 0.0;
    double M2 = 0.0;
    double M3 = 0.0;

    void update(double x) {
        size_t n1 = n;
        n++;

        double delta = x - mean;
        double delta_n = delta / n;
        double delta_n2 = delta_n * delta_n;
        double term1 = delta * delta_n * n1;

        mean += delta_n;
        M3 += term1 * delta_n2 * (n - 2)
              - 3.0 * delta_n * M2;
        M2 += term1;
    }

    double variance() const {
        return (n > 0) ? M2 / n : 0.0;
    }

    double skewness() const {
        if (n < 2) return 0.0;
        double var = variance();
        return (var > 0) ? (M3 / n) / std::pow(var, 1.5) : 0.0;
    }
};


LnmuStats compute_lnmu_stats_fast(
    double z,
    const CosmologyParams& cp,
    const SamplingParams& sp
) {
    cosmology C;

    // Cosmology parameters
    C.OmegaM = cp.OmegaM;
    C.sigma8 = cp.sigma8;
    C.h      = cp.h;

    C.OmegaB = cp.OmegaB;
    C.zeq    = cp.zeq;
    C.T0     = cp.T0;
    C.ns     = cp.ns;

    C.Mmin = cp.Mmin;
    C.Mmax = cp.Mmax;
    C.NM   = cp.NM;

    C.zmin = cp.zmin;
    C.zmax = cp.zmax;
    C.Nz   = cp.Nz;

    C.outdir = "dataL";

    int dm = 0;
    C.initialize(dm);

    lensing L;
    rgen mt(sp.seed);

    LensingConfig cfg;
    cfg.Nreal  = sp.Nreal;
    cfg.Nhalos = sp.Nhalos;
    cfg.fil    = sp.fil;
    cfg.bias   = sp.bias;
    cfg.ell    = sp.ell;
    cfg.write  = 0;

    // Get raw realizations (κ, γ1, γ2 per realization)
    auto raw = L.sample_lnmu_raw(C, z, mt, cfg);

    // Compute meankappa
    double meankappa = 0.0;
    for (auto &r : raw)
        meankappa += r.kappa;
    meankappa /= raw.size();

    // First pass: compute lnmu values and mean
    std::vector<double> lnmu_values;
    lnmu_values.reserve(raw.size());
    double sum_lnmu = 0.0;
    double sum_mu = 0.0;

    for (auto &r : raw) {
        double kappaj = r.kappa - meankappa;
        double gammaj = std::sqrt(r.gamma1*r.gamma1 + r.gamma2*r.gamma2);
        double muj = 1.0 / (std::pow(1.0-kappaj, 2.0) - std::pow(gammaj, 2.0));

        if (muj > 0.0 && gammaj > 0.0) {
            double lnmu = std::log(muj);
            lnmu_values.push_back(lnmu);
            sum_lnmu += lnmu;
            sum_mu += muj;
        }
    }

    size_t n = lnmu_values.size();
    if (n == 0) {
        LnmuStats stats;
        return stats;
    }

    double mean_lnmu = sum_lnmu / n;
    double mean_mu = sum_mu / n;

    // Second pass: compute variance and skewness
    double sum2 = 0.0;
    double sum3 = 0.0;

    for (double lnmu : lnmu_values) {
        double d = lnmu - mean_lnmu;
        sum2 += d * d;
        sum3 += d * d * d;
    }

    double variance = sum2 / n;
    double skewness = 0.0;
    if (variance > 0) {
        skewness = (sum3 / n) / std::pow(variance, 1.5);
    }

    LnmuStats stats;
    stats.mean = mean_lnmu;
    stats.variance = variance;
    stats.skewness = skewness;
    stats.mean_mu = mean_mu;

    return stats;
}




LnmuStats compute_lnmu_stats(
    double z,
    const CosmologyParams& cp,
    const SamplingParams& sp
) {
    // Reuse existing sampler
    auto y = sample_lnmu(z, cp, sp);

    LnmuStats stats;

    const size_t N = y.size();
    if (N == 0) return stats;

    // First pass: mean
    double sum = 0.0;
    double sum_mu = 0.0;
    for (double v : y) {
        sum += v;
        sum_mu += std::exp(v);
    }

    stats.mean = sum / N;
    stats.mean_mu = sum_mu / N;

    // Second pass: variance & skewness
    double var = 0.0;
    double skew = 0.0;

    for (double v : y) {
        double d = v - stats.mean;
        var += d * d;
        skew += d * d * d;
    }

    var /= N;
    skew /= N;

    stats.variance = var;

    if (var > 0)
        stats.skewness = skew / std::pow(var, 1.5);

    return stats;
}



std::vector<double> sample_lnmu(
    double z,
    const CosmologyParams& cp,
    const SamplingParams& sp
) {
    cosmology C;

    // cosmology parameters
    C.OmegaM = cp.OmegaM;
    C.sigma8 = cp.sigma8;
    C.h      = cp.h;

    C.OmegaB = cp.OmegaB;
    C.zeq    = cp.zeq;
    C.T0     = cp.T0;
    C.ns     = cp.ns;

    C.Mmin = cp.Mmin;
    C.Mmax = cp.Mmax;
    C.NM   = cp.NM;

    C.zmin = cp.zmin;
    C.zmax = cp.zmax;
    C.Nz   = cp.Nz;

    C.outdir = "dataL";

    int dm = 0;
    C.initialize(dm);

    lensing L;
    rgen mt(sp.seed);

    LensingConfig cfg;
    cfg.Nreal  = sp.Nreal;
    cfg.Nhalos = sp.Nhalos;
    cfg.fil    = sp.fil;
    cfg.bias   = sp.bias;
    cfg.ell    = sp.ell;
    cfg.write  = 0;

    return L.sample_lnmu(C, z, mt, cfg);
}
