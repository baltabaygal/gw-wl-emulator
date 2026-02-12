#include "lnmu_wrapper.h"
#include "cosmology.h"
#include "lensing.h"
#include <cmath>

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
