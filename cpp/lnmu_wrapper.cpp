#include "lnmu_wrapper.h"
#include "cosmology.h"
#include "lensing.h"
#include <cmath>
#include <limits>
#include <stdexcept>


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

namespace {

struct TheoryContext {
    cosmology C;
    LensingConfig cfg;
    double kappathrH = 0.0;
    double skappaW = 0.0;
    std::vector<std::vector<std::vector<double>>> dNH;
};

TheoryContext make_theory_context(
    double z,
    const CosmologyParams& cp,
    const SamplingParams& sp
) {
    if (sp.bias != 0) {
        throw std::runtime_error("theory helper currently requires bias=0");
    }
    if (sp.fil != 0) {
        throw std::runtime_error("theory helper currently requires filaments=0");
    }
    if (sp.ell != 0) {
        throw std::runtime_error("theory helper currently requires ell=0");
    }

    TheoryContext ctx;
    auto &C = ctx.C;

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

    ctx.cfg.Nreal  = sp.Nreal;
    ctx.cfg.Nhalos = sp.Nhalos;
    ctx.cfg.fil    = sp.fil;
    ctx.cfg.bias   = sp.bias;
    ctx.cfg.ell    = sp.ell;
    ctx.cfg.exact_poisson = sp.exact_poisson;
    ctx.cfg.write  = 0;

    function<double(double)> NfNFW = [&C, z](double kappa) {
        return NhfNFW(C, z, kappa);
    };
    ctx.kappathrH = findkappathr(ctx.cfg.Nhalos, NfNFW);
    ctx.skappaW = sigmakappaW(C, z, ctx.kappathrH);
    ctx.dNH = deltaNhfNFW(C, z, ctx.kappathrH);

    return ctx;
}

std::complex<double> integrate_midpoint_piecewise_linear_phase(
    const std::vector<double>& phase_values,
    double q
) {
    const size_t n = phase_values.size();
    if (n == 0) {
        return std::complex<double>(0.0, 0.0);
    }
    if (n == 1) {
        return std::exp(std::complex<double>(0.0, -q * phase_values[0]));
    }

    const double du = 1.0 / static_cast<double>(n);
    std::complex<double> total = 0.5 * du * std::exp(std::complex<double>(0.0, -q * phase_values.front()))
                               + 0.5 * du * std::exp(std::complex<double>(0.0, -q * phase_values.back()));

    for (size_t i = 0; i + 1 < n; ++i) {
        const double phi0 = phase_values[i];
        const double dphi = phase_values[i + 1] - phi0;
        const std::complex<double> base = std::exp(std::complex<double>(0.0, -q * phi0));
        const double arg = q * dphi;

        if (std::abs(arg) < 1.0e-10) {
            total += du * base;
        } else {
            total += du * base * (std::exp(std::complex<double>(0.0, -arg)) - std::complex<double>(1.0, 0.0))
                     / std::complex<double>(0.0, -arg);
        }
    }

    return total;
}

double interpolate_descending(
    const std::vector<double>& x_desc,
    const std::vector<double>& y_desc,
    double xq
) {
    if (x_desc.empty() || y_desc.empty() || x_desc.size() != y_desc.size()) {
        throw std::runtime_error("interpolate_descending received invalid arrays");
    }
    if (xq >= x_desc.front()) {
        return y_desc.front();
    }
    if (xq <= x_desc.back()) {
        return y_desc.back();
    }

    size_t lo = 0;
    size_t hi = x_desc.size() - 1;
    while (hi - lo > 1) {
        size_t mid = (lo + hi) / 2;
        if (x_desc[mid] >= xq) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    double x0 = x_desc[lo];
    double x1 = x_desc[hi];
    double y0 = y_desc[lo];
    double y1 = y_desc[hi];
    double t = (xq - x0) / (x1 - x0);
    return y0 + t * (y1 - y0);
}

std::complex<double> integrate_u_of_k_piecewise_linear(
    const std::vector<double>& k_desc,
    const std::vector<double>& u_desc,
    double q,
    int n_kappa,
    double cluster_power
) {
    if (k_desc.size() < 2 || k_desc.size() != u_desc.size()) {
        throw std::runtime_error("integrate_u_of_k_piecewise_linear received invalid arrays");
    }
    if (n_kappa < 2) {
        throw std::runtime_error("integrate_u_of_k_piecewise_linear requires n_kappa >= 2");
    }

    const double kmax = k_desc.front();
    const double kmin = k_desc.back();
    const double u0 = u_desc.front();

    std::vector<double> kg(static_cast<size_t>(n_kappa) + 1);
    std::vector<double> ug(static_cast<size_t>(n_kappa) + 1);
    for (int i = 0; i <= n_kappa; ++i) {
        double s = static_cast<double>(i) / static_cast<double>(n_kappa);
        double k = kmin + (kmax - kmin) * std::pow(s, cluster_power);
        kg[i] = k;
        ug[i] = interpolate_descending(k_desc, u_desc, k);
    }

    std::complex<double> total = u0 * std::exp(std::complex<double>(0.0, -q * kmax));
    for (int i = 0; i < n_kappa; ++i) {
        double dk = kg[i + 1] - kg[i];
        double du = ug[i + 1] - ug[i];
        if (std::abs(dk) < 1.0e-16) {
            continue;
        }
        double slope = du / dk;
        if (std::abs(q) < 1.0e-12) {
            total -= slope * dk;
        } else {
            total -= slope * (
                std::exp(std::complex<double>(0.0, -q * kg[i + 1])) -
                std::exp(std::complex<double>(0.0, -q * kg[i]))
            ) / std::complex<double>(0.0, -q);
        }
    }
    return total;
}

} // namespace


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
    cfg.exact_poisson = sp.exact_poisson;
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

        if (muj > 0.0) {
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

RawLensingSamples sample_lensing_raw(
    double z,
    const CosmologyParams& cp,
    const SamplingParams& sp
) {
    cosmology C;

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
    cfg.exact_poisson = sp.exact_poisson;
    cfg.write  = 0;

    auto raw = L.sample_lnmu_raw(C, z, mt, cfg);

    RawLensingSamples out;
    out.kappa.reserve(raw.size());
    out.gamma1.reserve(raw.size());
    out.gamma2.reserve(raw.size());

    for (const auto &r : raw) {
        out.kappa.push_back(r.kappa);
        out.gamma1.push_back(r.gamma1);
        out.gamma2.push_back(r.gamma2);
        out.mean_kappa += r.kappa;
    }

    if (!raw.empty()) {
        out.mean_kappa /= static_cast<double>(raw.size());
    }

    return out;
}

RawLensingEvents sample_lensing_events(
    double z,
    const CosmologyParams& cp,
    const SamplingParams& sp
) {
    cosmology C;

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
    cfg.exact_poisson = sp.exact_poisson;
    cfg.write  = 0;

    auto events = L.sample_lensing_events_raw(C, z, mt, cfg);

    RawLensingEvents out;
    out.realization.reserve(events.size());
    out.is_filament.reserve(events.size());
    out.z.reserve(events.size());
    out.M.reserve(events.size());
    out.b.reserve(events.size());
    out.kappa.reserve(events.size());
    out.gamma1.reserve(events.size());
    out.gamma2.reserve(events.size());

    for (const auto &e : events) {
        out.realization.push_back(e.realization);
        out.is_filament.push_back(e.is_filament);
        out.z.push_back(e.z);
        out.M.push_back(e.M);
        out.b.push_back(e.b);
        out.kappa.push_back(e.kappa);
        out.gamma1.push_back(e.gamma1);
        out.gamma2.push_back(e.gamma2);
    }

    return out;
}

TheoryKappaSupport theory_kappa_support(
    double z,
    const CosmologyParams& cp,
    const SamplingParams& sp,
    int n_u
) {
    if (n_u < 2) {
        throw std::runtime_error("theory_kappa_support requires n_u >= 2");
    }

    TheoryContext ctx = make_theory_context(z, cp, sp);
    auto &C = ctx.C;
    auto &dNH = ctx.dNH;

    TheoryKappaSupport out;
    out.kappa_threshold = ctx.kappathrH;
    out.sigma_kappaW = ctx.skappaW;

    for (int jz = 0; jz < C.Nz; ++jz) {
        double zl = C.zlist[jz];
        if (zl >= z) {
            continue;
        }
        for (int jM = 0; jM < C.NM; ++jM) {
            double lambda = dNH[jz][jM][0];
            double rmaxH = dNH[jz][jM][2];
            if (lambda <= 0.0 || rmaxH <= 0.0) {
                continue;
            }
            double M = C.Mlist[jM];
            double weight = lambda / static_cast<double>(n_u);
            for (int iu = 0; iu < n_u; ++iu) {
                double u = (static_cast<double>(iu) + 0.5) / static_cast<double>(n_u);
                double r = rmaxH * std::sqrt(u);
                double kappa = kappagammaNFW(C, z, zl, r, M, 0.0, 0.0)[0];
                out.kappa.push_back(kappa);
                out.weight.push_back(weight);
            }
        }
    }

    return out;
}

std::vector<std::complex<double>> theory_kappa_generator(
    double z,
    const CosmologyParams& cp,
    const SamplingParams& sp,
    const std::vector<double>& q_values,
    int n_u
) {
    if (n_u < 2) {
        throw std::runtime_error("theory_kappa_generator requires n_u >= 2");
    }
    TheoryContext ctx = make_theory_context(z, cp, sp);
    auto &C = ctx.C;
    auto &dNH = ctx.dNH;

    std::vector<std::complex<double>> out(q_values.size(), std::complex<double>(0.0, 0.0));

    for (size_t iq = 0; iq < q_values.size(); ++iq) {
        double q = q_values[iq];
        out[iq] = std::complex<double>(-0.5 * ctx.skappaW * ctx.skappaW * q * q, 0.0);
    }

    for (int jz = 0; jz < C.Nz; ++jz) {
        double zl = C.zlist[jz];
        if (zl >= z) {
            continue;
        }
        for (int jM = 0; jM < C.NM; ++jM) {
            double lambda = dNH[jz][jM][0];
            double rmaxH = dNH[jz][jM][2];
            if (lambda <= 0.0 || rmaxH <= 0.0) {
                continue;
            }

            double M = C.Mlist[jM];
            std::vector<double> kappa_samples;
            kappa_samples.reserve(static_cast<size_t>(n_u));

            for (int iu = 0; iu < n_u; ++iu) {
                double u = (static_cast<double>(iu) + 0.5) / static_cast<double>(n_u);
                double r = rmaxH * std::sqrt(u);
                double kappa = kappagammaNFW(C, z, zl, r, M, 0.0, 0.0)[0];
                kappa_samples.push_back(kappa);
            }

            for (size_t iq = 0; iq < q_values.size(); ++iq) {
                double q = q_values[iq];
                std::complex<double> mark_cf = integrate_midpoint_piecewise_linear_phase(kappa_samples, q);
                out[iq] += lambda * (mark_cf - std::complex<double>(1.0, 0.0));
            }
        }
    }

    return out;
}

std::vector<std::complex<double>> theory_kappa_generator_kspace(
    double z,
    const CosmologyParams& cp,
    const SamplingParams& sp,
    const std::vector<double>& q_values,
    int n_u,
    int n_kappa,
    double cluster_power
) {
    if (n_u < 2) {
        throw std::runtime_error("theory_kappa_generator_kspace requires n_u >= 2");
    }

    TheoryContext ctx = make_theory_context(z, cp, sp);
    auto &C = ctx.C;
    auto &dNH = ctx.dNH;

    std::vector<std::complex<double>> out(q_values.size(), std::complex<double>(0.0, 0.0));
    for (size_t iq = 0; iq < q_values.size(); ++iq) {
        double q = q_values[iq];
        out[iq] = std::complex<double>(-0.5 * ctx.skappaW * ctx.skappaW * q * q, 0.0);
    }

    for (int jz = 0; jz < C.Nz; ++jz) {
        double zl = C.zlist[jz];
        if (zl >= z) {
            continue;
        }
        for (int jM = 0; jM < C.NM; ++jM) {
            double lambda = dNH[jz][jM][0];
            double rmaxH = dNH[jz][jM][2];
            if (lambda <= 0.0 || rmaxH <= 0.0) {
                continue;
            }

            double M = C.Mlist[jM];
            std::vector<double> k_desc;
            std::vector<double> u_desc;
            k_desc.reserve(static_cast<size_t>(n_u) + 1);
            u_desc.reserve(static_cast<size_t>(n_u) + 1);

            for (int iu = 0; iu < n_u; ++iu) {
                double u = (static_cast<double>(iu) + 0.5) / static_cast<double>(n_u);
                double r = rmaxH * std::sqrt(u);
                double kappa = kappagammaNFW(C, z, zl, r, M, 0.0, 0.0)[0];
                k_desc.push_back(kappa);
                u_desc.push_back(u);
            }
            k_desc.push_back(kappagammaNFW(C, z, zl, rmaxH, M, 0.0, 0.0)[0]);
            u_desc.push_back(1.0);

            for (size_t iq = 0; iq < q_values.size(); ++iq) {
                double q = q_values[iq];
                std::complex<double> mark_cf = integrate_u_of_k_piecewise_linear(
                    k_desc, u_desc, q, n_kappa, cluster_power
                );
                out[iq] += lambda * (mark_cf - std::complex<double>(1.0, 0.0));
            }
        }
    }

    return out;
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
