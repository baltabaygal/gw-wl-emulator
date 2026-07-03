#include <random>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "lnmu_wrapper.h"
#include "cosmology.h"
#include "lensing.h"
#include <functional>

namespace py = pybind11;

double NhfNFW(cosmology &C, double zs, double kappathr);
double findkappathr(int N, std::function<double(double)> Nf);

static py::dict invalid_stats_to_dict(const InvalidSampleStats& s) {
    py::dict d;
    d["total_samples"] = s.total_samples;
    d["valid_samples"] = s.valid_samples;
    d["invalid_samples"] = s.invalid_samples;
    d["negative_detA"] = s.negative_detA;
    d["nonfinite_mu"] = s.nonfinite_mu;
    d["negative_mu"] = s.negative_mu;
    d["nan_kappa"] = s.nan_kappa;
    d["nan_gamma"] = s.nan_gamma;
    d["overflow_mu"] = s.overflow_mu;
    d["invalid_logmu"] = s.invalid_logmu;
    d["strict_weak_lensing_rejects"] = s.strict_weak_lensing_rejects;
    d["detA_min"] = s.detA_min;
    d["detA_max"] = s.detA_max;
    d["detA_mean"] = s.detA_mean;
    d["detA_near_zero_count"] = s.detA_near_zero_count;
    d["huge_threshold"] = 1.0e12;
    return d;
}

PYBIND11_MODULE(gwlensing, m) {
    m.doc() = "GW weak-lensing: ln(mu) sampler and stats (pybind11)";

    // ---- sample_lnmu --------------------------------------------------------
    m.def(
        "sample_lnmu",
        [](double z, double OmegaM, double sigma8, double h,
           int Nreal, std::uint64_t seed,
           bool filaments, bool bias, bool ell,
           int Nhalos, bool strict_weak_lensing,
           double Mmin, bool subhalo, double m_floor,
           int subhalo_threads, int subhalo_parallel_threshold,
           int subhalo_model, bool subhalo_brute, double subhalo_factor) {

            CosmologyParams cosmo;
            cosmo.OmegaM = OmegaM;
            cosmo.sigma8 = sigma8;
            cosmo.h      = h;
            cosmo.Mmin   = Mmin;

            SamplingParams samp;
            samp.Nreal  = Nreal;
            samp.seed   = seed;
            samp.fil    = filaments ? 1 : 0;
            samp.bias   = bias ? 1 : 0;
            samp.ell    = ell ? 1 : 0;
            samp.Nhalos = Nhalos;
            samp.strict_weak_lensing = strict_weak_lensing;
            samp.subhalo = subhalo;
            samp.m_floor = m_floor;
            samp.subhalo_threads = subhalo_threads;
            samp.subhalo_parallel_threshold = subhalo_parallel_threshold;
            samp.subhalo_model = subhalo_model;
            samp.subhalo_brute = subhalo_brute;
            samp.subhalo_factor = subhalo_factor;

            auto v = sample_lnmu(z, cosmo, samp);

            // Zero-copy: move vector onto heap and let NumPy own it via capsule.
            auto *heap_vec = new std::vector<double>(std::move(v));
            auto capsule = py::capsule(heap_vec, [](void *p) {
                delete reinterpret_cast<std::vector<double> *>(p);
            });

            return py::array_t<double>(
                {static_cast<ssize_t>(heap_vec->size())},
                {static_cast<ssize_t>(sizeof(double))},
                heap_vec->data(),
                capsule
            );
        },
        py::arg("z"),
        py::arg("OmegaM"),
        py::arg("sigma8"),
        py::arg("h"),
        py::arg("Nreal") = 50000,
        py::arg("seed") = 123,
        py::arg("filaments") = true,
        py::arg("bias") = true,
        py::arg("ell") = true,
        py::arg("Nhalos") = 100,
        py::arg("strict_weak_lensing") = false,
        py::arg("Mmin") = 1e7,
        py::arg("subhalo") = false,
        py::arg("m_floor") = 1e7,
        py::arg("subhalo_threads") = 1,
        py::arg("subhalo_parallel_threshold") = 200000,
        py::arg("subhalo_model") = 1,
        py::arg("subhalo_brute") = false,
        py::arg("subhalo_factor") = 1.0e-5
    );


    // ---- sample_lnmu_ml -----------------------------------------------------
    m.def(
        "sample_lnmu_ml",
        [](double z, double h, double OmegaM, double sigma8, int nsamples, py::object seed_obj, bool strict_weak_lensing, double Mmin, bool subhalo, double m_floor,
           int subhalo_threads, int subhalo_parallel_threshold, int subhalo_model, bool subhalo_brute, double subhalo_factor) {

            std::uint64_t seed = 0;
            if (seed_obj.is_none()) {
                std::random_device rd;
                seed = (static_cast<std::uint64_t>(rd()) << 32) | rd();
            } else {
                seed = seed_obj.cast<std::uint64_t>();
            }

            CosmologyParams cosmo;
            cosmo.OmegaM = OmegaM;
            cosmo.sigma8 = sigma8;
            cosmo.h      = h;
            cosmo.Mmin   = Mmin;

            SamplingParams samp;
            samp.Nreal  = nsamples;
            samp.seed   = seed;
            samp.fil    = 1;
            samp.bias   = 1;
            samp.ell    = 1;
            samp.Nhalos = 100;
            samp.strict_weak_lensing = strict_weak_lensing;
            samp.subhalo = subhalo;
            samp.m_floor = m_floor;
            samp.subhalo_threads = subhalo_threads;
            samp.subhalo_parallel_threshold = subhalo_parallel_threshold;
            samp.subhalo_model = subhalo_model;
            samp.subhalo_brute = subhalo_brute;
            samp.subhalo_factor = subhalo_factor;

            auto v = sample_lnmu(z, cosmo, samp);

            auto *heap_vec = new std::vector<double>(std::move(v));
            auto capsule = py::capsule(heap_vec, [](void *p) {
                delete reinterpret_cast<std::vector<double> *>(p);
            });

            return py::array_t<double>(
                {static_cast<ssize_t>(heap_vec->size())},
                {static_cast<ssize_t>(sizeof(double))},
                heap_vec->data(),
                capsule
            );
        },
        py::arg("z"),
        py::arg("h"),
        py::arg("OmegaM"),
        py::arg("sigma8"),
        py::arg("nsamples"),
        py::arg("seed") = py::none(),
        py::arg("strict_weak_lensing") = false,
        py::arg("Mmin") = 1e7,
        py::arg("subhalo") = false,
        py::arg("m_floor") = 1e7,
        py::arg("subhalo_threads") = 1,
        py::arg("subhalo_parallel_threshold") = 200000,
        py::arg("subhalo_model") = 1,
        py::arg("subhalo_brute") = false,
        py::arg("subhalo_factor") = 1.0e-5,
        "Simplified ML-facing wrapper API for raw ln(mu) sampling."
    );

    // ---- sample_lnmu_ml_with_diagnostics -----------------------------------
    m.def(
        "sample_lnmu_ml_with_diagnostics",
        [](double z, double h, double OmegaM, double sigma8, int nsamples, py::object seed_obj, bool strict_weak_lensing, double Mmin, bool subhalo, double m_floor,
           int subhalo_threads, int subhalo_parallel_threshold, int subhalo_model, bool subhalo_brute, double subhalo_factor) {
            std::uint64_t seed = 0;
            if (seed_obj.is_none()) {
                std::random_device rd;
                seed = (static_cast<std::uint64_t>(rd()) << 32) | rd();
            } else {
                seed = seed_obj.cast<std::uint64_t>();
            }

            CosmologyParams cosmo;
            cosmo.OmegaM = OmegaM;
            cosmo.sigma8 = sigma8;
            cosmo.h      = h;
            cosmo.Mmin   = Mmin;

            SamplingParams samp;
            samp.Nreal  = nsamples;
            samp.seed   = seed;
            samp.fil    = 1;
            samp.bias   = 1;
            samp.ell    = 1;
            samp.Nhalos = 100;
            samp.strict_weak_lensing = strict_weak_lensing;
            samp.subhalo = subhalo;
            samp.m_floor = m_floor;
            samp.subhalo_threads = subhalo_threads;
            samp.subhalo_parallel_threshold = subhalo_parallel_threshold;
            samp.subhalo_model = subhalo_model;
            samp.subhalo_brute = subhalo_brute;
            samp.subhalo_factor = subhalo_factor;

            auto diag = sample_lnmu_with_diagnostics(z, cosmo, samp);

            auto *heap_vec = new std::vector<double>(std::move(diag.lnmu));
            auto capsule = py::capsule(heap_vec, [](void *p) {
                delete reinterpret_cast<std::vector<double> *>(p);
            });

            py::dict out;
            out["lnmu"] = py::array_t<double>(
                {static_cast<ssize_t>(heap_vec->size())},
                {static_cast<ssize_t>(sizeof(double))},
                heap_vec->data(),
                capsule
            );
            out["invalid_stats"] = invalid_stats_to_dict(diag.invalid_stats);
            return out;
        },
        py::arg("z"),
        py::arg("h"),
        py::arg("OmegaM"),
        py::arg("sigma8"),
        py::arg("nsamples"),
        py::arg("seed") = py::none(),
        py::arg("strict_weak_lensing") = false,
        py::arg("Mmin") = 1e7,
        py::arg("subhalo") = false,
        py::arg("m_floor") = 1e7,
        py::arg("subhalo_threads") = 1,
        py::arg("subhalo_parallel_threshold") = 200000,
        py::arg("subhalo_model") = 1,
        py::arg("subhalo_brute") = false,
        py::arg("subhalo_factor") = 1.0e-5,
        "ML-facing sampler returning ln(mu) and invalid-sample diagnostics."
    );

    // ---- sample_lensing_raw_ml ---------------------------------------------
    m.def(
        "sample_lensing_raw_ml",
        [](double z, double h, double OmegaM, double sigma8, int nsamples, py::object seed_obj,
           bool filaments, bool bias, bool ell, int Nhalos, bool subhalo, double m_floor,
           int subhalo_threads, int subhalo_parallel_threshold, int subhalo_model, bool subhalo_brute,
           double subhalo_factor, double custom_kappathr, double Mmin) {
            std::uint64_t seed = 0;
            if (seed_obj.is_none()) {
                std::random_device rd;
                seed = (static_cast<std::uint64_t>(rd()) << 32) | rd();
            } else {
                seed = seed_obj.cast<std::uint64_t>();
            }

            cosmology C;
            C.OmegaM = OmegaM;
            C.sigma8 = sigma8;
            C.h      = h;
            C.OmegaB = 0.0493;
            C.zeq    = 3402.0;
            C.T0     = 2.7255;
            C.ns     = 0.965;
            C.Mmin   = Mmin;
            C.Mmax   = 1e17;
            C.NM     = 100;
            C.zmin   = 0.01;
            C.zmax   = 10.01;
            C.Nz     = 100;
            C.outdir = "dataL";

            int dm = 0;
            C.initialize(dm);

            lensing L;
            rgen mt(seed);
            LensingConfig cfg;
            cfg.Nreal = nsamples;
            cfg.Nhalos = Nhalos;
            cfg.fil = filaments ? 1 : 0;
            cfg.bias = bias ? 1 : 0;
            cfg.ell = ell ? 1 : 0;
            cfg.write = 0;
            cfg.subhalo = subhalo;
            cfg.m_floor = m_floor;
            cfg.subhalo_threads = subhalo_threads;
            cfg.subhalo_parallel_threshold = subhalo_parallel_threshold;
            cfg.subhalo_model = subhalo_model;
            cfg.subhalo_brute = subhalo_brute;
            cfg.subhalo_factor = subhalo_factor;
            cfg.custom_kappathr = custom_kappathr;

            auto raw = L.sample_lnmu_raw(C, z, mt, cfg);

            py::array_t<double> kappa(raw.size());
            py::array_t<double> gamma1(raw.size());
            py::array_t<double> gamma2(raw.size());
            py::array_t<double> kappa_nosub(raw.size());
            auto k = kappa.mutable_unchecked<1>();
            auto g1 = gamma1.mutable_unchecked<1>();
            auto g2 = gamma2.mutable_unchecked<1>();
            auto kns = kappa_nosub.mutable_unchecked<1>();

            for (ssize_t i = 0; i < static_cast<ssize_t>(raw.size()); ++i) {
                k(i) = raw[i].kappa;
                g1(i) = raw[i].gamma1;
                g2(i) = raw[i].gamma2;
                kns(i) = raw[i].kappa_nosub;
            }

            py::dict out;
            out["kappa"] = kappa;
            out["gamma1"] = gamma1;
            out["gamma2"] = gamma2;
            out["kappa_nosub"] = kappa_nosub;
            return out;
        },
        py::arg("z"),
        py::arg("h"),
        py::arg("OmegaM"),
        py::arg("sigma8"),
        py::arg("nsamples"),
        py::arg("seed") = py::none(),
        py::arg("filaments") = true,
        py::arg("bias") = true,
        py::arg("ell") = true,
        py::arg("Nhalos") = 100,
        py::arg("subhalo") = false,
        py::arg("m_floor") = 1e7,
        py::arg("subhalo_threads") = 1,
        py::arg("subhalo_parallel_threshold") = 200000,
        py::arg("subhalo_model") = 1,
        py::arg("subhalo_brute") = false,
        py::arg("subhalo_factor") = 1.0e-5,
        py::arg("custom_kappathr") = -1.0,
        py::arg("Mmin") = 1e7,
        "Return raw (kappa, gamma1, gamma2) realizations for diagnostics."
    );


    // ---- get_simulator_config -----------------------------------------------
    m.def("get_simulator_config", []() {
        py::dict d;
        d["filaments"] = true;
        d["bias"] = true;
        d["ell"] = true;
        d["Nhalos"] = 100;
        d["subhalo"] = false;
        d["m_floor"] = 1e7;
        d["subhalo_threads"] = 1;
        d["subhalo_parallel_threshold"] = 200000;
        d["subhalo_model"] = 1;
        d["subhalo_brute"] = false;
        d["subhalo_factor"] = 1.0e-5;
        return d;
    }, "Returns hardcoded default physics toggles used by sample_lnmu_ml");

    // ---- compute_lnmu_stats -------------------------------------------------
    m.def(
        "compute_lnmu_stats",
        [](double z, double OmegaM, double sigma8, double h,
           int Nreal, std::uint64_t seed,
           bool filaments, bool bias, bool ell,
           int Nhalos, bool strict_weak_lensing,
           bool fast, bool subhalo, double m_floor,
           int subhalo_threads, int subhalo_parallel_threshold,
           int subhalo_model, bool subhalo_brute, double subhalo_factor) {

            CosmologyParams cosmo;
            cosmo.OmegaM = OmegaM;
            cosmo.sigma8 = sigma8;
            cosmo.h      = h;

            SamplingParams samp;
            samp.Nreal  = Nreal;
            samp.seed   = seed;
            samp.fil    = filaments ? 1 : 0;
            samp.bias   = bias ? 1 : 0;
            samp.ell    = ell ? 1 : 0;
            samp.Nhalos = Nhalos;
            samp.strict_weak_lensing = strict_weak_lensing;
            samp.subhalo = subhalo;
            samp.m_floor = m_floor;
            samp.subhalo_threads = subhalo_threads;
            samp.subhalo_parallel_threshold = subhalo_parallel_threshold;
            samp.subhalo_model = subhalo_model;
            samp.subhalo_brute = subhalo_brute;
            samp.subhalo_factor = subhalo_factor;

            LnmuStats s = fast
                ? compute_lnmu_stats_fast(z, cosmo, samp)
                : compute_lnmu_stats(z, cosmo, samp);

            py::dict d;
            d["mean"]     = s.mean;
            d["variance"] = s.variance;
            d["skewness"] = s.skewness;
            d["mean_mu"]  = s.mean_mu;
            return d;
        },
        py::arg("z"),
        py::arg("OmegaM"),
        py::arg("sigma8"),
        py::arg("h"),
        py::arg("Nreal") = 50000,
        py::arg("seed") = 123,
        py::arg("filaments") = true,
        py::arg("bias") = true,
        py::arg("ell") = true,
        py::arg("Nhalos") = 100,
        py::arg("strict_weak_lensing") = false,
        py::arg("fast") = true,
        py::arg("subhalo") = false,
        py::arg("m_floor") = 1e7,
        py::arg("subhalo_threads") = 1,
        py::arg("subhalo_parallel_threshold") = 200000,
        py::arg("subhalo_model") = 1,
        py::arg("subhalo_brute") = false,
        py::arg("subhalo_factor") = 1.0e-5
    );

    m.def(
        "get_kappa_threshold",
        [](double z, double h, double OmegaM, double sigma8, int Nhalos) {
            cosmology C;
            C.OmegaM = OmegaM;
            C.sigma8 = sigma8;
            C.h      = h;
            C.OmegaB = 0.0493;
            C.zeq    = 3402.0;
            C.T0     = 2.7255;
            C.ns     = 0.965;
            C.Mmin   = 1e7;
            C.Mmax   = 1e17;
            C.NM     = 100;
            C.zmin   = 0.01;
            C.zmax   = 10.01;
            C.Nz     = 100;
            C.outdir = "dataL";
            C.initialize(0);

            std::function<double(double)> NfNFW = [&C, z](double kappa) {
                return NhfNFW(C, z, kappa);
            };
            return findkappathr(Nhalos, NfNFW);
        },
        py::arg("z"),
        py::arg("h"),
        py::arg("OmegaM"),
        py::arg("sigma8"),
        py::arg("Nhalos"),
        "Calculate the kappa threshold for a given number of resolved host halos."
    );

    m.def(
        "get_sigma_background",
        [](double z, double h, double OmegaM, double sigma8, double kappathr) {
            cosmology C;
            C.OmegaM = OmegaM;
            C.sigma8 = sigma8;
            C.h      = h;
            C.OmegaB = 0.0493;
            C.zeq    = 3402.0;
            C.T0     = 2.7255;
            C.ns     = 0.965;
            C.Mmin   = 1e7;
            C.Mmax   = 1e17;
            C.NM     = 100;
            C.zmin   = 0.01;
            C.zmax   = 10.01;
            C.Nz     = 100;
            C.outdir = "dataL";
            C.initialize(0);

            return sigmakappaW(C, z, kappathr);
        },
        py::arg("z"),
        py::arg("h"),
        py::arg("OmegaM"),
        py::arg("sigma8"),
        py::arg("kappathr"),
        "Calculate the background standard deviation for a given kappa threshold."
    );
}
