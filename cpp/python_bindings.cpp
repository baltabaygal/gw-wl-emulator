#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>

#include "lnmu_wrapper.h"

namespace py = pybind11;

PYBIND11_MODULE(gwlensing, m) {
    m.doc() = "GW weak-lensing: ln(mu) sampler and stats (pybind11)";

    auto build_cosmo = [](double OmegaM, double sigma8, double h) {
        CosmologyParams cosmo;
        cosmo.OmegaM = OmegaM;
        cosmo.sigma8 = sigma8;
        cosmo.h      = h;
        return cosmo;
    };

    auto build_sampling = [](int Nreal, std::uint64_t seed, bool filaments, bool bias, bool ell, bool exact_poisson, int Nhalos) {
        SamplingParams samp;
        samp.Nreal  = Nreal;
        samp.seed   = seed;
        samp.fil    = filaments ? 1 : 0;
        samp.bias   = bias ? 1 : 0;
        samp.ell    = ell ? 1 : 0;
        samp.exact_poisson = exact_poisson ? 1 : 0;
        samp.Nhalos = Nhalos;
        return samp;
    };

    // ---- sample_lnmu --------------------------------------------------------
    m.def(
        "sample_lnmu",
        [build_cosmo, build_sampling](double z, double OmegaM, double sigma8, double h,
           int Nreal, std::uint64_t seed,
           bool filaments, bool bias, bool ell, bool exact_poisson,
           int Nhalos) {
            auto cosmo = build_cosmo(OmegaM, sigma8, h);
            auto samp = build_sampling(Nreal, seed, filaments, bias, ell, exact_poisson, Nhalos);

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
        py::arg("exact_poisson") = false,
        py::arg("Nhalos") = 100
    );

    // ---- sample_lensing_raw -------------------------------------------------
    m.def(
        "sample_lensing_raw",
        [build_cosmo, build_sampling](double z, double OmegaM, double sigma8, double h,
           int Nreal, std::uint64_t seed,
           bool filaments, bool bias, bool ell, bool exact_poisson,
           int Nhalos) {
            auto cosmo = build_cosmo(OmegaM, sigma8, h);
            auto samp = build_sampling(Nreal, seed, filaments, bias, ell, exact_poisson, Nhalos);
            auto raw = sample_lensing_raw(z, cosmo, samp);

            auto array_from_vector = [](std::vector<double>&& v) {
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
            };

            py::dict out;
            out["kappa"] = array_from_vector(std::move(raw.kappa));
            out["gamma1"] = array_from_vector(std::move(raw.gamma1));
            out["gamma2"] = array_from_vector(std::move(raw.gamma2));
            out["mean_kappa"] = raw.mean_kappa;
            return out;
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
        py::arg("exact_poisson") = false,
        py::arg("Nhalos") = 100
    );

    // ---- sample_lensing_events ----------------------------------------------
    m.def(
        "sample_lensing_events",
        [build_cosmo, build_sampling](double z, double OmegaM, double sigma8, double h,
           int Nreal, std::uint64_t seed,
           bool filaments, bool bias, bool ell, bool exact_poisson,
           int Nhalos) {
            auto cosmo = build_cosmo(OmegaM, sigma8, h);
            auto samp = build_sampling(Nreal, seed, filaments, bias, ell, exact_poisson, Nhalos);
            auto events = sample_lensing_events(z, cosmo, samp);

            auto array_from_vector_double = [](std::vector<double>&& v) {
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
            };

            auto array_from_vector_int = [](std::vector<int>&& v) {
                auto *heap_vec = new std::vector<int>(std::move(v));
                auto capsule = py::capsule(heap_vec, [](void *p) {
                    delete reinterpret_cast<std::vector<int> *>(p);
                });
                return py::array_t<int>(
                    {static_cast<ssize_t>(heap_vec->size())},
                    {static_cast<ssize_t>(sizeof(int))},
                    heap_vec->data(),
                    capsule
                );
            };

            py::dict out;
            out["realization"] = array_from_vector_int(std::move(events.realization));
            out["is_filament"] = array_from_vector_int(std::move(events.is_filament));
            out["z"] = array_from_vector_double(std::move(events.z));
            out["M"] = array_from_vector_double(std::move(events.M));
            out["b"] = array_from_vector_double(std::move(events.b));
            out["kappa"] = array_from_vector_double(std::move(events.kappa));
            out["gamma1"] = array_from_vector_double(std::move(events.gamma1));
            out["gamma2"] = array_from_vector_double(std::move(events.gamma2));
            return out;
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
        py::arg("exact_poisson") = false,
        py::arg("Nhalos") = 100
    );

    // ---- theory_kappa_generator ---------------------------------------------
    m.def(
        "theory_kappa_generator",
        [build_cosmo, build_sampling](
            py::array_t<double, py::array::c_style | py::array::forcecast> q_values,
            double z, double OmegaM, double sigma8, double h,
            int Nreal, std::uint64_t seed,
            bool filaments, bool bias, bool ell, bool exact_poisson,
            int Nhalos, int n_u
        ) {
            auto cosmo = build_cosmo(OmegaM, sigma8, h);
            auto samp = build_sampling(Nreal, seed, filaments, bias, ell, exact_poisson, Nhalos);

            auto qbuf = q_values.request();
            const auto *qptr = static_cast<const double *>(qbuf.ptr);
            std::vector<double> q(qptr, qptr + qbuf.size);
            auto gen = theory_kappa_generator(z, cosmo, samp, q, n_u);

            auto *heap_vec = new std::vector<std::complex<double>>(std::move(gen));
            auto capsule = py::capsule(heap_vec, [](void *p) {
                delete reinterpret_cast<std::vector<std::complex<double>> *>(p);
            });

            return py::array_t<std::complex<double>>(
                {static_cast<ssize_t>(heap_vec->size())},
                {static_cast<ssize_t>(sizeof(std::complex<double>))},
                heap_vec->data(),
                capsule
            );
        },
        py::arg("q_values"),
        py::arg("z"),
        py::arg("OmegaM"),
        py::arg("sigma8"),
        py::arg("h"),
        py::arg("Nreal") = 50000,
        py::arg("seed") = 123,
        py::arg("filaments") = false,
        py::arg("bias") = false,
        py::arg("ell") = false,
        py::arg("exact_poisson") = false,
        py::arg("Nhalos") = 100,
        py::arg("n_u") = 256
    );

    // ---- theory_kappa_support -----------------------------------------------
    m.def(
        "theory_kappa_support",
        [build_cosmo, build_sampling](
            double z, double OmegaM, double sigma8, double h,
            int Nreal, std::uint64_t seed,
            bool filaments, bool bias, bool ell, bool exact_poisson,
            int Nhalos, int n_u
        ) {
            auto cosmo = build_cosmo(OmegaM, sigma8, h);
            auto samp = build_sampling(Nreal, seed, filaments, bias, ell, exact_poisson, Nhalos);
            auto support = theory_kappa_support(z, cosmo, samp, n_u);

            auto array_from_vector = [](std::vector<double>&& v) {
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
            };

            py::dict out;
            out["kappa"] = array_from_vector(std::move(support.kappa));
            out["weight"] = array_from_vector(std::move(support.weight));
            out["kappa_threshold"] = support.kappa_threshold;
            out["sigma_kappaW"] = support.sigma_kappaW;
            return out;
        },
        py::arg("z"),
        py::arg("OmegaM"),
        py::arg("sigma8"),
        py::arg("h"),
        py::arg("Nreal") = 50000,
        py::arg("seed") = 123,
        py::arg("filaments") = false,
        py::arg("bias") = false,
        py::arg("ell") = false,
        py::arg("exact_poisson") = false,
        py::arg("Nhalos") = 100,
        py::arg("n_u") = 256
    );

    // ---- theory_kappa_generator_kspace --------------------------------------
    m.def(
        "theory_kappa_generator_kspace",
        [build_cosmo, build_sampling](
            py::array_t<double, py::array::c_style | py::array::forcecast> q_values,
            double z, double OmegaM, double sigma8, double h,
            int Nreal, std::uint64_t seed,
            bool filaments, bool bias, bool ell, bool exact_poisson,
            int Nhalos, int n_u, int n_kappa, double cluster_power
        ) {
            auto cosmo = build_cosmo(OmegaM, sigma8, h);
            auto samp = build_sampling(Nreal, seed, filaments, bias, ell, exact_poisson, Nhalos);

            auto qbuf = q_values.request();
            const auto *qptr = static_cast<const double *>(qbuf.ptr);
            std::vector<double> q(qptr, qptr + qbuf.size);
            auto gen = theory_kappa_generator_kspace(z, cosmo, samp, q, n_u, n_kappa, cluster_power);

            auto *heap_vec = new std::vector<std::complex<double>>(std::move(gen));
            auto capsule = py::capsule(heap_vec, [](void *p) {
                delete reinterpret_cast<std::vector<std::complex<double>> *>(p);
            });

            return py::array_t<std::complex<double>>(
                {static_cast<ssize_t>(heap_vec->size())},
                {static_cast<ssize_t>(sizeof(std::complex<double>))},
                heap_vec->data(),
                capsule
            );
        },
        py::arg("q_values"),
        py::arg("z"),
        py::arg("OmegaM"),
        py::arg("sigma8"),
        py::arg("h"),
        py::arg("Nreal") = 50000,
        py::arg("seed") = 123,
        py::arg("filaments") = false,
        py::arg("bias") = false,
        py::arg("ell") = false,
        py::arg("exact_poisson") = false,
        py::arg("Nhalos") = 100,
        py::arg("n_u") = 256,
        py::arg("n_kappa") = 256,
        py::arg("cluster_power") = 4.0
    );

    // ---- compute_lnmu_stats -------------------------------------------------
    m.def(
        "compute_lnmu_stats",
        [build_cosmo, build_sampling](double z, double OmegaM, double sigma8, double h,
           int Nreal, std::uint64_t seed,
           bool filaments, bool bias, bool ell, bool exact_poisson,
           int Nhalos,
           bool fast) {
            auto cosmo = build_cosmo(OmegaM, sigma8, h);
            auto samp = build_sampling(Nreal, seed, filaments, bias, ell, exact_poisson, Nhalos);

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
        py::arg("exact_poisson") = false,
        py::arg("Nhalos") = 100,
        py::arg("fast") = true
    );
}
