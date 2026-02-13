#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "lnmu_wrapper.h"

namespace py = pybind11;

PYBIND11_MODULE(gwlensing, m) {
    m.doc() = "GW weak-lensing: ln(mu) sampler and stats (pybind11)";

    // ---- sample_lnmu --------------------------------------------------------
    m.def(
        "sample_lnmu",
        [](double z, double OmegaM, double sigma8, double h,
           int Nreal, std::uint64_t seed,
           bool filaments, bool bias, bool ell,
           int Nhalos) {

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
        py::arg("Nhalos") = 100
    );

    // ---- compute_lnmu_stats -------------------------------------------------
    m.def(
        "compute_lnmu_stats",
        [](double z, double OmegaM, double sigma8, double h,
           int Nreal, std::uint64_t seed,
           bool filaments, bool bias, bool ell,
           int Nhalos,
           bool fast) {

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
        py::arg("fast") = true
    );
}
