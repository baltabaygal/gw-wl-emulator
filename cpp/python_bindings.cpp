#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "lnmu_wrapper.h"  // your existing C++ wrapper

namespace py = pybind11;

static py::array_t<double> vec_to_numpy(std::vector<double>&& v) {
    // Move vector into a new heap allocation owned by a capsule
    auto *vec = new std::vector<double>(std::move(v));
    auto capsule = py::capsule(vec, [](void *p) {
        delete reinterpret_cast<std::vector<double>*>(p);
    });
    return py::array_t<double>(
        { static_cast<py::ssize_t>(vec->size()) },
        { static_cast<py::ssize_t>(sizeof(double)) },
        vec->data(),
        capsule
    );
}

PYBIND11_MODULE(gwlensing, m) {
    m.doc() = "GW weak-lensing ln(mu) sampler/stats (C++ backend)";

    m.def(
        "sample_lnmu",
        [](double z, double OmegaM, double sigma8, double h,
           int N, int seed,
           bool halos, bool bias, bool filaments) {
            // Match your CLI convention: enable/disable components
            auto v = sample_lnmu(z, OmegaM, sigma8, h, N, seed, halos, bias, filaments);
            return vec_to_numpy(std::move(v));
        },
        py::arg("z"),
        py::arg("OmegaM"),
        py::arg("sigma8"),
        py::arg("h"),
        py::arg("N") = 50000,
        py::arg("seed") = 123,
        py::arg("halos") = true,
        py::arg("bias") = true,
        py::arg("filaments") = true
    );

    m.def(
        "lnmu_stats",
        [](double z, double OmegaM, double sigma8, double h,
           int N, int seed,
           bool halos, bool bias, bool filaments,
           bool fast) {
            LnmuStats s = fast
                ? compute_lnmu_stats_fast(z, OmegaM, sigma8, h, N, seed, halos, bias, filaments)
                : compute_lnmu_stats(z, OmegaM, sigma8, h, N, seed, halos, bias, filaments);

            py::dict d;
            d["mean_lnmu"] = s.mean_lnmu;
            d["var_lnmu"]  = s.var_lnmu;
            d["skew_lnmu"] = s.skew_lnmu;
            d["mean_mu"]   = s.mean_mu;
            d["N_used"]    = s.N_used;   // if you have it; otherwise remove
            d["N_total"]   = s.N_total;  // if you have it; otherwise remove
            return d;
        },
        py::arg("z"),
        py::arg("OmegaM"),
        py::arg("sigma8"),
        py::arg("h"),
        py::arg("N") = 50000,
        py::arg("seed") = 123,
        py::arg("halos") = true,
        py::arg("bias") = true,
        py::arg("filaments") = true,
        py::arg("fast") = true
    );
}
