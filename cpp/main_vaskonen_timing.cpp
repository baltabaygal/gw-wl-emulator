#include "cosmology.h"
#include "lensing.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

cosmology make_planck_cosmology() {
    cosmology C;
    C.OmegaM = 0.315;
    C.OmegaB = 0.0493;
    C.zeq = 3402.0;
    C.sigma8 = 0.811;
    C.h = 0.674;
    C.T0 = 2.7255;
    C.ns = 0.965;
    C.Mmin = 1.0e7;
    C.Mmax = 1.0e17;
    C.NM = 100;
    C.zmin = 0.01;
    C.zmax = 10.01;
    C.Nz = 100;
    C.outdir = "dataL";
    C.initialize(0);
    return C;
}

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

void run_pdf_batch(const std::string &label, cosmology &C, lensing &L, const std::vector<double> &zs, rgen &mt) {
    auto start = std::chrono::steady_clock::now();
    std::cout << "{\n";
    std::cout << "  \"label\": \"" << label << "\",\n";
    std::cout << "  \"Nreal\": " << L.Nreal << ",\n";
    std::cout << "  \"redshifts\": [";
    for (size_t i = 0; i < zs.size(); i++) {
        if (i) std::cout << ", ";
        std::cout << zs[i];
    }
    std::cout << "],\n";
    std::cout << "  \"per_redshift_seconds\": [";
    for (size_t i = 0; i < zs.size(); i++) {
        auto zi_start = std::chrono::steady_clock::now();
        auto Plnmu = L.Plnmuf(C, zs[i], mt, 1, 1, 1, 0);
        double zi_seconds = seconds_since(zi_start);
        if (i) std::cout << ", ";
        std::cout << zi_seconds;
        std::cerr << label << " z=" << zs[i] << " bins=" << Plnmu.size()
                  << " seconds=" << zi_seconds << "\n";
    }
    std::cout << "],\n";
    std::cout << "  \"total_seconds\": " << seconds_since(start) << "\n";
    std::cout << "}\n";
}

} // namespace

int main(int argc, char **argv) {
    std::cout << std::setprecision(8) << std::fixed;

    std::string mode = (argc > 1) ? argv[1] : "likelihood6";
    unsigned long long seed = (argc > 2) ? std::stoull(argv[2]) : 42ULL;

    cosmology C = make_planck_cosmology();
    lensing L;
    L.Nhalos = 100;
    L.Nbins = 12;
    rgen mt(seed);

    if (mode == "fig2_single") {
        L.Nreal = 400000;
        run_pdf_batch(mode, C, L, {1.0}, mt);
    } else if (mode == "fig2_14") {
        L.Nreal = 400000;
        run_pdf_batch(mode, C, L, {0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0}, mt);
    } else if (mode == "likelihood6") {
        L.Nreal = 10000;
        run_pdf_batch(mode, C, L, {0.05, 0.44, 0.83, 1.22, 1.61, 2.0}, mt);
    } else if (mode == "likelihood10") {
        L.Nreal = 10000;
        run_pdf_batch(mode, C, L, {0.05, 0.44, 0.83, 1.22, 1.61, 2.0, 2.9, 4.2, 6.1, 10.0}, mt);
    } else {
        std::cerr << "Unknown mode: " << mode << "\n";
        return 1;
    }

    return 0;
}
