#include "cosmology.h"
#include "lensing.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

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

void write_profile_csv(const fs::path &path, bool subhalo, const LensingProfile &p) {
    std::ofstream out(path);
    out << std::setprecision(12);
    out << "subhalo,zs,Nreal,M_host,log10_M_host,host_events,smooth_host_seconds,"
           "subhalo_seconds,subhalo_calls,subhalo_clumps,mean_smooth_us_per_host,"
           "mean_subhalo_ms_per_call,mean_clumps_per_call,mean_fsub_seen,mean_Nsub_seen,"
           "subhalo_time_fraction\n";

    double measured = 0.0;
    for (double v : p.smooth_host_seconds) measured += v;
    for (double v : p.subhalo_seconds) measured += v;

    for (size_t i = 0; i < p.M_host.size(); i++) {
        if (p.host_events[i] == 0 && p.subhalo_calls[i] == 0) continue;
        double host_events = static_cast<double>(p.host_events[i]);
        double calls = static_cast<double>(p.subhalo_calls[i]);
        double mean_fsub = host_events > 0.0 ? p.fsub_weighted_sum[i] / host_events : 0.0;
        double mean_nsub = host_events > 0.0 ? p.nsub_mean_weighted_sum[i] / host_events : 0.0;
        double mean_smooth_us = host_events > 0.0 ? 1.0e6 * p.smooth_host_seconds[i] / host_events : 0.0;
        double mean_sub_ms = calls > 0.0 ? 1.0e3 * p.subhalo_seconds[i] / calls : 0.0;
        double mean_clumps = calls > 0.0 ? static_cast<double>(p.subhalo_clumps[i]) / calls : 0.0;
        double frac = measured > 0.0 ? p.subhalo_seconds[i] / measured : 0.0;
        out << (subhalo ? "true" : "false") << ","
            << p.zs << ","
            << p.Nreal << ","
            << p.M_host[i] << ","
            << std::log10(p.M_host[i]) << ","
            << p.host_events[i] << ","
            << p.smooth_host_seconds[i] << ","
            << p.subhalo_seconds[i] << ","
            << p.subhalo_calls[i] << ","
            << p.subhalo_clumps[i] << ","
            << mean_smooth_us << ","
            << mean_sub_ms << ","
            << mean_clumps << ","
            << mean_fsub << ","
            << mean_nsub << ","
            << frac << "\n";
    }
}

void write_profile_summary(const fs::path &path, bool subhalo, const LensingProfile &p) {
    double smooth = 0.0, sub = 0.0;
    uint64_t host_events = 0, sub_calls = 0, clumps = 0;
    for (size_t i = 0; i < p.M_host.size(); i++) {
        smooth += p.smooth_host_seconds[i];
        sub += p.subhalo_seconds[i];
        host_events += p.host_events[i];
        sub_calls += p.subhalo_calls[i];
        clumps += p.subhalo_clumps[i];
    }

    std::ofstream out(path);
    out << std::setprecision(12);
    out << "{\n";
    out << "  \"subhalo\": " << (subhalo ? "true" : "false") << ",\n";
    out << "  \"zs\": " << p.zs << ",\n";
    out << "  \"Nreal\": " << p.Nreal << ",\n";
    out << "  \"total_seconds\": " << p.total_seconds << ",\n";
    out << "  \"subhalo_precompute_seconds\": " << p.subhalo_precompute_seconds << ",\n";
    out << "  \"smooth_host_seconds_measured\": " << smooth << ",\n";
    out << "  \"subhalo_seconds_measured\": " << sub << ",\n";
    out << "  \"host_events\": " << host_events << ",\n";
    out << "  \"subhalo_calls\": " << sub_calls << ",\n";
    out << "  \"subhalo_clumps\": " << clumps << ",\n";
    out << "  \"mean_clumps_per_subhalo_call\": " << (sub_calls ? static_cast<double>(clumps) / sub_calls : 0.0) << ",\n";
    out << "  \"subhalo_carve_negatives\": " << p.subhalo_carve_negatives << "\n";
    out << "}\n";
}

void run_one(bool subhalo, int Nreal, double zs, uint64_t seed, const fs::path &outdir,
             int subhalo_threads, int subhalo_parallel_threshold, double subhalo_factor) {
    cosmology C = make_planck_cosmology();
    lensing L;
    LensingConfig cfg;
    cfg.Nreal = Nreal;
    cfg.Nhalos = 100;
    cfg.Nbins = 100;
    cfg.fil = 1;
    cfg.bias = 1;
    cfg.ell = 1;
    cfg.write = 0;
    cfg.subhalo = subhalo;
    cfg.m_floor = 1.0e7;
    cfg.subhalo_threads = subhalo_threads;
    cfg.subhalo_parallel_threshold = subhalo_parallel_threshold;
    cfg.subhalo_factor = subhalo_factor;

    LensingProfile profile;
    cfg.profile = &profile;
    rgen mt(seed);
    auto raw = L.sample_lnmu_raw(C, zs, mt, cfg);
    (void)raw;

    std::string stem = subhalo ? "subhalo_true" : "subhalo_false";
    write_profile_csv(outdir / (stem + "_by_mass.csv"), subhalo, profile);
    write_profile_summary(outdir / (stem + "_summary.json"), subhalo, profile);
    std::cerr << stem << " total_seconds=" << profile.total_seconds << "\n";
}

} // namespace

int main(int argc, char **argv) {
    int Nreal = (argc > 1) ? std::atoi(argv[1]) : 1000;
    double zs = (argc > 2) ? std::atof(argv[2]) : 1.0;
    uint64_t seed = (argc > 3) ? std::stoull(argv[3]) : 42ULL;
    fs::path outdir = (argc > 4) ? fs::path(argv[4]) : fs::path("plots/figures/subhalo_runtime/profile");
    int subhalo_threads = (argc > 5) ? std::atoi(argv[5]) : 1;
    int subhalo_parallel_threshold = (argc > 6) ? std::atoi(argv[6]) : 200000;
    double subhalo_factor = (argc > 7) ? std::atof(argv[7]) : 1.0e-2;

    fs::create_directories(outdir);
    run_one(false, Nreal, zs, seed, outdir, subhalo_threads, subhalo_parallel_threshold, subhalo_factor);
    run_one(true, Nreal, zs, seed, outdir, subhalo_threads, subhalo_parallel_threshold, subhalo_factor);
    return 0;
}
