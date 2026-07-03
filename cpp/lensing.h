
#pragma once
#include <vector>
#include <array>
#include <cstdint>
#include "invalid_stats.h"
#include "subhalo.h"

// NFW kappa/gamma kernels (defined in lensing.cpp; used by subhalo.cpp)
std::array<double,2> FgNFW(double x);
double kappa0NFW(double rs, double rhos, double Sigmac);
std::array<double,2> kappagammaNFWeps(double epsilon, double kappa0, double x, double phi);
double rmaxfNFW(cosmology &C, double zs, double zl, double M, double kappathr);
double sigmakappaW(cosmology &C, double zs, double kappathr);

struct LensingProfile {
  int Nreal = 0;
  double zs = 0.0;
  double total_seconds = 0.0;
  double subhalo_precompute_seconds = 0.0;
  std::vector<double> M_host;
  std::vector<double> smooth_host_seconds;
  std::vector<double> subhalo_seconds;
  std::vector<double> fsub_weighted_sum;
  std::vector<double> nsub_mean_weighted_sum;
  std::vector<uint64_t> host_events;
  std::vector<uint64_t> subhalo_calls;
  std::vector<uint64_t> subhalo_clumps;

  void reset(int NM) {
    total_seconds = 0.0;
    subhalo_precompute_seconds = 0.0;
    M_host.assign(NM, 0.0);
    smooth_host_seconds.assign(NM, 0.0);
    subhalo_seconds.assign(NM, 0.0);
    fsub_weighted_sum.assign(NM, 0.0);
    nsub_mean_weighted_sum.assign(NM, 0.0);
    host_events.assign(NM, 0);
    subhalo_calls.assign(NM, 0);
    subhalo_clumps.assign(NM, 0);
  }
};

struct LensingConfig {
  int Nreal = 400000;
  int Nhalos = 100;
  int Nbins = 100;

  // existing toggles
  int fil = 1;
  int bias = 1;
  int ell = 1;
  int write = 0;
  bool strict_weak_lensing = false;

  // subhalo substructure (OFF by default -> existing pipeline unchanged)
  bool subhalo = false;
  double m_floor = 1.0e7;
  int subhalo_threads = 1;
  int subhalo_parallel_threshold = 200000;
  LensingProfile *profile = nullptr;
  int subhalo_model = 1;       // 0 = Option A (gslope removal, legacy), 1 = Option B (host reduced to (1-f_s)M, correct)
  bool subhalo_brute = false;  // true = brute-force resolve down to m_floor (no dynamic floor)
  double subhalo_factor = 1.0e-5; // cross-redshift plateau choice, scripts/subhalo_factor_redshift_check.py (2026-07-03)

  // future nuisance params (Phase 2)
  // double c_norm = 1.0;
  // double filament_density_norm = 1.0;
  // double filament_fraction = 1.0;
  double custom_kappathr = -1.0;
};


class lensing {
    
public:
    int Nreal; // realizations
    int Nhalos; // number of halos in each realization
    int Nbins; // P(lnmu) bins
    
    struct RealizationRaw {
        double kappa;
        double gamma1;
        double gamma2;
        double kappa_nosub = 0.0;   // same realization minus the subhalo clumps (paired baseline)
    };
    
    // probability distribution of lnmu, {lnmu, dP/dlnmu}
    vector<vector<double> > Plnmuf(cosmology &C, double zs, rgen &mt, int fil, int bias, int ell, int write);
    
    std::vector<RealizationRaw> sample_lnmu_raw(cosmology &C, double zs, rgen &mt, const LensingConfig &cfg);
    
    vector<double> sample_lnmu(cosmology &C, double zs, rgen &mt, const LensingConfig &cfg);
    const InvalidSampleStats& get_last_invalid_stats() const { return last_invalid_stats_; }

    // MCMC likelihood analysis of the Hubble diagram
    void Hubble_diagram_fit(cosmology &C, double DLthr, vector<vector<double> > &data, vector<double> &initial, vector<double> &steps , vector<vector<double> > &priors, int Ns, int Nburnin, int lens, int dm, rgen &mt, fs::path filename);
    
private:
    InvalidSampleStats last_invalid_stats_;
    Subhalo subhalo_;          // substructure tables, rebuilt per run when cfg.subhalo
    
    // loglikelihood of the Hubble digram data
    double loglikelihood(cosmology &C, double DLthr, vector<vector<double> > &data, vector<double> &par, int lens, int dm, rgen &mt);
    
};
