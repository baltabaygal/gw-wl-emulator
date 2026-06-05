
#pragma once
#include <functional>
#include <vector>
#include <cstdint>

struct LensingConfig {
  int Nreal = 400000;
  int Nhalos = 100;
  int Nbins = 100;

  // existing toggles
  int fil = 1;
  int bias = 1;
  int ell = 1;
  int exact_poisson = 0;
  int write = 0;

  // future nuisance params (Phase 2)
  // double c_norm = 1.0;
  // double filament_density_norm = 1.0;
  // double filament_fraction = 1.0;
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
    };

    struct EventRecord {
        int realization;
        int is_filament;
        double z;
        double M;
        double b;
        double kappa;
        double gamma1;
        double gamma2;
    };
    
    // probability distribution of lnmu, {lnmu, dP/dlnmu}
    vector<vector<double> > Plnmuf(cosmology &C, double zs, rgen &mt, int fil, int bias, int ell, int write);
    
    std::vector<RealizationRaw> sample_lnmu_raw(cosmology &C, double zs, rgen &mt, const LensingConfig &cfg);
    std::vector<EventRecord> sample_lensing_events_raw(cosmology &C, double zs, rgen &mt, const LensingConfig &cfg);
    
    vector<double> sample_lnmu(cosmology &C, double zs, rgen &mt, const LensingConfig &cfg);

    // MCMC likelihood analysis of the Hubble diagram
    void Hubble_diagram_fit(cosmology &C, double DLthr, vector<vector<double> > &data, vector<double> &initial, vector<double> &steps , vector<vector<double> > &priors, int Ns, int Nburnin, int lens, int dm, rgen &mt, fs::path filename);
    
private:
    
    // loglikelihood of the Hubble digram data
    double loglikelihood(cosmology &C, double DLthr, vector<vector<double> > &data, vector<double> &par, int lens, int dm, rgen &mt);
    
};

double NhfNFW(cosmology &C, double zs, double kappathr);
double sigmakappaW(cosmology &C, double zs, double kappathr);
std::vector<std::vector<std::vector<double> > > deltaNhfNFW(cosmology &C, double zs, double kappathr);
std::vector<double> kappagammaNFW(cosmology &C, double zs, double zl, double r, double M, double phi, double epsilon);
double findkappathr(int N, function<double(double)> Nf);
