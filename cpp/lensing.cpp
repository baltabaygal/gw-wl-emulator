#include "cosmology.h"
#include "lensing.h"
// #include <functional>
#include <functional>  // For std::function
#include <algorithm>
#include <chrono>
#include <limits>

double Sigmacf(cosmology &C, double zs, double zl) {
    // angular diameter distances
    double DsA = C.DL(zs)/pow(1+zs,2.0);
    double DlA = C.DL(zl)/pow(1+zl,2.0);
    double DlsA = DsA - DlA*(1+zl)/(1+zs);
    
    // Sigma_c
    return 2.08871e16*DsA/(4.0*PI*DlA*DlsA);
}


/* ---------------------------------------------------------------------------------------------------------------------------------------------- */
/*                                                              NFW halos                                                                         */
/* ---------------------------------------------------------------------------------------------------------------------------------------------- */

// from astro-ph/0608153
array<double,2> FgNFW(double x) {
    if (x > 1) {
        double t = atan(sqrt((x-1)/(1+x))) / sqrt(x*x - 1);
        return {(1 - 2*t) / (x*x - 1), 2*t + log(x/2)};
    }
    if (x < 1) {
        double t = atanh(sqrt((1-x)/(1+x))) / sqrt(1 - x*x);
        return {(1 - 2*t) / (x*x - 1), 2*t + log(x/2)};
    }
    return {1.0/3.0, 1 + log(0.5)};
}

static inline double safeNFWGammaCore(double x, const array<double, 2> &Fg) {
    if (x < 1.0e-4) return 0.5;
    return 2.0 * Fg[1] / (x * x) - Fg[0];
}

double kappa0NFW(double rs, double rhos, double Sigmac) {
    return rs*rhos/Sigmac;
}

// fix ellipticity using the fit of astro-ph/0508497
double epsilonNFW(cosmology &C, double z, double M) {
    double s = 0.54*pow(M/exp(interpolate(z, C.logMcharlist)), -0.05);
    return max(0.0, (1.0-s)/(1.0+s));
}

// kappa and gamma for pseudo elliptical NFW halos
array<double,2> kappagammaNFWeps(double epsilon, double kappa0, double x, double phi) {
    double a1eps = 1.0-epsilon;
    double a2eps = 1.0+epsilon;
    double x1eps = sqrt(a1eps)*cos(phi)*x;
    double x2eps = sqrt(a2eps)*sin(phi)*x;
    double xeps = max(sqrt(pow(x1eps,2.0) + pow(x2eps,2.0)), 1.0e-12);
    double phieps = atan2(x2eps, x1eps);
    
    auto Fg = FgNFW(xeps);
    double kappaeps0 = 2.0*kappa0*Fg[0];
    double gammaeps0 = 2.0*kappa0*safeNFWGammaCore(xeps, Fg);

    double kappaeps = kappaeps0 + epsilon*cos(2.0*phieps)*gammaeps0;
    double gammaeps2 = pow(gammaeps0,2.0) + 2.0*epsilon*cos(2.0*phieps)*gammaeps0*kappaeps0 + pow(epsilon,2.0)*(pow(kappaeps0,2.0) - pow(cos(2.0*phieps)*gammaeps0,2.0));
    double gammaeps = sqrt(max(0.0, gammaeps2));
    
    return {kappaeps, gammaeps};
}
array<double,2> kappagammaNFW(cosmology &C, double zs, double zl, double r, double M, double phi, double epsilon) {
    double Sigmac = Sigmacf(C, zs, zl);
    
    // NFW scale radius and density
    vector<double> NFWparams = interpolate2(zl, M, C.zlist, C.Mlist, C.NFWlist);
    double rs = NFWparams[0];
    double rhos = NFWparams[1];
    
    double kappa0 = kappa0NFW(rs, rhos, Sigmac);
    
    return kappagammaNFWeps(epsilon, kappa0, r/rs, phi);
}

// maximal r so that kappa_NFW > kappa_thr
double rmaxfNFW(cosmology &C, double zs, double zl, double M, double kappathr) {
    double rmax;
    double logr1 = log(1.0e-6), logr2 = log(1.0e6);
    if (kappagammaNFW(C, zs, zl, exp(logr1), M, 0.0, 0.0)[0] > kappathr) {
        while (logr2-logr1 > 0.02) {
            rmax = exp((logr2+logr1)/2.0);
            if (kappagammaNFW(C, zs, zl, rmax, M, 0.0, 0.0)[0] > kappathr) {
                logr1 = log(rmax);
            } else {
                logr2 = log(rmax);
            }
        }
        rmax = exp((logr2+logr1)/2.0);
    } else {
        rmax = 0.0;
    }
    return rmax;
}

// number of halos with kappa_NFW > kappa_thr in each z and M bin
vector<vector<vector<double> > > deltaNhfNFW(cosmology &C, double zs, double kappathr) {
    vector<vector<vector<double> > > dNh(C.Nz, vector<vector<double> > (C.NM, vector<double> (3, 0.0)));
    double zl, dz, M, Mb, dlnM, dndlnM, rmax, sigma, sigmab;
    for (int jz = 1; jz < C.Nz; jz++) {
        zl = C.zlist[jz];
        dz = zl - C.zlist[jz-1];
        if (zl < zs) {
            for (int jM = 1; jM < C.NM; jM++) {
                M = C.Mlist[jM];
                sigma = C.sigmalist[jM][1];
                
                dlnM = log(M) - log(C.Mlist[jM-1]);
                dndlnM = C.HMFlist[jz][jM][0];
                rmax = rmaxfNFW(C, zs, zl, M, kappathr);
                dNh[jz][jM][0] = 306.535*PI*pow((1.0+zl)*rmax,2.0)/C.Hz(zl)*dndlnM*dlnM*dz;
                
                Mb = 2.0*PI*pow(rmax,2.0)*(C.dc(zl)-C.dc(zl-dz))*C.rhoM0;
                sigmab = interpolate(Mb, C.sigmalist);
                
                // see Baumann (5.129)
                dNh[jz][jM][1] = C.Dg(zl)*sigmab*C.halobias(zl, sigma);
                
                dNh[jz][jM][2] = rmax;
            }
        }
    }
    return dNh;
}

// total number of halos with kappa_NFW > kappa_thr
double NhfNFW(cosmology &C, double zs, double kappathr) {
    double Nh = 0.0;
    double zl, dz, M, dlnM, dndlnM, rmax;
    for (int jz = 1; jz < C.Nz; jz++) {
        zl = C.zlist[jz];
        dz = zl - C.zlist[jz-1];
        if (zl < zs) {
            for (int jM = 1; jM < C.NM; jM++) {
                M = C.Mlist[jM];
                dlnM = log(M) - log(C.Mlist[jM-1]);
                dndlnM = C.HMFlist[jz][jM][0];
                rmax = rmaxfNFW(C, zs, zl, M, kappathr);
                Nh += 306.535*PI*pow((1.0+zl)*rmax,2.0)/C.Hz(zl)*dndlnM*dlnM*dz;
            }
        }
    }
    return Nh;
}

// variance of kappa from weak lenses
double sigmakappaW(cosmology &C, double zs, double kappathr) {
    double Nh = 0.0, kappa1 = 0.0, kappa2 = 0.0;
    double zl, dz, M, dlnM, dndlnM, r, kappar;
    int Nr = 100;
    double dlnr = 0.01;
    double Edlnr = exp(dlnr);
    for (int jz = 1; jz < C.Nz; jz++) {
        zl = C.zlist[jz];
        dz = zl - C.zlist[jz-1];
        if (zl < zs) {
            for (int jM = 1; jM < C.NM; jM++) {
                M = C.Mlist[jM];
                dlnM = log(M) - log(C.Mlist[jM-1]);
                dndlnM = C.HMFlist[jz][jM][0];
                                
                r = rmaxfNFW(C, zs, zl, M, kappathr);
                if (r == 0.0) {
                    r = 1.0e-6;
                }

                double SigmacW = Sigmacf(C, zs, zl);
                vector<double> NFWpW = interpolate2(zl, M, C.zlist, C.Mlist, C.NFWlist);
                double rsW = NFWpW[0];
                double kappa0W = kappa0NFW(rsW, NFWpW[1], SigmacW);

                kappar = kappathr;
                while (kappar > 0.001*kappathr) {
                    kappar = kappagammaNFWeps(0.0, kappa0W, r/rsW, 0.0)[0];
                    Nh += 306.535*PI*pow((1.0+zl)*r,2.0)/C.Hz(zl)*dndlnM*dlnr*dlnM*dz;
                    kappa1 += 306.535*PI*pow((1.0+zl)*r,2.0)/C.Hz(zl)*dndlnM*kappar*dlnr*dlnM*dz;
                    kappa2 += 306.535*PI*pow((1.0+zl)*r,2.0)/C.Hz(zl)*dndlnM*pow(kappar,2.0)*dlnr*dlnM*dz;
                    r = r*Edlnr;
                }
            }
        }
    }
    return sqrt(kappa2 - pow(kappa1,2.0)/Nh);
}


/* ---------------------------------------------------------------------------------------------------------------------------------------------- */
/*                                                           Cylindrical filaments                                                                */
/* ---------------------------------------------------------------------------------------------------------------------------------------------- */

// axis in the lens plane
double kappa0CYL(double rs, double rhos, double Sigmac) {
    return rs*rhos/Sigmac;
}

// uniform density inside the cylinder
double kappaCYL1(double r, double rs, double kappa0) {
    if (r > rs) {
        return 0.0;
    }
    return kappa0*2.0*sqrt(1.0-pow(r/rs,2.0));
}
double gammaCYL1(double r, double rs, double kappa0) {
    return kappaCYL1(r, rs, kappa0);
}

// 1/(1+(r/r_s)^2) profile inside the cylinder
double kappaCYL2(double r, double rs, double kappa0) {
    if (r > rs) {
        return 0.0;
    }
    return kappa0*2.0*PI/sqrt(1.0+pow(r/rs,2.0));
}
double gammaCYL2(double r, double rs, double kappa0) {
    return kappa0*4.0*PI*(sqrt(1.0+pow(r/rs,2.0)) - 1.0)*pow(rs/r,2.0) - kappaCYL2(r, rs, kappa0);
}

array<double,2> kappagammaCYL(cosmology &C, double zs, double zl, double r, double M, double phi) {
    double Sigmac = Sigmacf(C, zs, zl);
    
    // cylinder radius and length, M = mass inside radius r_s (see astro-ph/0406665)
    double rs = 1000.0*pow(M/1.0e14, 1.0/3.0);
    double L = 20000.0*pow(M/1.0e14, 1.0/3.0);
    
    // average density inside radius r_s = 10*rho_c, rough approximation of rho_s density when it is not in the lens plane
    double rhos = 14.4*C.rhoc*L/max(2.0*rs, abs(cos(phi))*L);
    
    double kappa0 = kappa0CYL(rs, rhos, Sigmac);
    
    return {kappaCYL2(r, rs, kappa0), gammaCYL2(r, rs, kappa0)};
}

// maximal r so that kappa_CYL > kappa_thr
double rmaxfCYL(cosmology &C, double zs, double zl, double M, double kappathr) {
    double rmax;
    double logr1 = log(1.0e-6), logr2 = log(1.0e6);
    if (kappagammaCYL(C, zs, zl, exp(logr1), M, 0.0)[0] > kappathr) {
        while (logr2-logr1 > 0.02) {
            rmax = exp((logr2+logr1)/2.0);
            if (kappagammaCYL(C, zs, zl, rmax, M, 0.0)[0] > kappathr) {
                logr1 = log(rmax);
            } else {
                logr2 = log(rmax);
            }
        }
        rmax = exp((logr2+logr1)/2.0);
    } else {
        rmax = 0.0;
    }
    return rmax;
}

// number of halos with kappa_NFW > kappa_thr in each z and M bin
vector<vector<vector<double> > > deltaNhfCYL(cosmology &C, double zs, double kappathr) {
    vector<vector<vector<double> > > dNh(C.Nz, vector<vector<double> > (C.NM, vector<double> (3, 0.0)));
    double zl, dz, M, Mb, dlnM, dndlnM, rmax, sigma, sigmab;
    for (int jz = 1; jz < C.Nz; jz++) {
        zl = C.zlist[jz];
        dz = zl - C.zlist[jz-1];
        if (zl < zs) {
            for (int jM = 1; jM < C.NM; jM++) {
                M = C.Mlist[jM];
                sigma = C.sigmalist[jM][1];
                
                dlnM = log(M) - log(C.Mlist[jM-1]);
                dndlnM = C.FMFlist[jz][jM];
                rmax = rmaxfCYL(C, zs, zl, M, kappathr);
                dNh[jz][jM][0] = 306.535*PI*pow((1.0+zl)*rmax,2.0)/C.Hz(zl)*dndlnM*dlnM*dz;
                
                Mb = 2.0*PI*pow(rmax,2.0)*(C.dc(zl)-C.dc(zl-dz))*C.rhoM0;
                sigmab = interpolate(Mb, C.sigmalist);
                
                // see Baumann (5.129)
                dNh[jz][jM][1] = C.Dg(zl)*sigmab*C.halobias(zl, sigma);
                
                dNh[jz][jM][2] = rmax;
            }
        }
    }
    return dNh;
}

// number of filaments with kappa_CYL > kappa_thr
double NhfCYL(cosmology &C, double zs, double kappathr) {
    double Nh = 0.0;
    double zl, dz, M, dlnM, dndlnM, rmax;
    for (int jz = 1; jz < C.Nz; jz++) {
        zl = C.zlist[jz];
        dz = zl - C.zlist[jz-1];
        if (zl < zs) {
            for (int jM = 1; jM < C.NM; jM++) {
                M = C.Mlist[jM];
                dlnM = log(M) - log(C.Mlist[jM-1]);
                dndlnM = C.FMFlist[jz][jM];
                rmax = rmaxfCYL(C, zs, zl, M, kappathr);
                Nh += 306.535*PI*pow((1.0+zl)*rmax,2.0)/C.Hz(zl)*dndlnM*dlnM*dz;
            }
        }
    }
    return Nh;
}


/* ---------------------------------------------------------------------------------------------------------------------------------------------- */
/*                                                    PDF of amplifications                                                                       */
/* ---------------------------------------------------------------------------------------------------------------------------------------------- */

// find threshold kappa
double findkappathr(int N, function<double(double)> Nf) {
    double kappa1 = 1.0e-12, kappa2 = 1.0;
    double kappathr = pow(10.0, (log10(kappa1) + log10(kappa2))/2.0);
    while (log10(kappa2) - log10(kappa1) > 0.01) {
        if (Nf(kappathr) > N) {
            kappa1 = kappathr;
        } else {
            kappa2 = kappathr;
        }
        kappathr = pow(10.0, (log10(kappa1) + log10(kappa2))/2.0);
    }
    return kappathr;
}

// sample lnmu from the PDF of amplifications

vector<lensing::RealizationRaw> lensing::sample_lnmu_raw(cosmology &C, double zs, rgen &mt, const LensingConfig &cfg) {
    using Clock = std::chrono::steady_clock;
    auto elapsed_seconds = [](Clock::time_point start) {
        return std::chrono::duration<double>(Clock::now() - start).count();
    };
    auto total_start = Clock::now();
    LensingProfile *profile = cfg.profile;
    if (profile != nullptr) {
        profile->reset(C.NM);
        profile->Nreal = cfg.Nreal;
        profile->zs = zs;
        for (int jM = 0; jM < C.NM; jM++) {
            profile->M_host[jM] = C.Mlist[jM];
        }
    }
    
    // fix threshold kappa
    function<double(double)> NfNFW = [&C, zs](double kappa) {
        return NhfNFW(C, zs, kappa);
    };
    double kappathrH = (cfg.custom_kappathr > 0.0) ? cfg.custom_kappathr : findkappathr(cfg.Nhalos, NfNFW);
    
    if (cfg.write > 0) {
        cout << kappathrH << endl;
    }
        
    // distribution of kappa_NFW < kappa_thr
    double skappaW = sigmakappaW(C, zs, kappathrH);
    normal_distribution<double> PkappaW(0.0, skappaW);
    if (skappaW < 0.0) {
        cout << "Error: negative standard deviation." << endl;
    }
    
    vector<RealizationRaw> raw(cfg.Nreal);
    for (int j = 0; j < cfg.Nreal; j++) {
        raw[j].kappa = PkappaW(mt);
        raw[j].gamma1 = 0.0;
        raw[j].gamma2 = 0.0;
        raw[j].kappa_nosub = raw[j].kappa;   // paired baseline starts from the weak part
    }
    
    vector<vector<vector<double> > > dNH = deltaNhfNFW(C, zs, kappathrH);
    vector<vector<vector<double> > > dNF = deltaNhfCYL(C, zs, kappathrH);

    // subhalo substructure tables (rebuilt per run; zs-independent physics)
    if (cfg.subhalo) {
        auto precompute_start = Clock::now();
        subhalo_.m_floor = cfg.m_floor;
        subhalo_.precompute(C, zs, cfg.subhalo_factor * kappathrH);
        if (profile != nullptr) {
            profile->subhalo_precompute_seconds = elapsed_seconds(precompute_start);
        }
    }
    if (cfg.write > 0) {
        writeToFile(C.zlist, C.Mlist, dNH, C.outdir/"dNH.dat");
        writeToFile(C.zlist, C.Mlist, dNF, C.outdir/"dNF.dat");
    }
    
    array<double,2> kappagamma;
    normal_distribution<double> pG(0.0, 1.0);
    poisson_distribution<int> PN;
    double zl, M, rmaxH, rmaxF, r, phi, phiH, phiF, epsilon = 0.0, barNH, barNF, sigma, deltab, lambda;
    int NH, NF;
    int NtotH = 0, NtotF = 0;;
    for (int jz = 0; jz < C.Nz; jz++) {
        zl = C.zlist[jz];
        if (zl < zs) {
            for (int jM = 0; jM < C.NM; jM++) {
                M = C.Mlist[jM];
                
                // generate realizations
                barNH = dNH[jz][jM][0];
                sigma = dNH[jz][jM][1];
                rmaxH = dNH[jz][jM][2];
                
                barNF = dNF[jz][jM][0];
                rmaxF = dNF[jz][jM][2];
                
                double Sigmac = (barNH > 0.0 || barNF > 0.0) ? Sigmacf(C, zs, zl) : 0.0;
                double rsH = 1.0, kappa0H = 0.0;
                if (barNH > 0.0) {
                    vector<double> NFWp = interpolate2(zl, M, C.zlist, C.Mlist, C.NFWlist);
                    rsH = NFWp[0];
                    kappa0H = kappa0NFW(rsH, NFWp[1], Sigmac);
                    if (cfg.ell > 0) epsilon = epsilonNFW(C, zl, M);
                }
                double rsF = 0.0, LF = 0.0, kappa0baseF = 0.0;
                if (cfg.fil > 0 && barNF > 0.0) {
                    rsF = 1000.0*pow(M/1.0e14, 1.0/3.0);
                    LF = 20000.0*pow(M/1.0e14, 1.0/3.0);
                    kappa0baseF = rsF * 14.4*C.rhoc*LF / Sigmac;
                }

                auto add_host = [&](int j) {
                    auto smooth_start = Clock::now();
                    double rsH_eff = rsH;
                    double kappa0H_eff = kappa0H;
                    double epsilon_eff = epsilon;

                    if (cfg.subhalo && cfg.subhalo_model == 1) {
                        double g = subhalo_.gnorm[jz][jM];
                        double M_host_eff = M;
                        if (g > 0.0) {
                            double psi_lo = 0.0;
                            if (cfg.subhalo_brute) {
                                psi_lo = cfg.m_floor / M;
                            } else {
                                // dynamic floor keyed to host-center distance r; MUST match Subhalo::addClumps
                                // (r-vs-d bias is absorbed by tuning subhalo_factor to convergence)
                                const auto &rth = subhalo_.r_thr[jz];
                                int jlo = static_cast<int>(std::lower_bound(rth.begin(), rth.end(), r) - rth.begin());
                                if (jlo < C.NM) {
                                    psi_lo = std::max(C.Mlist[jlo], C.Mmin) / M;
                                } else {
                                    psi_lo = subhalo_.psi_max;
                                }
                            }
                            if (psi_lo < subhalo_.psi_max) {
                                double alpha = subhalo_.alpha;
                                double f_s_res = g * (pow(subhalo_.psi_max, 1.0 + alpha) - pow(psi_lo, 1.0 + alpha)) / (1.0 + alpha);
                                f_s_res = std::max(0.0, std::min(0.95, f_s_res));
                                M_host_eff = (1.0 - f_s_res) * M;
                            }
                        }
                        vector<double> NFWp = interpolate2(zl, M_host_eff, C.zlist, C.Mlist, C.NFWlist);
                        rsH_eff = NFWp[0];
                        kappa0H_eff = kappa0NFW(rsH_eff, NFWp[1], Sigmac);
                        if (cfg.ell > 0) epsilon_eff = epsilonNFW(C, zl, M_host_eff);
                    }

                    kappagamma = kappagammaNFWeps(epsilon_eff, kappa0H_eff, r/rsH_eff, phiH);
                    raw[j].kappa += kappagamma[0];
                    // single-angle gamma projection, matching the original Vaskonen code (kept by
                    // decision 2026-07-02; the spin-2 form would be -gamma_t (cos 2phi, sin 2phi) —
                    // see tmp/shear_convention_check.py; difference is ~0.5% on <gamma^2>)
                    raw[j].gamma1 += cos(phi)*kappagamma[1];
                    raw[j].gamma2 += sin(phi)*kappagamma[1];

                    // kappa_nosub tracks the full unperturbed host (NFW at M_total)
                    if (cfg.subhalo && cfg.subhalo_model == 1) {
                        auto kg_full = kappagammaNFWeps(epsilon, kappa0H, r/rsH, phiH);
                        raw[j].kappa_nosub += kg_full[0];
                    } else {
                        raw[j].kappa_nosub += kappagamma[0];
                    }

                    if (profile != nullptr) {
                        profile->smooth_host_seconds[jM] += elapsed_seconds(smooth_start);
                        profile->host_events[jM]++;
                        if (cfg.subhalo && subhalo_.fsub[jz][jM] > 0.0) {
                            profile->fsub_weighted_sum[jM] += subhalo_.fsub[jz][jM];
                            profile->nsub_mean_weighted_sum[jM] += subhalo_.Nsub[jz][jM];
                        }
                    }

                    if (cfg.subhalo) {
                        auto subhalo_start = Clock::now();
                        int Nc = subhalo_.addClumps(C, jz, jM, zl, M, Sigmac, r, phi, mt,
                                                     raw[j].kappa, raw[j].gamma1, raw[j].gamma2,
                                                     cfg.subhalo_model, cfg.subhalo_brute,
                                                     cfg.subhalo_threads, cfg.subhalo_parallel_threshold);
                        if (profile != nullptr) {
                            profile->subhalo_seconds[jM] += elapsed_seconds(subhalo_start);
                            profile->subhalo_calls[jM]++;
                            profile->subhalo_clumps[jM] += static_cast<uint64_t>(Nc);
                        }
                    }

                    NtotH++;
                };

                for (int j = 0; j < cfg.Nreal; j++) {
                    
                    // bias
                    deltab = sigma*pG(mt);
                    lambda = exp(deltab - pow(sigma,2.0)/2.0); // log-normal
                    
                    if (cfg.bias == 0) {
                        lambda = 1.0;
                    }
                    
                    // generate halos
                    if (lambda*barNH < 0.2) { // if lambda is small, compare to a random number U(0,1) (faster)
                        if (lambda*barNH > randomreal(0.0, 1.0, mt)) {
                            r = sqrt(randomreal(0.0,1.0,mt))*rmaxH; // distance from the line-of-sight
                            phi = randomreal(0.0,2*PI,mt); // polar angle of r vector
                            phiH = randomreal(0.0,2*PI,mt); // orientation of the halo ellipticity
                            add_host(j);
                        }
                    } else { // for larger lambda, generate number of halos from Poisson distribution (slower)
                        PN = poisson_distribution<int>(lambda*barNH);
                        NH = PN(mt);
                        if (NH > 0) {
                            for (int jH = 0; jH < NH; jH++) {
                                r = sqrt(randomreal(0.0,1.0,mt))*rmaxH; // distance from the line-of-sight
                                phi = randomreal(0.0,2*PI,mt); // polar angle of r vector
                                phiH = randomreal(0.0,2*PI,mt); // orientation of the halo ellipticity
                                add_host(j);
                            }
                        }
                    }
                    
                    if (cfg.fil > 0) {
                        // generate filaments
                        if (lambda*barNF < 0.2) { // if lambda is small, compare to a random number U(0,1) (faster)
                            if (lambda*barNF > randomreal(0.0, 1.0, mt)) {
                                r = sqrt(randomreal(0.0,1.0,mt))*rmaxF; // distance from the line-of-sight
                                phi = randomreal(0.0,2*PI,mt); // polar angle of r vector
                                phiF = randomreal(0.0,2*PI,mt); // orientation of the filament
                                double kappa0F = kappa0baseF / max(2.0*rsF, abs(cos(phiF))*LF);
                                kappagamma = {kappaCYL2(r, rsF, kappa0F), gammaCYL2(r, rsF, kappa0F)};
                                
                                raw[j].kappa += kappagamma[0];
                                raw[j].kappa_nosub += kappagamma[0];
                                raw[j].gamma1 += cos(phi)*kappagamma[1];
                                raw[j].gamma2 += sin(phi)*kappagamma[1];

                                NtotF++;
                            }
                        } else { // for larger lambda, generate number of halos from Poisson distribution (slower)
                            PN = poisson_distribution<int>(lambda*barNF);
                            NF = PN(mt);
                            if (NF > 0) {
                                for (int jF = 0; jF < NF; jF++) {
                                    r = sqrt(randomreal(0.0,1.0,mt))*rmaxF; // distance from the line-of-sight
                                    phi = randomreal(0.0,2*PI,mt); // polar angle of r vector
                                    phiF = randomreal(0.0,2*PI,mt); // orientation of the filament
                                    double kappa0F = kappa0baseF / max(2.0*rsF, abs(cos(phiF))*LF);
                                    kappagamma = {kappaCYL2(r, rsF, kappa0F), gammaCYL2(r, rsF, kappa0F)};
                                    
                                    raw[j].kappa += kappagamma[0];
                                    raw[j].kappa_nosub += kappagamma[0];
                                    raw[j].gamma1 += cos(phi)*kappagamma[1];
                                    raw[j].gamma2 += sin(phi)*kappagamma[1];

                                    NtotF++;
                                }
                            }
                        }
                    }
                    
                }
            }
        }
    }
    if (profile != nullptr) {
        profile->total_seconds = elapsed_seconds(total_start);
    }
    return raw;
}


vector<double> lensing::sample_lnmu(cosmology &C, double zs, rgen &mt, const LensingConfig &cfg) {
    
    auto raw = sample_lnmu_raw(C, zs, mt, cfg);
    
    double meankappa = 0.0;
    for (auto &r : raw)
        meankappa += r.kappa;
    meankappa /= raw.size();
    
    vector<double> lnmulist;
    lnmulist.reserve(raw.size());

    InvalidSampleStats stats;
    stats.total_samples = raw.size();
    double detA_sum = 0.0;
    bool detA_seen = false;
    constexpr double huge_threshold = 1.0e12;
    constexpr double detA_near_zero_threshold = 1.0e-6;

    double kappaj, gammaj, muj, detA, logmu;
    for (auto &r : raw) {
        kappaj = r.kappa - meankappa;
        gammaj = sqrt(r.gamma1*r.gamma1 + r.gamma2*r.gamma2);
        detA = (1.0 - kappaj) * (1.0 - kappaj) - (gammaj * gammaj);

        if (!detA_seen) {
            stats.detA_min = detA;
            stats.detA_max = detA;
            detA_seen = true;
        } else {
            stats.detA_min = std::min(stats.detA_min, detA);
            stats.detA_max = std::max(stats.detA_max, detA);
        }
        detA_sum += detA;
        if (std::abs(detA) < detA_near_zero_threshold) {
            stats.detA_near_zero_count++;
        }

        bool invalid = false;
        if (std::isnan(kappaj)) {
            stats.nan_kappa++;
            invalid = true;
        }
        if (std::isnan(gammaj)) {
            stats.nan_gamma++;
            invalid = true;
        }
        if (detA <= 0.0) {
            stats.negative_detA++;
            invalid = true;
            if (cfg.strict_weak_lensing) {
                stats.strict_weak_lensing_rejects++;
            }
        }

        if (detA == 0.0) {
            muj = std::numeric_limits<double>::infinity();
        } else {
            muj = 1.0 / detA;
        }

        if (!std::isfinite(muj)) {
            stats.nonfinite_mu++;
            invalid = true;
        }
        if (muj <= 0.0) {
            stats.negative_mu++;
            invalid = true;
        }
        if (std::isfinite(muj) && std::abs(muj) > huge_threshold) {
            stats.overflow_mu++;
            invalid = true;
        }

        if (!invalid) {
            logmu = log(muj);
            if (!std::isfinite(logmu)) {
                stats.invalid_logmu++;
                invalid = true;
            } else {
                lnmulist.push_back(logmu);
                stats.valid_samples++;
            }
        } else {
            // Keep invalid_logmu classification explicit when log(mu) is undefined.
            if (muj <= 0.0 || !std::isfinite(muj)) {
                stats.invalid_logmu++;
            }
        }
    }

    stats.invalid_samples = stats.total_samples - stats.valid_samples;
    stats.detA_mean = (stats.total_samples > 0) ? (detA_sum / static_cast<double>(stats.total_samples)) : 0.0;
    if (!detA_seen) {
        stats.detA_min = 0.0;
        stats.detA_max = 0.0;
    }
    last_invalid_stats_ = stats;
    
    return lnmulist;
}


// probability distribution of lnmu, {lnmu, dP/dlnmu}
vector<vector<double> > lensing::Plnmuf(cosmology &C, double zs, rgen &mt,
                                       int fil, int bias, int ell, int write) {

    LensingConfig cfg;
    cfg.Nreal = this->Nreal;
    cfg.Nhalos = this->Nhalos;
    cfg.Nbins = this->Nbins;
    cfg.fil = fil;
    cfg.bias = bias;
    cfg.ell = ell;
    cfg.write = write;

    vector<double> lnmulist = sample_lnmu(C, zs, mt, cfg);
    vector<vector<double> > Plnmu = binSample(lnmulist, cfg.Nbins);
    
    int jmin = 0, jmax = Plnmu.size()-1;
    for (int j = 0; j < Plnmu.size()-4; j++) {
        if (Plnmu[j][1] > 0 && Plnmu[j+1][1] > 0 && Plnmu[j+2][1] > 0 && Plnmu[j+3][1] > 0 && Plnmu[j+4][1] > 0) {
            jmin = j;
            j = Plnmu.size();
        }
    }
    for (int j = max(4,jmin); j < Plnmu.size(); j++) {
        if (Plnmu[j-4][1] == 0 && Plnmu[j-3][1] == 0 && Plnmu[j-2][1] == 0 && Plnmu[j-1][1] == 0 && Plnmu[j][1] == 0) {
            jmax = j;
            j = Plnmu.size();
        }
    }
    //cout << jmin << "   " << jmax << endl;
    Plnmu.erase(Plnmu.begin() + jmax, Plnmu.end());
    Plnmu.erase(Plnmu.begin(), Plnmu.begin() + jmin);
    
    // convert from image plane to source plane (P_S ~ P_I/mu) and normalize
    double dlnmu = Plnmu[1][0] - Plnmu[0][0];
    double norm = 0.0;
    for (int j = 0; j < Plnmu.size(); j++) {
        Plnmu[j][1] *= exp(-Plnmu[j][0]);
        norm += Plnmu[j][1]*dlnmu;
    }
    for (int j = 0; j < Plnmu.size(); j++) {
        Plnmu[j][1] *= 1.0/norm;
    }
    return Plnmu;
}

// loglikelihood of the Hubble digram data
double lensing::loglikelihood(cosmology &C, double DLthr, vector<vector<double> > &data, vector<double> &par, int lens, int dm, rgen &mt) {
    
    // initialize cosmology
    C.OmegaM = par[0];
    C.sigma8 = par[1];
    C.h = par[2];
    C.initialize(dm, pow(10.0, par[3]));
    
    double z, DL0, DL, sigmaDL, Y, dY, Pdet;
    
    // compute loglikelihood
    double logL = 0.0;
    if (lens == 0) { // model without lensing
        for (int j = 0; j < data.size(); j++) {
            z = data[j][0];
            DL0 = C.DL(z);
            DL = data[j][1];
            sigmaDL = data[j][2];
            
            // compute P_det
            dY = sigmaDL/10.0;
            Y = min(DL0, DLthr) - 3.0*sigmaDL;
            Pdet = 0.0;
            while (Y <= min(DLthr, DL0 + 3.0*sigmaDL)) {
                Pdet += dY*NPDF(Y, DL0, sigmaDL);
                Y += dY;
            }
            logL += logNPDF(DL, DL0, sigmaDL) - log(Pdet);
        }
    }
    else { // model with lensing
        
        // compute the lensing distributions
        vector<vector<vector<double> > > Plnmuz(C.Zlist.size());
        for (int jz = 0; jz < C.Zlist.size(); jz++) {
            z = C.Zlist[jz];
            Plnmuz[jz] = Plnmuf(C, z, mt, 1, 1, 1, 0);
        }
        
        int jz;
        vector<vector<double> > Plnmu;
        double L, dlnmu;
        for (int j = 0; j < data.size(); j++) {
            z = data[j][0];
            
            DL = data[j][1];
            sigmaDL = data[j][2];
            
            // find the closest z at which P(lnmu) is computed
            jz = lower_bound(C.Zlist.begin(), C.Zlist.end(), z) - C.Zlist.begin();
            if ((jz > 0 && C.Zlist[jz]-z > z-C.Zlist[jz-1]) || jz >= C.Zlist.size()) {
                jz--;
            }
            Plnmu = Plnmuz[jz];
            dlnmu = Plnmu[1][0] - Plnmu[0][0];
            
            // integrate over lnmu
            dY = sigmaDL/10.0;
            Pdet = 0.0, L = 0.0;
            for (int i = 0; i < Plnmu.size(); i++) {
                DL0 = C.DL(z)/exp(Plnmu[i][0]/2.0);
                
                L += dlnmu*Plnmu[i][1]*NPDF(DL, DL0, sigmaDL);
                
                // integrate over Y
                Y = min(DL0, DLthr) - 3.0*sigmaDL;
                while (Y <= min(DLthr, DL0 + 3.0*sigmaDL)) {
                    Pdet += dlnmu*Plnmu[i][1]*dY*NPDF(Y, DL0, sigmaDL);
                    Y += dY;
                }
            }
            
            logL += log(L/Pdet);
        }
    }
    
    return logL;
}

// MCMC inference of the Hubble diagram data
void lensing::Hubble_diagram_fit(cosmology &C, double DLthr, vector<vector<double> > &data, vector<double> &initial, vector<double> &steps , vector<vector<double> > &priors, int N, int Nburnin, int lens, int dm, rgen &mt, fs::path filename) {
    
    // loglikelihood
    function<double(vector<double>&)> logpdf = [this, &C, DLthr, &data, lens, dm, &mt](vector<double> &par) {
        return loglikelihood(C, DLthr, data, par, lens, dm, mt);
    };
    
    // no cut
    function<double(vector<double>&)> cut = [](vector<double> &par) {
        return 1.0;
    };
    
    MCMC_sampling(N, Nburnin, logpdf, initial, steps, priors, cut, mt, 1, 0, filename);
}
