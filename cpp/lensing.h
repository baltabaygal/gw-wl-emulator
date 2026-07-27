
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
// eps_floor = convergence floor fraction: the outward radial integration stops when a
// halo's kappa drops below eps_floor*kappathr (sets the faintest halo / largest impact
// parameter included). Default 0.001 reproduces the original result; K2 is converged
// w.r.t. it (see playground/k2_vs_floor.cpp).
double sigmakappaW(cosmology &C, double zs, double kappathr, double eps_floor = 0.001);

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
  // carve guard: # encounters where M - Sum m_i - M_u fell below Mmin and the host mass
  // was floored (mass-conserving carve, scheme A). Expected ~0 (needs a ~9-sigma SHMF
  // excursion); a nonzero value flags a resolution/floor pathology.
  uint64_t subhalo_carve_negatives = 0;

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
    subhalo_carve_negatives = 0;
  }
};

struct LensingConfig {
  int Nreal = 400000;
  int Nhalos = 100;   // used only when kappathr_flat <= 0 (legacy <N>=Nhalos rule)
  int Nbins = 100;

  // Explicit-halo threshold rule. Default (<= 0) is the legacy <N>=Nhalos rule:
  // kappa_thr is chosen so exactly Nhalos are explicit at every z_s (this inflates
  // kappa_thr with z_s: 1.28e-4 at z_s=1 -> 1.37e-3 at z_s=10). Set > 0 to use a
  // fixed (z_s-independent) flat threshold instead, which lets <N> grow with path
  // length and keeps the explicit/Gaussian split at a fixed physical kappa scale.
  // Decision history: flat 1e-3 was the default 2026-07-09; reverted to fixed-<N>
  // on 2026-07-10 for predictability + continuity with Vaskonen's convention (the
  // flat rule stays converged uniformly in z_s but fixed-<N> gives a bounded,
  // predictable per-LOS cost; the high-z accuracy cost is small, JSD ~1.2e-3 at
  // z_s=10 vs ~3.4e-4 flat). custom_kappathr (sweep override) takes precedence.
  double kappathr_flat = -1.0;

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
  int subhalo_model = 3;       // 0 = gslope removal (legacy), 1 = reduced-host resolved-only,
                               // 2 = diagnostic bare clumps, 3 = reduced-host + Wsub term
                               // (mu_unres(y) + Gaussian; exact mean/variance at any factor;
                               // DEFAULT since 2026-07-09, see docs/subhalo/wsub_gaussian_term_derivation.md),
                               // 4 = brute-to-floor + carve: every subhalo sampled down to the
                               //     absolute floor psi_min = m_floor/M, host carved to M - Sum_i m_i,
                               //     NO unresolved term (no M_u/kappa_u/Wsub, no dynamic floor, no
                               //     subhalo_factor). Supervisor's simplified production model
                               //     (2026-07-23); == model 1 + subhalo_brute + subhalo_carve as one
                               //     named model. Requires subhalo_carve = true (throws otherwise);
                               //     ignores subhalo_brute/subhalo_factor. Marginal cost ~45 ms/ray
                               //     vs ~0.2 for model 3 (~1e6 clumps rendered per ray at z_s=1);
                               //     floor 1e7/M validated converged. STAGED: not yet the shipped
                               //     default (pending supervisor sign-off + emulator retrain).
                               //     docs/subhalo/mass_conserving_carve_note.md §10.
                               // 5 = model 4 + per-clump convergence threshold kappa_thr,sub:
                               //     SAME population as model 4 (every subhalo still exists down to
                               //     psi_min = m_floor/M, no unresolved/Gaussian stand-in), but only
                               //     clumps whose kappa AT THE RAY clears subhalo_kappathr are
                               //     rendered, drawn from the RESTRICTED Poisson intensity so the
                               //     rejects are never instantiated. Same resolution rule the HOST
                               //     halos already obey (r_thr / kappa_thr), applied to subhalos.
                               //     Requires subhalo_carve = true. ~1e4x fewer clumps for a 0.1%
                               //     loss in sigma_kappa at the default threshold, which puts it at
                               //     the subhalo-off cost floor. STAGED, needs a P(lnmu) JSD gate.
                               //     data/results/subkappathr_population/report.md
  bool subhalo_brute = false;  // true = brute-force resolve down to m_floor (no dynamic floor)
  // model 5 only: per-clump convergence threshold kappa_thr,sub. > 0 sets it absolutely;
  // <= 0 (default) uses subhalo_kappathr_factor * kappathr_host, i.e. it tracks the host
  // counting threshold as z_s changes. The population-weighted sweep gives, at factor 0.1,
  // clump reductions of 1.9e4/1.3e4/4.1e3 for sigma losses 0.08/0.11/0.21% at z_s=0.5/1/5;
  // factor 1.0 (the fully self-consistent "same rule as hosts") costs 0.24/0.41/1.29%.
  double subhalo_kappathr = -1.0;
  double subhalo_kappathr_factor = 0.1;
  double subhalo_factor = 1.0e-2; // PDF-level brute acceptance, scripts/convergence/subhalo_factor_jsd.py (2026-07-12); ~10x cheaper than the old 1e-5, indistinguishable vs brute at z_s=1,5
  // Mass-conserving host carve (scheme A, 2026-07-22, docs/subhalo/mass_conserving_carve_note.md):
  // true (default) reduces a subhalo-bearing host by the REALIZED resolved clump mass
  // Sum_i m_i plus the mean unresolved mass M_u(r), so the total halo mass is M exactly
  // in every realization (not merely in the mean). Applies to subhalo_model 3 and the
  // brute reference (model 1/2 + subhalo_brute; M_u = 0 there). false reproduces the
  // pre-2026-07-22 deterministic (1 - f_s,b)M reduction, kept as the bitwise determinism
  // reference and for A/B. The RNG stream is identical either way (the host build draws
  // no random numbers; the carve only reorders the host build after the clump draw).
  bool subhalo_carve = true;
  // Diagnostic override (2026-07-23): > 0 fixes psi_min everywhere the subhalo module would
  // otherwise use m_floor/M (buildWsubBin, unresolvedMass, addClumps' subhalo_brute path).
  // E.g. psi_min_fixed = 1e-4 = psi_res tests "no extrapolation below the SHMF's own
  // calibration point", host-mass-independent by construction. <= 0 (default) keeps the
  // existing m_floor/M behavior, bitwise unchanged. See docs/subhalo/mass_conserving_carve_note.md.
  double psi_min_fixed = -1.0;

  // future nuisance params (Phase 2)
  // double c_norm = 1.0;
  // double filament_density_norm = 1.0;
  // double filament_fraction = 1.0;
  double custom_kappathr = -1.0;

  // Mean-kappa anchor for the flux-conservation compensation in sample_lnmu
  // (2026-07-13, batch-anchor bug: the legacy empirical batch mean lets one
  // kappa >> 1 monster ray shift the WHOLE batch by -2*kappa/n; see
  // docs/convergence_mmin_nz_note.md Addendum and
  // data/results/floor_permutation_null/report.md).
  //   0 = legacy: empirical mean over ALL rays (default; bit-identical to the
  //       pre-2026-07-13 behavior, seed-for-seed).
  //   1 = robust: empirical mean over rays with kappa <= kappa_anchor_cut only
  //       (recommended fix; kappa > 1 rays are outside weak-lensing validity
  //       and already tracked by InvalidSampleStats). Residual coupling
  //       O(kappa_anchor_cut/n). Falls back to mode 0 if every ray exceeds
  //       the cut (pathological).
  //   2 = external: use kappa_anchor_value directly (e.g. an analytically
  //       derived <kappa>, or 0.0 for no compensation). The only mode with
  //       exactly independent realizations within one call.
  int kappa_anchor = 0;
  double kappa_anchor_cut = 1.0;
  double kappa_anchor_value = 0.0;

  // Bias (clustering) layer model (2026-07-16, docs/bias_field_design_note.md):
  //   0 = legacy: independent log-normal count modulation per (jz,jM) cell with
  //       sigma_b = Dg(z) b(M,z) sigma(M_b) keyed to the tube-segment mass M_b
  //       (default; bit-identical to the pre-change behavior, seed-for-seed).
  //       Known pathology: no continuum limit — refining Nz (or raising
  //       kappa_thr via rmax) makes the iid draws wilder without bound
  //       (docs/nz_bias_convergence_note.md).
  //   1 = correlated 1D field: ONE Gaussian delta_1D(chi) along the LOS
  //       (KP91 pencil projection of the code's own linear P(k) through a
  //       transverse disk window of comoving radius bias_Rperp), periodic mode
  //       spectrum with L = 1.05 chi(z_s), modes k_n = 2 pi n / L up to
  //       k_max = 2 pi / bias_Rperp (N_max = L/R_perp, floored at 4). Every
  //       (jz,jM) cell reads the segment average of its z-shell and rides the
  //       field via b(M,z) Dg(z); lambda = exp(bDg dbar - (bDg)^2 sig2/2) keeps
  //       <N> = Nbar exact. Realized via the exact shell-covariance Cholesky
  //       (equal in law to the mode sum).
  int bias_model = 0;
  // Comoving transverse window radius, kpc; only used when bias_model = 1.
  // Default = R_L(1e14 Msun) at the fiducial cosmology (2026-07-16 decision):
  // the clustering-variance-weighted signal scale at z_s <= 1 (where the
  // clustering share of Var(kappa) and the GW source population sit), PBS-
  // valid for the median signal-carrying cell; turboGL's calibrated
  // lambda_L agrees at the z~1 pivot only (their k_L = exp(3.9-4.6z)/Mpc is
  // z-dependent — weak corroboration of the scale, none of the fixedness).
  // Deliberately NOT chosen by matching the legacy layer (it
  // happens to agree with legacy at z_s = 1 — observation, not criterion).
  // Fixed number, NOT recomputed per cosmology (predictability).
  double bias_Rperp = 8441.0;

  // Smoothing window shape of the bias field (2026-07-20,
  // docs/bias_window_design_plan.md); only used when bias_model = 1:
  //   0 = transverse disk, W(x) = 2 J1(x)/x with x = k_perp bias_Rperp — the
  //       window acts on k_perp ALONE, so LOS power is suppressed only by the
  //       numerical mode cutoff k_max = 2 pi / bias_Rperp (legacy; bitwise
  //       default).
  //   1 = spherical top-hat, W(x) = 3 (sin x - x cos x) / x^3, and
  //   2 = Gaussian,          W(x) = exp(-x^2/2),
  //       both acting on the FULL modulus x = |k| bias_Rperp,
  //       |k| = sqrt(k_par^2 + k_perp^2): isotropic smoothing, as the
  //       peak-background split reading of b(M) wants, and (top-hat) the
  //       window the paper's sigma(R) already refers to.
  // NOTE (window 2): bias_Rperp is used AS GIVEN — no internal variance
  // matching. To compare a Gaussian against a top-hat of radius R_TH, solve
  // sigma^2_G(R_G) = sigma^2_TH(R_TH) offline and pass R_G explicitly
  // (expect R_G ~ 0.4-0.5 R_TH).
  // The mode convention (N_max = L/bias_Rperp floored at 4, L = 1.05 chi(z_s))
  // is UNCHANGED for all windows. Nonzero requires bias_model = 1 (throws).
  int bias_window = 0;

  // Weak (sub-threshold) arm of the correlated field (2026-07-16): when true,
  // the background kappa_W is drawn CONDITIONALLY on the same per-shell field
  // as the explicit counts (Cox-process split of Campbell's theorem):
  //   E[kappa_W|delta] = sum_iM lambda_iM m_iM,  Var[kappa_W|delta] = sum_iM lambda_iM v_iM,
  //   m_iM = int nbar kappa, v_iM = int nbar kappa^2 over sub-threshold annuli
  //   (same integrand/measure/floor as sigmakappaW; sum v = sigma_W^2 checked
  //   at build), lambda_iM = the count layer's mean-1 lognormal.
  // Sampled as kappa_W = sum_i S_i(dbar_i) + N(0,1) sqrt(sum_i V_i(dbar_i)),
  // S_i = sum_M m(lambda-1), V_i = sum_M v lambda (per-shell delta tables).
  // Adds the 2-halo variance of the weak layer + the weak<->count covariance
  // (both read the same realized field). Requires bias_model = 1 (throws
  // otherwise); no-op when bias = 0. Default OFF: bias_model=1 alone stays
  // the counts-only field model, stream-for-stream. NFW halos only (same
  // scope as the legacy weak Gaussian) — filaments carry no weak arm.
  bool bias_weak = false;

  // Filament clustering bias (2026-07-23, docs/filament_bias_note.md). When true,
  // the filament population is modulated by the filament bias b_fil(M,z) =
  // cosmology::filbias (PBS of the pFCfil barrier, q = 0.7) instead of sharing the
  // halo bias b(M,z) with the NFW population. Same one-dimensional field delta_1D
  // and the same lognormal count modulation lambda = exp(bDg dbar - bDg^2 sig2/2);
  // only the per-cell amplitude bDg switches from Dg*halobias to Dg*filbias for the
  // filament counts. filbias < halobias at every mass (lower, flatter barrier), so
  // filaments cluster more weakly. Requires the correlated field (bias_model = 1);
  // in the legacy iid layer (bias_model = 0) this flag is a no-op and filaments keep
  // the halo modulation. Default OFF: bias_model = 1 alone reproduces the pre-change
  // behavior (filaments ride the halo bias), stream-for-stream. Orientation of the
  // filaments is unchanged (isotropic) — this flag is a number-density bias only.
  bool fil_bias = false;
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
        double kappa_weak = 0.0;    // the weak-background component alone (legacy Gaussian or
                                    // bias_weak conditional draw); kappa - kappa_weak = explicit part
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
