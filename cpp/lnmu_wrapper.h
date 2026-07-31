#pragma once
#include <vector>
#include <cstdint>
#include "invalid_stats.h"

struct CosmologyParams {
    double OmegaM = 0.315;
    double sigma8 = 0.811;
    double h      = 0.674;

    // primordial amplitude: As > 0 normalizes P(k) directly (sigma8 ignored);
    // As <= 0 keeps the sigma8 normalization.
    double As     = -1.0;
    double kpivot = 5.0e-5;   // comoving kpc^-1 (= 0.05 Mpc^-1)

    // sigma8 normalization filter (2026-07-30, "option b"): true = real-space
    // top-hat at 8 Mpc/h (conventional sigma8, DEFAULT); false = legacy smooth-k Ws.
    // Ignored in As-mode. Only the amplitude anchor moves, never sigma_M(M).
    bool sigma8_tophat = true;

    // grid defaults
    double OmegaB = 0.0493;
    double zeq    = 3402.0;
    double T0     = 2.7255;
    double ns     = 0.965;

    double Mmin = 1e7;
    double Mmax = 1e17;
    int NM      = 100;

    // Source-redshift grid. The halo/z grid is log-spaced over [zmin, zmax] with Nz
    // nodes (cosmology.h: `zlist = loglist(zmin, zmax, Nz)`), and a z_s ABOVE zmax is
    // silently CLAMPED to the top node -- there is no throw and no warning, so an
    // out-of-range source redshift returns a duplicate of the zmax PDF.
    //
    // Raised 2026-07-30 from (10.01, 100) to cover the confirmed training range
    // z_s <= 12 (ml/params.py TRAINING_RANGE). This pair is NOT an arbitrary
    // rescaling: it EXTENDS the old grid at unchanged resolution. loglist builds by
    // the recurrence x_{j+1} = exp(log(x_j) + dlogz) from zmin, and
    //     dlogz = (log(zmax) - log(zmin)) / (Nz - 1)
    // evaluates to the SAME double for (10.01, 100) and for (ZMAX_DEFAULT, 103):
    // 0.06978540181126486, hex 3fb1dd74c284818c. So nodes 0..99 are bitwise
    // identical to the old grid and three new nodes are appended at
    // z = 10.7335, 11.5093, 12.3412. Anything with z_s <= 10.01 is unaffected as
    // long as no loop integrates over the full grid irrespective of z_s -- that is
    // what tests/test_cosmology_params.py::test_backward_compat_bitwise checks.
    //
    // ⚠ zmax and Nz remain INDEPENDENT knobs, so overriding Nz alone still rescales
    // the grid exactly as before (the Nz convergence studies keep their meaning).
    // If you change one, recompute the other from dlogz rather than picking a round
    // number, or the bitwise-extension property is lost.
    static constexpr double ZMAX_DEFAULT = 12.341169644129371;
    static constexpr int    NZ_DEFAULT   = 103;

    double zmin = 0.01;
    double zmax = ZMAX_DEFAULT;
    int Nz      = NZ_DEFAULT;
};

struct SamplingParams {
    int Nreal   = 1000;
    uint64_t seed = 123;
    int fil     = 1;
    int bias    = 1;
    int ell     = 1;
    int Nhalos  = 100;
    bool strict_weak_lensing = false;
    bool subhalo = true;             // PAPER DEFAULT (2026-07-29): the draft's model includes subhalos
    double m_floor = 1.0e7;
    int subhalo_threads = 1;
    int subhalo_parallel_threshold = 200000;
    int subhalo_model = 5;   // PAPER DEFAULT (2026-07-29): brute population + per-clump kappa threshold
    bool subhalo_brute = false;
    double subhalo_factor = 1.0e-2;  // PDF-level brute acceptance, scripts/convergence/subhalo_factor_jsd.py (2026-07-12)
    bool subhalo_carve = true;       // mass-conserving realized-clump host carve (scheme A, 2026-07-22); false = legacy (1-f_s,b)M
    bool subhalo_virial = true;      // PAPER DEFAULT (2026-07-29): JvdB14 virial convention, psi = m/M_vir, profile to r_vir; model 4/5 only
    // model 5 only (2026-07-27): per-clump convergence threshold kappa_thr,sub.
    // > 0 absolute; <= 0 uses subhalo_kappathr_factor * host kappa_thr.
    double subhalo_kappathr = -1.0;
    double subhalo_kappathr_factor = 0.1;
    double psi_min_fixed = -1.0;    // diagnostic: > 0 fixes psi_min (overrides m_floor/M); e.g. 1e-4 = psi_res
    double kappathr_flat = -1.0;   // <= 0 = legacy <N>=Nhalos rule (default, reverted 2026-07-10); > 0 = flat explicit-halo threshold
    int kappa_anchor = 1;          // PAPER DEFAULT (2026-07-29): robust anchor. 0 = legacy batch mean,
    double kappa_anchor_cut = 1.0; // 1 robust (exclude kappa > cut), 2 external value below
    double kappa_anchor_value = 0.0; // used only when kappa_anchor == 2
    int bias_model = 1;            // PAPER DEFAULT (2026-07-29): correlated 1D field (draft sec. II.A); 0 = legacy iid cell bias
    double bias_Rperp = 20000.0;   // PAPER DEFAULT (2026-07-29): R_s = 20 Mpc, confirmed by Ville (bias_model = 1 only)
    bool bias_weak = true;         // PAPER DEFAULT (2026-07-29): sub-threshold background rides the field (requires bias_model = 1)
    int bias_window = 1;           // PAPER DEFAULT (2026-07-29): real-space spherical top-hat. 0 = legacy transverse disk;
                                   // 1 spherical top-hat, 2 Gaussian (both on |k|; require bias_model = 1)
    bool fil_bias = true;          // PAPER DEFAULT (2026-07-29): filaments use filbias (PBS of pFCfil, q=0.7); requires bias_model = 1
};

struct LnmuSampleDiagnostics {
    std::vector<double> lnmu;
    InvalidSampleStats invalid_stats;
};

std::vector<double> sample_lnmu(
    double z,
    const CosmologyParams& cosmo,
    const SamplingParams& sampling
);

LnmuSampleDiagnostics sample_lnmu_with_diagnostics(
    double z,
    const CosmologyParams& cosmo,
    const SamplingParams& sampling
);

struct LnmuStats {
    double mean = 0.0;
    double variance = 0.0;
    double skewness = 0.0;
    double mean_mu = 0.0;   // <mu> for flux conservation
};

LnmuStats compute_lnmu_stats(
    double z,
    const CosmologyParams& cosmo,
    const SamplingParams& sampling
);

LnmuStats compute_lnmu_stats_fast(
    double z,
    const CosmologyParams& cosmo,
    const SamplingParams& sampling
);
