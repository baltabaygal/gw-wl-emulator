# Comparison with the Desktop `halos` reference

Comparison date: 2026-07-10  
Reference checkout: `/Users/baltabay/Desktop/halos` at `739210d` (`subhalo-production-integration`)  
Emulator checkout: `/Users/baltabay/Desktop/gw-wl-emulator`

The `halos` checkout was treated as read-only. No source file, build product, commit, or branch in that checkout was changed.

## Intended primary differences

### 1. Subhalo model

The emulator retains the opt-in subhalo addition in `cpp/subhalo.cpp`, `cpp/subhalo.h`, and the host-sampling path in `cpp/lensing.cpp`.

Compared with the reference port, the emulator also contains the research and production controls needed to validate the model:

- resolved-host and legacy mass-removal modes;
- brute-force sampling for reference comparisons;
- the model-3 unresolved-clump `Wsub` mean/variance tables;
- configurable subhalo mass floor, threshold factor, profiling, and clump-loop threading;
- paired `kappa_nosub` output for variance comparisons.

Subhalos remain off by default, so the ordinary halo-only path does not consume subhalo random draws.

### 2. Different `kappa_thr` handling

The reference chooses `kappa_thr` by solving for `<N> = Nhalos`. The emulator keeps that rule as its default and adds two explicit alternatives:

- `kappathr_flat > 0`: use a source-redshift-independent threshold;
- `custom_kappathr > 0`: override both the solved and flat rules for scans.

The weak-lens integration floor is anchored to the default threshold while `custom_kappathr` is swept. This prevents a threshold scan from changing both the explicit/weak split and the absolute faint-end cutoff at the same time. The subhalo resolution threshold is separately derived as `subhalo_factor * kappa_thr_host`.

## Shared baseline reviewed

The common C++ halo/cosmology path was compared file by file across `basics`, `cosmology`, `lensing`, and `main_lensing`.

The following behavior is shared with the reference:

- cosmology grids, transfer functions, halo mass functions, concentration relation, and redshift-dependent NFW density convention;
- NFW and cylindrical lens kernels and the single-angle shear projection convention;
- the fixed-`<N>` threshold solver used by the default path;
- the corrected Campbell weak-lensing variance (`2 pi r^2 dlnr`, without subtracting a squared mean);
- the `CLIGHT` constant in distance and lens-count calculations.

The emulator necessarily retains non-subhalo infrastructure that has no counterpart in the small reference executable: the library configuration API, Python bindings, raw-sample diagnostics, invalid-sample accounting, primordial-`A_s` parameterization, and ML-facing entry points. It also retains numerically stable NFW core evaluation and cached per-bin quantities. These are support/stability changes, not alternate halo-population physics.

## Changes applied during this review

1. Restored `cosmology::Dfstarperfstar` to the reference formula. The previous emulator expression contained `beta*alpha - alpha*beta`, so its mass-dependent double-power-law term canceled identically.
2. Restored the shared `readdataCSV` contract: the first CSV row is treated as column names and skipped before numeric parsing.
3. Added this comparison record. No files in `/Users/baltabay/Desktop/halos` were edited.

## Deliberately retained implementation differences

- `randomreal` is inline in the emulator and preserves its existing regression stream; changing it would invalidate bitwise sampler baselines.
- `sample_lnmu_raw` factors repeated host/filament work into cached quantities and adds diagnostics, while retaining the same halo-only sampling structure.
- `main_lensing.cpp` remains a compatibility executable; the emulator's supported interface is the configured C++/Python sampling API.
- The `A_s`, expanded cosmology, magnetic-spectrum, and emulator plumbing are outside the subhalo/threshold comparison and were not removed.

## Verification contract

The expected acceptance checks are:

- the project builds against the Python 3.12 `test` environment;
- the backward-compatible default sampler remains bitwise equal to its checked-in pre-6d reference;
- cosmology-parameter tests pass;
- enabling subhalos is opt-in and the reference `halos` checkout remains untouched.
