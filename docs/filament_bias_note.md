# Filament clustering bias — design + implementation note (2026-07-23)

Companion to `docs/bias_field_design_note.md` (the halo clustering field) and
`docs/bias_window_design_plan.md`. Adds a filament-specific bias to the correlated
1D field so the filament population clusters with its own bias `b_fil(M,z)` instead
of borrowing the halo bias `b(M,z)`.

## 1. Motivation

The correlated bias field (`bias_model = 1`) modulates every (jz,jM) cell's Poisson
mean by `lambda = exp(bDg dbar - 1/2 bDg^2 sig2)` with `bDg = Dg(z) b(M,z)`. Before
this change the SAME `lambda` (halo bias amplitude) multiplied both the NFW halo
counts and the cylindrical filament counts — filaments inherited the halo bias.

Filaments collapse from a lower, nearly scale-independent barrier (two-axis collapse;
Yan & Fan 2011, `papers/filaments_biasing/1101.3847v1.pdf`), so they are **less
strongly biased** than halos of the same mass. The code already encodes that barrier
as `pFCfil` (`cosmology.cpp`, flat barrier `p = 0, q = 0.7`, vs the halo `pFC`
`p = 0.3, q = 0.8`). The self-consistent filament bias is therefore the
peak-background split (PBS) of `pFCfil`, exactly as `halobias` is the PBS of `pFC`.

## 2. The bias

The Sheth-Mo-Tormen PBS bias is `b = 1 + (q nu^2 - 1)/deltac0 + 2p/[deltac0 (1 + (q nu^2)^p)]`.
The `p = 0` filament barrier kills the last term, leaving the **flat-barrier bias**

    b_fil(M,z) = 1 + (q nu^2 - 1)/deltac0,   nu = deltac(z)/sigma(M),   q = 0.7,

implemented as `cosmology::filbias(z, sigma)` (mirrors `cosmology::halobias`). `q`
here MUST track `pFCfil`'s `q` so the bias is the PBS of the code's own filament mass
function.

Verified against the compiled library (`cpp/probe.cpp`, throwaway): `filbias < halobias`
at every (z, sigma). Ratio `halobias/filbias` runs ~1.3 at low mass (z=0) down to ~1.07
at high mass / high z. Matches the analytic formula to 4 decimals (z=0, nu=0.817:
filbias 0.6839, halobias 0.8999).

## 3. Implementation (files touched)

- `cosmology.h` / `cosmology.cpp` — new `double filbias(double z, double sigma)`.
- `lensing.h` — `LensingConfig::fil_bias` (bool, **default false**).
- `lensing.cpp::sample_lnmu_raw`:
  - per cell (bias_model 1): `bDgF = Dg*filbias`, `bcompF = 1/2 bDgF^2 sig2[bsh]`
    (computed only when `fil_bias`);
  - per realization: `lambdaF = exp(bDgF*bfvals[j][bsh] - bcompF)`, defaulting to the
    halo `lambda`;
  - the filament block uses `lambdaF*barNF` (halos unchanged, still `lambda*barNH`).
- `lnmu_wrapper.{h,cpp}` — `SimParams::fil_bias` -> `cfg.fil_bias`.
- `python_bindings.cpp` — `fil_bias` kwarg (default false) on `sample_lnmu`,
  `sample_lnmu_ml`, `sample_lnmu_ml_with_diagnostics`, `sample_lensing_raw_ml`;
  echoed in `get_simulator_config`.

**Same field, no new RNG.** `lambdaF` reuses the already-realized `bfvals[j][bsh]`;
only the filament Poisson mean changes. `fil_bias = false` -> `lambdaF == lambda`
everywhere, so the default is bitwise-identical to the pre-change code (verified in the
Linux sandbox: default == explicit `fil_bias=false`, seed-for-seed). When
`fil_bias = true` the filament counts change, which shifts the shared RNG stream
downstream — this is the intended model change, gated behind the flag.

## 4. Scope / caveats

- **Correlated field only.** `fil_bias` requires `bias_model = 1`; in the legacy iid
  layer (`bias_model = 0`) it is a no-op (filaments keep the halo modulation). The
  paper describes the `bias_model = 1` field, so this is the relevant path.
- **Number-density bias only — orientation is NOT modelled.** Filament axes are still
  drawn isotropically (`phiF ~ U(0,2pi)`). Tested separately (2026-07-23,
  `plots/filament_orientation_bracket.png`): under realistic per-tidal-cell coherence
  the orientation effect on P(lnmu) is JSD ~1e-3 (sigma ~+3-4%), sub-dominant to the
  emulator KL ~7e-3 — so the isotropic-orientation approximation is justified.
- **Small effect on P(lnmu).** Filaments are subdominant, so switching their bias from
  `halobias` to `filbias` moves P(lnmu) below MC noise at 70k samples (body sigma
  Delta < 1e-4 at z_s=1). Real but small; quote from a high-N ensemble if needed.
- **Default staged OFF** (matches `bias_weak` / `bias_window`). Paper/production config
  = `bias_model=1, bias_window=1, bias_Rperp=20000, fil_bias=true`. Flip the default
  only bundled with the other bias sign-offs.

## 5. STILL PENDING — Mac gates

Built and smoke-tested in the **Linux sandbox** only (GSL replaced by a local shim;
Python 3.10). Before production use, on the Mac `test` env:
1. `make build` (Python 3.12 `.so`);
2. `tests/test_cosmology_params.py` incl. `test_backward_compat_bitwise` (fil_bias
   default must stay bitwise);
3. a fil_bias on/off A/B at `bias_model=1` on a high-N ensemble to log the P(lnmu)
   shift + the emulator edge/flux impact (expected negligible; filaments carry no weak
   arm and the shift is < emulator KL).
