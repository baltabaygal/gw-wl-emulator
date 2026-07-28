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
  = `bias_model=1, bias_window=1, bias_Rperp=20000, bias_weak=true, fil_bias=true`.
  Flip the default only bundled with the other bias sign-offs. Since 2026-07-27 the ML
  pipeline no longer depends on that flip: the config is pinned explicitly in
  `ml.params.PRODUCTION_CONFIG` and threaded through `python/generate_dataset.py`.

## 5. Mac gates — ALL PASSED (2026-07-27)

Previously built and smoke-tested in the Linux sandbox only. Now closed on the Mac
`test` env. Full evidence: **`data/results/fil_bias/report.md`**.

1. `make build` (Python 3.12 `.so`) — clean.
2. `tests/test_cosmology_params.py` **11/11** incl. `test_backward_compat_bitwise`.
   Its stored reference (`tests/data/reference_lnmu_pre6d.npz`) predates `fil_bias`,
   so passing it IS the bitwise-default gate — no new reference capture needed.
3. `tests/test_fil_bias.py` (new) **10/10** — default bitwise across four configs,
   inert at `bias_model=0`, live at `bias_model=1` and in the production config
   (dead-flag guard), deterministic/finite for all three windows, wired and acted on
   by all five entry points.
4. A/B at the full production config (`ml.params.PRODUCTION_CONFIG`),
   `scripts/convergence/fil_bias_ab.py`, 480k rays/arm at z_s = 0.5/1/5 plus a 2M
   rays/arm deep run at z_s = 1:
   - **JSD at the sampling floor everywhere**, and the cross-JSD tracks 1/N
     (6.22e-5 → 1.53e-5 for a ×4.17 in rays), which is the signature of a null and
     bounds any true offset at z_s=1 to J\* ≲ 1e-5 — ~500× under the emulator KL.
   - clipped σ shifts −0.4 to −0.8%, negative at every z_s (the predicted direction,
     `filbias < halobias`), but only ~2σ and it does NOT firm up with statistics.
     Quote it as "of order −0.4% at ~2σ, unresolved"; do **not** quote −0.78%.
   - edge Δq01 = +2–4e-4 (35–65× below the `bias_weak` arm's −1.4e-2) and flux
     Δln⟨1/μ⟩ ~ 1e-6–2e-4 ⇒ **no new emulator edge/flux refit** beyond the weak arm's.

⚠ A null was the *expected* outcome here (filaments are subdominant), so this bounds
the effect rather than measuring it — same caveat as the model-5 gate. The case for
`fil_bias=true` in production is physical correctness, not numerics: the draft states
filaments collapse from the flatter (0, 0.7) barrier, and with the flag off they ride
`halobias` and that sentence is false in the code.
