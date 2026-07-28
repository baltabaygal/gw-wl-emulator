# fil_bias acceptance — Mac gates + production-config A/B (2026-07-27)

Closes the three items left pending in `docs/filament_bias_note.md` §5. `fil_bias`
switches the filament count modulation in the correlated bias field from the halo
bias `b(M,z)` to the PBS bias of the code's own filament barrier,
`cosmology::filbias` with `(p,q) = (0, 0.7)`, which is 10–20% lower.

This was the only setting in `ml.params.PRODUCTION_CONFIG` with no gate behind it.

## 1. Mac build + bitwise default

- `make build` on the Mac `test` env (Python 3.12) — clean.
- `tests/test_cosmology_params.py` **11/11**, including `test_backward_compat_bitwise`.
  That test replays the no-kwarg default path against
  `tests/data/reference_lnmu_pre6d.npz`, a reference captured **before** `fil_bias`
  existed, so it passing is exactly the "default is bitwise-clean" gate the note asked
  for — no separate reference capture was needed.
- `tests/test_fil_bias.py` (new) **10/10**:
  - default == explicit `fil_bias=False`, bit-for-bit, across four configs
    (bias_model=1; +bias_weak; +top-hat/20 Mpc; full production config);
  - inert at `bias_model=0` (bitwise) — the legacy iid layer has no field to ride;
  - **live** at `bias_model=1` and in the production config (guards a dead flag);
  - deterministic + finite for all three windows;
  - accepted **and acted on** by all five py entry points.

  Note the two argument orders: `sample_lnmu` / `compute_lnmu_stats` take
  `(z, OmegaM, sigma8, h, Nreal, ...)` while the `*_ml` family takes
  `(z, h, OmegaM, sigma8, nsamples, ...)`. Swapping h and OmegaM silently kills the
  field power (CLAUDE.md gotcha); the test passes them by keyword for that reason.

## 2. A/B at the production config

`scripts/convergence/fil_bias_ab.py`. Both arms run the full
`PRODUCTION_CONFIG` (subhalo model 5 + carve, correlated field, spherical top-hat at
R_s = 20 Mpc, conditional weak arm, robust κ anchor); the ONLY difference is
`fil_bias`. Sharded over subprocesses (never `mp.Pool`), 8 shards/arm, seed-split
floors, clipped/trimmed statistics per the standing rules.

### 480k rays/arm (8 × 60k), `ab_n60000.json`

| z_s | σ_clip off | σ_clip on | Δσ/σ | JSD(off,on) | floors | edge Δq01 | flux Δln⟨1/μ⟩ |
|-----|-----------|----------|------|-------------|--------|-----------|----------------|
| 0.5 | 0.032939 | 0.032797 | −0.433 ± 0.550 % (−0.8σ) | 5.85e-5 | 8.4 / 9.9e-5 | +2.14e-4 | −1.08e-5 |
| 1.0 | 0.072707 | 0.072140 | −0.781 ± 0.361 % (−2.2σ) | 6.22e-5 | 7.9 / 10.0e-5 | +4.09e-4 | −6.75e-7 |
| 5.0 | 0.214395 | 0.214312 | −0.039 ± 0.224 % (−0.2σ) | 7.94e-5 | 17.7 / 18.9e-5 | +3.33e-4 | −1.92e-4 |

### 2M rays/arm at z_s = 1 (8 × 250k), `ab_n250000.json`

| z_s | σ_clip off | σ_clip on | Δσ/σ | JSD(off,on) | floors | edge Δq01 | flux Δln⟨1/μ⟩ |
|-----|-----------|----------|------|-------------|--------|-----------|----------------|
| 1.0 | 0.072710 | 0.072436 | −0.376 ± 0.192 % (−2.0σ) | 1.53e-5 | 1.80 / 2.16e-5 | +2.95e-4 | −9.31e-7 |

## 3. Verdict

**JSD is at the sampling floor at every z_s and at both depths, and the cross-JSD
tracks 1/N.** This is the load-bearing result. Going 480k → 2M rays at z_s = 1
(×4.17) took the cross-JSD 6.22e-5 → 1.53e-5, a factor 4.07. A *real* P(lnμ)
difference would have plateaued at a constant offset J\* once the floor dropped below
it; scaling as 1/N instead is the signature of a null. It bounds any true offset at
z_s = 1 to **J\* ≲ 1e-5**, i.e. ~500× below the emulator's median KL of 7.3e-3.

**The clipped-σ shift is suggestive but NOT resolved.** It is negative at all three
redshifts and at both depths, which is the physically predicted direction
(`filbias < halobias` ⇒ weaker filament clustering ⇒ less variance). But it sits at
~2σ and does not firm up with statistics: quadrupling the rays at z_s = 1 halved the
central value (−0.78% → −0.38%) while halving the error, leaving the significance
flat. A genuine −0.78% would have returned at ~4σ in the deep run and did not. Fair
statement: **a shift of order −0.4% in σ_clip at z_s = 1, at ~2σ, consistent with the
predicted sign, unresolved at 3σ even with 2M rays/arm.** Do not quote −0.78%.

⚠ Do not oversell the "at floor" verdicts. Filaments are subdominant, so a null was
the *expected* outcome — this bounds the effect, it does not measure it. Same caveat
as the model-5 gate.

**Emulator impact: none beyond what the weak arm already forces.** The low-μ edge
moves by +2–4e-4, ~35–65× smaller than the `bias_weak` arm's −1.4e-2, and the flux
normalization moves by ~1e-6 to 2e-4 in ln⟨1/μ⟩. So `fil_bias` adds **no new**
edge/flux refit requirement on top of the one the weak arm already triggers.

**`fil_bias=True` therefore belongs in the production config on physical-correctness
grounds, not numerical ones** — the draft states filaments use the (0, 0.7) barrier
bias, and without the flag they ride `halobias` and that sentence is false in the
code. Its cost in P(lnμ) fidelity is bounded at the 1e-5 JSD level.

## 4. Reproduce

```bash
PY=/Users/baltabay/miniforge3/envs/test/bin/python
$PY scripts/convergence/fil_bias_ab.py --nray 60000  --nshard 8 --zs 0.5 1.0 5.0
$PY scripts/convergence/fil_bias_ab.py --nray 250000 --nshard 8 --zs 1.0
```

Shards cache in `tmp/filbias_ab_zs{zs}_n{nray}/`; both the shard dir and the output
JSON are keyed by depth, so a deeper rerun neither reuses shallow shards nor clobbers
a shallower sweep. Reruns at an existing depth reload from cache and recompute the
statistics without resampling.
