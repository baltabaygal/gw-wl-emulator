# Permutation-null calibration of the JSD floors

R = 4000 random re-splits per test, multivariate-hypergeometric on the shared 120-bin histograms (exactly equivalent to permuting the samples). floor pctl = percentile of the ORIGINAL seed-split floor_half within the null of random truth re-splits (>=97.5 flags an unlucky split). p_null = P(J_perm >= J_obs) under candidate==truth pooling; p_null >= 0.05 -> statistically converged. Materiality vs the emulator KL 7.3e-3 is a separate (physics) criterion.

## mmin_convergence / z_s=0.2

original floor_half = 3.026e-04, null median = 2.407e-04 [68%: 2.14e-04..2.70e-04], **original split at pctl 98**

| config | J_obs | floor_pred | excess | p_null | verdict |
|:--|--:|--:|--:|--:|:--|
| m1e5 | 1.125e-04 | 1.513e-04 | 0.000e+00 | 0.823 | converged (noise-level) |
| m1e6 | 2.272e-04 | 1.513e-04 | 7.593e-05 | 0.000 | resolved residual |
| m1e7 | 5.667e-04 | 1.513e-04 | 4.154e-04 | 0.000 | resolved residual |
| m1e8 | 1.156e-03 | 1.513e-04 | 1.005e-03 | 0.000 | resolved residual |
| m1e9 | 2.623e-03 | 1.513e-04 | 2.472e-03 | 0.000 | resolved residual |
| nm200ctl | 1.656e-04 | 1.513e-04 | 1.433e-05 | 0.008 | resolved residual |

## mmin_convergence / z_s=10

original floor_half = 2.290e-04, null median = 2.605e-04 [68%: 2.31e-04..2.93e-04], **original split at pctl 14**

| config | J_obs | floor_pred | excess | p_null | verdict |
|:--|--:|--:|--:|--:|:--|
| m1e5 | 1.497e-04 | 1.145e-04 | 3.520e-05 | 0.109 | converged (noise-level) |
| m1e6 | 2.131e-04 | 1.145e-04 | 9.862e-05 | 0.000 | resolved residual |
| m1e7 | 4.060e-04 | 1.145e-04 | 2.915e-04 | 0.000 | resolved residual |
| m1e8 | 8.060e-04 | 1.145e-04 | 6.915e-04 | 0.000 | resolved residual |
| m1e9 | 1.693e-03 | 1.145e-04 | 1.579e-03 | 0.000 | resolved residual |
| nm200ctl | 6.930e-04 | 1.145e-04 | 5.785e-04 | 0.000 | resolved residual |

## mmin_convergence / z_s=1

original floor_half = 2.351e-04, null median = 2.433e-04 [68%: 2.16e-04..2.72e-04], **original split at pctl 38**

| config | J_obs | floor_pred | excess | p_null | verdict |
|:--|--:|--:|--:|--:|:--|
| m1e5 | 1.419e-04 | 1.175e-04 | 2.433e-05 | 0.195 | converged (noise-level) |
| m1e6 | 2.921e-04 | 1.175e-04 | 1.746e-04 | 0.000 | resolved residual |
| m1e7 | 6.196e-04 | 1.175e-04 | 5.021e-04 | 0.000 | resolved residual |
| m1e8 | 9.900e-04 | 1.175e-04 | 8.725e-04 | 0.000 | resolved residual |
| m1e9 | 1.701e-03 | 1.175e-04 | 1.584e-03 | 0.000 | resolved residual |
| nm200ctl | 2.798e-04 | 1.175e-04 | 1.623e-04 | 0.000 | resolved residual |

## mmin_convergence / z_s=5

original floor_half = 2.957e-04, null median = 2.553e-04 [68%: 2.26e-04..2.88e-04], **original split at pctl 89**

| config | J_obs | floor_pred | excess | p_null | verdict |
|:--|--:|--:|--:|--:|:--|
| m1e5 | 1.192e-04 | 1.479e-04 | 0.000e+00 | 0.692 | converged (noise-level) |
| m1e6 | 2.065e-04 | 1.479e-04 | 5.869e-05 | 0.000 | resolved residual |
| m1e7 | 4.353e-04 | 1.479e-04 | 2.874e-04 | 0.000 | resolved residual |
| m1e8 | 7.532e-04 | 1.478e-04 | 6.053e-04 | 0.000 | resolved residual |
| m1e9 | 1.422e-03 | 1.478e-04 | 1.274e-03 | 0.000 | resolved residual |
| nm200ctl | 6.376e-04 | 1.479e-04 | 4.897e-04 | 0.000 | resolved residual |

## mmin_pd_convergence / z_s=0.2

original floor_half = 3.216e-04, null median = 2.354e-04 [68%: 2.08e-04..2.65e-04], **original split at pctl 100**

| config | J_obs | floor_pred | excess | p_null | verdict |
|:--|--:|--:|--:|--:|:--|
| m1e5 | 1.153e-04 | 1.608e-04 | 0.000e+00 | 0.690 | converged (noise-level) |
| m1e6 | 1.442e-04 | 1.608e-04 | 0.000e+00 | 0.085 | converged (noise-level) |
| m1e7 | 5.372e-04 | 1.608e-04 | 3.764e-04 | 0.000 | resolved residual |
| m1e8 | 9.735e-04 | 1.608e-04 | 8.127e-04 | 0.000 | resolved residual |
| m1e9 | 2.298e-03 | 1.608e-04 | 2.137e-03 | 0.000 | resolved residual |

## mmin_pd_convergence / z_s=10

original floor_half = 2.166e-04, null median = 2.533e-04 [68%: 2.24e-04..2.84e-04], **original split at pctl 10**

| config | J_obs | floor_pred | excess | p_null | verdict |
|:--|--:|--:|--:|--:|:--|
| m1e5 | 1.299e-04 | 1.083e-04 | 2.157e-05 | 0.368 | converged (noise-level) |
| m1e6 | 1.577e-04 | 1.083e-04 | 4.939e-05 | 0.035 | resolved residual |
| m1e7 | 1.958e-04 | 1.083e-04 | 8.750e-05 | 0.000 | resolved residual |
| m1e8 | 2.753e-04 | 1.083e-04 | 1.670e-04 | 0.000 | resolved residual |
| m1e9 | 6.790e-04 | 1.083e-04 | 5.707e-04 | 0.000 | resolved residual |

## mmin_pd_convergence / z_s=1

original floor_half = 6.753e-04, null median = 2.288e-04 [68%: 2.02e-04..2.60e-04], **original split at pctl 100**

| config | J_obs | floor_pred | excess | p_null | verdict |
|:--|--:|--:|--:|--:|:--|
| m1e5 | 4.036e-04 | 3.377e-04 | 6.592e-05 | 0.000 | resolved residual |
| m1e6 | 8.904e-04 | 3.377e-04 | 5.527e-04 | 0.000 | resolved residual |
| m1e7 | 1.337e-03 | 3.377e-04 | 9.996e-04 | 0.000 | resolved residual |
| m1e8 | 2.008e-03 | 3.377e-04 | 1.671e-03 | 0.000 | resolved residual |
| m1e9 | 2.397e-03 | 3.377e-04 | 2.060e-03 | 0.000 | resolved residual |

## mmin_pd_convergence / z_s=5

original floor_half = 2.196e-03, null median = 2.445e-04 [68%: 2.16e-04..2.76e-04], **original split at pctl 100**

| config | J_obs | floor_pred | excess | p_null | verdict |
|:--|--:|--:|--:|--:|:--|
| m1e5 | 7.227e-04 | 1.098e-03 | 0.000e+00 | 0.000 | resolved residual |
| m1e6 | 9.539e-04 | 1.098e-03 | 0.000e+00 | 0.000 | resolved residual |
| m1e7 | 1.077e-03 | 1.098e-03 | 0.000e+00 | 0.000 | resolved residual |
| m1e8 | 1.527e-03 | 1.098e-03 | 4.293e-04 | 0.000 | resolved residual |
| m1e9 | 1.998e-03 | 1.098e-03 | 9.002e-04 | 0.000 | resolved residual |

## nz_convergence / z_s=0.2

original floor_half = 3.277e-04, null median = 2.373e-04 [68%: 2.09e-04..2.68e-04], **original split at pctl 100**

| config | J_obs | floor_pred | excess | p_null | verdict |
|:--|--:|--:|--:|--:|:--|
| nz25 | 4.019e-03 | 1.639e-04 | 3.855e-03 | 0.000 | resolved residual |
| nz50 | 1.254e-03 | 1.639e-04 | 1.091e-03 | 0.000 | resolved residual |
| nz100 | 2.855e-04 | 1.639e-04 | 1.216e-04 | 0.000 | resolved residual |
| nz200 | 1.355e-04 | 1.639e-04 | 0.000e+00 | 0.237 | converged (noise-level) |

## nz_convergence / z_s=10

original floor_half = 2.142e-04, null median = 2.580e-04 [68%: 2.27e-04..2.92e-04], **original split at pctl 7**

| config | J_obs | floor_pred | excess | p_null | verdict |
|:--|--:|--:|--:|--:|:--|
| nz25 | 6.795e-04 | 1.071e-04 | 5.724e-04 | 0.000 | resolved residual |
| nz50 | 2.957e-04 | 1.071e-04 | 1.886e-04 | 0.000 | resolved residual |
| nz100 | 2.166e-04 | 1.071e-04 | 1.095e-04 | 0.000 | resolved residual |
| nz200 | 2.039e-04 | 1.071e-04 | 9.681e-05 | 0.000 | resolved residual |

## nz_convergence / z_s=1

original floor_half = 2.352e-04, null median = 2.398e-04 [68%: 2.13e-04..2.69e-04], **original split at pctl 44**

| config | J_obs | floor_pred | excess | p_null | verdict |
|:--|--:|--:|--:|--:|:--|
| nz25 | 1.586e-03 | 1.176e-04 | 1.468e-03 | 0.000 | resolved residual |
| nz50 | 2.902e-04 | 1.176e-04 | 1.726e-04 | 0.000 | resolved residual |
| nz100 | 1.822e-04 | 1.176e-04 | 6.458e-05 | 0.002 | resolved residual |
| nz200 | 1.301e-04 | 1.176e-04 | 1.248e-05 | 0.431 | converged (noise-level) |

## nz_convergence / z_s=5

original floor_half = 2.620e-04, null median = 2.514e-04 [68%: 2.22e-04..2.84e-04], **original split at pctl 63**

| config | J_obs | floor_pred | excess | p_null | verdict |
|:--|--:|--:|--:|--:|:--|
| nz25 | 7.868e-04 | 1.310e-04 | 6.558e-04 | 0.000 | resolved residual |
| nz50 | 3.480e-04 | 1.310e-04 | 2.170e-04 | 0.000 | resolved residual |
| nz100 | 1.784e-04 | 1.310e-04 | 4.735e-05 | 0.000 | resolved residual |
| nz200 | 1.083e-04 | 1.310e-04 | 0.000e+00 | 0.889 | converged (noise-level) |


---

# ADDENDUM (2026-07-13): the permutation test found shard contamination

The null calibration above flagged the original truth splits as non-exchangeable
(floor_half at pctl 98-100 of the permutation null in 5 groups). KS tests on the
raw samples confirmed real distributional differences between truth halves, and
a per-shard scan (8 shards per array, flag |shard mean - median| > 5 sigma_mean)
localized the cause: **9 whole shards, always body-shifted NEGATIVE (up to -63
sigma), matching NO cached config (KS >= 0.24 vs everything)**:

| study | z | config | shard | z-score |
|:--|--:|:--|--:|--:|
| mmin | 0.2 | truthA | 0 | -5.9 |
| mmin | 10 | nm200ctl | 3 | -18.9 |
| mmin_pd | 1 | m1e5 | 1 | -13.1 |
| mmin_pd | 1 | truthA | 0 | -8.3 |
| mmin_pd | 1 | truthB | 7 | -28.4 |
| mmin_pd | 5 | m1e5 | 7 | -5.7 |
| mmin_pd | 5 | truthB | 4 | -63.1 |
| nz | 0.2 | truthB | 5 | -8.8 |
| nz | 1 | nz50 | 2 | -5.0 |

Suspected cause: stale shard-cache files from aborted mid-debug runs on
2026-07-11/12 (the scan loads shards/<key>_<s>.npy blindly; the finish()
concatenation bug was also live until 2026-07-12). NOT yet verified — run
`scripts/convergence/verify_flagged_shards.py` on the Mac (test env, ~2 min):
shards are deterministic per seed, so regenerating settles stale-cache vs live
bug, and `--fix` writes corrected arrays.

## Corrected results (flagged shards excluded; PROVISIONAL until verified)

The note's "inflated floors from sparse tail bins" explanation is superseded --
the floors were inflated by contaminated shards:

| group | floor_half | corrected | default's excess | corrected |
|:--|--:|--:|--:|--:|
| mmin_pd z=1 | 6.75e-4 | **2.37e-4** | 1.00e-3 (m1e7) | **4.05e-4** |
| mmin_pd z=5 | 2.20e-3 | **2.87e-4** | ~0 (masked) | **2.1e-5** |
| mmin z=0.2 | 3.03e-4 | 2.72e-4 | 4.15e-4 (m1e7) | 3.29e-4 |
| nz z=0.2 | 3.28e-4 | 3.12e-4 | 1.22e-4 (nz100) | 6.6e-5 |
| mmin z=10 nm200ctl | (truth clean) | - | 5.79e-4 | **8.19e-4** |

Impact on conclusions:
- Verdicts UNCHANGED (Mmin default: not converged but immaterial; Nz: converged)
  -- margins to the 7.3e-3 budget only grow, except the NM confound at z=10
  which is ~40% LARGER than reported.
- Specific numbers in docs/convergence_mmin_nz_note.md, the onepager, and
  CLAUDE.md sec 12 need updating after verification: worst-case Mmin excess is
  ~4e-4 (pd z=1) and ~8e-4 (NM z=10), not 1.0e-3; the mmin_pd z=5 column is now
  resolvable and shows a clean monotone ladder.


---

# RESOLUTION (2026-07-13, same day): root cause found — NOT corruption

Mac verification (`verify_flagged_shards.py` v2): 7/9 flagged shards REPRODUCE
deterministically with the current build (bitwise or to FP noise) and are
body-shifted vs siblings; 2/9 were false alarms of the 5-sigma flagging
heuristic (tail-luck means). The stale-cache hypothesis is DEAD.

**Mechanism (established from data, then located in source):** each flagged
shard is exactly (sibling distribution) + (uniform lnmu shift) + (one monster
ray). Predicted shift -2*kappa_monster/n from the shard's own most extreme ray
matches the observed median shift to ~10% in ALL 9 cases (e.g. z5 truthB s4:
kappa ~= 1014 from lnmu_min = -13.84, predicted -0.1354, observed -0.1290;
after removing the ray and un-shifting, KS vs siblings collapses 0.288 ->
0.032). Source: `cpp/lensing.cpp::sample_lnmu` (lines 646-663) subtracts the
EMPIRICAL batch mean kappa (flux-conservation anchor) instead of the analytic
expectation -- one kappa >> 1 ray (weak-lensing formula invalid there anyway)
shifts the whole batch by -2*kappa/n.

Consequences:
- Realizations within one sampler call are weakly coupled (O(kappa_max/n)) --
  "independent realizations" holds only between calls. Affects ALL ensembles
  ever drawn through sample_lnmu, at a rate set by P(monster kappa | config, n);
  negligible for typical production batches, material for refined-grid truth
  configs at 15-30k/batch.
- The permutation-null p-values and the "corrected (shards excluded)" tables
  above remain PROVISIONAL: exclusion over-corrects (the shards are valid
  physics up to the batch shift). Proper repair = fix the compensation and
  regenerate affected configs, or post-hoc un-shift each batch by its own
  monster-ray estimate.
- Fix options (supervisor decision): (a) analytic mean-kappa compensation;
  (b) robust mean excluding kappa > 1 rays from the anchor (minimal, physically
  motivated: those rays are outside weak-lensing validity and already tracked
  by the invalid-sample stats); (c) mitigation only: larger batches (impact
  ~ kappa^2/n). Option (b) recommended.
- The two false alarms (mmin_pd z5 m1e5 s7, nz z1 nz50 s2) carry small real
  shifts too (-0.006, -0.002, same mechanism, moderate rays) -- the flagging
  threshold, not the mechanism, is what separated them.


---

# BLOCK-AWARE RECALIBRATION + CROSS-STUDY EXPOSURE (2026-07-13, final)

Full tables: `data/results/shard_screen/report.md`; figure:
`plots/batch_anchor_diagonal.png`.

## Correction to this report's own percentiles

The "original split at pctl 98-100" results ABOVE are an artifact of the
sample-level permutation: under batch coupling the exchangeable unit is the
SHARD, and permuting individual samples destroys the shared anchor shift,
making the null too narrow. Exact block enumeration (all C(16,8)=12870
8v8-shard splits of the 16 truth shards) puts the original A|B splits at
pctl 4-90 across ALL 20 groups -- i.e. TYPICAL. The sample-level p_null
columns above are likewise anti-conservative and superseded. (The mechanism
diagnosis is unaffected: it rests on deterministic reproduction and the
-2*kappa_max/n law, not on those p-values.)

## What the block floors say instead

The mmin_pd z=1 / z=5 groups have block-null MEDIANS of 8.3e-4 / 2.4e-3 vs
~2.5e-4 elsewhere: with monster shards in the truth pool, those two groups
simply have LOW RESOLUTION -- their floors are genuinely that big until the
anchor is fixed and the configs regenerated. Their excesses should be quoted
as order-of-magnitude only.

## Exposure of published conclusions (the screen, all 5 caches)

- **kappathr_subhalo_jsd: CLEAN.** No detectable anchor shards; block floor ~=
  sample floor (pctl 51-90). The fixed-<N> default decision stands on clean data.
- **subhalo_factor_jsd: CLEAN.** Same (pctl 15-51). The factor=1e-2 default
  flip stands on clean data.
- **nz_convergence: essentially clean.** One small shard each in z0.2 truthB
  (-0.0006) and z1 nz50 (-0.002); conclusions unaffected.
- **mmin_convergence: mostly clean**, but (a) z10 nm200ctl shard 3 carries
  -0.035 => the z>=5 "NM confound" NUMBER is the single most exposed published
  value; (b) z5 m1e7 shard 1 carries -0.004 (mild) -- note this is the
  PRODUCTION-DEFAULT config, showing the mechanism is not confined to refined
  configs; (c) z5 m1e5 shard 5 similar.
- **mmin_pd_convergence: low resolution** (above); z1/z5 floors and excesses
  order-of-magnitude until regeneration.

Diagonal check: Delta_obs tracks -2*kappa_max/n across two decades
(-1e-4..-0.13) for every detectable shard, across studies -- one mechanism.
Additional screen artifact to be aware of: in groups containing one monster
shard, the OTHER shards show mirror +shifts vs siblings (the monster drags the
sibling median); these are reflections, not independent anomalies.

## Standing recommendations

1. Fix the anchor (robust mean excluding kappa>1; supervisor decision), then
   regenerate: both mmin_pd truths, mmin z10 nm200ctl, mmin z5 m1e5/m1e7,
   nz z0.2 truthB, nz z1 nz50, mmin z1_sub truthB (cheap: ~10 shards).
2. Until then, quote mmin_pd numbers as order-of-magnitude and flag the z>=5
   NM confound as provisional.
3. Future ensembles: shard-level resampling for floors (this report's block
   enumeration) and record kappa_max per shard at generation time.
