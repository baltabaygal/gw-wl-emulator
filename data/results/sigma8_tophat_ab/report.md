# sigma8 amplitude convention: "option b" (2026-07-30)

User decision. `cosmology::sigma8_tophat` (default **true**) anchors the P(k)
amplitude with the **real-space top-hat at 8 Mpc/h** -- the standard definition of
sigma8 -- instead of inverting the smooth-k excursion-set filter `Ws` at M8.

## Why

The engine took the input number `sigma8` and solved `deltaH8 = sigma8/sigmaC(M8,1)`,
where `sigmaC` integrates against `Ws(x) = 1/(1+(0.43x)^6)`. So "sigma8 = 0.811"
described a universe whose *conventional* (top-hat) sigma8 was **0.7786**:

| | value |
|---|---|
| input label | 0.811 |
| true top-hat sigma8 of that universe | 0.778574 |
| sigma offset | **+4.17%** |
| P(k) amplitude offset | **+8.50%** |

That made our sigma8 incomparable to Planck's, to ACE-Lensing's, and to any reader's
expectation -- on the one parameter the paper is named for. It also dwarfed every
physics effect certified this month (the subhalo radial-profile fix was +1.5%).

⚠ Vaskonen (2026) sec. 2 states a real-space top-hat in the TEXT, and the `halos`
code even defines `W(x) = 3(x cos x - sin x)/x^3` -- but never calls it (dead code in
both repos). So there is no version in which the top-hat was live; his text and code
have always disagreed. This change follows his text for the anchor only.

## Scope: the anchor ONLY

`sigma_M(M)` for the HMF, the collapse barrier and the halo bias keeps `Ws`. Those
`(p,q) = (0.3, 0.8)` come from random-walk first-crossing fits (Vaskonen fn. 3),
which require a Markovian filter that a real-space top-hat does not provide. The
paper's literal "top-hat everywhere" would break that calibration, so it is
deliberately NOT implemented. New `cosmology::sigmaTH` exists solely for the anchor.

## Verification

* Round trips exact in BOTH conventions: `sigma8_tophat_derived == 0.811` (new) and
  `sigma8_derived == 0.811` (legacy), each to 1e-12.
* `deltaH8` ratio new/legacy = **1.041648** (+4.16%), squared **1.085030** (+8.50%) --
  matches the independent estimate in `tmp/engine_checkpoints.txt` to 2 digits.
* **Attribution proof:** with the change in place, explicit `sigma8_tophat=false`
  reproduces `tests/data/reference_lnmu_pre_paper_defaults.npz` **bit-for-bit at all
  three points**, so only the amplitude anchor moved. The default reference was
  re-baselined on that basis (third re-baseline; protocol in
  `tests/capture_reference_lnmu.py`).
* `tests/test_sigma8_tophat.py` -- 13 tests: default, round trips, the filter ratio,
  legacy bitwise, dead-flag guard, As-mode inertness, all five entry points.
* ⚠ `sigma8_tophat` was added to **both** `PRODUCTION_CONFIG` and `LEGACY_CONFIG`.
  Without it in LEGACY_CONFIG, the legacy dict silently ran old physics at the NEW
  amplitude the moment the default moved -- the same class of bug as an
  "inherit the defaults" empty dict.
* Two As-round-trip tests broke for real reasons and were fixed, not relaxed:
  the "conventional" sigma8 in As-mode is now `sigma8_tophat_derived`; and the legacy
  arm must re-derive `As` under `sigma8_tophat=False` or it compares two amplitudes.

## Physics effect (clipped sd(lnmu), |lnmu| <= 1, 8 shards x 25k rays/arm)

Both arms at full `PRODUCTION_CONFIG`; SEM is the ACROSS-SHARD spread, never
`1/sqrt(2N)`.

| z_s | legacy anchor | top-hat anchor | change | sigma |
|---|---|---|---|---|
| 0.5 | 0.033885 | 0.035052 | **+3.44% +/- 0.75%** | 4.6 (under the 5-sigma bar) |
| 1 | 0.073463 | 0.077693 | **+5.76% +/- 0.50%** | 11.6 RESOLVED |
| 5 | 0.217527 | 0.225638 | **+3.73% +/- 0.30%** | 12.4 RESOLVED |

This is **first-order** -- comparable to the entire legacy->paper composition shift
(+5.96/+5.16/+1.79%). Consistency check against the independent scaling study
(`data/results/sigma8_shape/`: sd(lnmu) ~ sigma8^0.96 at z_s=1, ^0.93 at z_s=5): a
+4.17% effective amplitude predicts ~+3.9-4.0%, matching at z_s=5 (3.73 vs 3.87) and
z_s=0.5 (3.44 vs ~4.0). **z_s=1 runs high (5.76 vs 4.0) and was not chased** -- do not
present the scaling law as a quantitative prediction of this shift.
z_s=0.5 sits at 4.6 sigma, just under the repo's 5-sigma bar; the effect is
nonetheless unambiguous from the other two redshifts and from the deterministic
+8.50% in P(k). Shards in this directory.

## Consequences

* **Every pre-2026-07-30 result is at the old amplitude.** Any stored reference,
  cache, or figure must be regenerated, not compared across the switch.
* **Emulator training data must be generated after this change** -- it moves every ray.
* Ville's published sigma8 number is redefined by this. Flag it to him with the
  bundle; `sigma8_tophat=false` reverts exactly.
