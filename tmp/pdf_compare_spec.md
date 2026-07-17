# PDF comparison — flat vs adaptive κ_thr (spec)

## What we're testing and why
The threshold κ_thr is a **pure computational split**: halos with κ > κ_thr are drawn
explicitly (non-Gaussian, exact), everything below is folded into the Gaussian weak-lensing
field. By Campbell's theorem the **mean and variance of total κ are threshold-independent**
(already proven analytically + measured, `plots/sigma_partition_vs_kthr.png`). So switching
the rule *cannot* move ⟨μ⟩ or σ_κ. The ONLY thing a threshold change can do is alter how much
sub-threshold **non-Gaussianity** (skew/kurtosis) gets discarded by Gaussianization.

Goal of the run: confirm that **flat κ_thr = 1e-4** produces a magnification PDF statistically
indistinguishable from the old adaptive ⟨N⟩=100 rule — i.e. the switch is science-neutral,
especially in the high-μ tail that carries the GW distance information.

## Configs to generate (subhalo OFF — isolates the threshold variable)
For each z_s ∈ {0.2, 1.0, 10.0}, generate lnμ samples under three rules:
| label     | kwarg                | note |
|-----------|----------------------|------|
| adaptive  | `kappathr_flat=-1.0` | legacy ⟨N⟩=100 rule (bit-identical to pre-change) |
| flat 1e-4 | `kappathr_flat=1e-4` | the chosen value |
| flat 1e-3 | `kappathr_flat=1e-3` | sensitivity check (older choice) |

Use **large N** (≥ 2e5, ideally 1e6 at z_s=0.2/1 where it's cheap) so tail quantiles converge.
`subhalo=False`, defaults otherwise, `seed=123`.

**Do NOT expect seed-for-seed identity** — changing κ_thr changes how many explicit halos
are drawn, which shifts the RNG stream. Compare as **ensembles/distributions only**.

## Metrics (per z_s)
1. **⟨1/μ⟩** — must be 1.000 ± few×1e-3 under every rule (flux conservation; sanity).
2. **σ_κ** (or σ of lnμ) — must agree across rules to < ~0.5% (Campbell prediction).
3. **KL divergence** of the μ (or lnμ) histogram, flat-1e-4 vs adaptive — expect ≪ the
   model's own KL floor (~0.007). Anything > ~1e-3 is worth a look.
4. **Tail quantiles** of μ: 99, 99.9, 99.99 percentiles — the science-bearing tail. These
   should match within Monte-Carlo error, because the tail is built from *explicit* halos
   under every rule.

## Pass criterion
flat 1e-4 vs adaptive: KL ≲ 1e-3, tail quantiles overlap within MC error, ⟨1/μ⟩ and σ_κ
match. If so → flat 1e-4 is confirmed faithful; lock it in. If the low-z_s core (|κ|≲1e-3)
shows a small KL bump, that's the expected dropped-skew effect (predicted ~0.4% at z_s=0.2,
sub-percent) — irrelevant to μ tail, still a pass.

## Optional spot check (subhalo ON)
One run at z_s=1, `subhalo=True, subhalo_model=3`, flat 1e-4 vs adaptive, N~3e4. Slower (~5
min each). Confirms the Wsub model-3 term (anchored to a z_s-independent κ_thr) behaves under
both rules. Not required for the threshold decision, but good belt-and-suspenders.
