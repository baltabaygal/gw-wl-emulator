# Phase 3B Mixed-Catalog Posterior Mismatch Diagnosis

## Technical Summary

The mixed-catalog smoke failure is **not primarily a mixed-redshift aggregation bug or a coarse-grid artifact**. The same posterior disagreement appears in fixed-redshift catalogs and in each redshift slice of the smoke catalog. Grid refinement from 12x12 to 24x24 does not recover overlap; it makes the mixed posterior disagreement essentially maximal.

The strongest diagnosis is that the NSF density model and the simulator reference likelihood are not posterior-compatible in the current likelihood formulation. The low-redshift regime is explicitly bad and contributes the same low-OmegaM/high-sigma8 direction seen in the mixed smoke NSF mode, but mid- and high-redshift slices also fail. The failure is therefore deeper than one isolated low-z leak.

**Final verdict: Option C.** The NSF likelihood formulation is not compatible with posterior-level inference in its current form. Revisit the likelihood aggregation or emulator target before rerunning Phase 3 smoke or any 20-catalog coverage study.

## Smoke Failure Recap

The central mixed-redshift smoke catalog failed with:

- JSD: `0.607845`
- Total variation: `0.947312`
- 68% overlap: `0.000`
- 95% overlap: `0.167`
- Normalized shifts: OmegaM `8.027`, sigma8 `7.788`

The simulator posterior favored higher OmegaM and lower sigma8 than the NSF posterior, while the NSF mode moved toward low OmegaM and high sigma8.

## Likelihood Audit Results

The audit found no event ordering, event weighting, prior-bound, or catalog-slicing mismatch for the discrete smoke catalog. Both paths sum catalog contributions over the same events.

Two likelihood-construction differences remain material:

- The simulator path evaluates a histogram approximation with clamping into the fixed bin range.
- The simulator path applies a `1e-12` zero-probability floor, while the NSF path evaluates unclamped continuous flow density and raises on non-finite values.

This means the catalog-level rule is sum-like in both paths, but tail and edge events do not receive identical leverage.

## Single-z Isolation Results

Fixed-redshift catalogs fail before redshift mixing:

| z | JSD | TV | 68% overlap | 95% overlap | MAP offset (OmegaM, sigma8) |
|---:|---:|---:|---:|---:|---|
| 0.5 | 0.693147 | 1.000000 | 0.000 | 0.000 | (-0.1091, +0.2182) |
| 1.5 | 0.663012 | 0.987604 | 0.000 | 0.059 | (+0.0545, -0.1818) |
| 2.5 | 0.688028 | 0.998655 | 0.000 | 0.000 | (+0.0545, -0.1091) |

This rules out “mixed catalog only” as the primary cause.

## Grid Refinement Results

The central mixed-catalog posterior was rerun on a 24x24 grid.

| Grid | JSD | TV | 68% overlap | 95% overlap | MAP offset (OmegaM, sigma8) |
|---|---:|---:|---:|---:|---|
| 12x12 smoke | 0.607845 | 0.947312 | 0.000 | 0.167 | (-0.1091, +0.2909) |
| 24x24 refined | 0.693146 | 1.000000 | 0.000 | 0.000 | (-0.1826, +0.3130) |

Refinement does not fix the discrepancy. The mismatch is real at posterior level, not a coarse-grid artifact.

## Redshift-Slice Decomposition

Each smoke-catalog redshift slice fails independently:

| Slice | z | JSD | TV | 68% overlap | 95% overlap | MAP offset (OmegaM, sigma8) |
|---|---:|---:|---:|---:|---:|---|
| low_z | 0.5 | 0.645227 | 0.980663 | 0.000 | 0.059 | (-0.1273, +0.2182) |
| mid_z | 1.5 | 0.687678 | 0.998553 | 0.000 | 0.000 | (+0.0182, -0.1091) |
| high_z | 2.5 | 0.583010 | 0.941186 | 0.056 | 0.054 | (+0.0545, -0.1455) |

Low redshift is explicitly bad and shifts in the same direction as the full mixed smoke posterior. However, mid- and high-z are also not posterior-compatible, so a low-z-only repair is unlikely to be sufficient.

## Local Density Residual Analysis

At truth, the simulator smoke MAP, and the NSF smoke MAP, NSF event log densities are systematically higher than simulator histogram event log-probabilities for all events. The positive offset is partly the expected density-versus-bin-mass constant, but it also changes with theta:

| Theta point | Sum residual | Mean residual | Positive fraction |
|---|---:|---:|---:|
| truth | 5102.579 | 5.1026 | 1.000 |
| sim_smoke_map | 5148.754 | 5.1488 | 1.000 |
| nsf_smoke_map | 5492.187 | 5.4922 | 1.000 |

The theta-dependent residual is largest at the NSF smoke mode, consistent with small density-shape errors accumulating into a large catalog posterior shift.

## Cache and Reproducibility Status

Cache safety remained strict:

- Smoke catalog commit matches the current simulator commit.
- Smoke grid commit matches the current simulator commit.
- Catalog hash matches grid metadata.
- NSF checkpoint hash and preprocessing-stats hash are present and match current files.
- Checked catalog and grid arrays contain no NaNs or Infs.

No stale-cache workaround, prior change, simulator physics change, or coverage-study run was introduced.

## QA Summary

Added Phase 3B regression tests:

- `tests/test_phase3b_likelihood_audit.py`
- `tests/test_phase3b_single_z.py`
- `tests/test_phase3b_grid_refinement.py`
- `tests/test_phase3b_redshift_decomposition.py`
- `tests/test_phase3b_local_density.py`

The targeted Phase 3B tests pass. The full `tests/` command is recorded in `docs/phase3b_qa_report.md`.

## Recommendation

Do not clear the NSF for production posterior replacement and do not run the 20-catalog coverage study yet.

Recommended next step:

1. Define a posterior-compatible reference likelihood target: either compare NSF to a continuous simulator density estimate with matched units and tail treatment, or train/evaluate the NSF against the same binned/clamped/floored likelihood target used by the simulator posterior reference.
2. Add a calibrated low-z stress repair, but treat it as necessary rather than sufficient.
3. Rerun Phase 3 smoke only after the likelihood target and tail handling are made identical enough for posterior inference.
