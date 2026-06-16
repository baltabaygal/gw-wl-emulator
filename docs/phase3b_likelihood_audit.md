# Phase 3B Likelihood Construction Audit

Verdict: **FAIL**

Both benchmark paths sum catalog log-likelihood contributions, but two implementation details are not identical: simulator-side histogram clamping and the `1e-12` zero-probability floor.

## Audit Matrix

| Topic | Status | Impact |
|---|---|---|
| event_ordering | pass | Ordering does not affect either sum. |
| redshift_handling | conditional_pass | The smoke catalog uses exact support z={0.5,1.5,2.5}, so rounding is not active there; this would be unsafe for continuous-z catalogs. |
| prior_truncation | pass | No prior-bound mismatch for grid posterior benchmarks. |
| normalization_constants | conditional_pass | The bin-width constant should not move a posterior on fixed bins, but simulator and NSF likelihood values are not absolute-density comparable. |
| zero_probability_handling | fail | The simulator has an explicit low-probability floor and the NSF does not; tail events can receive systematically different leverage. |
| clamping | fail | Out-of-range or edge-near events are treated differently by the two likelihoods. |
| per_event_accumulation | conditional_pass | Aggregation is a sum in both paths; the density approximation and tail handling differ. |
| event_weighting | pass | No explicit weighting mismatch found. |
| catalog_slicing | pass | For fixed discrete redshifts, both implement the same catalog membership. |

## Cache and Grid Integrity

- Catalog hash matches grid metadata: `True`
- Checkpoint hash matches file: `True`
- Preprocessing stats hash matches file: `True`
- Catalog finite: `True`

## Interpretation

The catalog-level sum is consistent, but the simulator reference applies histogram clamping and a 1e-12 tail floor while the NSF evaluates an unclamped continuous density. Phase 3B should therefore treat the smoke failure as a real density/tail-handling mismatch unless single-z isolation points to a bookkeeping-only issue.
