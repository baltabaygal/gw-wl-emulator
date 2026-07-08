# Phase 3 Cache Refresh

Refresh timestamp: 2026-06-15

Current simulator checkout:

`59c107efc320264b2a6d91bf31a89e62760b5ca7`

## Stale Cache Inspection

Three active simulator-grid cache files were found in `data/cache/` with embedded simulator metadata from the previous checkout:

- `sim_grid_0a9f08ef4b03800542435c4c5f73cf80454d96a9c2676f748fb45148e4a07395.npz`
- `sim_grid_54ed7712c62c5fdd7d44eac1a5b1d05c62e3f316aaab659d73814e42d4446c27.npz`
- `sim_grid_b9bd4b80e0f46340339cfcde7b601b0492ddf5411b337484464b7472465abdaa.npz`

Their embedded simulator commit was:

`f8c837cd4177133d8c5c390f90be2ffb929e2bbe`

This mismatch was the expected full-suite failure and confirms that metadata-safe cache validation is working.

## Invalidation Action

The stale files were quarantined under:

`data/cache/stale_phase2c_f8c837/`

No cache validation rules were loosened. Phase 3 grid caches are written under `data/results/phase3_posterior_grids/` with keys that include catalog hash, grid definition, prior bounds, simulator git commit, likelihood version, NSF checkpoint hash, and preprocessing stats hash.

## Safety Checks

- Simulator commit used for new Phase 3 caches must match `59c107efc320264b2a6d91bf31a89e62760b5ca7`.
- NSF cache keys include `nsf_checkpoint_hash`.
- NSF cache keys include `preprocessing_stats_hash`.
- Stale metadata loads continue to fail loudly through `CacheManager`.

