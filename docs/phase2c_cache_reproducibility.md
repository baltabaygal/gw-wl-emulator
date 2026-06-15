# Phase 2C — NSF Caching & Reproducibility Report

Generated on: 2026-06-15 19:04:00 UTC
Utility Implementation: [cache_utils.py](file:///Users/baltabay/Desktop/gw-wl-emulator/ml/cache_utils.py)
Unit Tests: [test_cache_utils.py](file:///Users/baltabay/Desktop/gw-wl-emulator/tests/test_cache_utils.py)

---

## 1. Motivation & Scope

Evaluating the C++ weak-lensing Monte Carlo simulator over grid configurations is computationally expensive, requiring ~200 seconds per redshift for a 20x20 parameter grid. To avoid wasting simulator execution time during iterative validations and failure mapping, we implement a lightweight, metadata-safe caching layer.

To prevent stale-cache scientific errors, the caching system follows two strict constraints:
1. **Cache Only Reference Objects**: We cache only the mock catalogs (raw samples) and simulator likelihood grids. We do NOT cache emulator (NSF or MLP) outputs, which must always be evaluated dynamically.
2. **Metadata-Safe Validation**: The cache key contains a cryptographic hash of all parameters and environment metadata. The cache manager performs full verification upon load and **fails loudly** (refuses to load) if any metadata mismatch is detected.

---

## 2. Cache Key Definition

The cache key is computed as a SHA256 hash of a serialized, sorted JSON dictionary. It includes the following fields:

| Field | Type | Description |
|---|---|---|
| `z` | `float` | Redshift |
| `h` | `float` | Hubble parameter |
| `OmegaM` | `float` | Matter density parameter $\Omega_M$ |
| `sigma8` | `float` | Amplitude of matter fluctuations $\sigma_8$ |
| `nsamples` | `int` | Number of mock catalog samples ($N = 10,000$) |
| `seed` | `int` | Random seed used for generation |
| `catalog_hash` | `str` | SHA256 hash of the catalog samples (for grids) |
| `simulator_git_commit` | `str` | Current Git commit hash of the emulator codebase |
| `likelihood_version` | `str` | Likelihood formulation version (e.g., `v1`) |
| `grid_definition` | `dict` | Details of the 2D evaluation grid (min, max, steps) |

If any of these fields change (e.g., if the simulator C++ code is updated, modifying the git commit), the cache key will hash differently, ensuring a cache miss and forcing simulator regeneration.

---

## 3. Loud Failure Safeguard

If a cached `.npz` file exists on disk with a filename corresponding to the requested parameters, but the internal metadata saved inside the `.npz` file does not match the expected parameters exactly, the loader will raise a `ValueError` rather than silently loading bad data. 

This prevents silent data corruption or stale cache lookups if cache files are manually renamed or mismatched.

```python
# Mismatch verification snippet in cache_utils.py
for k, val in cleaned_expected.items():
    if k not in metadata:
        raise ValueError(f"Cache metadata key '{k}' missing from cached file.")
    cached_val = metadata[k]
    if cached_val != val:
        raise ValueError(
            f"Cache metadata mismatch for key '{k}': "
            f"Expected {val}, found {cached_val} in cache."
        )
```

---

## 4. Command Line Interface

The validation and failure mapping scripts expose two CLI flags:
- `--use_cache`: Attempts to load generated mock catalogs and simulator grids from `data/cache/` using the CacheManager.
- `--overwrite_cache`: Ignores existing cache files and forces fresh runs of the C++ simulator, overwriting cached files on disk.

---

## 5. Verification & Tests

The unit test suite `tests/test_cache_utils.py` verifies:
1. **Save/Load Integrity**: Verifies that arrays (mock catalogs, grids) and metadata are correctly serialized to and deserialized from `.npz` files.
2. **Metadata Mismatch Handling**: Manually mocks a cached file with altered metadata and verifies that `CacheManager.load()` raises a `ValueError`.
3. **Environment Fetching**: Verifies that the Git commit hash is successfully retrieved via subprocess.
