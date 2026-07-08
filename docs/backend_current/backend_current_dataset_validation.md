# Statistical Stability and Dataset Validation Report

Generated on: 2026-06-15 11:41:17 UTC
Dataset directory: `datasets_backend_current_1k`
Validation threshold (`DEFAULT_MIN_VALID_FRACTION`): **0.50**
Reproducibility status: **PASSED**

## 1. Split Summary

| Split | Configurations | Samples/Config | Total Valid | Total NaN | Valid Fraction | Pass/Fail |
|---|---:|---:|---:|---:|---:|---|
| train | 544 | 10000 | 4555180 | 884820 | 0.8373 | PASS |
| validation | 457 | 10000 | 3669229 | 900771 | 0.8029 | PASS |
| test | 458 | 10000 | 3806204 | 773796 | 0.8310 | PASS |

## 2. Parameter Space Coverage (Measured Ranges)

| Split | z Range | h Range | OmegaM Range | sigma8 Range |
|---|---|---|---|---|
| train | [0.0562, 9.9786] | [0.5918, 0.7599] | [0.2008, 0.3999] | [0.6504, 1.0495] |
| validation | [0.0296, 9.9919] | [0.5902, 0.7597] | [0.1505, 0.4499] | [0.4005, 1.3968] |
| test | [0.0239, 9.9082] | [0.5900, 0.7594] | [0.1507, 0.4491] | [0.4060, 1.3993] |

## 3. Split Configuration Types

| Split | Train Configs | Interpolation Configs | OoD Configs |
|---|---|---|---|
| train | 544 | 0 | 0 |
| validation | 0 | 263 | 194 |
| test | 0 | 264 | 194 |

## 4. Metadata Provenance

### Train Split
- **Git Commit**: `cf5b73ee43b23f525be7ebead75a7493d3250a48`
- **Git Branch**: `phase_1-4665179221458693490`
- **Dataset Schema Version**: `1.1`
- **Generation Seed**: `1123`
- **Generation Timestamp**: `2026-06-15 11:17:56`

### Validation Split
- **Git Commit**: `cf5b73ee43b23f525be7ebead75a7493d3250a48`
- **Git Branch**: `phase_1-4665179221458693490`
- **Dataset Schema Version**: `1.1`
- **Generation Seed**: `2123`
- **Generation Timestamp**: `2026-06-15 11:22:31`

### Test Split
- **Git Commit**: `cf5b73ee43b23f525be7ebead75a7493d3250a48`
- **Git Branch**: `phase_1-4665179221458693490`
- **Dataset Schema Version**: `1.1`
- **Generation Seed**: `3123`
- **Generation Timestamp**: `2026-06-15 11:27:04`

