# Statistical Stability and Dataset Validation Report

Generated on: 2026-06-05 13:03:48 UTC
Dataset directory: `datasets_tiny`
Validation threshold (`DEFAULT_MIN_VALID_FRACTION`): **0.50**
Reproducibility status: **PASSED**

## 1. Split Summary

| Split | Configurations | Samples/Config | Total Valid | Total NaN | Valid Fraction | Pass/Fail |
|---|---:|---:|---:|---:|---:|---|
| train | 6 | 1000 | 3938 | 2062 | 0.6563 | PASS |
| validation | 4 | 1000 | 2404 | 1596 | 0.6010 | PASS |
| test | 5 | 1000 | 3826 | 1174 | 0.7652 | PASS |

## 2. Parameter Space Coverage (Measured Ranges)

| Split | z Range | h Range | OmegaM Range | sigma8 Range |
|---|---|---|---|---|
| train | [1.1384, 9.9865] | [0.6431, 0.7272] | [0.2682, 0.3498] | [0.6559, 0.9488] |
| validation | [2.0774, 8.1667] | [0.6109, 0.7077] | [0.1717, 0.4299] | [0.5979, 1.1670] |
| test | [0.9166, 8.7874] | [0.6020, 0.7224] | [0.2240, 0.4370] | [0.7394, 1.3663] |

## 3. Split Configuration Types

| Split | Train Configs | Interpolation Configs | OoD Configs |
|---|---|---|---|
| train | 6 | 0 | 0 |
| validation | 0 | 2 | 2 |
| test | 0 | 3 | 2 |

## 4. Metadata Provenance

### Train Split
- **Git Commit**: `cf5b73ee43b23f525be7ebead75a7493d3250a48`
- **Git Branch**: `phase_1-4665179221458693490`
- **Dataset Schema Version**: `1.1`
- **Generation Seed**: `1123`
- **Generation Timestamp**: `2026-06-05 12:58:25`

### Validation Split
- **Git Commit**: `cf5b73ee43b23f525be7ebead75a7493d3250a48`
- **Git Branch**: `phase_1-4665179221458693490`
- **Dataset Schema Version**: `1.1`
- **Generation Seed**: `2123`
- **Generation Timestamp**: `2026-06-05 12:58:34`

### Test Split
- **Git Commit**: `cf5b73ee43b23f525be7ebead75a7493d3250a48`
- **Git Branch**: `phase_1-4665179221458693490`
- **Dataset Schema Version**: `1.1`
- **Generation Seed**: `3123`
- **Generation Timestamp**: `2026-06-05 12:58:45`

