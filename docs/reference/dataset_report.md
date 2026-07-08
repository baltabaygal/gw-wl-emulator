# Dataset Report

Dataset directory: `datasets`

## Split Summary

| Split | Points | Samples/Point | Total Valid | Valid Fraction | NaN Fraction |
|---|---:|---:|---:|---:|---:|
| train | 10 | 1000 | 6715 | 0.6715 | 0.3285 |
| validation | 5 | 1000 | 4669 | 0.9338 | 0.0662 |

## Train Details

### Parameter Ranges
- z: [1.8945, 9.7806]
- h: [0.5930, 0.7349]
- OmegaM: [0.1600, 0.4655]
- sigma8: [0.5577, 1.3543]

### Valid Count Stats
- min valid_count: 250
- max valid_count: 988

### Normalization Metadata
- lnmu_mean: 0.006722
- lnmu_std: 0.204986
- lnmu_min: -0.698013
- lnmu_max: 5.062414

### Resource Usage
- generation_time_sec: 25.402
- throughput_samples_per_sec: 264.354
- mem_before_mb: 111.952
- mem_after_mb: 124.043
- peak_memory_proxy_mb: 124.043
- cpu_count: 10

### Wasserstein Smoothness (Adjacent in sigma8 order)
- mean distance: 0.046909
- std distance: 0.029836

## Validation Details

### Parameter Ranges
- z: [0.3023, 7.8399]
- h: [0.6626, 0.7554]
- OmegaM: [0.1937, 0.4261]
- sigma8: [0.4289, 1.3354]

### Valid Count Stats
- min valid_count: 748
- max valid_count: 999

### Normalization Metadata
- lnmu_mean: 0.006162
- lnmu_std: 0.200912
- lnmu_min: -1.651133
- lnmu_max: 2.653586

### Resource Usage
- generation_time_sec: 13.448
- throughput_samples_per_sec: 347.183
- mem_before_mb: 124.223
- mem_after_mb: 54.395
- peak_memory_proxy_mb: 124.223
- cpu_count: 10

### Wasserstein Smoothness (Adjacent in sigma8 order)
- mean distance: 0.102974
- std distance: 0.057393

## Regression Test Summary
- Physics regression status: **PASS**

```text
.......                                                                  [100%]
7 passed in 47.02s
```
