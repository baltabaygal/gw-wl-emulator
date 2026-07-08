# Phase 3 Max-N Characterization

Model: `data/models/conditional_nsf_backend_current.pt`  |  converged sim nsim=40000  |  16 catalog realizations averaged  |  z slices [0.5, 1.5, 2.5]  |  grid 2d res 12

Systematic bias = |mean over realizations of (NSF mean - converged-sim mean)| (realization noise averaged out).
Width = mean simulator posterior std. Emulator bias is sub-dominant while ratio < 1; it dominates once ratio >= 1.

| N | OmegaM syst-bias | OmegaM width | OmegaM ratio | sigma8 syst-bias | sigma8 width | sigma8 ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 150 | 0.0016 | 0.0422 | 0.04 | 0.0014 | 0.0999 | 0.01 |
| 300 | 0.0022 | 0.0383 | 0.06 | 0.0025 | 0.0938 | 0.03 |
| 600 | 0.005 | 0.0315 | 0.16 | 0.0066 | 0.0776 | 0.09 |
| 1200 | 0.0039 | 0.0262 | 0.15 | 0.0006 | 0.0613 | 0.01 |
| 2400 | 0.013 | 0.0197 | 0.66 | 0.0356 | 0.0475 | 0.75 |
| 4800 | 0.0164 | 0.0123 | 1.33 | 0.0451 | 0.0307 | 1.47 |
| 9600 | 0.0124 | 0.0105 | 1.18 | 0.0387 | 0.0256 | 1.51 |

## Crossover (max usable N before emulator bias persistently dominates statistical width)
- OmegaM: 4800
- sigma8: 4800

Below the crossover, the emulator is safe for inference (systematic bias < statistical width).

