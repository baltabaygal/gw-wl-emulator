# Phase 3D Control: sim-vs-sim N-scaling

Max JSD (ln 2) = 0.693147. Two simulator references at seeds 100/777, nsim=10000.
Catalogs are nested subsamples of one truth draw.

| N | sim-vs-sim JSD | sim-vs-sim TV | nsf-vs-sim JSD | nsf-vs-sim TV |
|---:|---:|---:|---:|---:|
| 250 | 0.344347 | 0.676816 | 0.625687 | 0.946558 |
| 1000 | 0.639278 | 0.968621 | 0.676534 | 0.991672 |
| 5000 | 0.185719 | 0.438566 | 0.693144 | 0.999999 |

If the sim-vs-sim column also rises toward ln 2 with N, the Phase 3D
scaling curve is a metric artifact of comparing over-concentrated posteriors,
not evidence of systematic NSF bias.

