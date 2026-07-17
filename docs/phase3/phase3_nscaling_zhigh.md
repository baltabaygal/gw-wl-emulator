# Phase 3D Control: sim-vs-sim N-scaling

Max JSD (ln 2) = 0.693147. Two simulator references at seeds 100/777, nsim=10000.
Catalogs are nested subsamples of one truth draw.

| N | sim-vs-sim JSD | sim-vs-sim TV | nsf-vs-sim JSD | nsf-vs-sim TV |
|---:|---:|---:|---:|---:|
| 1000 | 0.483522 | 0.822872 | 0.453204 | 0.756702 |
| 5000 | 0.693147 | 1.0 | 0.680693 | 0.996213 |

If the sim-vs-sim column also rises toward ln 2 with N, the Phase 3D
scaling curve is a metric artifact of comparing over-concentrated posteriors,
not evidence of systematic NSF bias.

