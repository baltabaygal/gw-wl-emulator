# Weak-arm MC: PDF-level effect of the clustered weak background

Prototype MC (kappa only, Nz=100, fixed-<N>=100 rule), 2x100000 realizations per arm per zs. base = shipped bias_model=1 (counts-only field); weakw/weakp = same draws + windowed/pencil clustered weak term; old = legacy iid layer. JSD in nats on 120 body bins (pooled-base quantile range 1e-4..1-1e-4) + under/overflow; floor = JSD(baseA, baseB) at N/2 vs N/2 (the pooled arm-vs-base comparisons have ~half that floor).

## 1. Body statistics (pooled seeds)

| zs | arm | Var(|k|<0.5) | dVar vs base | q99 | q99.9 | f(k>1) |
|--:|:--|--:|:--|--:|--:|--:|
| 0.2 | base | 1.3871e-05 | +0.0% | 0.0182 | 0.0433 | 0.00e+00 |
| 0.2 | weakw | 1.5418e-05 | +11.2% | 0.0185 | 0.0436 | 0.00e+00 |
| 0.2 | weakp | 2.2106e-05 | +59.4% | 0.0196 | 0.0443 | 0.00e+00 |
| 0.2 | old | 1.5402e-05 | +11.0% | 0.0186 | 0.0465 | 0.00e+00 |
| 1 | base | 8.2779e-04 | +0.0% | 0.1732 | 0.2964 | 5.00e-06 |
| 1 | weakw | 1.0159e-03 | +22.7% | 0.1776 | 0.2994 | 5.00e-06 |
| 1 | weakp | 1.4579e-03 | +76.1% | 0.1863 | 0.3046 | 5.00e-06 |
| 1 | old | 8.0151e-04 | -3.2% | 0.1711 | 0.3160 | 1.50e-05 |
| 5 | base | 4.7337e-03 | +0.0% | 0.6291 | 0.8814 | 4.10e-04 |
| 5 | weakw | 7.1762e-03 | +51.6% | 0.6570 | 0.9093 | 4.80e-04 |
| 5 | weakp | 9.7290e-03 | +105.5% | 0.6886 | 0.9358 | 6.00e-04 |
| 5 | old | 4.8958e-03 | +3.4% | 0.6417 | 0.9266 | 6.15e-04 |

## 2. JSD vs base (nats)

| zs | arm | JSD(pooled) | floor(baseA,baseB) | per-seed-half |
|--:|:--|--:|--:|:--|
| 0.2 | weakw | 1.965e-02 | 2.755e-04 | 1.940e-02 / 2.006e-02 |
| 0.2 | weakp | 9.292e-02 | 2.755e-04 | 9.195e-02 / 9.414e-02 |
| 0.2 | old | 1.192e-02 | 2.755e-04 | 1.215e-02 / 1.201e-02 |
| 1 | weakw | 1.576e-02 | 2.767e-04 | 1.590e-02 / 1.588e-02 |
| 1 | weakp | 5.759e-02 | 2.767e-04 | 5.763e-02 / 5.786e-02 |
| 1 | old | 1.687e-03 | 2.767e-04 | 1.715e-03 / 1.995e-03 |
| 5 | weakw | 1.917e-02 | 3.175e-04 | 1.933e-02 / 1.934e-02 |
| 5 | weakp | 4.582e-02 | 3.175e-04 | 4.597e-02 / 4.598e-02 |
| 5 | old | 3.456e-04 | 3.175e-04 | 6.048e-04 / 4.871e-04 |

## 3. Measured weak-arm variance budget vs analytic sizing

(S_ew measured on the body-clipped counts arm; analytic from bias_field_joint sizing.npz)

| zs | S_ww win MC/an | 2S_ew win MC/an | S_ww pen MC/an | 2S_ew pen MC/an |
|--:|:--|:--|:--|:--|
| 0.2 | 3.70e-07 / 3.72e-07 | 1.18e-06 / 1.21e-06 | 4.25e-06 / 4.27e-06 | 3.99e-06 / 8.32e-06 |
| 1 | 5.90e-05 / 5.90e-05 | 1.29e-04 / 1.29e-04 | 3.26e-04 / 3.26e-04 | 3.04e-04 / 4.64e-04 |
| 5 | 1.74e-03 / 1.74e-03 | 1.63e-03 / 2.01e-03 | 4.70e-03 / 4.69e-03 | 2.73e-03 / 4.49e-03 |
