# Cascade correction — mode (SVD) analysis [lowz]

M_i[θ,lnμ] = logP_base+i − logP_base on body+shoulder (edge MASKED, kept as separate scalar; far tail >q99 cut). Light Hann-smoothed; MC noise floor from the multi-seed block (95th pct). Sym: +1 symmetric/width, −1 antisym/skew.

## A. Per-ingredient SVD — modes above the noise floor (UNCENTERED)

| z_s | ing | S1 | S2 | S3 | floor(S1..) | n>floor | PC1 sym | PC1·H2 | PC1·H3 |
|----|----|----|----|----|----|----|----|----|----|
| 0.5 | sub | 4.060 | 1.328 | 1.274 | 1.515 | 1 | +0.09 | 0.66 | 0.56 |
| 0.5 | bias | 4.970 | 1.804 | 1.256 | 1.518 | 2 | +0.03 | 0.46 | 0.35 |
| 1.0 | sub | 4.495 | 1.375 | 1.201 | 1.390 | 1 | +0.08 | 0.76 | 0.63 |
| 1.0 | bias | 2.881 | 1.463 | 1.097 | 1.402 | 2 | -0.19 | 0.45 | 0.30 |
| 5.0 | sub | 6.132 | 1.206 | 1.120 | 1.513 | 1 | +0.09 | 0.78 | 0.64 |
| 5.0 | bias | 4.357 | 1.295 | 1.128 | 1.390 | 1 | +0.35 | 0.84 | 0.75 |

## B. Shared vs unique deformation (the honest orthogonality test)

u_i = unit MEAN-ΔlogP shape of ingredient i. cos(u_sub,u_bias) = how parallel the leading corrections are (both broaden ⇒ expected high). The real H2 test: remove the shared subhalo direction from clustering, u_bias⊥ = u_bias − (u_bias·u_sub)u_sub; report ‖u_bias⊥‖ (fraction of clustering NOT shared with subhalos) and cos(u_bias⊥, H3) (is that unique part SKEW). Symmetrically u_sub⊥ vs H2. noise‖ = MC ‖·‖ of a noise-only mean shape (floor for the residual norms).

| z_s | cos(u_sub,u_bias) | ‖u_bias⊥‖ | cos(u_bias⊥,H3) | ‖u_sub⊥‖ | cos(u_sub⊥,H2) | noise‖ |
|----|----|----|----|----|----|----|
| 0.5 | +0.903 | 0.429 | 0.27 | 0.429 | 0.46 | 0.139 |
| 1.0 | +0.869 | 0.494 | 0.45 | 0.494 | 0.69 | 0.223 |
| 5.0 | -0.889 | 0.457 | 0.37 | 0.457 | 0.10 | 0.149 |

## C. SVD-vs-parametric (H3): PC1·H2 (width) and PC1·H3 (skew) — see table A.

