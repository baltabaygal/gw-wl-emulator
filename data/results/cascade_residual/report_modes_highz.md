# Cascade correction — mode (SVD) analysis [highz]

M_i[θ,lnμ] = logP_base+i − logP_base on body+shoulder (edge MASKED, kept as separate scalar; far tail >q99 cut). Light Hann-smoothed; MC noise floor from the multi-seed block (95th pct). Sym: +1 symmetric/width, −1 antisym/skew.

## A. Per-ingredient SVD — modes above the noise floor (UNCENTERED)

| z_s | ing | S1 | S2 | S3 | floor(S1..) | n>floor | PC1 sym | PC1·H2 | PC1·H3 |
|----|----|----|----|----|----|----|----|----|----|
| 6.0 | sub | 6.143 | 1.267 | 1.147 | 1.271 | 1 | +0.12 | 0.75 | 0.62 |
| 6.0 | bias | 5.327 | 1.423 | 1.232 | 1.287 | 2 | +0.32 | 0.86 | 0.77 |
| 8.0 | sub | 6.900 | 1.241 | 1.170 | 1.408 | 1 | +0.06 | 0.78 | 0.65 |
| 10.0 | sub | 7.148 | 1.191 | 1.177 | 1.347 | 1 | +0.17 | 0.80 | 0.68 |

## B. Shared vs unique deformation (the honest orthogonality test)

u_i = unit MEAN-ΔlogP shape of ingredient i. cos(u_sub,u_bias) = how parallel the leading corrections are (both broaden ⇒ expected high). The real H2 test: remove the shared subhalo direction from clustering, u_bias⊥ = u_bias − (u_bias·u_sub)u_sub; report ‖u_bias⊥‖ (fraction of clustering NOT shared with subhalos) and cos(u_bias⊥, H3) (is that unique part SKEW). Symmetrically u_sub⊥ vs H2. noise‖ = MC ‖·‖ of a noise-only mean shape (floor for the residual norms).

| z_s | cos(u_sub,u_bias) | ‖u_bias⊥‖ | cos(u_bias⊥,H3) | ‖u_sub⊥‖ | cos(u_sub⊥,H2) | noise‖ |
|----|----|----|----|----|----|----|
| 6.0 | -0.893 | 0.451 | 0.50 | 0.451 | 0.05 | 0.114 |

## C. SVD-vs-parametric (H3): PC1·H2 (width) and PC1·H3 (skew) — see table A.

