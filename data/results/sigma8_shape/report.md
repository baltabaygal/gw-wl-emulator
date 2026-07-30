# Where does sigma_8 move dP_I/dmu?  (2026-07-29)

Run to check a sentence before it went into Sec. II.B of the paper. **The sentence
was false for our model and has been rewritten.**

Driver `paper_prod/scripts/check_sigma8_dependence.py`, data `sigma8_shape.json`.
sigma_8 varied ALONE (0.65 / 0.811 / 1.05) at the fiducial `OmegaM=0.315, h=0.674`,
full `PRODUCTION_CONFIG` (hash `0d50caf91c75`), 2e5 realizations per point, seed 1,
`z_s = 1` and `5`.

## The claim under test

> "at fixed source redshift its shape depends mainly on sigma_8, whose imprint sits
> almost entirely on the low-magnification side while the region mu > 1 is left
> nearly unchanged [Premadi:2001]"

## Result — the mu > 1 side is the MOST sigma_8-sensitive part, not the least

Power-law index fitted over the three sigma_8 points:

| quantity | z_s = 1 | z_s = 5 |
|---|---|---|
| `sd(ln mu)`, clipped to mu <= 3 | sigma_8^0.96 | sigma_8^0.93 |
| low-mu edge, abs(0.1% quantile of ln mu) | sigma_8^0.87 | sigma_8^0.92 |
| `P(mu > 1.5)` | sigma_8^3.30 | sigma_8^2.43 |
| `P(mu > 2)` | sigma_8^3.88 | sigma_8^3.45 |

Concretely at z_s = 1, going sigma_8 0.811 -> 1.05: the 0.1% edge moves
-0.1229 -> -0.1536 in ln mu (25%), while `P(mu > 2)` goes 5.1e-4 -> 1.69e-3, i.e.
**3.3x**. Count-weighted mean `dP_I/dmu` ratio over 1.5 < mu < 3 is **1.84**,
against 1.02 over 0.7 < mu < 1.0. Going down to sigma_8 = 0.65 halves the tail
(ratio 0.53) and again barely touches the low side (0.99).

So the low-magnification EDGE POSITION does move with sigma_8, which is the half of
Premadi et al. that we reproduce and still cite. Their other half -- that mu > 1 is
nearly unchanged -- does not hold here. Plausible reason: our lens population carries
substructure and filaments that Premadi's ray tracing does not, and the tail is built
from exactly those small, dense objects, whose abundance is steeply sigma_8-dependent
through the HMF. Not chased further; the paper does not need it.

## Knock-on: a recorded number in CLAUDE.md is off by ~2x

CLAUDE.md sec. 15 converts the subhalo profile fix's `+1.54%` on `sigma(ln mu)` into
"~0.75% in sigma_8", which assumes `sigma(ln mu) ~ sigma_8^2`. Measured here the
width goes as **sigma_8^0.93-0.96**, i.e. nearly LINEAR, so the same `+1.54%` is
**~1.6% in sigma_8** -- about twice the recorded sensitivity. The direction of the
argument is unchanged (state the shift, do not absorb it) but it is twice as large
as recorded.

## Caveats

- Single seed per point, 2e5 rays. The width exponent is safe -- `sd` changes 58%
  across the sigma_8 range, far above any noise. The tail exponents rest on 102
  (z_s=1, mu>2) to 8600 (z_s=5, mu>1.5) events, so read them as "roughly cubic",
  not as 3.88 +/- nothing.
- `sd` is clipped to mu <= 3, the certified support. Raw sd is monster-ray junk
  (`paper_writer.md` sec 4).
- Do NOT read a sigma-ratio SEM off 1/sqrt(2N) here (CLAUDE.md sec. 18).
