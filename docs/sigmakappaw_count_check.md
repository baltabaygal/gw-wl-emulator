# A pure counting check of the `sigmakappaW` radial measure

Companion to `sigmakappaw_measure_note.md`; this is the minimal, assumption-free
version. Code: `playground/sigmakappaw_count_check.cpp` → `build/sigmakappaw_count_check [zs]`.

## The question

Fix a lens mass bin M (width dlnM) and a redshift shell [z, z+dz]. Ask a question
with **no κ weighting, no variance, no randomness**:

> How many such lenses have their transverse distance r from the line of sight
> inside the band [r_max, r_stop]?

The code contains two answers. They must agree; they differ by exactly 2.

## The band edges (both defined by the code itself)

- **r_max** — inner edge. `rmaxfNFW(M, z; κ_thr)`: the transverse distance at
  which this lens's κ at the sightline equals κ_thr. Eq. (3) of the paper places
  resolved lenses at r < r_max (the θ(r_max − r) factor); everything beyond r_max
  is what `sigmakappaW` integrates. So r_max is where the weak band starts.
- **r_stop** — outer edge. `sigmakappaW`'s while-loop walks outward from r_max in
  steps of dlnr = 0.01 and stops when the lens's κ has fallen below 10⁻³·κ_thr.
  The last grid point defines r_stop (≈ √1000 ≈ 32× r_max, since κ ~ r⁻² there).
  We let the loop itself define r_stop, so both answers below count the
  **identical** set of lenses; the κ profile plays no other role in this check.

## Answer (a): eq. (3) of the paper

Eq. (3): dP_l/dr ∝ 2π(1+z)² r / H(z) · dn/dlnM. Integrating r from r_max to r_stop:

    N_band = pref · π [ ((1+z) r_stop)² − ((1+z) r_max)² ],
    pref   = 306.535 / H(z) · dn/dlnM · dlnM · dz .

This is the same "difference of disks" the code itself uses to count *resolved*
lenses: `NhfNFW` (lensing.cpp:149) is exactly pref·π((1+z)r_max)², and it is what
`findkappathr` calibrates to ⟨N⟩ = 100 (footnote 4). The check program prints
`NhfNFW(κ_thr) = 100.57` to confirm: **formula (a) is the convention the whole
model is anchored to.**

## Answer (b): the `sigmakappaW` loop, with the κ weights deleted

The loop (lensing.cpp:183–189) accumulates three sums with weights 1, κ, κ².
The weight-1 sum is its own `Nh` accumulator — which appears in the returned
`sqrt(kappa2 − kappa1²/Nh)`, so this is not an artificial construct; the code
already computes it:

    N_band^(loop) = Σ_i  pref · π ((1+z) r_i)² · dlnr ,   r_i = r_max e^{i·dlnr}.

## The exact prediction for the ratio

With r_i = r_max·e^{i·dlnr} (i = 0…K−1) and r_stop = r_max·e^{K·dlnr}, the sum is
a geometric series:

    Σ_{i=0}^{K−1} r_i² = r_max² (e^{2K·dlnr} − 1)/(e^{2·dlnr} − 1),

so

    (b)/(a) = [Σ r_i² · dlnr] / (r_stop² − r_max²) = dlnr / (e^{2·dlnr} − 1)
            = 0.01 / (e^{0.02} − 1) = 0.4950166 …  →  1/2 as dlnr → 0.

Every cell-dependent quantity (r_max, K, pref, cosmology) cancels. If the loop
measure were the correct annulus element the ratio would be 1 (up to O(dlnr)).
Instead the loop's measure π r² dlnr is half the annulus differential
d(π r²) = 2π r² dlnr — the paper's own 2π in eq. (3).

## Measured (z_s = 1, κ_thr = 1.27775e-4)

|  z_l | M [M⊙] | r_max [kpc] | r_stop [kpc] | steps | (b) loop | (a) eq. 3 | ratio |
|---|---|---|---|---|---|---|---|
| 0.201 | 9.1e12 | 1750.5 | 57 969 | 350 | 21.49 | 43.41 | 0.495017 |
| 0.498 | 1.1e10 | 51.7 | 1 748 | 352 | 30.50 | 61.62 | 0.495017 |
| 0.498 | 9.1e12 | 1872.7 | 61 397 | 349 | 71.33 | 144.1 | 0.495017 |
| **total, all cells** | | | | | **56 110** | **113 350** | **0.495017** |

The measured ratio equals dlnr/(e^{2dlnr}−1) = 0.495017 to all printed digits,
cell by cell and in total (and at every z_s) — so there is no room for a
"different domain / discretization" explanation: the two counts differ by the
measure alone.

## What it means

"Moving dr to the other side and integrating" kills the 2 only in the cumulative
count ∫2πr dr = πr² — that is formula (a), and it is correct. The loop, however,
needs the *differential* of that count at each grid step, d(πr²) = 2πr²·dlnr,
because each radius carries its own weight κ(r)ⁿ in the actual σ_W integrals.
The loop uses πr²·dlnr — half the lenses at every radius. The κ-weighted sums
κ₁ and κ₂ inherit the same factor, hence σ²_W is low by ×2 (see the companion
note for the Monte-Carlo confirmation and the impact assessment: ≤0.5% of
Var(κ_total), so no scientific conclusions change).

*(2026-07-08)*
