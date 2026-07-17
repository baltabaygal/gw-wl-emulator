# Analytic structure of σ_κ(z_s) — the full Campbell convergence variance

*(2026-07-09; companion to `docs/sigmakappaw_measure_note.md`. Probes:
`playground/campbell_reduced.cpp` (1D reduction), `playground/campbell_moments.cpp`
(moment decomposition, asymptotics); figure `plots/campbell_asymptotics.png`;
data `playground/campbell_P_of_z.txt`, `campbell_sigma_dense.txt`.)*

Setup: halos only, Poisson positions (no clustering), untruncated NFW, the code's HMF
(ellipsoidal pFC) and cons14 concentrations — i.e. exactly `sigmakappaW` in the
κ_thr → ∞, floor → 0 limit.

## 1. Kernel factorization (exact)

NFW self-similarity: κ(r) = κ₀(M,z) K(r/r_s), κ₀ = ρ_s r_s/Σ_c, K = 2F_g. Hence

∫₀^∞ κ² 2πr dr = 2π κ₀² r_s² C₂,  **C₂ = ∫₀^∞ K² x dx = 1.4674011002723397**

C₂ is a pure number (independent of M, z, cosmology). PSLQ over
{1, ln2, ln²2, π², π²ln2, ζ(3)} finds no closed form. Convergence of the x-integral
(Σ ∝ r⁻² outside r_s) is also why the eps_floor truncation error is ∝ ε.

## 2. Geometry factorization and the moment decomposition (exact)

`lensing.cpp::Sigmacf` (flat universe): 1/Σ_c = A₀ χ_l(χ_s−χ_l)/(χ_s(1+z_l)),
A₀ = 4π/2.08871e16 (code units) — verified to 4×10⁻¹⁶. The (1+z_l)² path factor
cancels the (1+z_l)⁻² of Σ_c⁻², so **all z_s dependence enters through χ_s alone**:

σ²(z_s) = ∫₀^{z_s} P(z) [χ(z)(1 − χ(z)/χ_s)]² dz     (exact)

P(z) = 2π C₂ A₀² (306.535/H(z)) B(z),  B(z) = ∫dlnM (dn/dlnM)[ρ_s r_s²]²

Expanding the square gives the **moment decomposition**:

**σ²(z_s) = M₂(z_s) − 2 M₃(z_s)/χ_s + M₄(z_s)/χ_s²,  M_k(z_s) = ∫₀^{z_s} P χ^k dz**

Three cumulative moments of ONE fixed function P(z). Validated against brute
`sigmakappaW` to ≤ 5×10⁻⁵ over z_s = 0.2–30 (`campbell_moments.cpp`). P(z) is
monotonically decreasing: flat to z ≈ 0.5, then HMF-collapse-driven decline
(3 decades down by z = 20).

## 3. Low-z_s limit (proof)

As z_s → 0: P → P(0), χ → χ'(0) z with χ'(0) = c/H₀. Then
σ² → P(0) χ'(0)² z_s³ ∫₀¹ u²(1−u)² du = P(0) χ'(0)² z_s³/30, i.e.

**σ(z_s) → √(P(0)/30) (c/H₀) z_s^{3/2} = 0.04431 z_s^{3/2}**  (no free parameters)

The 3/2 splits as: 1 from ∫w² dz over the path, ½ from w ∝ z_l(z_s−z_l)/z_s per unit
z_s. The BPL fit's A = 0.0431 is this derived 0.0443 minus what the break absorbs.
First correction is O(z_s) and negative (kernel curvature + P decline): exact/asymptote
= 0.970 at z_s = 0.05, 0.907 at 0.2.

## 4. High-z_s limit (proof): saturation, not a power law

χ_s → χ_hor = 1.402×10⁷ kpc (finite comoving horizon) and P(z)'s support is exhausted
(HMF collapse), so the moments freeze and σ² becomes **exactly a parabola in
u = 1/χ_s**:

σ²(z_s) = M₂(∞) − 2M₃(∞) u + M₄(∞) u²  →  **σ_∞ = 0.1542** at u = 1/χ_hor

(moments frozen at z = 30; M₂ still grows 4.9% between z = 20 and 30, so σ_∞ carries
~% -level uncertainty from the deep-z HMF tail). The exact curve merges onto the
frozen parabola for z_s ≳ 10 (panel d). σ(10) = 0.109 is only 71% of saturation — the
approach is slow because χ_hor − χ_s ∝ (1+z_s)^{−1/2} in the matter era. The local
slope → 0, NOT a constant power.

## 5. Verdict on the broken power law

σ = A z^{3/2}[1+(z/z₀)^q]^{−p/q} is a **phenomenological interpolant**: the z^{3/2}
prefactor is exact (§3) but the break shape is not derived and its z → ∞ behavior
(σ ∝ z^{3/2−p} = z^{0.09}, unbounded) contradicts the true saturation (§4). Numerically
it stays within 0.11% on its fit range (0.2–10) and, by accident of the small residual
slope mimicking the slow saturation, within 1.6% out to z = 30 — but it has no analytic
standing beyond the fit range. **The analytically correct compact form is the
moment decomposition of §2** (three cumulative-moment functions, or at high z_s three
frozen constants + χ(z_s)).

## 6. What blocks a fully closed form (and the path to one)

Already exact/closed: the kernel factorization (C₂), the geometry (χ_l(χ_s−χ_l)/χ_s,
exact in flat FRW), the moment decomposition, both asymptotics. The irreducible
obstructions sit inside P(z):

1. **σ(M) — the mass variance / transfer function.** No elementary form; this is the
   same obstruction that prevents closed-form HMFs generally. Locally a power law
   σ ∝ M^{−α} ⇒ the B(z) mass integral (a 4/3-moment: B ∝ (Δρ_ref)^{2/3}
   ∫dlnM n M^{4/3} c⁴/m(c)²) reduces to incomplete Γ-functions.
2. **m(c) = ln(1+c) − c/(1+c)** in ρ_s — a log obstruction; power-law-approximable.
3. **C₂** — pure number, no closed form found (quote numerically).
4. **χ(z), D_g(z)** in ΛCDM — hypergeometric ₂F₁ closed forms exist; acceptable.

So a "closed form in special functions" σ(z_s) is achievable via: PS/ST f(ν) +
local power-law σ(M) + power-law DM14 c(M,z) + power-law m(c) ⇒ B(z) in incomplete
Γ ⇒ M_k via the ₂F₁ χ(z). Controlled approximations, not pursued; the exact 1D
moment form is already < 1 ms and machine-accurate.

## Caveats

Halo-only, unclustered (no 2-halo term, no filaments); fiducial cosmology; inherits
pFC HMF, cons14, M ∈ 10⁷–10¹⁷. C₂ and the exponent 3/2 are universal; P(z), the
moments, σ_∞, and all fit coefficients are model/cosmology-dependent.
