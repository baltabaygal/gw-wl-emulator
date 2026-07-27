# Semi-analytic magnification PDF for bright standard sirens — full specification

**Purpose.** Exact semi-analytic solution of the unbiased, spherical, halos-only Poisson lensing model (the scalar reduction of the Kainulainen–Marra / Vaskonen stochastic pipeline). Chain:

```
dn/dM  →  R(ξ)  →  Λ(k)  →  P(µ)
```

"Semi-analytic" = every object has a derived closed form; the computer evaluates quadratures only (σ(M) integral, M and z integrals in R, one Fourier inversion). No Monte Carlo, no fitting, no simulation input anywhere.

**Model scope (v1):** spherical NFW lenses, exact projected profile; Sheth–Tormen mass function; Poisson lens counts, **no** linear bias, **no** filaments, **no** ellipticity, **no** subhalos. Source-plane statistics via the 1/µ weighting. Ensemble-mean subtraction via the compensated Lévy exponent.

---

## 0. Conventions and constants

Planck 2018 benchmark: h = 0.674, Ω_M = 0.315, Ω_b = 0.0493, Ω_Λ = 1−Ω_M, σ8 = 0.811, n_s = 0.965, T_CMB = 2.7255 K, δ_c = 1.686.

Units: masses in M_⊙, distances in **Mpc (not Mpc/h)**, Σ in M_⊙/Mpc², H in km/s/Mpc.

- c = 299792.458 km/s; H₀ = 100h; ρ_c,0 = 2.77536627×10¹¹ h² M_⊙/Mpc³
- ρ_m = Ω_M ρ_c,0 (comoving matter density, used in σ(M) and mass function)
- G = 4.30091×10⁻⁹ Mpc (km/s)² / M_⊙
- E(z) = √(Ω_M(1+z)³ + Ω_Λ); ρ_crit(z) = ρ_c,0 E²(z) (**physical**)
- Comoving distance χ(z) = ∫₀ᶻ (c/H₀) dz′/E(z′); angular diameter D_A = χ/(1+z)
- Growth factor: D(z) ∝ H(a) ∫₀ᵃ da′/[a′H(a′)/H₀]³, normalised D(0)=1. Checkpoint: **D(0.5) = 0.7689**.

Critical surface density (physical D_A's):

```
Σ_cr(z_l, z_s) = c² D_A(z_s) / [4πG D_A(z_l) D_A(z_l→z_s)]
D_A(z_l→z_s) = (χ_s − χ_l)/(1+z_s)      (flat universe)
```

## 1. Linear power spectrum and σ(M)

Eisenstein & Hu (1998) **no-wiggle** transfer function. With ω_m = Ω_M h², ω_b = Ω_b h², θ = T_CMB/2.7:

```
s = 44.5 ln(9.83/ω_m) / √(1 + 10 ω_b^0.75)                      [Mpc]
α_Γ = 1 − 0.328 ln(431 ω_m)(ω_b/ω_m) + 0.38 ln(22.3 ω_m)(ω_b/ω_m)²
Γ_eff(k) = Ω_M h [α_Γ + (1−α_Γ)/(1 + (0.43 k s)⁴)]
q = k θ² / (Γ_eff h)          [k in Mpc⁻¹]
L = ln(2e + 1.8q);  C = 14.2 + 731/(1 + 62.5q)
T(k) = L/(L + Cq²)
```

P(k) ∝ kⁿˢ T²(k). Top-hat variance:

```
σ²(R) = (1/2π²) ∫ dk k² P(k) W²(kR),   W(x) = 3(sin x − x cos x)/x³
R(M) = [3M/(4π ρ_m)]^{1/3}
```

Normalise so σ(R = 8/h Mpc) = σ8. σ(M, z) = σ(M)·D(z).

Numerics: k-grid logspace(10⁻⁵, 10³) Mpc⁻¹, 3000 pts; tabulate σ(M) on logspace(10⁴, 10¹⁷) M_⊙, 400 pts; cubic-spline lnσ(lnM); dlnσ/dlnM by central difference on the spline.

**Checkpoints:** σ(10¹² M_⊙, z=0) = 2.223; ν(10¹², z=0) = (δ_c/σ)² = 0.575.

## 2. Sheth–Tormen mass function (Vaskonen Eq. 2 convention)

**ν ≡ δ_c²/σ²(M,z)** (note: the *square*). Halo parameters p = 0.3, q = 0.8; normalisation A = [1 + 2⁻ᵖ Γ(½−p)/√π]⁻¹ = **0.3222**.

```
dn/dlnM = (ρ_m/M) · A (1 + (qν)⁻ᵖ) √(qν/2π) e^{−qν/2} · (dlnν/dlnM)
dlnν/dlnM = −2 dlnσ/dlnM
```

Comoving number density per lnM. Integration range M ∈ [10⁷, 10¹⁶] M_⊙ (M_min is the physical dial; 10¹⁶ upper cut is inert).

## 3. NFW lens (exact, Wright & Brainerd 2000)

Mass definition **M = M_200c** (Δ = 200 × ρ_crit(z), physical). Concentration: Dutton & Macciò 2014 (200c, NFW):

```
log₁₀C = a + b log₁₀(M h / 10¹² M_⊙)
a = 0.520 + (0.905 − 0.520) exp(−0.617 z^1.21),   b = −0.101 + 0.026 z
```

⚠ **Convention risk #1:** verify what C(M,z) and mass definition the `halos` repo uses. If it uses Δ=200m or a different concentration relation, replace this block to match before comparing. κ_s ∝ C²/f(C) makes this a leading-order sensitivity.

Halo structure (all **physical** lengths):

```
r_200 = [3M/(4π·200·ρ_crit(z))]^{1/3};   r_s = r_200/C
f(C) = ln(1+C) − C/(1+C);   ρ_s = (200/3) ρ_crit(z) C³/f(C)
κ_s = ρ_s r_s / Σ_cr
Identity:  κ_s r_s² = M / (4π f(C) Σ_cr)      ← used in all closed forms
```

Projected profile at x = b/r_s (b physical):

```
κ(x) = 2κ_s F(x),   κ̄(x) = 4κ_s h(x)/x²,   γ(x) = κ̄(x) − κ(x)

x<1: F = (1/(x²−1))[1 − (2/√(1−x²)) artanh√((1−x)/(1+x))]
     h = ln(x/2) + arccosh(1/x)/√(1−x²)
x>1: F = (1/(x²−1))[1 − (2/√(x²−1)) arctan√((x−1)/(x+1))]
     h = ln(x/2) + arccos(1/x)/√(x²−1)
x=1: F = 1/3;  h = 1 + ln(1/2)
```

Use series near x=1 (F ≈ 1/3 − 2(x−1)/5; h ≈ ln(x/2)+1−(x−1)/3) and the small-x series h ≈ (x²/2)(ln(2/x) − ½) for stability.

**Checkpoint:** M = 10¹², z_l = 0.5, z_s = 2 → C = 6.76, r_s = 25.98 kpc, κ_s = 0.0495.

## 4. Single-lens jump and cross-section

Exact jump of one halo at impact parameter x:

```
ξ(x; M, z) = −ln[(1−κ)² − γ²]        (ξ = ln µ, weak-lensing Jacobian)
```

Exclude points where (1−κ)² − γ² ≤ 0 (strong-lensing interior; measure ~e^{−1/κ_s}, negligible). Differential cross-section per unit ξ:

```
dσ/dξ = 2π r_s² x² / |dξ/dlnx|
```

**Do not invert ξ(x) analytically.** Evaluate ξ on x-grid logspace(10⁻⁶, 10³·⁵), 900 pts; compute dξ/dlnx by `np.gradient`; interpolate ln(dσ/dξ) vs ln ξ linearly onto the output ξ-grid. This handles the x ~ 1 bridge exactly (no Padé needed).

## 5. Jump measure R(ξ)

Poisson intensity of jumps per unit ξ per line of sight to z_s:

```
R(ξ; z_s) = ∫₀^{z_s} dz (1+z)² (c/H(z)) ∫ dlnM (dn/dlnM)(M, z) · dσ/dξ(ξ; M, z, z_s)
```

Measure bookkeeping: dn/dlnM comoving × (1+z)³ physical conversion × physical path dl = c dz/[(1+z)H] → net (1+z)² c/H; cross-section physical (r_s physical). Σ_cr from §0 with physical D_A's.

Grids: ξ-grid `XI = logspace(−7, log10(4), 420)`; z: 40 linear pts on (10⁻³, z_s−10⁻³), rectangle rule; M: 48 log pts, trapezoid in lnM.

The floor ξ_min = 10⁻⁷ plays the role of the code's κ_thr. **Convention risk #2:** for MC comparison set this to match the code's actual κ_thr (Vaskonen: chosen so N̄ = 100 lenses; κ_thr ≈ 7×10⁻⁴ at z_s = 5). σ_DL shifts at the percent level with this choice.

**Checkpoints (z_s = 2, σ8 = 0.811):**

| ξ | R(ξ) | −dlnR/dlnξ |
|---|---|---|
| 10⁻⁴ | 8.35×10⁶ | 2.02 |
| 10⁻³ | 7.82×10⁴ | 2.06 |
| 10⁻² | 618 | 2.11 |
| 0.05 | 15.1 | 2.39 |
| 0.1 | 2.47 | 2.69 |
| 0.2 | 0.323 | 3.46 |
| 0.4 | 2.81×10⁻² | 3.90 |
| 0.8 | 1.25×10⁻³ | 3.60 |

## 6. Lévy–Khintchine assembly

Source-plane weighting is an Esscher tilt: **R_s(ξ) = e^{−ξ} R(ξ)**. Compensated exponent (implements ensemble-mean subtraction, ⟨ξ⟩ = 0):

```
Λ(k) = ∫ dξ R_s(ξ) (e^{ikξ} − 1 − ikξ)
```

Real-argument version (for exact moments): Λ̃(t) = ∫ R_s (e^{−tξ} − 1 + tξ) dξ.

Distance scatter, exactly (D_L ∝ µ^{−1/2} = e^{−ξ/2}):

```
(σ_DL/D̄_L)² = exp[Λ̃(1) − 2Λ̃(1/2)] − 1
```

**Checkpoints:** σ_DL/DL = 0.0114, 0.0233, 0.0412, 0.0682, 0.0848 at z_s = 0.5, 1, 2, 5, 10. Compare: Vaskonen full-model Hirata fit (c,β,α = 0.22, 1.09, 1.87) gives 0.0137, 0.0286, 0.0478, 0.0703, 0.0812. v1 sits ~10–15% low, as it must (no filaments/bias/ellipticity), except z_s = 10 where it crosses — open item, see §10.

⚠ **Convention risk #3:** the compensation enforces E[ξ]=0; the C++ code subtracts the ensemble mean of κ per realization. These differ at O(κ²). For a clean MC comparison, either subtract mean ξ in the MC, or add the corresponding drift here.

## 7. PDF inversion

```
P(ξ) = (1/π) ∫₀^∞ dk Re[e^{−ikξ + Λ(k)}]
dP/dµ = P(ln µ)/µ        (source plane)
```

Numerics: estimate decay from slope of −ReΛ between k = 50 and 200; set k_max ≈ 35/slope (clip to [2×10³, 4×10⁴]); linear k-grid, 6×10⁴ pts; trapezoid; evaluate on ξ ∈ [−0.9, 1.6], 420 pts.

**Checks:** ∫P dξ = 1.0000; ⟨ξ⟩ = 0 (compensation); min P ≳ −3×10⁻⁷ (ringing floor).

## 8. Closed-form asymptotics — validation theorems

These are pen-and-paper results the numerics must reproduce; use them as unit tests.

1. **Universal ξ⁻² window** (mass conservation, profile-independent): for ξ ≪ 4κ_s/3 of the dominant masses,
   `dσ/dξ = M/[f(C) Σ_cr ξ²]` exactly (from κ = A/b², A = 2κ_s r_s²). Hence R(ξ) = C_ξ/ξ² with C_ξ carrying all cosmology. Measured C_ξ(z_s=2) ≈ 0.078 (R·ξ² at ξ=10⁻³).
2. **Exponential window** (inner log profile): 4κ_s ≲ ξ ≲ 1: `dσ/dξ = (2π/e²)(r_s²/κ_s) e^{−ξ/2κ_s}`.
3. **Temper from the mass-function saddle:** R ∝ ξ⁻ˢ exp[−(ξ/ξ_c)^{ν_str}], ν_str = b/(a+b) with a = dlnκ_s/dlnM ≈ 0.19, b = dlnν^{1/2}... (b = −dlnσ/dlnM ≈ 0.30 at cluster scales) → **ν_str ≈ 0.55–0.65**. ξ_c(z_s=2) ≈ 0.045. Saddle mass at µ=1.5: **M ≈ 6×10¹³ M_⊙ at b ≈ 40 kpc** (x ≈ 0.26).
4. **Running slope:** −dlnR/dlnξ runs 2 → ~2.7 → ~4 (table §5). No plateau exists; any single-α fit is window-dependent.
5. **Lévy exponent:** ReΛ ∝ −|k|^α with α running ~1.9 (k≲10, Gaussian side) → ~1.1 (k≳400, Landau side). Measured: α = 1.90, 1.52, 1.30, 1.21, 1.15, 1.10 across decades k = 1…800. Crossover parameter N_eff = C_ξ/ξ_c ≈ 0.8: the PDF sits between Gaussian and Landau fixed points.
6. **Single-jump tail:** for µ ≳ 1.3, `dP/dµ = R(ln µ)/µ²` (subexponential temper ⇒ tail = jump measure; mean-subtraction makes the many-jump correction ~4%).
7. **σ8 sensitivity runs** (z_s = 2): γ(ξ) ≡ ∂lnR/∂lnσ8 = 0.73 (ξ=10⁻⁴), 0.89 (10⁻²), 1.28 (0.1), **1.97 (µ=1.5)**, 2.36 (0.6). σ_DL scales ≈ σ8^0.65. The emulator's rank-1 amplitude exponent 1.45 is the PDF-weighted mean of this running function.

## 9. Comparison protocol vs the `halos` MC

The decisive figure is **R_MC(ξ) vs R_analytic(ξ)**, not PDF vs PDF (the inversion compresses errors; R exposes them).

1. Configure the MC to v1 scope: spherical (ellipticity off), filaments off, bias off, same M_min/M_max, and — after resolving risk #1 — same mass definition and C(M,z).
2. Match κ_thr (risk #2) and record it.
3. Instrument the MC: per realization, for every lens i record the **single-halo** ξ_i = −ln[(1−κ_i)² − γ_i²] with that halo's own κ, γ (shear included). Histogram per unit ξ, normalised per line of sight. That is R_MC.
4. Overlay on log–log axes; compare amplitude (C_ξ), turnover location (~4κ_s/3 of dominant mass), running slope, and cutoff shape. Point-by-point agreement here guarantees Λ and hence P(µ).
5. Secondary: σ_DL/DL(z) from MC moments of e^{−ξ/2} vs Λ̃ formula (§6); PDF overlay at matched z_s; check mean-subtraction convention (risk #3).
6. Repeat R comparison at two σ8 values to test the running γ(ξ) (theorem 7) — this is the science payoff, do it once conventions match.

Known places a mismatch is *convention, not physics*: mass definition / concentration; κ_thr; Mpc vs Mpc/h internally; per-halo κ definition in the C++ (does it already subtract a background?); image- vs source-plane at the histogram stage (R_MC as defined above is image-plane; tilt by e^{−ξ} before comparing to R_s, or compare untilted to R).

## 10. Known gaps and open items (do not silently fix)

- **Scalar reduction:** exact process is vector (κ, γ₁, γ₂); this chain folds shear into the single-lens ξ (exact for the single-jump tail, O(γ²) approximation in the core). Vector completion is written down but unevaluated.
- z_s = 10 crossing above the full-model fit (§6) — suspect high-z mass function or z-grid resolution; unresolved.
- ⟨1/f(C)⟩ weighting and real vs eyeballed σ(M) shifted earlier hand-estimates by tens of %; the quadrature supersedes them, but any *new* closed-form claims must be checked against the quadrature.
- The strong-lensing region (two images, µ⁻³ regime) is excluded; enters only at µ ≳ κ_s⁻² ~ 500 with weight ~10⁻¹¹.
- KM09/11 literature check pending: which of theorems 1–7 are new vs rediscovered is not yet established.

## 11. Reference implementation

`analytic_pdf.py` (module: cosmology, EH98, ST, WB profile, `R_of_xi`, `Lambda_of_k`, `invert_to_Pxi`, `sigma_DL_over_DL`, `set_sigma8`) and `make_figure.py` (six-panel behaviour figure + tables) reproduce every checkpoint in this document with numpy/scipy/matplotlib only, runtime ~1 min. Any reimplementation should hit the §1, §3, §5, §6 checkpoints to 3 significant figures before touching the MC comparison.
