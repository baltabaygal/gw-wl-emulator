# pyHalo internals vs our pipeline — exact expressions and differences

Companion to `docs/optionB_check.md`. Everything below was read directly from the
installed pyHalo v0.2.8 source (`.venv_pyhalo/.../pyHalo/`), file:function cited, not
from memory. Their reference papers: Gilman et al. 2020 (MNRAS 491, 6077, flux-ratio
Σ_sub inference), Gannon et al. 2025 (host/z scaling), Ludlow et al. 2016 (c–m),
Baltz, Marshall & Oguri 2009 (tNFW).

## 1. pyHalo's pipeline (CDM preset, defaults)

**Geometry** (`Cosmology/geometry.py`): a DOUBLE_CONE rendering volume of full opening
angle θ (default 6″) centered on the lens axis — built for strong-lens image modeling;
subhalos are rendered only inside the projected aperture R < (θ/2)·kpc/″ (~10–20 kpc),
NOT throughout the host.

**Subhalo mass function** (`Rendering/subhalos.py::normalization_sigmasub`,
`MassFunctions/mass_function_base.py::CDMPowerLaw`): a PROJECTED power law in
**infall mass**,

    d²N / (dm dA) = Σ_sub · F(M_host, z_l) · (m/m₀)^α ,   m₀ = 10⁸ M⊙,  α = −1.9

with Σ_sub = 0.025 kpc⁻² (default) — an *observationally anchored* amplitude (quasar
flux-ratio-anomaly inference, Gilman+2020) — and a host/redshift scaling

    F = 10^( k₁ log₁₀(M_host/10¹³) + k₂ log₁₀(z + 0.5) )

with (k₁, k₂) = (0.55, 0.37) in the CDM preset signature (Gannon et al. 2025; the
module-level defaults are an older calibration (0.88, 1.7) — worth knowing both exist).
Mean count = aperture area × Σ_sub F × ∫(m/m₀)^α dm (analytic), Poisson-drawn.

**Infall redshift** (`Halos/accretion.py::InfallDistributionHybrid`, preset
HYBRID_INFALL): each subhalo draws z_infall from a Galacticus-merger-tree-calibrated
distribution (conditioned on the host mass); this sets its time-since-infall.

**Concentration** (`concentration_models.py`, default LUDLOW2016 via colossus): c(m, z_infall)
from Ludlow et al. 2016, with log-normal scatter 0.2 dex. Host concentration is FIXED at
c_host = 6.0 by default (not drawn from a c–M relation).

**Spatial distribution** (`Rendering/SpatialDistributions/`): default `UNIFORM` in the
projected cone (justified: the strong-lensing aperture ≪ host r_s, where the projected
subhalo density is nearly flat). Alternative `PROJECTED_NFW`: a cored NFW
(core = r_tidal = 0.25 r_s of the host) sampled in projection + a line-of-sight coordinate.
No anti-bias function; the 3D radius mainly feeds the truncation model.

**Tidal evolution — their f_s analogue** (`Halos/tidal_truncation.py::TruncationGalacticus`):
the bound mass is computed PER HALO from an emulator of Galacticus merger-tree runs,

    m_bound = m_infall × 10^{ 𝓘(log₁₀ c_infall, t_since_infall, c_host) }

and the tNFW truncation radius r_t (τ = r_t/r_s) is then solved so the truncated profile
encloses m_bound. So the bound-mass fraction is **emergent per object** — pyHalo has no
prescribed f_s; the population-level bound fraction arises from (infall SHMF) ×
(stripping statistics).

**Subhalo profile**: tNFW (Baltz–Marshall–Oguri), ρ ∝ [R(R+r_s)²]⁻¹ · r_t²/(R²+r_t²).

**Field (LOS) halos** (`Rendering/line_of_sight.py`,
`MassFunctions/density_peaks.py::ShethTormen`): Sheth–Tormen HMF evaluated with
colossus, rendered per redshift plane in the double cone; a two-halo boost near the main
deflector (Lazar et al. 2021 correction); and **negative convergence sheets** per plane
subtract the mean mass added in halos — their analogue of our `meankappa` subtraction.

**Output**: a lenstronomy profile list; their native observable is strong-lens image
perturbations (flux ratios, astrometry), not a magnification PDF.

## 2. Our pipeline (for contrast; details in docs/subhalo_combining.md + variance_derivation.md)

Hosts: pFC ellipsoidal-collapse first-crossing HMF (p=0.3, q=0.8) over the FULL line of
sight, Poisson-with-log-normal-bias (Cox) sampling, threshold κ_thr(Nhalos=100).
Per host: **evolved** SHMF dN/dlnψ = γ ψ^α e^{−βψ^ω} (ψ = m/M, α = −0.82 ⇒ dN/dm ∝ m^{−1.82},
β = 50, ω = 4, ψ_max = 1) with γ anchored to the bound fraction f_s(N_τ) from the EPS
formation redshift (JvdB14 eqs. 23–26; Giocoli+2007 w̃_f) — i.e. our masses are ALREADY
bound masses; no infall bookkeeping. Untruncated NFW clumps with cons14 (Dutton–Macciò)
c(m, z_l), no scatter. Han+16 anti-biased 3D radial profile throughout the host (to r200).
Mass conserved via the (1 − f_s,res)·M host reduction; resolution gated by subhalo_factor;
mean removed via meankappa. Observable: P(μ) along random rays.

## 3. Side-by-side

| ingredient | pyHalo (defaults) | this work |
|---|---|---|
| mass variable | infall mass + per-halo stripping → bound | bound (evolved SHMF) directly |
| SHMF form | Σ_sub·F(M,z)·(m/10⁸)^−1.9 per kpc² (projected) | γ(M,z) ψ^−0.82 e^{−50ψ⁴} per host (3D) |
| normalization anchor | observed flux-ratio anomalies (Σ_sub) | EPS formation-time physics (f_s(N_τ)) |
| bound fraction f_s | emergent (Galacticus emulator per halo) | prescribed f_s = 0.3563/N_τ^0.6 − 0.075 |
| profile | tNFW, r_t from bound mass | NFW untruncated (Option A: <10% on Var κ) |
| concentration | Ludlow16 at z_infall, 0.2 dex scatter | cons14 at z_l, no scatter |
| host c | fixed 6.0 | cons14(M, z) |
| spatial | uniform in ~15 kpc cone (or cored proj. NFW) | Han+16 anti-biased, full r200 |
| field-halo HMF | Sheth–Tormen (colossus) | pFC ellipsoidal first-crossing (p=0.3, q=0.8) — same ST family |
| mean-mass removal | negative convergence sheets | meankappa subtraction |
| LOS clustering | two-halo term + Lazar+21 | log-normal halo-bias Cox layer |
| native observable | strong-lens image perturbations | weak-lensing P(μ) |

Structural summary: the two codes take **different routes to the same intermediate
object** — the bound-subhalo population of a host at z_l. pyHalo: observationally
normalized infall SHMF → simulate stripping per object. Us: simulation-calibrated
*evolved* SHMF with the stripping already integrated into (γ, f_s). The right endpoint
comparison is therefore our SHMF vs their **bound**-mass function (not their infall one),
which is what `plots/figures/pyhalo_vs_ours.png` shows.

## 4a. Visual realization comparison (2026-07-07)

`plots/figures/pyhalo_visual_compare.png` (`tmp/pyhalo_visual_compare.py`): same host,
one realization from each code, in the style of `subhalo_resolved_demo_default_factor.png`.
(a) our full-host population (N≈4000 with m≥10⁷, Han+16 over r200=378 kpc); (b) our inner
38 kpc (N≈83); (c) pyHalo's inner 38 kpc — ~1000 rendered *remnants*, but only N≈61 with
bound mass ≥10⁷ (median stripping m_b/m_inf ≈ 0.002); (d) the projected bound/infall MF
comparison; (e) cumulative N(>m) in the shared aperture: at equal BOUND mass the two codes
nearly coincide. Key visual takeaways: pyHalo renders only the strong-lensing core (1% of
the host area); its population is a cloud of heavily-stripped remnants whose sub-10⁷ tail
we simply never instantiate; above any common bound-mass threshold the realizations look
statistically alike.

## 4. Figure and numbers (2026-07-05)

`plots/figures/pyhalo_vs_ours.png`; script `tmp/pyhalo_vs_ours_plot.py`, data
`tmp/pyhalo_vs_ours.npz`. Host 10¹³ M⊙, z_l = 0.5, aperture R < 38 kpc (12″ cone),
30 realizations at pyHalo defaults; infall window extended to 10^11.5 M⊙ so their
bound-mass function is complete over [10⁷, 10⁹] (with the default 10¹⁰ ceiling it was
truncation-incomplete: median stripping is m_b/m_inf ≈ 0.002 in this inner region, so
bound masses ≥ 10⁷ descend from infall masses ≫ 10¹⁰).

Projected mass function d²N/(dln m dA) [kpc⁻²]:

| m [M⊙] | pyHalo infall | pyHalo bound | ours (evolved) | ours/bound |
|---|---|---|---|---|
| 1.4e7 | 1.49e-1 | 8.6e-3 | 1.10e-2 | 1.28 |
| 1.1e8 | 2.25e-2 | 1.37e-3 | 2.02e-3 | 1.47 |
| 8.9e8 | 3.64e-3 | 2.46e-4 | 3.69e-4 | 1.50 |
| 7.1e9 | 5.0e-4 | 3.2e-5 | 6.7e-5 | 2.1 (their window edge) |

**Reading:**
- **Shape: agrees.** The ours/bound ratio drifts only 1.28 → 1.50 over two decades —
  the two codes' bound-mass functions have essentially the same logarithmic slope.
- **Amplitude: ours is a uniform factor ≈ 1.3–1.5 above pyHalo's inner-region bound MF.**
  Expected sign and plausible magnitude: pyHalo strips *per object conditioned on
  position* (everything rendered here lives at R < 38 kpc of a 378-kpc host, where
  stripping is maximal — median m_b/m_inf ≈ 0.002), while our JvdB14 evolved SHMF is the
  *halo-averaged* bound MF, position-independent by construction. An inner-conditional
  bound MF should sit below the halo average; factor ~1.4 is that radial-conditioning
  difference plus normalization-anchor differences (their Σ_sub from flux-ratio
  observations, ours from EPS theory).
- **Convergent middle:** their infall MF is ~10× above our evolved MF; their very
  aggressive stripping brings it down to ~0.7× ours. Two opposite modeling strategies —
  high observational infall normalization + strong simulated stripping vs
  pre-integrated evolved MF — land within ~40% of each other. For context, Σ_sub itself
  is only constrained to factor ~2–3 by current flux-ratio data.
- **Radial profile:** both codes are flat in the inner 38 kpc (pyHalo uniform by
  construction; our projected Han+16 varies by <5% there) — consistent with each other
  and with the DRCD18 constant-density convention.

**Caveat for the earlier `optionB_check.md` m_eff test:** that run compared our evolved
SHMF to pyHalo's *infall* MF at matched slope over a window where ψ ≪ 1 — a valid
*implementation-level* check of mass-function sampling (slope, moments), but not an
endpoint physics comparison. The physics-level statement is this section's: bound-mass
functions agree in shape with a ~1.4× amplitude offset attributable to inner-region
conditioning, well inside literature systematics.
