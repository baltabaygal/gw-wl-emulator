# Draft memo — "Learning weak lensing magnifications": comment resolutions

2026-07-20. Every claim below was checked against the implementation (file:line refs
are to this repo, current working tree), not from memory. New figures + scripts:
`plots/fig_clustering_field.{pdf,png}`, `plots/fig_subhalo_population.{pdf,png}`
(`paper_prod/scripts/plot_fig_{clustering_field,subhalo_population}.py`, numpy-only,
built on the port-validated `scripts/convergence/bias_field_prototype.py` +
`playground/bias_field/validate_field_covariance.py`; the covariance layer re-passed
its validation gates in this run: |dCov|/diag ≤ 4e-6, ⟨λ⟩=1 to 0.2%).

---

## R1. "Should we show some plot here, e.g. P_1D or realizations of λ?"

Yes — both, one figure. Generated: `plots/fig_clustering_field.pdf`.
Panel (a): P_1D(k∥) for the pencil limit R⊥→0 (KP91), the production default
R⊥ = 8441 kpc = R_L(10^14 M⊙), and 2× the default; dots mark the sampler's mode
cutoff k_max = 2π/R⊥. Panel (b): three realizations of λ(M=10^13 M⊙, z) along a
z_s = 3 sightline, drawn exactly as production (per-shell segment averages via the
exact-covariance Cholesky), with the per-shell ±1σ lognormal band.

```latex
\begin{figure}
    \centering
    \includegraphics[width=\columnwidth]{plots/fig_clustering_field.pdf}
    \caption{Top: power spectrum of the one-dimensional density field
    $\delta_{\rm 1D}$ for the pencil-beam limit $R_\perp\to0$~\cite{1991ApJ...379..482K}
    and for the transverse smoothing radius used in this work,
    $R_\perp = 8.44\,{\rm Mpc}$, together with a two times larger window. The dots
    mark the mode cutoff $k_{\rm max} = 2\pi/R_\perp$ of the discrete realization.
    Bottom: three realizations of the count modulation $\lambda(M,z)$ for
    $M = 10^{13}\Msun$ along a $z_s=3$ line of sight (piecewise constant over the
    redshift shells of the simulation grid), with the per-shell $\pm1\sigma$ range
    of the mean-one lognormal (gray band).}
    \label{fig:clustering_field}
\end{figure}
```

**But first fix the text — three places where the draft disagrees with the code:**

1. **The window is a transverse disk, not an isotropic 3D top-hat.** The code
   (`cpp/lensing.cpp:325-327, 488-499`) computes
   P_1D(k∥) = (1/2π) ∫ dk⊥ k⊥ P(√(k∥²+k⊥²)) W̃²(k⊥R⊥) with W̃(x) = 2J₁(x)/x —
   the 2D disk window acting on k⊥ only (KP91 eq. 3.8 with a uniform beam profile).
   The draft's Ŵ²(R_s k) (argument = |k|) and the sentence "smoothing the
   three-dimensional density field isotropically … suppresses … both along and
   perpendicular" describe a different construction. Along the LOS the only
   suppression is the sharp mode cutoff k_max = 2π/R⊥ (N_max = L/R⊥ modes,
   `lensing.cpp:481`). Replacement equation:

   ```latex
   \be
       P_{\rm 1D}(k_\parallel) = \frac{1}{2\pi} \int_0^\infty \!\td k_\perp \,
       k_\perp \, P\big(\sqrt{k_\parallel^2+k_\perp^2}\,\big)\,
       \tilde{W}^2(R_\perp k_\perp) \,,
   \ee
   ```
   with "the field is averaged over a transverse disk of comoving radius
   $R_\perp$, $\tilde W(x) = 2J_1(x)/x$; modes with $k_\parallel > 2\pi/R_\perp$
   are excluded from the discrete realization." The KP91 footnote stays correct
   (W̃→1 recovers their pencil beam).

2. **R_s = 20 Mpc is stale.** The production default is `bias_Rperp = 8441 kpc`
   (comoving) = the Lagrangian radius R_L(10^14 M⊙) at the fiducial cosmology
   (`cpp/lensing.h:127-137`, decision 2026-07-16; supervisor sign-off was still
   pending — Ville, this is the number to bless or change). Suggested text: "We
   use a transverse disk window of comoving radius $R_\perp = 8.4\,$Mpc, the
   Lagrangian radius of $M = 10^{14}\Msun$ halos, the clustering-variance-weighted
   signal scale of the model; we show later how changing $R_\perp$ impacts our
   results" (that later section can draw on `data/results/rperp_pdf_scan/report.md`).
   Magnitude of the difference: P_1D(k→0) = 26.3 Mpc (R⊥=8.44 Mpc) vs 13.7 Mpc
   (20 Mpc); point-field σ_1D = 1.11 vs 0.64.

3. **"Padded well beyond the source distance" oversells.** The code uses
   L = 1.05 χ(z_s) (`lensing.cpp:480`; known kept deviation from the design note's
   ≳2χ, wrap leakage ≤1.6%). Either soften ("on a periodic path of length
   $L = 1.05\,d_c(z_s)$; the residual periodic correlation between the observer
   and source ends is at the per-cent level") or bump the pad in code first.

4. *(presentation)* The compensation in λ uses the **shell's own segment-averaged
   variance**, not the global point variance σ²_1D (`lensing.cpp:838`,
   `bcomp = ½(bD_g)²σ²_i`), so ⟨λ⟩=1 holds exactly per shell. Suggested fix: after
   the λ equation, "where $\delta_{\rm 1D}$ is understood as the average of the
   field over the redshift shell of the lens and $\sigma_{\rm 1D}^2$ as the
   variance of that shell average". (The current eq. with the point-field σ²_1D
   normalizes only the idealized continuum field.)

Also: b(M,z) should be specified — the code uses the Sheth–Tormen peak-background
split bias with (p,q) = (0.3, 0.75) (`cpp/cosmology.cpp:462-469`):
b = 1 + (qν²−1)/δ_c + 2p/[δ_c(1+(qν²)^p)], ν = δ_c(z)/σ(M). Cite Sheth & Tormen
1999 (`Sheth:1999mn`; note q here differs from the HMF's first-crossing q = 0.8;
for the HMF itself the ellipsoidal-collapse cite is Sheth, Mo & Tormen 2001,
`Sheth:1999su`).

---

## R2. "Describe field halos here … section about filaments. What clustering bias for filaments? See 1101.3847."

**1101.3847 = Yan & Fan 2011, ApJ 730, 33** (Inspire texkey `Yan:2011ux`),
"Statistical Properties of Supercluster-Like Filaments from Cosmological
Simulations": filaments = linking-density 1+δ=16 groups, whose mass function
matches an excursion-set prediction with a **nearly flat (lowered) barrier**. That
is exactly what the code implements: the filament mass function uses the same
first-crossing distribution as halos but with (p,q) = (0, 0.7)
(`cpp/cosmology.cpp:299-311` `pFCfil`) — i.e. an effective barrier √0.7 δ_c.

**Filament bias — recommendation.** Currently the code multiplies the filament
counts by the *same* λ as halos of the same mass bin, i.e. the halo bias
(`cpp/lensing.cpp:828-840, 986-1005`). The PBS-consistent choice is the bias of the
filament first-crossing barrier, b_fil(ν) = 1 + (qν²−1)/δ_c with q = 0.7, p = 0 —
one line to add next to `halobias`. Size of the change at z = 0.5:
b_fil/b_halo = 0.80/1.02 (10^12), 1.21/1.44 (10^13), 2.33/2.61 (10^14),
6.11/6.63 (10^15 M⊙) — a 10–20% lower filament bias. Either implement it (small,
isolated change; retest bitwise default) or state explicitly that filaments share
the halo-bias modulation as an approximation. My suggestion: implement, since the
paper now makes the clustering layer a headline improvement.

**Drop-in subsection text** (verified against the code; details cite Vaskonen 2026):

```latex
\subsubsection{Field halos and filaments}
The mean field-halo abundance is given by the halo mass function computed from
the ellipsoidal-collapse first-crossing distribution~\cite{Sheth:1999su} (with
$p=0.3$, $q=0.8$), and the number of halos in each mass and redshift cell is
Poisson distributed with mean $\lambda \bar N$, where $\lambda$ is the
clustering modulation of Eq.~\eqref{eq:n}. Each halo is assigned an NFW profile
with the concentration--mass relation of Ref.~\cite{Dutton:2014xda} (halos
defined at 200 times the critical density), projected with an elliptical
distortion whose axis ratio follows the $N$-body calibration of
Ref.~\cite{Allgood:2005eu}, $s = 0.54\,(M/M_*(z))^{-0.05}$,
$\epsilon = (1-s)/(1+s)$. The convergence and shear of each halo follow from
the closed-form NFW lensing functions~\cite{Wright:1999tv}.

Filaments are modeled as in Ref.~\cite{Vaskonen:2026ubd}: uniform-density
cylinders of radius $r_{\rm F} = 1\,{\rm Mpc}\,(M/10^{14}\Msun)^{1/3}$, length
$L_{\rm F} = 20\,{\rm Mpc}\,(M/10^{14}\Msun)^{1/3}$ and mean density
$14.4\rho_{\rm c}$, with random orientation. Their abundance is obtained from
the first-crossing distribution with a lowered, scale-independent barrier
($p=0$, $q=0.7$), consistent with the filament mass function measured in
$N$-body simulations~\cite{Yan:2011ux}. The filament counts are modulated by
the large-scale density field in the same way as the halo counts, with the
peak-background-split bias of the filament barrier,
$b_{\rm F}(M,z) = 1 + [\,0.7\nu^2 - 1\,]/\delta_c$.
\end{...}
```
(Last sentence assumes the recommended change; otherwise "…with the same bias as
halos of equal mass, which we checked is a 10–20\% overestimate of the
peak-background-split filament bias.") If you keep a κ-profile equation for the
cylinder, the code's normalization is
κ₀F = r_F·14.4ρ_c·L_F / [Σ_c · max(2r_F, |cosφ| L_F)] (`lensing.cpp:823-825, 993`).

---

## R3. "Why not M − Σᵢmᵢ instead of (1−f_s)M?"

Three code-grounded reasons; suggested replacement text below.

1. **The realized sum is never available.** Only clumps above the dynamic
   resolution floor m_res(r) are ever drawn (`subhalo.cpp::addClumps:418-443`);
   the unresolved band enters only through its Campbell mean + variance
   (model 3, `lensing.cpp:927-935`). "M − Σ_realized" would misconserve by
   exactly the unresolved fraction, and obtaining the full Σ down to m_floor is
   the brute-force cost the scheme exists to avoid.
2. **Exactness of the split.** With the deterministic reduction by the full bound
   fraction f_s,b (`subhalo.cpp::buildWsubBin:258-263`, `lensing.cpp:882-894`),
   κ_halo = κ_NFW[(1−f_s,b)M] + resolved clumps + μ_unres(y) + N(0,σ_unres(y))
   is exact in mean *and* variance at every impact parameter for any resolution
   parameter (docs/subhalo/wsub_gaussian_term_derivation.md). A host that
   re-scales per realization would break the precomputed (z,M)-grid tables and
   the exactness proof.
3. **The scatter is physical and already carried by the clumps.** ⟨Σᵢmᵢ⟩ = f_s,b M;
   the Poisson fluctuations about it are dominated by the few largest subhalos,
   which are explicitly resolved wherever they matter — their direct κ
   contribution is first order, while the host-profile response to the same
   fluctuation is second order. Real halos of fixed M do scatter in bound
   substructure mass (JvdB14 model this); pinning Σᵢmᵢ = f_sM would delete
   scatter the model is built to capture.

Also fix the over-claim in the same paragraph: conservation holds **in
expectation**, not "along every sightline". Replacement:

```latex
with $\langle \sum_i m_i \rangle = f_{\rm s} M$, so that the total halo mass is
conserved in expectation; the Poisson fluctuations of the substructure about
this mean are part of the halo-to-halo scatter the model is designed to
capture. We reduce the smooth component by the deterministic bound fraction
rather than by the realized $\sum_i m_i$ because only subhalos above a
resolution floor are explicitly sampled (Sec.~\ref{...}); the unresolved
remainder enters through its mean and variance, and the decomposition is exact
in both precisely for this split.
```

---

## R4. "m_floor here is not needed, is it?"

Correct — **not in the SHMF equation**, but it cannot be deleted from the model.
In the code m_floor = 10^7 M⊙ is (i) the brute-mode sampling floor
(`addClumps:420`), (ii) the lower edge of the unresolved band in the model-3
tables (`buildWsubBin:255`), and (iii) the lower limit of the full bound fraction
f_s,b used for the host reduction (`buildWsubBin:258-263`). Below m_floor,
substructure is simply left in the smooth host. Note the two cutoffs play
different roles: ψ_res = 10⁻⁴ is part of the *definition* of f_s (the JvdB14 fit
counts mass in ψ > 10⁻⁴), while m_floor is our absolute substructure floor. For
hosts with M > 10^11 M⊙, f_s,b extrapolates the SHMF below its calibrated range;
the extrapolated share is ~20% of the bound mass for cluster hosts (verified:
f_s,b = 0.289 vs f_s = 0.232 at M = 10^15, z = 0.5 — exactly the 19.7%
sub-ψ_res mass share of the SHMF integrated from ψ = 10⁻⁸).

Suggested edit: in Eq. (shmf) give the range simply as ψ ≤ 1 (the exponential
cutoff does the rest), keep Eq. (fsnorm) as the normalization statement, and move
m_floor to the resolution paragraph:

```latex
Subhalos below an absolute floor $m_{\rm floor} = 10^7\Msun$ are never treated
as substructure and remain part of the smooth host; the bound fraction used
for the host reduction in Eq.~\eqref{eq:reducedhost} is correspondingly
evaluated over $m_{\rm floor} \leq m \leq M$, which extrapolates
Eq.~\eqref{eq:shmf} below the resolution $\psi_{\rm res}$ of the simulations it
was calibrated on (a $\lesssim 20\%$ addition to the bound mass fraction for
cluster-scale hosts).
```

---

## R5. "Expression for Δ_vir and a reference?"

Bryan & Norman (1998) fitting formula, as implemented (`subhalo.cpp:174-176`):

```latex
with $\Delta_{\rm vir}(z) = 18\pi^2 + 82\,d - 39\,d^2$, $d \equiv \Omega_M(z)-1$,
the virial overdensity relative to the critical density~\cite{Bryan:1997dn}
```

(Δ_vir = 103.2 at z = 0, 139.5 at z = 0.5 for the fiducial cosmology.) The
prefactor 6.006 is not arbitrary either — it is JvdB14's eq. (2)
τ_dyn ≃ 1.628 h⁻¹Gyr (Δ_vir/178)^(−1/2) (H/H₀)^(−1) inserted into
N_τ = ∫dt/τ_dyn (H cancels): 6.006 = 9.78/1.628 (using H₀ = h/9.78 Gyr⁻¹). Worth a half-sentence
so the constant is traceable: "…where the coefficient follows from
$\tau_{\rm dyn} \simeq 1.628\,h^{-1}{\rm Gyr}\,(\Delta_{\rm vir}/178)^{-1/2}
(H/H_0)^{-1}$~\cite{Jiang:2014nsa}."

---

## R6/G1. Radial profile: citation ("the Bolshoi") + c200 question

**c200: yes to everything.** c200 = `cosmology::cons14` = Dutton & Macciò 2014
(arXiv:1402.7073, `Dutton:2014xda`), their NFW c200(M,z) relation
(`cosmology.cpp:365-370`); halos are built at 200ρ_c(z) with r200 = c200·r_s
(`cosmology.cpp:434-436`); the subhalo module reuses exactly that concentration
and r200h = r_s·c for the host geometry (`subhalo.cpp:198-200`), and the clump
NFW parameters come from the same tables at (m, z_l) (`subhalo.cpp:95`,
`interpolateNFWMass`). So your red text is correct as written; add
\cite{Dutton:2014xda} and, for the clumps, "with the same concentration–mass
relation evaluated at the subhalo mass".

**Citation for the profile — resolved by `plots/bias_fit_comparison.png`.** That
plot documents the provenance of the constants: the adopted
B(x) = [(x/0.54)^(−2.5)+1]^(−1/2) is the "transition fit" to the **Green, van den
Bosch & Jiang 2021** withering+disruption radial bias curve (arXiv:2103.01227;
fitted values (0.54, 2.48), rounded to 2.5 in `subhalo.cpp`), shown against the
**Bolshoi Figure 7 data points** (Klypin, Trujillo-Gomez & Primack 2011,
arXiv:1002.3660, `Klypin:2010qw`); the direct fit to the Bolshoi points would be
(0.94, 2.52) — a stronger depletion. So cite Green+21 as the source of the
adopted fit, with Bolshoi as the simulation comparison. Two nits to decide:
(i) the fit plot's x is r/r_vir while Eq. (radial) and the code use x = r/r200
(r_vir > r200, so the code places the depletion at ~25–35% smaller physical
radius than the calibration — either convert 0.54 r_vir → r200 units or state
the approximation); (ii) if the (0.94, 2.52) Bolshoi variant vs (0.54, 2.48)
Green variant matters at the paper's precision, say which is adopted and why.

```latex
The subhalos are distributed within their host following the radial profile
motivated by $N$-body results~\cite{Green:2021,Klypin:2010qw}: subhalos
selected by bound mass trace the host density profile in the outskirts but are
strongly depleted in the central region by tidal stripping. We use the
transition fit $B(x) = [1+(x/0.54)^{-5/2}]^{-1/2}$ to the radial bias of
Ref.~\cite{Green:2021},
```
and after Eq. (radial): "The depletion … reflects their **tidal stripping and
disruption**" (currently "disruption by tidal disruption"). For scale: the model
gives n_sub/(mass-follows-NFW) ≈ 0.16 at r = 0.2 r200, and a median subhalo
radius of 0.72 r200 (c200 = 4.5).

---

## R7/G2. "Two-panel figure: SHMF + radial profile" (+ Gala's "add this fit plot?")

Generated: `plots/fig_subhalo_population.pdf` — panel (a) evolved SHMF
dN/dlnψ for M = 10^12–10^15 M⊙ at z = 0.5 with the f_s(N_τ) normalization
computed exactly as `subhalo.cpp::precompute` (f_s = 0.11, 0.14, 0.17, 0.23;
dot marks ψ = m_floor/M where visible); panel (b) the anti-biased radial profile
for the same hosts (c200 = 6.8…3.7) with the mass-follows-NFW comparator dashed.

```latex
\begin{figure}
    \centering
    \includegraphics[width=\columnwidth]{plots/fig_subhalo_population.pdf}
    \caption{Top: the evolved subhalo mass function, Eq.~\eqref{eq:shmf}, for
    host masses $M = 10^{12}$--$10^{15}\Msun$ at $z=0.5$, normalized by the
    bound fraction $f_{\rm s}(N_\tau)$ of Eq.~\eqref{eq:fsnorm}; the dot marks
    the absolute floor $m_{\rm floor} = 10^7 \Msun$. Bottom: the subhalo radial
    distribution, Eq.~\eqref{eq:radial}, for the same hosts
    ($c_{200}$ from~\cite{Dutton:2014xda}); the dashed curve shows the
    mass-follows-NFW expectation, illustrating the central depletion by tidal
    stripping.}
    \label{fig:subhalo_population}
\end{figure}
```

Useful numbers for the surrounding text: ⟨N⟩(m > 10^7 M⊙) ≈ 500 (10^12),
3.3×10^4 (10^14), 2.9×10^5 (10^15 M⊙ at z = 0.5) — so "can reach
$\mathcal{O}(10^5\text{--}10^6)$" is the safer phrasing than O(10^6).

---

## G3. Comparison-with-earlier-works section — material that already exists

1. **ACE-Lensing** (Türker et al., `Turker:2025rdt`). Two-level comparison:
   (i) emulator quality: our post-calibration median KL 0.0073 vs ACE's reported
   0.007 (ml/autoresearch/REPORT.md); (ii) *model* difference: Vaskonen-model
   ⟨κ²⟩ is 36–42% above ACE at z_s = 1 for ALL Mmin 10^4–10^9 (the mass floor
   cannot close it — wrong sign; `docs/convergence_mmin_nz_note.md`), while the
   full-MC substructure effect is +14.9% on ⟨κ²⟩ at z_s = 5 — real but below the
   ACE−Vaskonen gap (scripts/subhalo_gate/ace_gap.py,
   plots/gate_verdict.png). Standing rule applies: compare body-weighted
   quantities (P(lnμ), JSD, clipped moments), not raw tails.
2. **pyHalo**: the side-by-side is already written —
   `docs/subhalo/pyhalo_pipeline_comparison.md` (infall-SHMF + per-object
   stripping vs our evolved SHMF with f_s(N_τ); observational Σ_sub anchor vs
   EPS formation-time anchor; tNFW vs untruncated NFW; etc.) with the endpoint
   comparison figure `plots/figures/pyhalo_vs_ours.png` (our SHMF vs their
   *bound*-mass function — the right object, not their infall SHMF).
3. **Mpetha 24** = Mpetha, Congedo & Taylor, PRD 110, 023502 (2024),
   arXiv:2402.19476, "Impact of weak lensing on bright standard siren analyses".
   Natural hook: they show the σ_lnμ(z; Ω_M, H₀…) prescription used in siren
   pipelines is outdated/underestimated; our Fig. (variance_D_L) provides the
   replacement σ_DL(z, Θ), and the subhalo contribution is part of the budget
   they cannot capture with N-body-calibrated fits.
4. Possible extras: Takahashi 2011 fitting formula (already in intro refs) and
   the DES Y6 supernova-magnification measurement `DES:2025omh` for the
   resolution argument; turboGL / stochastic-approach lineage
   (Kainulainen & Marra — `papers/misc/1101.4143` is their halo-model
   observables paper) as the method ancestor.

---

## Other issues found while checking (not flagged in the draft)

1. **Fig. 1 (`fig:kappa_convergence`) points at the pre-bias-era figure.**
   `plots/sigma_partition_vs_kthr.pdf` is the *legacy* (shot-only) partition.
   The bias-era production figure is
   `paper_prod/plots/figures/sigma_partition_vs_kthr_bias.pdf`
   (+ `data/sigma_partition_vs_kthr_bias_z1.npz`); it shows measured
   weak/strong/total on the κ_tot ≤ 1 core, σ_tot flat to ~2% over
   κ_min ≤ κ_thr ≤ 10³, plateau σ_κ = 0.0313 at z_s = 1. Caption must say the
   estimator (clipped core, 8-seed ensembles) and that the plotted domain ends
   at the model floor κ_min (user decision 2026-07-17). Beware the known trap:
   `paper_prod/scripts/plot_sigma_k_vs_kappa_threshold.py` plots a *subhalo-factor*
   sweep npz with a wrong x-label — don't reuse it.
2. **Fig. 2 filename doesn't exist**: draft includes
   `plots/sigma_partition_vs_subhalo_factor.png`; the actual file is
   `plots/sigma_k_vs_subhalo_factor.png` (placeholder anyway, per its caption).
   ε_sub in the text = `subhalo_factor`, default 10⁻² (PDF-level brute
   acceptance, `data/results/subhalo_factor_jsd/report.md`) — worth stating the
   value and that runs before 2026-07-12 used 10⁻⁵.
3. **The sub-threshold section describes the code correctly** (checked term by
   term: `weakMomentsNFW`, `BiasField1D::buildWeak/weakSV`, the draw at
   `lensing.cpp:770-786` = κ_W = Σᵢ Sᵢ(δ̄ᵢ) + N(0,1)·√(ΣᵢVᵢ(δ̄ᵢ))). Two small
   additions: (i) the r-integration in Eq. (wk) is truncated at
   κ = 10⁻³ κ_thr — an effective outer truncation of the NFW halos; κ_min is a
   model parameter, not a convergence knob (2026-07-17 finding) — one sentence
   recommended; (ii) state that filaments carry no weak arm (NFW halos only,
   `lensing.h:150-152`).
4. **Subhalo resolution paragraph describes model 1, not the default model 3.**
   "The unresolved subhalos remain part of the smooth halo component … f_s
   evaluated with the same distance-dependent mass floor" is the resolved-only
   scheme. The production default (`lensing.h:78`, subhalo_model = 3) reduces
   the host by the *full* bound fraction f_s,b and adds the unresolved band as
   its azimuthally-averaged mean profile plus a Gaussian:
   ```latex
   The unresolved subhalos are not lost: the host is reduced by the full bound
   fraction, and along each sightline the unresolved population contributes its
   exact mean convergence profile $\mu_{\rm u}(r)$ plus a Gaussian fluctuation
   with the matching Campbell variance $\sigma_{\rm u}^2(r)$, evaluated at the
   host-centric distance $r$ of the ray. This decomposition is exact in the
   mean and variance of the convergence for any value of
   $\epsilon_{\rm sub}$~[App./doc ref], so $\epsilon_{\rm sub}$ controls only
   which part of the scatter is carried by explicitly sampled clumps.
   ```
5. **Giocoli citation**: the α_f = 0.815 e^(−1/4) 2^0.707 median-weight formula
   is Giocoli, Moreno, Sheth & Tormen 2007, MNRAS 376, 977 (eq. 10;
   `Giocoli:2006yz`); `Giocoli:2011hz` (= Giocoli et al. 2012, MNRAS 422, 185)
   is the follow-up. Cite `Giocoli:2006yz` (optionally both). The formula itself
   matches the code (`subhalo.cpp:142-144`, the fixed e^(−1/4) version) ✓.
6. **Eq. (shmf) tilde-γ vs Eq. (fsnorm)**: consistent with the code
   (`subhalo.cpp:141-146,182`) ✓; state (α,β,ω) source as JvdB14 eqs. 21–26.
7. **κ_thr rule**: "expected number … N̄_l = 100" matches the legacy fixed-⟨N⟩
   default ✓ (`lensing.h:50-63`). If you want a robustness sentence: the PDF is
   converged w.r.t. the threshold (plateau below κ_thr ≈ 10⁻⁴; fixed-⟨N⟩ carries
   an accepted ~1.2×10⁻³ JSD z-correlated error at z_s = 10).
8. **Mean-κ anchor**: if the paper quotes ⟨κ⟩ = 0 flux conservation anywhere,
   remember the batch-mean compensation bug note (CLAUDE.md item 12 ⚠): quote
   trimmed/clipped statistics only, or use kappa_anchor = 1 runs.
9. Typos/notation: "magnifies the signals by factor μ induced by structures" →
   rephrase; Eq. (mueq) ends "\,." then "where" — comma; "Number of halos" →
   "The number of halos"; "effect on can be" → "effect can be"; "quiet
   expensive" → "quite"; "dispalyed" → "displayed"; "$\bar N_l$" vs "$\bar N$"
   notation mismatch between Eq. (dN) text and the clustering section; shear:
   define $\gamma_\epsilon$/$\kappa_\epsilon$ subscript ε (ellipticity) at first
   use; tikz ML diagram: $\vect\Theta$ list should match ml/params.py naming
   (z_eq in units? — CONTEXT_KEYS uses zeq/1000 internally, irrelevant for the
   paper but keep the symbols consistent with §"1+6d parameterization").
10. **Shear convention** (if a referee asks): the single-angle γ projection is a
    deliberate, documented choice (kept 2026-07-02; ~0.5% on ⟨γ²⟩,
    `lensing.cpp:898-901`); consider one endnote sentence.

## Figure/script inventory for the revision

| draft slot | file to use | status |
|---|---|---|
| Fig. clustering (new) | `plots/fig_clustering_field.pdf` | generated now |
| Fig. subhalo two-panel (new) | `plots/fig_subhalo_population.pdf` | generated now |
| Fig. `fig:kappa_convergence` | `paper_prod/plots/figures/sigma_partition_vs_kthr_bias.pdf` | exists; caption to write |
| Fig. `fig:subhalo-factor` | `plots/sigma_k_vs_subhalo_factor.png` | placeholder, regenerate |
| Fig. `fig:variance_DL` | `plots/variance_D_L.pdf` | caption: fix spacing, move conclusion to text |
| pyHalo comparison | `plots/figures/pyhalo_vs_ours.png` | exists |
