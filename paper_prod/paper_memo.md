# paper_memo.md — draft ↔ code map ("Learning weak lensing magnifications")

Purpose: for any paper section/equation, find the implementation fast (and vice
versa). The .tex itself is NOT in this repo (Overleaf); `paper_prod/` holds plot
style, figure scripts, and memos. Line numbers are as of 2026-07-20 — they drift,
so function names are authoritative; `grep -n` the name if a line is off.
Companion: `paper_prod/draft_comments_memo.md` (2026-07-20) = per-comment
resolutions of Ville's \R{} notes + ready LaTeX + known draft↔code mismatches.

Update this file when the paper or the touched code changes (same living-document
rule as CLAUDE.md; date your edits).

## Global objects (used by every section)

| thing | where |
|---|---|
| cosmology tables: grids (M 1e7–1e17 ×100, z 0.01–10.01 ×100), σ(M) smooth-k window, δ_c(z), D_g(z), d_c(z), EH98 T(k), P(k) normalization (σ8- and A_s-mode) | `cpp/cosmology.{h,cpp}`; A_s-mode `initialize_normalization` in `cosmology.h` |
| HMF (ellipsoidal first-crossing p=0.3, q=0.8) | `cosmology.cpp::pFC` (:190), `HMFlistf` (:206) |
| halo bias b(M,z) (ST99, p=0.3, q=0.75) | `cosmology.cpp::halobias` (:462) |
| concentration c200(M,z) (Dutton–Macciò 14) | `cosmology.cpp::cons14` (:365); NFW rs/ρs/c tables at 200ρ_c(z): `NFWlistf` (:419, r200 = c·rs) |
| NFW lensing kernels κ0, Fg, pseudo-elliptical κ/γ | `lensing.cpp`: `kappa0NFW` (:45), `FgNFW` (:28), `kappagammaNFWeps` (:56); ellipticity ε(M,z) Allgood+06: `epsilonNFW` (:50) |
| Python mirror of all of the above (numpy-only; use when the C++ build is unavailable, e.g. Linux sandbox — validated vs gwlensing helpers) | `scripts/convergence/bias_field_prototype.py::Cosmo` |
| entry points / defaults (Nreal, Nhalos=100, kappathr_flat, subhalo_*, bias_*, kappa_anchor) | `cpp/lensing.h::LensingConfig` (:48–154); pybind: `cpp/python_bindings.cpp` |

## §II Weak lensing

- D_L(z) = D̃_L/√μ: applied in `lensing.cpp::loglikelihood` (:1209) /
  `Hubble_diagram_fit` (:1289); D_L: `cosmology.h` (`DL`), python port `Cosmo.DL`.
- Eq. (mueq) μ = 1/[(1−κ)²−γ²]: `lensing.cpp::sample_lnmu` (:1036–1160) — detA,
  ⟨κ⟩ flux anchor (:1040–1067; kappa_anchor modes 0/1/2, batch-mean bug note),
  invalid-sample bookkeeping `cpp/invalid_stats.h`; P(lnμ) hist `Plnmuf` (:1163).
- κ, γ single-angle sums over lenses: host add `lensing.cpp` add_host (:842–944,
  γ projection comment :898–901); realization struct `lensing.h::RealizationRaw`
  (κ, γ1, γ2, κ_nosub, κ_weak).

### §II.A.1 Clustering (correlated 1D field, bias_model=1)

- Eq. P_1D(k∥): `lensing.cpp::BiasField1D::build` + `biasWindow2` (:390–550).
  Window is a selectable enum `bias_window`: 0 = transverse disk 2J1(x)/x on k⊥
  (bitwise default), 1 = spherical 3D top-hat 3(sinx−xcosx)/x³, 2 = Gaussian,
  the latter two on |k|=√(k∥²+k⊥²).
  **PAPER/PRODUCTION = top-hat (window 1), R_s = 20 Mpc** (user decision
  2026-07-20; **SETTLED — Gala confirmed the decision 2026-07-23**, no longer
  "pending sign-off" for the paper value; the code-default flip is a separate
  repo matter still tracked in CLAUDE.md). §II.A.1 uses the isotropic form
  P_1D = (1/2π)∫_{k∥}^∞ dk k P(k) W̃²(R_s k), W̃ = 3(sinx−xcosx)/x³,
  σ²_1D = σ²(R_s). Gaussian robustness at matched σ²: R_G = 9.45 Mpc.
- ⚠ R_s = 20 Mpc, NOT R_L(1e14) = 8.44 Mpc (2026-07-20): R_L is a Lagrangian
  radius, not a smoothing scale; 20 Mpc is larger/PBS-cleaner and matches
  Ville's original draft. σ_TH 0.972 (8.44) → **0.530 (20 Mpc)**, clustering
  var ~30%. Window-shape robustness (top-hat vs Gaussian) is R-independent
  and stands. `bias_window_sigmaR.py --R 20000`; §5b of
  `data/results/bias_window/report.md`.
- **R_s dependence MEASURED at 20 Mpc (2026-07-20)** — this is what backs the
  draft's "We show later how changing $R_s$ impacts our results", which until
  now had NO result behind it (R_s appears at :72–76, :99, :105, caption, then
  nowhere). Full config (with the §II.A.3 conditional weak arm), 240k/arm,
  R_s = 5/10/20/40/80 Mpc: Var(lnμ)/no-clustering = 1.99/1.45/**1.15**/1.05/1.02
  at z_s=0.5, 1.62/1.30/**1.11**/1.04/1.01 at z_s=1, 1.32/1.18/**1.08**/1.04/1.02
  at z_s=5 (bold = production). ⚠ The falloff is only APPROXIMATELY σ²(R_s):
  good to ~10% for z_s≲1 at R_s≲20 Mpc, shallower at R_s≥40 Mpc and at z_s=5
  — do not write it as a σ²(R_s) law.
  **PAPER FIGURE: `paper_prod/scripts/plot_fig_clustering_rs.py` →
  `plots/fig_clustering_rs.{pdf,png}`** (single column, paper style, joint arms,
  R_s=20 marked; z_s ramp validated: CVD ΔE 16.3, contrast ≥3:1). Study version
  (2-panel, incl. JSD + counts-only): `plots/bias_window_rs_scan.png`; table
  `data/results/bias_window/rs_dependence.md`. ⚠ Quote the JOINT numbers:
  counts-only arms give only 1.07/1.04/1.02 at 20 Mpc and would misrepresent
  clustering as nearly inert.
- ⚠ Default STAGED (user 2026-07-20): shipped default stays `bias_model=0`
  (legacy iid) + `bias_window=0` (disk). Production config
  `bias_model=1, bias_window=1, bias_Rperp=20000` passed EXPLICITLY. Full flip
  (incl. bias_model 0→1 = clustering on by default, and the 8.44→20 reversal)
  bundled for Ville's sign-off. `docs/bias_window_design_plan.md`.
- Paper/fig updated: `paper_prod/draft_revised_2026-07-20.tex` clustering block;
  `plots/fig_clustering_field.{pdf,png}` (headline 20 Mpc + 8.44 comparison).
- Eq. δ_1D mode sum / σ_n²: realized as exact shell-covariance + Cholesky
  (equal in law): `BiasField1D::build` mode sum (:526–~600); L = 1.05χ(z_s)
  (:480), N_max = L/R⊥ (:481).
- Eq. (n) + λ lognormal (Coles–Jones): sampling loop `lensing.cpp` (:946–959);
  amplitude bDg + per-SHELL compensation bcomp (:828–840) — compensation uses the
  shell's own segment variance, not global σ²_1D.
- Legacy iid layer (bias_model=0, bitwise default): `deltaNhfNFW` (:108, σ_b at
  :128) + loop (:955–958).
- Python replicas: `playground/bias_field/validate_field_covariance.py::cpp_field`
  (verbatim build), `bias_field_prototype.py::LOSField`; design:
  `docs/bias_field_design_note.md`; R⊥ decision: `data/results/rperp_pdf_scan/`.
- Paper figure (P_1D + λ realizations):
  `paper_prod/scripts/plot_fig_clustering_field.py` → `plots/fig_clustering_field.pdf`.

### §II.A.2 Field halos and filaments

- Eq. (dN) expected counts per (z, M, r): `lensing.cpp::deltaNhfNFW` (:108),
  total `NhfNFW` (:138), rmax(κ_thr) `rmaxfNFW` (:88).
- κ_thr rule (N̄_l = 100 legacy default / flat override): `findkappathr` (:614),
  selection logic in `sample_lnmu_raw` (:668–680); config `lensing.h:50–63`.
- Filaments: mass function `cosmology.cpp::pFCfil` (:299, p=0, q=0.7 — Yan & Fan
  2011 barrier) + `dndlnMfil` (:328); cylinder kernels `kappaCYL2/gammaCYL2`
  (:220/:226), counts `deltaNhfCYL` (:266); geometry r_F, L_F ∝ M^{1/3}, 14.4ρ_c
  and sampling: `lensing.cpp` (:822–826, :986–1021). Filament-specific PBS bias
  `b_F(M,z)` RESOLVED 2026-07-23: `cosmology::filbias` + `lensing.h::fil_bias`
  (default false, staged like `bias_weak`/`bias_window`; production config sets
  it true) — `docs/filament_bias_note.md`; draft text updated, decision-pending
  note removed.

### §II.A.3 Sub-threshold contribution (κ_W)

- Eq. (wk) moments w_n per cell: `lensing.cpp::weakMomentsNFW` (:351–388);
  legacy unconditional σ_W: `sigmakappaW` (:158; eps_floor semantics :14–18,
  absolute floor 0.001·κ_thr_default at :689 and :772).
- Eq. (weakcond) conditional E/Var + the Cox-split draw
  κ_W = ΣS_i(δ̄_i) + N(0,1)√(ΣV_i(δ̄_i)): tables `BiasField1D::buildWeak`
  (:430–467) + `weakSV` (:415–426); draw in `sample_lnmu_raw` (:765–786);
  bias_weak config `lensing.h:139–153` (NFW only, no filament weak arm).
- Eqs. (sigmaW/sigmashot/sigma2h/xi1D) partition: analytic probe
  `playground/sigma_partition_bias_vs_kthr.cpp` (includes lensing.cpp →
  production BiasField1D); measured sweep
  `playground/sweep_sigma_partition_components.py`.
- Fig. `fig:kappa_convergence`: USE the bias-era figure
  `paper_prod/scripts/plot_sigma_partition_vs_kthr_bias.py` →
  `paper_prod/plots/figures/sigma_partition_vs_kthr_bias.{png,pdf}` +
  `data/sigma_partition_vs_kthr_bias_z1.npz` (NOT plots/sigma_partition_vs_kthr.pdf
  = legacy shot-only; NOT plot_sigma_k_vs_kappa_threshold.py = mislabeled npz).
  ⟨N⟩-axis companion: `plot_sigma_partition_vs_N_bias.py`.

### §II.A.4 Subhalos

All in `cpp/subhalo.{h,cpp}` + the host-side hooks in `lensing.cpp`:

- SHMF constants (α,β,ω, ψ_res=1e-4, ψ_max=1, m_floor=1e7): `subhalo.h:20–21`.
- Eq. (shmf) sampling (power-law inverse CDF + exact Poisson thinning of the
  exp cutoff): `subhalo.cpp::addClumps` (:491–533).
- Eq. (fsnorm) normalization γ̃ (JvdB14 eq. 23): `precompute` (:141–146, :182);
  full bound fraction f_s,b over [m_floor/M, 1] (model-3 host reduction):
  `buildWsubBin` (:258–263).
- z_f (Giocoli+2007 w̃_f = 1.1925, fixed e^{−1/4}): `precompute` (:142–144,
  bisection :160–168).
- N_τ + Δ_vir (Bryan–Norman; 6.006 = 9.78/1.628 from JvdB14 eq. 2):
  `precompute` (:170–178).
- f_s = 0.3563 N_τ^{−0.6} − 0.075 (JvdB14 eq. 26): `precompute` (:180).
- Eq. (reducedhost) host reduction: model 1 (resolved-only f_s_res, dynamic
  floor) `lensing.cpp` (:848–881); model 3 (FULL f_s,b) (:882–894); reduced-host
  NFW at (1−f_s)M `subhalo.cpp` (:195–197).
- Eq. (radial) anti-biased profile x²/(1+cx)²·[1+(x/0.54)^{−5/2}]^{−1/2}:
  inverse-CDF table `precompute` (:202–227); same B(x) in `buildWsubBin` p3
  (:268–272). c200/r200 of host from NFWlist: (:198–200). ⚠ provenance of
  (0.54, 5/2) unconfirmed — see draft_comments_memo R6 before citing.
- Resolution ε_sub (= `subhalo_factor`, default 1e-2): clump-reach table r_thr
  `precompute` (:130–139); dynamic floor `addClumps` (:418–433); host-side
  mirror (:856–864).
- Unresolved term (model 3, default): tables `buildWsubBin` (:245–386), lookup
  `wsubTerm` (:389–402), applied μ_u + σ_u·N(0,1) in `lensing.cpp` (:927–935);
  derivation `docs/subhalo/wsub_gaussian_term_derivation.md`; acceptance
  `data/results/subhalo_factor_jsd/report.md`. ⚠ draft currently describes
  model 1 here — fix per draft_comments_memo item 4.
  **Mass-conserving carve (scheme A) — draft AND code now agree (2026-07-22):**
  Eq.(reducedhost) reads `κ_NFW[M − Σ_i m_i − M_u]` (explicit form, f̂_s dropped;
  exact per-realization total-mass conservation, supervisor's requirement) + the
  Campbell μ_u/σ_u² integrals as Eq.(kappau_moments); the three "exact mean+variance
  for any floor/ε_sub" claims softened to "mean exact, variance sub-percent."
  **C++ IMPLEMENTED** behind `subhalo_carve` (default on), gated to model 3 + the
  brute reference: flag in `lensing.h` (:83–92) + `lnmu_wrapper.{h,cpp}` +
  `python_bindings.cpp` (5 entry points + config dict); reorder + carved host mass
  in `lensing.cpp::add_host` (early branch :886–960); realized Σm_i via
  `subhalo.cpp::addClumps` `mass_out` out-param; `M_u(r)` via new
  `subhalo.cpp::unresolvedMass`; guard counter `LensingProfile.subhalo_carve_negatives`.
  Conditioning C/D NOT implemented (inert/harmful). Full design + sandbox
  verification: `docs/subhalo/mass_conserving_carve_note.md` §9; harness
  `tmp/carve_verify.cpp`; pytest `tests/test_subhalo_carve.py`. Single-host POC
  (`playground/analytic/{mass_budget_carve_demo,carve_single_host_kappa_pdf,carve_var_seeds}.py`):
  old model 3 sat +1.48%±0.15% above carved brute in Var(κ); carve closes it to
  +0.40%; A≈C≈D. **Still pending on the user's Mac:** `make build` + `pytest
  tests/test_subhalo_carve.py`, `subhalo_factor_jsd` acceptance rerun +
  Fig.(subhalo-factor) recheck, ML-data regen + emulator edge/flux re-check. Figs:
  `plots/subhalo_mass_budget.png`, `plots/subhalo_carve_var_gap.png`,
  `plots/single_host_kappa_pdf.png`.
- Paper figure (SHMF + radial, incl. python port of the f_s chain):
  `paper_prod/scripts/plot_fig_subhalo_population.py` →
  `paper_prod/plots/figures/fig_subhalo_population.{pdf,png}` (promoted into the
  paper figure dir 2026-07-20, facecolor=white dpi=300 like the siblings; NO
  LONGER writes repo-root `plots/`). **Reworked 2026-07-20 into a comparison
  figure, then simplified to a SINGLE 10¹³ M⊙ host at z=0.5** (per-model colors,
  not per-host; axes x∈[1e-4,1], y∈[1e-2,1e3]). Panel (a) overlays two external
  bound-mass references on the same host: Diffhalos (Zacharegkas+26
  arXiv:2607.10419; exact numpy port of their CCSHMF sig-slope kernel, verified
  vs the JAX package <1e-6; native masses are PEAK/unevolved — no evolved MF
  exists in Diffhalos) **converted to bound at runtime by the EMERGENT stripping
  factor s_eff = f_s,b / f_peak ≈ 0.113** (both integrated over [m_floor/M,1];
  parameter-free — our f_s sets the conversion; a ψ→s·ψ shift since dN/dlnψ is
  number-conserving; shown dashed to ψ≤0.5·s_eff where the differential-of-
  sigmoid is still clean, ~5% rolloff wiggle beyond is invisible on the log
  axis); and pyHalo bound masses (evolved; `data/pyhalo_vs_ours.npz`, now the
  **2000-realization** matched run — host 1e13, z_l=0.5, z_s=2, m_infall up to
  the host mass, R<38 kpc aperture, deprojected by our f_ap=0.0198; reaches
  ψ≈0.34 with a smooth rolloff; per-host ratio to ours ≈0.5–0.7 at low ψ,
  ~1.0–1.6 through the rolloff, consistent with
  `docs/subhalo/pyhalo_pipeline_comparison.md` §4). Regenerate more realizations
  with `.venv_pyhalo/bin/python tmp/pyhalo_vs_ours_plot.py --mhi 1e13 --nreal N
  --suffix …` (~0.65 s/real) then copy the tmp npz onto `data/pyhalo_vs_ours.npz`.
  Panel (b) is the radial BIAS function B(x)=n_sub/n_host log-log (subhalo
  number density over host NFW; replacing the dN/dx panel; old version in git).
  **Digitization minimized to ONE dataset 2026-07-21** (searched SatGen
  sheridan branch/DASH — model code only, no data tables; Green+21 curve = model
  output, Bolshoi points exist only in Green+21 Fig. 7): the digitized Green+21
  "withering+disruption" curve was DROPPED (the adopted fit already stands in
  for it — cite Green:2021 in the caption for the fit target); Bolshoi points
  (Klypin:2010qw) KEPT as the sole digitized set, taken from Fig. 7 of Green:2021
  (`scripts/subhalo_gate/fit_bias_profile.py`; GREEN_LOGX/B arrays removed from
  the plot script). Analytic curves (NOT digitized, from published params):
  adopted fit [1+(x/0.54)^{-5/2}]^{-1/2}; Springel+08 Aq-A-1 Einasto/NFW
  (α=0.678, r₋₂=0.81 r200, c_NFW=16.11, verified from arXiv:0809.0898
  §3.2+Tab.2, add bib); Han+16 x^1.3 (Aq-A, their Fig. 1; model γ=αβ~1, add
  bib). Caveats for the caption: diffhalos CCSHMF is z-independent (JvdB16-tuned,
  hosts 1e11–1e15); pyHalo deprojection assumes our radial profile (slight
  underestimate of their halo average); the (0.54, 5/2) x is r/r200 vs r_vir
  calibration nit (draft_comments_memo R6) still applies.
- Figs. `fig:subhalo-factor-mean` / `fig:subhalo-factor-scatter` (formerly the single
  two-panel `fig:subhalo-factor`, split 2026-07-23 into two single-panel figures per
  user request):
  `paper_prod/scripts/plot_fig_subhalo_sigma_decomposition.py` →
  `paper_prod/plots/figures/fig_subhalo_sigma_decomposition_{mean,scatter}.{pdf,png}`.
  Host/subhalo convergence + scatter decomposition for a single M=1e13 Msun host
  (z_l=0.5, z_s=1) under subhalo_model=4 (brute to psi_min=m_floor/M, carved host);
  REPLACES the obsolete eps_sub partition figure `sigma_partition_vs_subhalo_factor.png`
  (dropped 2026-07-23 with the kappa_u/unresolved apparatus). Uses
  `paper_prod/plot_style.py` (`apply_style`, `FIGURE_SIZES["single"]`,
  `format_log_axis_decimal` for the 0.1/1/10 log-tick convention).
- Design/derivation docs: `docs/subhalo/subhalo_combining.md`,
  `docs/subhalo/variance_derivation.md`; papers in `papers/context/`
  (JvdB14, vdB05, Han+16, BMO09, Green+21).

## §II.B Magnification PDF

- Raw sampler + PDF: `sample_lnmu_raw` (:630), `sample_lnmu` (:1036),
  `Plnmuf` (:1163); python API `python/testing_api.py`, bindings
  `cpp/python_bindings.cpp` (`sample_lnmu`, `sample_lnmu_ml`,
  `sample_lensing_raw_ml` incl. per-ray `kappa_weak`, `compute_lnmu_stats`,
  `get_simulator_config`).
- Tail/edge/flux facts for the text: `docs/edge_tail_flux_note.md` (μ⁻² tail,
  empty-beam edge, flux calibration target).
- Figs. `fig:magpdf_zs` / `fig:magpdf_ingredients` (added 2026-07-24, Sec. II.B):
  dP/dμ vs μ for z_s=0.5/1/2/5/10 (full config), and dP/dμ at z_s=5 for
  baseline/+subhalos/+clustering. Generator
  `paper_prod/scripts/plot_fig_magnification_pdf.py` →
  `paper_prod/plots/figures/fig_magnification_pdf_{zs,ingredients}.{pdf,png}`;
  run guide `paper_prod/scripts/README_magnification_pdf.md`. **Heavy MC run by
  the USER** (needs the model-4 Mac build + test env; smoothness via --nreal or
  seed shards + --combine). Configs: full = bias_model=1/window=1/Rperp=20000/
  bias_weak/fil_bias + subhalo_model=4; baseline = bias_model=0, no subhalos;
  kappa_anchor=1 (robust). Caches histograms to plots/data/magpdf_<tag>.npz for
  instant --replot. Plotted window mu≤1.8 only (far tail uncertified).

## §III Machine learning

- Θ = {h, z_eq, Ω_M, Ω_B, **σ8**, n_s} + z_s: single source of truth
  `ml/params.py` (FIDUCIAL, PRIOR_6D, WIDE_6D, CONTEXT_KEYS). **Amplitude
  switched A_s → σ8 on 2026-07-27** to match Vaskonen's paper (θ = {Ω_M, h, σ8},
  priors σ8 ∈ [0.4,1.4]) and his code; A_s-mode retained as an unused escape
  hatch (`cosmology.h::initialize_normalization`, Bunn–White). If the tikz Θ
  diagram or §III text still shows A_s, update it.
  ⚠ **σ8 window caveat — do NOT write "top-hat" in the paper yet.** The engine
  normalizes with the smooth-k window `Ws`, so σ8 = 0.811 here is a smooth-k σ8;
  the top-hat value is 0.7786. Vaskonen's §2 claims a real-space top-hat, which
  his code does not do. Switching the normalization to top-hat (option b) is
  agreed in principle but pending Ville's sign-off — until then, state the
  window convention explicitly or say nothing about it. See CLAUDE.md §1+6d.
- Data: `python/generate_dataset.py` (LHS 6d wide box, HDF5 schema 2.1, σ8-mode),
  `ml/data.py` (schema autodetect). ⚠ pre-2026-07-12 subhalo-ON data used
  factor 1e-5; 2026-07-09/10 data used flat-1e-3 κ_thr — regenerate/flag.
- Production model (NSF): **`gw-wl-emulator-ar` worktree**, branch
  autoresearch/jun16 — `ml/autoresearch/smooth_model.py::load_smooth_fn`
  (flow body models/flow_smooth_lowz.pt + POT tail + empty-beam edge EDGE_W=0.01
  + flux calibration `cache/flux_target_fit.json`; do NOT use flow_ar.pt).
  Validation numbers (KL 0.0073, PIT ≤2.1%, posterior recovery): 
  `ml/autoresearch/REPORT.md`. Retraining recipe: CLAUDE.md §NSF.
- Fig. `fig:variance_DL` (`plots/variance_D_L.pdf`): **no generating script or
  file found in the repo** (2026-07-20) — lives outside or TBD; add here when
  created.

## §IV Comparison with earlier works

- ACE-Lensing: `ace_lensing/` (import gotcha: sys.path.insert the inner pkg dir),
  gap analysis `scripts/subhalo_gate/ace_gap.py`, verdict
  `scripts/figures/plot_gate_verdict.py` → `plots/gate_verdict.png`; numbers:
  ⟨κ²⟩ Vaskonen 36–42% above ACE at z_s=1 ∀ Mmin (`docs/convergence_mmin_nz_note.md`),
  subhalo effect +14.9%/+4.5% on ⟨κ²⟩/⟨κ³⟩ at z_s=5 (`validate_split_vs_brute.py`).
- pyHalo: full side-by-side `docs/subhalo/pyhalo_pipeline_comparison.md`,
  figures `plots/figures/pyhalo_vs_ours*.png`.
- Mpetha, Congedo & Taylor 2024 (arXiv:2402.19476): hook = their outdated
  σ_lnμ(z) prescriptions vs our σ_DL(z, Θ) emulator.

## Figures + style

- Style: `paper_prod/plot_style.py::apply_style` (3.37×2.6 single / 7.1×2.8
  double, usetex if available; sandbox guard: `guard_broken_latex` in
  `paper_prod/scripts/plot_fig_subhalo_population.py`).
- Inventory table: end of `paper_prod/draft_comments_memo.md`.

## Standing rules that constrain paper claims (from CLAUDE.md)

Do not quote: bias-tail quantities (f(κ>1), q≳99.9 at z_s≳5, wide-support ⟨κ²⟩)
as converged; raw ⟨1/μ⟩/max/moments (monster-ray junk — clipped/trimmed
ensembles only, κ_tot≤1 is the certified estimator); Mmin as an ACE-gap closer
(wrong sign); κ_min as a convergence knob (it is a model parameter). Legacy-match
is not a correctness criterion (user ruling 2026-07-16).
