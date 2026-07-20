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

- Eq. P_1D(k∥) [transverse DISK window 2J1(x)/x on k⊥, NOT isotropic 3D top-hat —
  see draft_comments_memo R1]: `lensing.cpp::BiasField1D::build` (:469–524,
  header comment :317–342); R⊥ default 8441 kpc = R_L(1e14): `lensing.h:137`.
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
  and sampling: `lensing.cpp` (:822–826, :986–1021). Filaments share the halo λ
  (same bias) — PBS filament-bias recommendation: draft_comments_memo R2.

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
- Paper figure (SHMF + radial, incl. python port of the f_s chain):
  `paper_prod/scripts/plot_fig_subhalo_population.py` →
  `plots/fig_subhalo_population.pdf`.
- Fig. `fig:subhalo-factor`: existing file is `plots/sigma_k_vs_subhalo_factor.png`
  (draft's filename doesn't exist); regenerate via
  `scripts/figures/plot_variance_vs_factor.py` lineage.
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

## §III Machine learning

- Θ = {h, z_eq, Ω_M, Ω_B, A_s, n_s} + z_s: single source of truth
  `ml/params.py` (FIDUCIAL, PRIOR_6D, WIDE_6D, CONTEXT_KEYS); A_s-mode mapping
  `cosmology.h::initialize_normalization` (Bunn–White; σ8 smooth-k gotcha:
  derived σ8 = 0.860 at Planck A_s — expected).
- Data: `ml/generate_dataset.py` (LHS 6d wide box, HDF5 schema 2.0),
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
