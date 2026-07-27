# Delegated task — cascade decomposition: mode analysis (understanding phase)

**Purpose:** Understand *why* the parametric decomposition behaves as it does, BEFORE
building clustering v2. Turn the result into paper figures. This is analysis-only — do not
build or train anything. Prior context: `docs/cascade_phase4_brief.md`,
`data/results/cascade_residual/{report,report_param,report_compose,report_metric}.md`,
`ml/cascade/reconstruction_report.md`.

## What is already settled (don't re-litigate)
- Pointwise-logP emulation is dead (MC-noise-limited, no CRN). We emulate parameter shifts.
- Subhalos load cleanly on **width** (2nd moment); the additive cascade tracks them
  (+22/44/47/46% floor-subtracted KL closed at z_s=2/3/4/5).
- Clustering does **not** reduce to a width correction: its effect sits in the **shoulder**
  and moves **skew** (3rd moment); a width-only warp makes reconstruction worse.
- σ is metric-fragile for clustering → measure on the **production clipped-Var / κ≤1
  estimator with a robust anchor (mean excluding κ>1)**, never IQR.

## The question this task answers
Is each ingredient's correction a low-dimensional, physically-interpretable deformation,
and are the two ingredients' dominant deformations **orthogonal in shape-space** (subhalos
= width, clustering = skew)? If yes, that orthogonality is the mathematical statement of
"each ingredient is a distinct physical effect" — and explains why additive composition
worked (#3). This is the paper's central figure.

## Hypotheses to test (state these up front; report confirm/refute)
- H1: subhalo correction ≈ **one** mode (width) at z_s≲3; possibly a weak **second**
  tail-steepening mode emerging by z_s=5.
- H2: clustering is **also nearly 1D**, but its dominant mode is **skew/shoulder**, in a
  direction **orthogonal to** the subhalo width mode — i.e. the finding is "different mode,"
  not "more modes."
- H3: the SVD shape modes match the analytic parametric directions: PC1 ≈ ∂P/∂σ (width),
  PC2 ≈ ∂P/∂skew — cross-validating the {σ, edge, skew, tail} representation.

## Environment
`test` conda env (Python 3.12). Use the **raw-κ sampler** (`sample_lensing_raw_ml`,
production-currency scalars incl. κ≤1-core σ). Reuse the existing 4-arm shards
(base / halo+sub / halo+bias / full × Om,σ8,h axes × z_s∈{0.5,1,5}, multi-seed) where
possible; regenerate only for the z_s extension below.

## Method — and the trap that must be avoided
Build, per ingredient i∈{sub, bias} and per z_s, the correction matrix
`M_i[θ, lnμ] = logP_{base+i}(lnμ; θ) − logP_base(lnμ; θ)` on a **shared fixed lnμ grid**.

**TRAP (do not skip):** raw pointwise ΔlogP is MC-noise-limited — a naive SVD returns
noise modes and inflates the mode count. Mandatory precautions:
1. **Restrict to body+shoulder.** **Mask the moving edge** (ΔlogP diverges at the empty-beam
   wall and would dominate the singular values). Keep the edge as a **separate scalar**
   (edge-location shift), consistent with the parametric decomposition — do NOT feed it to
   the SVD.
2. **Noise-weight the SVD.** Estimate the per-bin seed-to-seed variance of ΔlogP from the
   multi-seed blocks; do a noise-weighted / whitened SVD (or smooth-then-SVD), and **plot
   the noise floor on the singular-value spectrum** so retained modes are visibly above it.
   A mode is only "real" if its singular value clears the floor.
3. **Be explicit about centering** — report both readings and label which "% variance
   explained" each refers to:
   - **Uncentered** SVD → dominant *correction shape* (answers "does the correction look
     like a width change"; the leading right-singular-vector includes the fiducial shift).
   - **θ-centered** SVD → *cosmology-dependence* dimensionality (how the correction varies
     across Om/σ8/h).

## Analyses to run
A. **Per-ingredient SVD** (each ingredient, each z_s): singular-value spectrum vs noise
   floor → how many modes clear it. Plot the leading 2–3 right-singular-vectors (lnμ-shape
   modes). Do they look like width (symmetric broadening) and skew (antisymmetric
   shoulder)? Report the θ-loadings (left vectors) vs cosmology.
B. **Stacked orthogonality test** (the key figure): stack subhalo and clustering rows into
   one `[ingredient × θ × lnμ]` matrix, SVD it, and report the **overlap/loading matrix** —
   do subhalo rows load on one singular vector and clustering rows on a different one?
   Quantify the overlap (cosine) between the subhalo-dominant and clustering-dominant shape
   modes; near-zero overlap = orthogonal deformations = the thesis.
C. **Parametric cross-check (H3):** analytically construct the base-PDF shape derivatives
   ∂P/∂σ, ∂P/∂skew, ∂P/∂(tail index) at each z_s (finite-difference the parametric operators
   is fine), and project the empirical SVD modes onto them. Report cosine overlaps — a high
   overlap independently validates the {σ, edge, skew, tail} representation.
D. **z_s = 6, 8, 10 extension (scientific narrative only):** regenerate base + halo+sub
   arms at z_s∈{6,8,10} (clustering optional — expected near-null; include base+bias at
   z_s=6 only as a spot check). Repeat A. Look specifically for an **emergent second
   subhalo mode** (tail steepening) as the effect grows.
   **HARD GUARDRAIL:** at z_s≳5 the absolute tail is model-uncertified in this method class
   and the clustering 2-halo variance has no sane top-mass limit — **restrict ALL z_s≳5
   statements to body/shoulder shape and width behavior; make NO tail-amplitude claims.**
   This extension is paper-understanding, NOT gate-relevant (the event population caps at
   z_s=4).

## Deliverables
- Scripts under `scripts/cascade_diag/` (env-`test` runnable).
- `data/results/cascade_residual/report_modes.md` (+ tables) with: per-ingredient
  singular-value spectra + noise floors, mode-count verdicts, H1/H2/H3 confirm-or-refute,
  the stacked-overlap number, and the z_s→10 mode-structure summary.
- Figures (paper-candidate): ΔlogP_sub and ΔlogP_cluster curves per z_s; singular-value
  spectra with noise floor; leading mode shapes (width vs skew); the ingredient×mode
  overlap matrix; SVD-vs-parametric-derivative overlay.

## Do NOT
- Do not build clustering v2 or any emulator/training — this is analysis only.
- Do not SVD un-denoised ΔlogP; do not include the edge region; no IQR σ anywhere.
- Do not make tail claims at z_s≳5.
- Do not touch `~/Desktop/halos`. No commits/pushes (user handles git).
