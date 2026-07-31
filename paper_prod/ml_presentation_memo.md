# ml_presentation_memo.md — how §III should present the emulator and its results

Companion to `paper_writer.md` (writing contract), `paper_memo.md` (draft ↔ code map)
and `draft_comments_memo.md` (per-comment resolutions). This file covers one thing:
**how the machine learning and its validation are presented**, benchmarked against
`papers/core/Trker_2025_AccurateCosmologicalEmulatorProbabilityDistribution.pdf`
(ACE, Türker et al. 2026, A&A) and against the norms of cosmological emulator papers.

Created 2026-07-30. Living document — date every edit.

**Scope note.** Everything here is about *presentation*. It does not change physics,
and it does not licence quoting any number that the standing rules in
`paper_writer.md` §4 forbid. All emulator numbers currently in the draft are
**pre-retrain and on the old physics** — see §7.

---

## 1. Diagnosis

§III of `draft_revised_2026-07-20.tex` (l.414–712) is ~1100 words of architecture
and calibration mechanics, **one results paragraph** (l.683–693), **one headline
number** (median KL = 0.0073), and **zero validation figures**.

The section's three figures are:

| figure | what it shows | evidential value |
|---|---|---|
| `fig:flow_transformation` | the generative path, base → output | pedagogical; shows *how it works* |
| `fig:emulator_arch` | the three-branch composite | pedagogical; shows *how it is built* |
| `fig:variance_DL` | σ_DL(z) decomposition | **a physics result, not an emulator result** — misplaced in §III |

So the emulator is described carefully and demonstrated barely at all. ACE devotes
**seven figures (9–15) plus two tables** to nothing but performance. That asymmetry
is the entire difference in how "finished" the two sections read.

The imbalance is also internal: §III.A spends a paragraph on `K=3`, `B=10`,
`[-16,16]`, `2×128` hidden features, Student-$t$(2) — more space than the accuracy
gets. In this genre the ratio should run the other way.

---

## 2. Questions §III currently cannot answer

Each of these is a routine referee question for an emulator paper. None is
answerable from the current text.

1. **Over what domain is the emulator valid?** The training box appears nowhere in
   either `.tex`. It is now confirmed (`ml/params.py::TRAINING_RANGE`, Ville
   2026-07-30) and must be tabulated. ACE's Table 1 is exactly this.
   ⚠ Quote the **reachable** range, not the intended one: `TRAINING_RANGE` says
   `z=(0.2, 12.0)` but the engine clamps at `ENGINE_ZMAX = 10.01`
   (`cpp/lnmu_wrapper.h:27`), so the paper must say 10 (or the clamp must be fixed
   first). Writing 12 would be false.
2. **How much training data?** "Monte Carlo realizations drawn over a Latin
   hypercube" — how many configurations, how many rays each? ACE: 1447 PDFs,
   80/20 split → 1157 train / 290 test, stated plainly. The unverified
   "10⁵ realizations across a 70-point Latin hypercube" was correctly deleted
   2026-07-29 (draft comment l.636) and nothing replaced it.
3. **What does the error *distribution* look like?** The draft quotes medians only.
   ACE gives 16–84 percentiles on every point of Fig. 9, a full histogram in
   Fig. 14, and says outright that the median "does not fully capture the spread or
   presence of outliers". For an emulator entering a likelihood the tail of the
   error distribution matters more than its centre.
4. **Is the emulator error below the Monte Carlo noise floor of its own training
   data?** This is the most persuasive single number an emulator paper can give,
   and it is computable here — the shard-level floor methodology is already used
   throughout the physics sections (`scripts/convergence/*`). ACE's honest closing
   remark is that "performance is primarily limited by noise in the ML
   predictions, which could be reduced with a larger training set."
5. **How much faster?** The abstract says "orders of magnitude" with no number.
   Needs: *X* ms per PDF evaluation vs *Y* CPU-seconds per Monte Carlo evaluation,
   on stated hardware, at stated `Nreal`.
6. **Does the error matter for the science?** Nobody acts on a KL value. They act
   on whether the emulator biases σ₈. The measurement exists (posterior recovery,
   §4) and is not in the paper.

---

## 3. What to take from ACE, and what to leave

### Take

| ACE | our analogue | why it earns its space |
|---|---|---|
| Table 1 — parameters + simulation setup | `TRAINING_RANGE` table + N configs + rays/config + `PRODUCTION_CONFIG_HASH` | first thing a referee checks; also fixes the "what is z_s max" question |
| Fig. 13 — emulated vs true PDFs at representative panels, KL in the legend | **the single most important missing figure** | it is the only figure that lets a reader see the fit rather than trust a scalar |
| Fig. 14 — KL histogram over the test set | same | shows the spread and the outliers the median hides |
| Fig. 15 — KL vs each parameter and z | same, 7 axes | shows *where* it degrades; preempts "did you only test near the fiducial?" |
| Fig. 9 — accuracy vs a complexity knob, separating compression error from ML error | **learning curve**: KL vs training-set size | tells the referee whether more data helps — the honest statement of the current limit |

Two upgrades on ACE while copying:

- **Add a residual/ratio subpanel to the overlay figure.** ACE's Fig. 13 is a
  log-scale PDF overlay with no residuals, which hides errors of tens of per cent
  in the wings. A lower panel of emulator/simulator − 1 costs nothing and is
  strictly more informative.
- **One histogram coloured by z_s**, not ACE's four separate per-z-bin panels
  (their Fig. 14). Same information, a quarter of the float space, and the trend
  with z_s becomes visible instead of inferred across panels.

Their pred-vs-true idea (Figs. 10–12) has one worthwhile translation even though a
flow has no regressed coefficients: **predict vs simulated summary statistics** —
σ(lnμ), q₀₁, q₉₉ — across held-out configurations. Those are the quantities the σ₈
inference actually consumes, and σ(lnμ) is where the 2026-07 physics work landed
(profile fix +1.54%, assembled config +5.2% at z_s=1). An emulator that gets the
KL right but the width wrong is useless here, and only this figure would catch it.

### Leave

- **Fig. 8** (triangular Spearman matrix, PCA components vs parameters) —
  method-specific bookkeeping, no flow analogue, decorative even in their paper.
- **Fig. 6** (PCA explained-variance ratio) — same.
- **Table 2** (XGBoost hyperparameter grid) — defensible for a grid search over
  tree ensembles; for us a sentence and a small table of the final config suffice.
- **Figs. 10–12** as *six* pred-vs-true panels with R² — inflated. R² near 1 on a
  smoothly varying target flatters itself, and the content is one table. Take the
  *idea* (§ above), not the six panels.

---

## 4. Where we can beat ACE rather than match it

ACE regresses PCA coefficients; ours is a **density** emulator with a tractable
likelihood. Two validations follow that they structurally cannot perform, and both
have already been run in this project:

1. **PIT / coverage.** A probability-integral-transform histogram plus its KS
   statistic proves *calibration*, not merely mean accuracy — that the emulated
   density is right as a density, not just close on average. Script:
   `gw-wl-emulator-ar/ml/autoresearch/validate_pit.py`.
2. **Posterior recovery on mock catalogues.** Run the actual inference on mocks
   drawn from the simulator, with the emulator as the likelihood, and show the
   recovered parameters are unbiased *in units of the posterior width*. Script:
   `validate_posterior.py`. **This is the argument that retires the accuracy
   question**, because Vaskonen's σ₈ forecast is 10–30% and a bias of a fraction
   of σ_post is manifestly harmless. ACE never makes this argument.

**Lead §III's results with the posterior-recovery result, not with the KL.** KL is
the number that makes us comparable to ACE; posterior recovery is the number that
makes the emulator *fit for purpose*. They answer different questions and the
second is the one the rest of the paper depends on.

### The KL comparison needs one honest clause

Our KL measures fidelity to **our simulator**; ACE's measures fidelity to
**N-body**. "0.0073 versus 0.007" is a comparable *scale*, not a competition —
beating them on self-fidelity says nothing about whose physics is right. That
caveat currently lives only in a LaTeX comment (draft l.631–634). It costs one
clause in the text and removes a referee's easiest objection. The head-to-head
belongs in §IV (`sec:comparison`), not §III.

---

## 5. Field norms — what is load-bearing, ranked

What referees of cosmological emulator papers actually check, in order:

1. **Domain of validity**, stated as a table, plus an explicit sentence on
   behaviour outside it (does it extrapolate, refuse, or silently degrade?).
2. **Data volume and sampling design** — LHS or otherwise, N configurations,
   realizations per configuration, and how the training data's own MC noise
   compares with the emulator error. An emulator cannot be better than the noise
   of what it was trained on; say so before a referee does.
3. **An honest split.** What was used for hyperparameter selection, what for the
   edge/tail fits, what was never touched. The draft already words this correctly
   ("held-out configurations that enter neither the flow nor the edge and tail
   fits") — it just needs numbers attached.
4. **Metric defined precisely, and its distribution shown** — not only a median.
   Report median, 68% interval, and the worst case.
5. **Error translated into the science tolerance.** Convert emulator error into a
   bias on the inferred parameters. Highest-value item in the whole section.
6. **Speed, quantified**, with hardware and settings.
7. **Calibration**, for a density emulator: PIT / coverage.
8. **Reproducibility**: code and weights availability. Vaskonen 2026 footnotes his
   code link; do the same.

### Commonly overweighted — keep short or omit

- Architecture minutiae (layer counts, activations, optimizer, LR schedule).
  Nobody reproduces from prose. A paragraph or a compact table.
- **Training/validation loss curves.** Never publish these in a cosmology paper.
- Framework and version names in body text.
- Long justification of the ML method against alternatives. ACE gets this right in
  one sentence ("neural networks are unlikely to provide significant advantages"
  at their dataset size). One sentence is the correct length.
- "Ours beats theirs" framing when the metrics are not measured against the same
  reference (see §4).
- Generative-path illustrations. `fig:flow_transformation` is pleasant and of low
  evidential value — keep it, but it must not remain one of only two ML figures.

---

## 6. Proposed §III structure

```
III. MACHINE LEARNING
  intro ¶                          (as-is; add the speedup number)
  A. Neural spline flow            (compress ~30%; hyperparameters -> small table)
       fig:flow_transformation     (keep, demoted)
       fig:emulator_arch           (keep; this is better than ACE's Fig. A.1)
  B. Composite PDF & calibration   (as-is — this subsection is good)
  C. Training data and validation  (NEW — currently one paragraph's worth of §III.B)
       Table: training box + data volume + config hash
       fig:emulator_panels         overlay + residuals
       fig:emulator_kl             KL histogram, coloured by z_s
       fig:emulator_kl_params      KL vs each of the 7 context axes
       fig:emulator_moments        predicted vs simulated sigma, q01, q99
       fig:emulator_pit            PIT + KS
       fig:emulator_posterior      posterior recovery on mocks
       speedup, and the MC-noise-floor statement
```

Move `fig:variance_DL` **out of §III** — it is a physics result and belongs in §II
or in its own results section.

Six new figures is more than the section can carry at ~12–14 pp. Priority order if
space forces a cut:

1. `fig:emulator_panels` (overlay + residuals) — non-negotiable
2. `fig:emulator_posterior` (posterior recovery) — the fitness-for-purpose argument
3. `fig:emulator_kl` (histogram) — cheap, answers "what about the outliers"
4. `fig:emulator_kl_params` — cheap, answers "where does it break"
5. `fig:emulator_moments` — the σ(lnμ) check
6. `fig:emulator_pit` — can be folded into the posterior figure as a subpanel
7. learning curve — first to drop; can be a sentence quoting the trend

`paper_writer.md` §5 records that the appendix question was closed ("NO APPENDIX",
user 2026-07-28b) but explicitly reopens it if validation material is written up.
**This is that trigger.** Items 4–7 are natural appendix material if the body gets
tight; items 1–3 are body figures.

### Scripts to write

Convention: `paper_prod/scripts/plot_fig_*.py`, style via
`paper_prod/plot_style.py::apply_style()`, log-axis ticks via
`format_log_axis_decimal`, registered in `paper_prod/scripts/run_all.py`.

| new script | data source (in `gw-wl-emulator-ar`) |
|---|---|
| `plot_fig_emulator_panels.py` | `ml/autoresearch/validate_kl.py` + `smooth_model.load_smooth_fn()` |
| `plot_fig_emulator_kl.py` | `validate_kl.py` output over the held-out set |
| `plot_fig_emulator_kl_params.py` | same, plotted against each context axis |
| `plot_fig_emulator_moments.py` | new — sigma/q01/q99 from held-out sims vs emulator |
| `plot_fig_emulator_pit.py` | `validate_pit.py` |
| `plot_fig_emulator_posterior.py` | `validate_posterior.py` (16 x 1000-event mocks) |
| `plot_fig_emulator_learning_curve.py` | new — retrain at 1/8, 1/4, 1/2, 1x of the training set |

⚠ All of these read from the **-ar worktree**, whose `build/` is a symlink to the
main repo's, so a `make build` here updates both (CLAUDE.md, corrected 2026-07-27).
The -ar pipeline now splats `SIM_CONFIG = PRODUCTION_CONFIG` at all 7 call sites
(`ml/autoresearch/simcfg.py`, 2026-07-29) — do not add a bare simulator call.

---

## 7. Every emulator number in the draft is stale — regenerate before writing §III.C

Do not build the figures above on the shipped checkpoint. The numbers currently in
the draft predate:

- the σ₈ re-parameterization (2026-07-27) — the 6d fiducial **point moved**, −11% in
  P(k) amplitude;
- the subhalo radial-profile shape fix (2026-07-28) — **+1.54% on σ(lnμ)** at
  z_s = 1, 7.9σ, a real effect;
- `halobias` q 0.75 → 0.8 and `subhalo_virial` (2026-07-28);
- the paper-default flip (2026-07-29) — assembled-config composition measured
  **+5.96 / +5.16 / +1.79%** on σ(lnμ) at z_s = 0.5 / 1 / 5;
- the flux target switch to `"unit"` (2026-07-29);
- the confirmed training range (2026-07-30) — **every axis moved**.

Affected: median KL 0.0073; the 0.0051 → 0.0034 and 0.0091 → 0.0035 panel numbers;
PIT KS ≤ 2.1%; posterior recovery h −0.20 ± 0.22 σ_post, Ω_M +0.34 ± 0.21 σ_post;
and `fig:flow_transformation`, still plotted from the legacy `K=6` Gaussian
checkpoint (declared in its own caption, draft l.436–440).

Retrain order is in CLAUDE.md §NSF recipe. Note `fit_flux_target.py` drops out of
the recipe under `FLUX_TARGET = "unit"`.

⚠ **The OoD split is currently a no-op** — `WIDE_6D == PRIOR_6D` since the range
confirmation, so every configuration is in-distribution. If §III.C wants to say
anything about extrapolation beyond the training box (norm §5.1), `WIDE_6D` must
be widened first. Never widen `PRIOR_6D`.

---

## 8. Defects found in `production.tex` while reading (user-only file — report, do not fix)

Per `paper_writer.md` §1, these are reported for the user to merge; Claude does not
edit `production.tex`.

1. **`fig:variance_DL` is included twice**, at `:332–337` and `:442–447`, with a
   **duplicated `\label{fig:variance_DL}`**. LaTeX takes the last definition, so
   every `\ref` points at the second float and the first is orphaned. The draft has
   it once, at `:695`.
2. **§III.A contradicts `fig:emulator_arch` in the same file.** The prose at `:342`
   describes `K = 6`, `B = 8`, `[-3, 3]`, Gaussian base; the tikz figure at
   `:353–354`/`:385` says 3 transforms, 10 bins, Student-$t$. These are two
   different checkpoints — `data/models/conditional_nsf_backend_current.pt` and
   `ml/autoresearch/models/flow_smooth_lowz.pt`. `flow_smooth_lowz.pt` is the
   production body and the only one the composite and the KL headline exist for.
   Fixed in the draft (l.466–468); not yet merged.
3. **Missing `\usetikzlibrary{arrows.meta}`** in the preamble. With
   `fig:emulator_arch` present, production **does not compile**:
   `! Package pgf Error: Unknown arrow tip kind 'Stealth'`. Already flagged in the
   draft preamble comment (l.10–12); still absent from Overleaf as of 2026-07-30.
4. **Flux wording is internally inconsistent**: `:329` says the calibration
   "enforces $\langle\mu^{-1}\rangle = 1$", while `:438` correctly describes the
   support-restricted target and `:407` says "shift to enforce". Under the user's
   2026-07-29 decision (`paper_writer.md` §5b) all four sites flip together to the
   sum rule once the code targets `"unit"`; until then they must at least agree.
5. `\Gala{i can also add some fancy ML related plots like ACE paper does}` at
   `:330` — **this memo is the answer.** §3 and §6 say which plots, §3 says which
   of ACE's are not worth the space.

---

## 9. Open questions for the user

- **Speedup number**: which hardware and which `Nreal` do we quote the simulator
  cost at? The per-ray cost varies by ~500× across subhalo models
  (model 5 ≈ 0.21 ms/ray vs model 4 ≈ 45 ms/ray), so the honest comparison is
  emulator vs **`PRODUCTION_CONFIG` at the production `Nreal`**, stated explicitly.
- **Body vs appendix** for validation items 4–7 of §6 — reopens the closed
  appendix decision (`paper_writer.md` §5).
- **Code/weights release** alongside the paper? Affects §5.8 and the ACE
  comparison, since `ace_lensing` is public.
