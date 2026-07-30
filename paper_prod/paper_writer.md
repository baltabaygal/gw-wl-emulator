# paper_writer.md — how to write this paper

Companion to `paper_memo.md` (draft ↔ code map) and `draft_comments_memo.md`
(per-comment resolutions). This file is the **writing** contract: file ownership,
markup conventions, prose style, and the claim-level rules that constrain what may
be written. Living document — date every edit, delete anything that turns out wrong.

Created 2026-07-28.

---

## 1. File ownership — hard rules

| file | who edits | role |
|---|---|---|
| `paper_prod/production.tex` | **user only** | pasted from Overleaf; the authoritative manuscript |
| `paper_prod/draft_revised_2026-07-20.tex` | Claude | working copy; proposals land here |

Claude **reads** `production.tex` to compare, recommend and report, and **never
writes to it** — not even a one-character fix. Surface the recommendation; the user
merges on their own schedule.

### Comment macros

- `\R{...}` — **Ville's** comments. Claude never removes, resolves or rewrites
  these, in either file, *even when the underlying model has since changed*. Only
  Ville closes an `\R{}`.
- `\Gala{...}` — the user's own notes. Claude does not delete these either; answer
  them in surrounding text or in chat.
- `\B{...}` — see §2.

---

## 2. The `\B{}` convention (re-baselined 2026-07-28, again 2026-07-28b)

**`\B{...}` in the draft means: this text differs from `production.tex`.**
Nothing else. It is a merge-candidate marker, not a "new since some date" marker.

Invariant to preserve on every edit:

> Delete every `\B{...}` block from the draft, delete every `\Gala{}`/`\R{}` block
> from `production.tex`, and the two files agree word-for-word (float ordering
> aside).

Mechanical check (run after any batch of draft edits):

```bash
# brace-matched deletion of a macro + its argument, then word diff
python3 - <<'EOF'
import sys,re,difflib
def delete_cmd(s,cmd):
    out=[];i=0;n=len(s);tag='\\'+cmd+'{'
    while i<n:
        j=s.find(tag,i)
        if j<0: out.append(s[i:]);break
        out.append(s[i:j]);k=j+len(tag);d=1
        while k<n and d:
            if s[k]=='{':d+=1
            elif s[k]=='}':d-=1
            if d:k+=1
        i=k+1
    return ''.join(out)
def prep(p,cmds):
    s=open(p).read()
    for c in cmds:
        while '\\'+c+'{' in s: s=delete_cmd(s,c)
    return ' '.join(re.sub(r'%.*','',s).split()).split()
a=prep('paper_prod/production.tex',['Gala','R','B'])
b=prep('paper_prod/draft_revised_2026-07-20.tex',['B'])
for t,x1,x2,y1,y2 in difflib.SequenceMatcher(None,a,b,autojunk=False).get_opcodes():
    if t=='equal': continue
    A=' '.join(a[x1:x2]); B=' '.join(b[y1:y2])
    if A.strip() or B.strip(): print(f"[{t}]\n  PROD :{A[:200]}\n  DRAFT:{B[:200]}\n")
EOF
```

Also assert after editing: `\B{}` braces balanced, `figure`/`figure*`/`equation`/
`tikzpicture` environments balanced, no undefined `\ref{}`, no duplicate `\label{}`,
and **no stray `\\<controlword>`** — the 2026-07-28 re-baseline turned
`\subsection{Magnification PDF}` into `\\subsection{…}` (caught and fixed
2026-07-28b; regex `\\\\[a-zA-Z]+`).

⚠ **Third trap, hit 2026-07-29c — SHORT BLOCKS ALIGN SPURIOUSLY.** Alignment
fixes the containment trap for *paragraphs*, not for one- or two-token blocks. A
`\B{\cite{Sheth:1999su}}` was classified STALE at `frac = 1.00` and unwrapped,
even though `Sheth:1999su` appears **zero** times in production — the lone token
landed inside some unrelated `equal` opcode. Production actually had
`\cite{Baumann:2022mni} \Gala{maybe you have a better citation}` there, i.e. the
draft's cite was a live *answer* to the user's question, and unwrapping it
silently destroyed the proposal. **Never trust a STALE verdict on a block under
~8 words — grep production for a distinctive token from the block and confirm a
non-zero count before unwrapping.** The reverse check is cheap and decisive.

⚠ **Fourth: `\B{}` cannot mark a DELETION.** When the draft removes words that
production still has, there is no text to wrap, and the block's remaining words
all align, so the classifier reports `frac = 1.00` (STALE) for a block that is
genuinely a difference. Two of these live in the `fig:flow_transformation`
caption (production's "Gaussian" and "$K=6$", removed here because they name the
legacy checkpoint). Mark deletions with a `%` comment next to the site, and make
sure some nearby *added* text carries the change to the merge.

⚠ Two traps in the check script itself, both hit on 2026-07-28b:
- **Containment is not alignment.** Testing whether a `\B{}` block's text appears
  *anywhere* in production gives false MERGED verdicts — `$(p,q)=(0.3,\,0.8)$` and
  `\sigma_8` both matched at unrelated positions. Align the two word streams with
  `difflib` and require the block's words to land in an `equal` opcode.
- **Strip `\B{}` from production by UNWRAPPING, not deleting**, whenever production
  contains leaked markers (see §7): deleting the content makes production look like
  it is missing text it actually has.

### Which way to sync

- production is the **newer version of a shared passage** → sync the draft **to**
  production, leave it black. (Draft staleness is noise, not a proposal.)
- draft has content production **lacks or contradicts** → keep it, wrap in `\B{}`.
- production has content the draft lacks → import it into the draft, black, so the
  draft stays a superset. Exception: never import `\R{}` or `\Gala{}` blocks.

`\B{}` is `\newcommand` and therefore `\long`, so blank lines inside are legal.
For a sectioning command put the macro *inside* the argument —
`\subsection{\B{Title}}`, never `\B{\subsection{Title}}`.

---

## 3. Prose style

No prose-style guide existed in the repo before this file (`plot_style.md` covers
figures only). **Target = synthesis of two references:**

- `papers/core/Vaskonen_2026_GWWeakLensingSigma8.pdf` — the predecessor/companion
  (Vaskonen 2026, MNRAS 547, 7 pp.)
- `2607.01333v1` — Baltabay, D'Eramo & Vaskonen, *Axion Misalignment Across
  First-Order Phase Transitions* (PRD, 18 pp.), the user's first paper

### Take from Vaskonen: compression and directness

- **Short declarative sentences in the model sections.** "We compute the lensing
  magnification distribution using the stochastic approach of X, which we have
  implemented in C++." Subject–verb–object. No throat-clearing.
- **Get to the model fast.** His §2 opens with the method in sentence one.
- **Define every symbol immediately after the equation that introduces it**, in a
  `where ...` clause, and never again.
- **Footnotes carry implementation choices**, keeping the main text clean: his
  footnotes 1–4 are the code link, why the non-linear relation is retained, how
  `p` and `q` were fixed, and why a grid size was chosen. Use footnotes the same
  way here for `κ_thr`, `m_floor`, `R_s`, `κ_thr,sub`, thread counts, etc.
- **Headline numbers in the abstract**, concrete: "30% accuracy, assuming a
  population of 300 neutron-star binaries."
- **No intensifiers.** He does not write "crucially", "importantly", "it is
  interesting to note". Neither should we.

### Take from the axion paper: PRD structure and signposting

- **REVTeX PRD** with roman-numeral sections in caps and lettered subsections —
  which is what `production.tex` already uses.
- **Subsection titles carry the physics, not just the topic.** "Fast FOPT
  (β ≫ M_φ): Delayed Misalignment", not "The fast case". Applied here: prefer
  "Sub-threshold contribution (κ_W)" over "Background", and name the mechanism
  when a subsection is about one.
- **Organize by regime or ingredient, and say so up front.** The axion paper's
  spine is "we identify two distinct regimes"; ours is "we extend the model in two
  ways" (subhalos, clustering) — keep that pairing visible in the abstract,
  introduction, conclusions, and in `fig:magpdf_ingredients`.
- **Forward signposting with first-person plural, active voice.** "We begin by
  discussing…", "To quantify this effect, we…", "We find that the numerical
  results shown in Fig. 3 are well described by…". Use it at the head of a
  derivation, not in every paragraph.
- **Equations woven into the sentence**, not dropped in: "…defined by", "…reads",
  "…with spectral coefficients and expectation value". Punctuate equations as
  sentence parts (`\,,` / `\,.`).
- **Appendices for validation and for illustrative special cases.** The axion
  paper's Appendix B is "Homogeneous Dynamics Validation". This paper has a lot of
  validation material (`data/results/*/report.md`) that does not belong in the
  body — an appendix is the right home for the κ_thr insensitivity, the model-5
  gate, and the split-invariance checks.

### Where they differ, and what we do

| | Vaskonen 2026 | axion paper | **this paper** |
|---|---|---|---|
| length | 7 pp., 4 sections | 18 pp. + appendices | ~12–14 pp.; body compact, validation in appendices |
| introduction | ~1.5 columns, minimal | ~2 pp., full literature narrative | intermediate — the ML/emulator angle needs the N-body-vs-stochastic contrast spelled out |
| citations | author–year (MNRAS) | numeric (PRD) | **numeric**, `\cite{key}` |
| subsections | almost none | many, physics-named | as in the axion paper |

### Mechanical conventions

- `\Msun`, `\td` for the differential, `\vect{}` for vectors — already in the
  preamble; use them, do not hand-roll `{\rm d}`.
- `\be/\ee`, `\bea/\eea` shortcuts, not raw `equation`/`aligned`.
- Reference figures as `Fig.~\ref{}` and equations as `Eq.~\eqref{}` with
  non-breaking spaces.
- American spelling ("modeled", "normalization"), consistent with both references.
- Log-axis tick convention for figures: decades −1/0/1 print as `0.1`/`1`/`10`,
  everything else as `$10^{n}$`; call `format_log_axis_decimal(ax, axis=...)` from
  `paper_prod/plot_style.py` per axis — it is **not** automatic.

---

## 4. Verification contract — before writing any number

Every quantitative claim in the paper must be traceable to the implementation or a
report in `data/results/`. Practically:

1. **Check the code, not memory or the draft.** Equations in the draft have been
   wrong (see §5). `grep -n` the function named in `paper_memo.md` and read it.
2. **Quote the file:line** in the memo when recording a resolution.
3. **Lift fitting-function constants from the source PDFs** in `papers/`, never
   from recollection.
4. When the paper text and the code disagree, **report it and let the user
   decide** which is wrong — do not silently change either to match the other.
   Changing sampling code is a physics decision and needs explicit sign-off.

### Standing rules that constrain claims (from CLAUDE.md)

Do **not** quote as converged or certified:

- bias-tail quantities: `f(κ>1)`, `q ≳ 99.9` at `z_s ≳ 5`, wide-support `⟨κ²⟩`
- raw `⟨1/μ⟩`, maxima, or raw moments — monster-ray junk. Use clipped/trimmed
  ensembles; `κ_tot ≤ 1` is the certified estimator, `kappa_anchor=1` for runs
- `M_min` as a closer of the ACE gap (wrong sign)
- `κ_min` as a numerical convergence knob — it is a **model parameter**

Legacy-match is not a correctness criterion (user ruling 2026-07-16).

A null result **bounds**, it does not measure. When an A/B lands at the sampling
floor, say "consistent with zero at this sensitivity, bounding the effect to X",
never "we showed there is no effect". Applies to `fil_bias` and the model-5 gate.

---

## 5. Decisions made 2026-07-28 (this session)

**(p,q) unified at (0.3, 0.8).** One barrier, one bias. The PBS bias is now
evaluated with the same barrier parameters that set the corresponding abundance:
`(0.3, 0.8)` for field halos matching `pFC`, `(0, 0.7)` for filaments matching
`pFCfil` — mirroring how `cosmology::filbias` already tracks `pFCfil`. Applied to
the draft **and** to `cosmology.cpp::halobias` (q 0.75 → 0.8). **Not bitwise
compatible**: `test_cosmology_params.py::test_backward_compat_bitwise` reference
and cached clustering runs (`bias_window`, `rperp_pdf_scan`, `fil_bias`) need
regeneration; b rises a few per cent so clustering strengthens slightly and the
8–15% Var(lnμ) numbers in the Conclusions will move.

**Subhalo radial profile: `x`, not `x²`.** Green+21 define the bias function as a
ratio of **volume** number densities (their §3.2.1, p.10, and the Fig. 7 caption,
p.11), so `B` multiplies the host *density*:

```
n_sub(x) = ρ_NFW(x)·B(x) ∝ B(x)/[x(1+cx)²]
dN/dx    = 4πr²n_sub      ∝ x·B(x)/(1+cx)²
```

The shell-volume `x²` cancels one power against the NFW `1/x` cusp. The code had
`x²/(1+cx)²·B` at three sites — a cored profile with one extra power of `x` of
central suppression (inner bias slope 2.25 instead of 1.25) and an `x⁻²` rather
than `x⁻³` outskirt, contradicting both `B→1` ("subhalos trace the host outside")
and the Han+16 `x^1.3` that `B` is fitted to. Fixed in `cpp/subhalo.cpp`
(`buildRestrictedBin` p3, `precompute` inverse CDF, `buildWsubBin` p3, plus the
`RADIAL SAMPLING CONVENTION` header block), `plot_fig_subhalo_sigma_decomposition.py`,
`playground/analytic/subhalo_factor_analytic_deficit.py`, and
`docs/subhalo/wsub_gaussian_term_derivation.md`.

Verification: `n_sub/ρ_NFW − B(x)` = 5.6e-16 (was 0.28); inner slope 1.25 (was
2.25); projection and rejection envelope unchanged in behaviour. Impact: median
clump radius 0.756 → 0.658 r₂₀₀, fraction inside 0.2 r₂₀₀ 0.87% → **3.83%**.
Requires `make build`, regenerated test references, regenerated
`fig:subhalo-factor-{mean,scatter}` and `fig:subhalo-kappa`, and re-running the
Fig. 4 probe cross-check.

**N_τ coefficient stays `6`** in the paper (user ruling: the 6.006 of
`subhalo.cpp:404` is an approximation and 6 reads better).

**Amplitude parameter is σ₈**, not `A_s`, everywhere including the tikz context
node. ⚠ Do **not** write "top-hat σ₈" — the engine normalizes through the smooth-k
window `Ws`, so σ₈ = 0.811 here is a smooth-k σ₈ (top-hat value 0.7786). State the
window convention explicitly or say nothing about it, pending Ville's sign-off.

### §II.B Magnification PDF — shape settled 2026-07-28

Written to answer the `\Gala{}` at `production.tex:282`. The subsection's job is to
be the **bridge**: it is the only place that earns the three structural priors §III
imposes on the emulator (edge cutoff, μ⁻² POT tail, flux calibration). Evidence for
all three lives in the body — see the appendix decision below.

**NO APPENDIX (user, 2026-07-28b).** `app:edge_tail` was written, then deleted
(577 words). Its three blocks resolved as follows. The *low-μ edge* block was
redundant once §II.B carried the argument. The *flux sum rule* block became a
footnote on the "we do not impose Eq. (fluxsumrule)" sentence, keeping the two
numbers that matter (raw ⟨1/μ⟩ = 1.08 at z_s=1 is monster-ray junk; imposing the
rule costs KL 0.50 vs 0.0034). The *high-μ tail* block was kept, compressed to
one clause in §II.B (Hill index 1.95–2.03, μ⁻² over μ⁻³ by 3–4 orders of
magnitude in likelihood) — it is the only demonstration that the simulator
satisfies the exponent §III imposes, so it cannot be dropped. `fig:magpdf_tail`
moved into §II.B alongside `fig:magpdf_zs`. ⚠ This supersedes the §3 guidance
below about appendices being the right home for validation material — if the
κ_thr insensitivity, model-5 gate or split-invariance checks are ever written up,
the appendix question reopens and should be decided then, not assumed.

**Keep the low-μ edge discussion SHORT (user, 2026-07-28b).** It was a full
paragraph with three floors (Dyer–Roeder, the model's own −2ln(1+⟨κ⟩), the
measured quantile) at two redshifts. Now one sentence in the feature list plus a
one-sentence footnote. The load-bearing content is only "the edge is not the
empty beam, so we fit it rather than imposing the analytic value" — everything
else was detail §III does not need.

**Restructured 2026-07-28b (user request) on the model of Killedar, Lasky, Lewis &
Fluke 2012 §2.1** (`papers/misc/mnras0420-0155.pdf`, p.157) — a short, equation-light
section that *enumerates the features* of the μPDF in prose with one citation each,
carrying only two equations (normalization, flux conservation). Ours keeps one
displayed equation (the sum rule) and runs four paragraphs — synthesis+estimator
(κ anchor and plane convention in a footnote), the feature list, the empty-beam
caveat, and what §III does and does not impose. Killedar's own list is worth
re-reading before editing this section further: normalization, flux conservation,
empty beam, peak below μ=1 with skew, μ⁻³ power-law tail (source plane, Vietri &
Ostriker), spread growing with z_s, low-μ shape set mainly by σ₈ and Ω_Λ (Premadi
et al. 2001), and low number statistics in the high-μ region.

**⚠ The empty beam is NOT a feature of our distributions — say so.** The draft
previously claimed the PDF drops "near a lower cutoff corresponding to the emptiest
lines of sight (the empty-beam limit)". It does not. With N̄_l = 100 explicit lenses
per sightline the empty configuration has probability e⁻¹⁰⁰, so the observed edge is
a *convolution* edge sitting above the empty-beam value, by a margin that widens with
z_s — measured 0.1% quantile lnμ = −0.12 (z_s=1) / −0.48 (z_s=8) against Dyer–Roeder
−0.13 / −1.65, with our own hard floor −2ln(1+⟨κ⟩) at −0.18 / −0.83
(`docs/edge_tail_flux_note.md` §1, which calls the offset "paper material, not a
model component"). The emulator imposes a **ridge-fitted** 0.1% quantile, never the
analytic DR value. ⚠ Those numbers are from the 2026-07-11 study on the -ar build and
predate the halobias and subhalo-profile fixes — refresh after the rebuild (TODO
comment carried at `fig:magpdf_zs` in the draft).

**Plane convention: IMAGE plane throughout** (user, 2026-07-28), with
`P_S ∝ μ⁻¹ P_I` given in the footnote for comparability with Vaskonen's Fig. 2 and
with ACE. ⚠ Note the repo contains BOTH conventions: `lensing.cpp::Plnmuf` (:1396)
applies the 1/μ source-plane conversion (Vaskonen's Fig. 2), while
`sample_lnmu` — used by `plot_fig_magnification_pdf.py` and by the ML pipeline —
returns image-plane samples. Do not mix them in one figure.

**Ingredient decomposition moved off the PDF overlay** onto the σ_DL(z) variance
figure (`fig:variance_DL`), following Vaskonen's Fig. 3, which decomposes
spherical/+bias/+elliptical/+filaments as one number per z rather than as
overlaid PDFs. `fig:magpdf_ingredients` deleted from the draft.
⚠ `fig:variance_DL` still has **no generating script in the repo** and now needs
three cumulative curves — that is a new MC sweep, not a replot.

**~~Do not impose ⟨1/μ⟩ = 1 on the emulator.~~ OVERTURNED BY THE USER 2026-07-29 —
see §5b below. The evidence in this block is still valid and is now the COST of
the decision, not an argument against it. Do not revert the draft on the strength
of what follows.** Recurring proposal; the answer has
three parts. (i) The constraint is **plane-invariant**: with source-plane weights
w_i ∝ 1/μ_i, ⟨μ⟩_S = N/Σ(1/μ_i) = 1/⟨1/μ⟩_I, so switching plane does not improve
the estimator — it only moves the heavy-tailed sum from the sum rule into the
normalization of P_S. (ii) It was **tried**: KL 0.50 vs 0.0034 shipped
(`docs/edge_tail_flux_note.md` §3). (iii) The principle that separates it from the
edge and tail: **impose structure the simulator satisfies, never theory it
violates.** The tail μ⁻² is measured in the sim (Hill α = 1.95–2.03); the edge is
*fitted from* the sim (Dyer–Roeder is off by 0.5 in lnμ at z_s=8 — never
substitute it); flux = 1 is neither, since on certified support the model gives
1.005–1.02 at z_s ≳ 3.5. An emulator that silently corrects its own training model
biases the σ₈ posterior undiagnosably.

## 5b. Flux sum rule — USER DECISION 2026-07-29: the paper imposes it

**Decision.** §II.B ¶3 now reads "The sharp low-$\mu$ edge, the $\mu^{-2}$ tail and
the flux sum rule of Eq.~(fluxsumrule) are structural … and in Sec.~III we build
**all three** into the emulator". The user chose this over a neutral phrasing,
knowing the cost below. The two sentences that said the opposite ("We do not impose
Eq. (fluxsumrule)…" plus its footnote) were deleted from the draft at the user's
request the same day.

**⚠ The paper now claims something the code does not do.** This is a live
inconsistency, deliberately entered, and it must be closed in §III:

- `gw-wl-emulator-ar/ml/autoresearch/smooth_model.py:134` targets `F_trim(ctx)`,
  the simulator's support-restricted flux (~1.000–1.02), and its own comment says
  **"NOT raw 1 — the raw sim mean is corrupted by rare kappa>1 rays and exactly-1
  over-shifts at high z"**.
- §III already contradicts itself independently of this: its intro paragraph says
  the calibration "enforces $\langle\mu^{-1}\rangle=1$" while its item 4 correctly
  describes the support-restricted target. One of the two has to go.
- Measured cost of actually imposing exactly 1, last time it was tried:
  **KL 0.0034 → 0.50** (`docs/edge_tail_flux_note.md` §3).

**The prerequisite question, still open: why is `F_trim ≠ 1`?** §7 records the
deviation as an *excess* (1.005–1.02 at z_s ≳ 3.5), which a support restriction
alone cannot produce — it would give a deficit. Suspects there: the κ-anchoring
convention, the NFW outer truncation at κ_min, the retained κ>1 rays. §7 says this
"cannot be measured until the module is rebuilt" — **that is no longer true, the
module was rebuilt 2026-07-28 13:32**, so measure it before the §III rewrite. If
the excess is a simulator artifact, imposing the rule is a fix and the KL 0.50
result should be re-derived after the artifact is removed. If it is genuine, the
paper is committing to a 1–2% distortion and the user should be told the number.

The user was offered "measure `F_trim` first, then decide" and chose to commit
without waiting — so this measurement is a §III task, not a blocker on §II.B.

### Decisions made 2026-07-29 — §II.B trim

**The empty-beam ARGUMENT is out of the paper (user: "i dont like it at all").**
§II.B states only that the limiting case is the empty beam, so the magnification
cannot be made arbitrarily small. §III uses that to justify the low-$\mu$ cutoff.
⚠ This knowingly drops the "our edge sits *above* the empty beam" nuance that
`docs/edge_tail_flux_note.md` §1 establishes, together with the $\bar N_l$ /
$e^{-\bar N_l}$ probability argument and the $-0.48$ vs $-1.65$ comparison. **Do
not reintroduce any of it.** The shipped emulator still *fits* $\mu_{\rm min}$
from the simulation rather than imposing Dyer--Roeder, so only the paper changed.

**Plane subscripts are now carried explicitly: $\td P_I/\td\mu$.** Vaskonen
writes plain $\td P/\td\mu$ and his Fig. 2 is the SOURCE plane, so a reader
arriving from his paper would otherwise read our figure as his. Applied in §II.B,
both §II.B captions, `fig:flow_transformation`, §III, and the figure y-axis
labels. The direction is $\td P_S/\td\mu = \mu^{-1}\td P_I/\td\mu$ — confirmed
three ways (`cpp/lensing.cpp:1396` multiplies by `exp(-lnmu)`; Vaskonen 2026 p.3;
and image $\mu^{-2}\times\mu^{-1}$ = the source $\mu^{-3}$ of Vietri--Ostriker).
⚠ The inverse, $\mu\,\td P/\td\mu$, is a natural-looking guess and is **wrong**.

**SUPERSEDED 2026-07-29b (user): the subscript is now carried EVERYWHERE,
including the abstract, introduction and conclusions.** The earlier carve-out —
that in those sections $\td P/\td\mu$ names the magnification distribution as a
general object, including other authors' $N$-body work — is dropped; the user's
ruling is "explicit $\td P_I$ and so on, to not create confusion". Ten bare sites
converted in the draft (abstract ×2, introduction ×5, §III ×2 incl.
$\td P_I/\td\ln\mu$, conclusions ×1). The draft now has 24 `P_I` and 2 `P_S`, the
latter only where the conversion is stated. ⚠ Those lines are shared with
`production.tex`, which still writes plain $\td P/\td\mu$ and declares the
convention verbally in its §II.B footnote instead — so the draft and production
now differ on notation in the abstract/intro/conclusions. Not wrapped in `\B{}`:
a notation sweep is not a merge proposal per passage.

**The $\mu^{-2}$ demonstration is the FIGURE, not estimators** (user: "just a
plot and stating that it follows the $\mu^{-2}$ is good enough"). In
$\mu^2\,\td P_I/\td\mu$ a flat line *is* the claim. The Hill index (1.95--2.03)
and the Poisson likelihood ratio against $\mu^{-3}$ stay in `TODO.md` §4 for a
referee, not in the body.

**MEASURED 2026-07-29 — the high-$\mu$ tail is a SHEAR effect, not $\kappa\to1$.**
A draft sentence claimed the tail comes from sightlines "where $\kappa$ approaches
unity". That is false in this simulator and was cut. Measured at $z_s=5$,
high-structure, 300k raw rays (`sample_lensing_raw_ml`, robust anchor): for
$\mu>20$ the median $\kappa$ is **0.691** and the median $\gamma$ is **0.301**,
with only **3.8%** of tail rays at $\kappa>1$ — and $\gamma \simeq 1-\kappa$
(0.301 vs 0.309). So $\mu = 1/|(1-\kappa-\gamma)(1-\kappa+\gamma)|$ diverges
through the **tangential** eigenvalue $1-\kappa-\gamma \to 0$. These are sightlines
near a tangential critical curve. Stable across the cut: $\kappa$ median
0.646/0.691/0.705 for $\mu>8/20/50$. ⚠ Do not reintroduce a $\kappa\to1$ story.

**Captions target 20--50 words**, on production's model. Whatever else is cut,
`fig:magpdf_tail` must keep its high-structure cosmology statement — two body
figures at different cosmologies is a trap and the caption is the only warning.

---

## 6. Citation notes established here

**Yan & Fan 2011 (`Yan:2011ux`, arXiv:1101.3847) — what it does and does not
support.** Page references are arXiv page numbers.

- p.8, right col., Eq. (8): barrier `B = √a δ_c[1 + β(aν)^{-α}]`. Their `a` is our
  `q`, their `α` is our `p`. Halos: `α≈0.615, β≈0.485`; and "`a ≈ 0.7` is often
  required" — **0.7 is quoted there as the halo value.**
- p.9, top left, Eqs. (9)–(10): Shen+06 ellipsoidal-collapse barriers. Filament
  `α ≈ 0.28, β ≈ −0.012` — β≈0 is the "nearly flat" result. **This is the passage
  that backs `p = 0`.**
- p.9, left, §4 ¶1: fitting the *raw* 1+δ=16 mass function gives a barrier with
  "significant slopes", **not** flat — contradicting Eq. (10).
- p.10, left, "modified MF" ¶: only after correcting for the **peak-exclusion
  effect** does the fitted barrier become consistent with flat, with amplitude
  **`a ≈ 0.5`**, not 0.7. This is their actual filament result.
- **No filament clustering bias anywhere in the paper.** The only bias discussion
  is p.9 (halo bias, non-Markovian corrections) and p.12 (a sampling-bias remark).

So the paper supports the *barrier shape*; the PBS filament bias is our own
derivation. Two open caveats: our `q = 0.7` is the halo-calibrated `a` whereas
their filament fit gives `a ≈ 0.5`; and their filaments are 1+δ=16 percolation
groups, a different object from our cylinders.

**Green+21 (`Green:2021vtc`) caveat.** They conclude the Bolshoi central depletion
is largely a *resolution artifact*: "the chief cause of the dearth of subhaloes in
the central regions of haloes is the limiting mass resolution of the simulation.
It is neither physical nor primarily a manifestation of artificial disruption",
and their unresolved-limit profile is *nearly unbiased*, agreeing with Han+16. Any
caption claiming the depletion "reflects tidal stripping" is in tension with the
source it is fitted to — either qualify it or drop the causal claim.

### §II.B keys checked against the PDFs (2026-07-28b)

All downloaded to `papers/misc/` under their ADS bibcode. **Two were wrong and are
removed from the draft.** Full per-key evidence lives in the comment block at the foot
of the draft — read that before touching any of these cites.

| key | verdict |
|---|---|
| `Dyer:1972` | ✅ empty "tube"/beam D(z). Distance relation ONLY — not a minimum magnification |
| `Weinberg:1976` | ✅ p.L3 "mean inverse-square luminosity distance", = ⟨μ⟩_S = 1 = our ⟨1/μ⟩_I = 1 |
| `Blandford:1986` | ✅ eq. (5.16) σ(>M) ∝ M⁻², folds dominate high magnification. Never states a PDF exponent |
| `Vietri:1983` | ✅ eq. (6c) `dP = const × dA/A³` — **μ⁻³, source plane**. The old annotation saying μ⁻² was wrong |
| `Premadi:2001` | ⚠️ §5.1.1 — σ₈ moves the LOW edge, leaves μ>1 nearly unchanged. They vary λ₀, not Ω_M. **The second half is FALSE for our model** — measured 2026-07-29, `data/results/sigma8_shape/`: μ>1 is the MOST σ₈-sensitive region here, P(μ>2) ∝ σ₈^3.5–3.9 against width ∝ σ₈^0.95. Cite it only for the edge moving |
| `Killedar:2012` | ✅ for "the tail is method-dependent" — §2.1 p.157 verbatim: "The high-magnification region suffers from low number statistics and since it represents the effects of strong lensing, it cannot be probed by ray-bundle methods". ⚠️ but their σ_μ = 0.205 (multiple-lens-plane) vs 0.021 (3D) is the scatter of the WHOLE distribution at a single z_s = 0.5, the slope comparison is quoted only for **1.1 < μ < 1.5**, they attribute the gap partly to RBM **shot noise** rather than projection alone, and their own abstract calls the comparison preliminary. Cite for the qualitative claim, never for a number |
| `Schneider:1992` | ⚠️ **unverified** — a book, no copy in repo. Claim already carried by `Blandford:1986` |
| `Seitz:1997` | ❌ **REMOVED** — A&A 318, 687 is cluster mass reconstruction, not flux conservation |
| `Zakharov:1995` | ❌ **REMOVED** — A&A 293, 1 is about **cusps**, no PDF, no tail exponent |

Two lessons. **A secondary source can be internally inconsistent** — Killedar's intro and
their §2.1 say apparently opposite things about Premadi and σ₈, and only the primary
resolves it (both are true, of `P(μ>1)` and of the low-μ shape respectively). And
**`adsabs.harvard.edu/pdf/<bibcode>` works from here while
`articles.adsabs.harvard.edu/pdf/<bibcode>` hangs** — use the former for pre-arXiv
papers. ADS scans of ApJ carry an OCR text layer that `pdftotext` reads; A&A scans
often do not.

**Other keys settled earlier:** HMF → `Sheth:1999su` (SMT01); PBS bias →
`Sheth:1999mn` (ST99), though production deliberately cites `Baumann:2022mni`
because Vaskonen's own paper does (user ruling — do not "fix" it);
`Giocoli:2006yz` not `Giocoli:2011hz` for `α_f`; `Green:2021vtc` consistently.

---

## 7. Open items (2026-07-28)

- ~~**§ Subhalos contains a false cost claim.**~~ **CLOSED 2026-07-28b** — the
  "costs only a modest factor over the smooth-halo computation and requires no
  resolution parameter" sentence is gone from both files (grep confirms
  2026-07-29). The paragraph now states the real figure, more than two orders of
  magnitude over the smooth-halo computation, and introduces `κ_thr,sub` as what
  buys it back, with `fig:subhalo-kappa` referenced. Shortened again 2026-07-29.
- **§ III states the KL twice** — production's short paragraph (0.0073 vs ACE
  0.007) and the draft's expanded subsection (0.0073 → 0.0034). Pick one; both are
  pre-retraining numbers anyway.
- ~~**Eq. (radial) in `production.tex`** still has `x/(1+cx)²` with a literal
  `0.54`~~ **CLOSED 2026-07-28b** — user merged the draft version; production now
  carries `x_0` with the η rescaling text.
- **Three `\B{}` markers leaked into `production.tex`** during that merge
  (2026-07-28b), at `:79` (`\,\B{,}` in Eq. (mueq)), `:265` (`(x/\B{x_0})` in the
  radial fit) and `:361` (the whole `fig:emulator_arch` caption). Harmless to
  compilation — `\B` is defined at `:29` as blue text — but they render blue and
  they **break the §2 verification script**, which deletes `\B{}` *content* from
  production and so reports the enclosed text as missing. User-only file: report,
  do not fix. Same merge left two typos at `:265`, a stray `\.` after
  `\cite{Green:2021vtc})`, and a **duplicated explanation** — the new sentence
  "The first factor is the radial distribution of an NFW host…" is followed by the
  old "The first factor corresponds to the radial NFW distribution of the host halo
  and accounts for the depletion…", which the fix makes wrong (the depletion is
  carried by `B(x)`, not by the first factor). Deliberately NOT imported into the
  draft despite the §2 superset rule.
- **`fig:subhalo-factor-{mean,scatter}`** are `\ref{}`d in production but not
  defined there → two `??`. The draft supplies them.
- **Comparison section** stays a placeholder in production until the ML retraining
  lands (user + Ville, 2026-07-23).
- **σ₈ normalization window** (option b, top-hat anchor) pending Ville; bundle with
  the `bias_model`/`bias_window`/`R_⊥`/subhalo default flips.
- **The μ⁻² asymptote sets in near μ ≈ 20–30, not μ = 8** (measured 2026-07-28 on
  the user's REAL 798k/series new-grid run, high-structure cosmology). Survival
  exponent `a = 1 − dlnS/dlnμ`: 2.25/2.13/1.99/1.98 above μ = 8/16/32/64 at
  z_s = 10, and 2.33/2.01/2.05/2.06 at z_s = 5. So μ⁻² holds, but at μ > 8 the
  exponent is still ~2.25 and a plateau level taken from μ > 8 sits above the
  true one. `--tail-fit-mu` default moved 8 → 20. ⚠ This is the HIGH-STRUCTURE
  cosmology; re-check the onset at the fiducial. Mild tension with
  `docs/edge_tail_flux_note.md`, whose Hill fit used μ ≥ 8 (it still returned
  1.95–2.03, so the contamination is small).
  ⚠ **Methodological trap, hit here:** a count-weighted log-density fit over
  half-decade bins gave slopes of −1.0 to −1.7 in the far tail, i.e. an absurd
  non-normalizable tail. That estimator is biased shallow at low counts. **Use
  the survival function for tail exponents**, never binned log-density.
- **Our σ_DL/D_L sits 10–37% above Vaskonen's own Eq. (8) fit, and the gap is in
  the BASELINE, not in our additions** (measured 2026-07-28 on the 480k cache,
  clipped μ ≤ 3, σ computed directly as the scatter of μ^{-1/2}):

  | z_s | ours (image) | ours (source) | Vaskonen Eq. (8) | ratio |
  |---|---|---|---|---|
  | 0.5 | 0.0158 | 0.0151 | 0.0137 | 1.10 |
  | 1 | 0.0345 | 0.0321 | 0.0286 | 1.12 |
  | 2 | 0.0625 | 0.0573 | 0.0478 | 1.20 |
  | 5 | 0.1023 | 0.0919 | 0.0703 | 1.31 |
  | 10 | 0.1244 | 0.1113 | 0.0812 | 1.37 |

  Decomposition at z_s = 5: the image→source plane conversion accounts for
  −11% (σ ×0.891, mode 0.907→0.861); our added physics accounts for **+1.8%**
  (base 0.0903 → full 0.0919); the remaining **×1.28 is our `CONFIG_BASE`,
  which is supposed to BE his model.** Candidates to check: the `sigmakappaW`
  ×2 measure fix (documented ≤1% on Var(κ_total), so probably not),
  `kappa_anchor=1` vs his plain batch mean, the μ ≤ 3 clip (which if anything
  suppresses ours), and that Eq. (8) is a fit to his numerics rather than the
  numerics. Context: at z_s = 5 we sit 15% ABOVE Takahashi 2011 where he sits
  12% below, so we bracket his N-body reference from the other side. Belongs in
  §IV; do not write the ingredient comparison until this is understood.
- **The ingredient arms are not one-variable-at-a-time.** `CONFIG_SUBH` uses
  `bias_model=0` while `CONFIG_FULL` uses the correlated field at R_s = 20 Mpc,
  so base→subh→full changes two things between the last two arms and the
  effects do not add (σ_DL/D_L 0.0903 → 0.0946 → 0.0919 at z_s = 5, i.e.
  clustering appears to REDUCE the scatter). Fix the arm definitions before
  `fig:variance_DL` is generated, or the decomposition will be misread.
- **Flux sum rule: F_trim ≠ 1 is unexplained** (opened 2026-07-28, user: log for
  after the rebuild). The simulator's support-restricted flux is 1.005–1.02 at
  z_s ≳ 3.5 — an *excess*, not the deficit a support restriction alone would give,
  so it is not simply truncation. Suspects: the κ-anchoring convention (which
  directly sets where the distribution sits, CLAUDE.md item 12), the NFW outer
  truncation at κ_min, and the retained κ>1 population. This is a §II question, not
  a §III one: if the simulator satisfied the sum rule, imposing it in the emulator
  would be free. **Cannot be measured until the module is rebuilt** — the current
  `.so` predates both the halobias and the subhalo-profile fixes, and the profile
  fix moves substructure inward, which moves σ and hence the flux.
- **§II.B figures must be REGENERATED** (2026-07-28). ⚠ Correction to an earlier
  note in this file that said they had never been run — they were: 8 shards ×
  60k = **480k realizations, 2026-07-25**, cached in `paper_prod/plots/data/`
  (`magpdf_combined.npz` + `magpdf_shard2*.npz`, gitignored) with the figures
  committed. They are nonetheless stale for three independent reasons:
  1. **physics** — the run predates the `halobias` and subhalo-profile changes,
     and used `subhalo_model=4` without `subhalo_virial`, i.e. a different
     subhalo population from the one the draft describes. The script now imports
     `PRODUCTION_CONFIG` from `ml/params.py` (hash `0d50caf91c75`) instead of
     restating it, so this class of drift cannot recur;
  2. **grid** — the cache is 2000 LINEAR bins over μ ∈ [0.4, 5], which cannot
     show the μ⁻² tail (fitted above μ ≥ 8). Now 4000 log bins over
     [0.05, 200], and the run also records under/overflow counts so the 0.1%
     edge quantile is exact;
  3. **content** — the figure now carries the low-μ edge markers and the log-log
     tail inset.

  The caption's `$N_{\rm real}$` placeholder is filled at that point. Old-grid
  caches still replot correctly (plotting uses the cache's own edges) and the
  inset auto-disables with a warning, but they cannot be combined with new ones.
  Measured 0.1% edge on the 480k run, for reference: lnμ = −0.048/−0.119/−0.238/
  −0.411/−0.506 at z_s = 0.5/1/2/5/10 (the z_s=1 value reproduces the −0.111 in
  `docs/edge_tail_flux_note.md`). ⚠ At z_s=10 the edge sits at μ = 0.603, right
  on the default left limit `--mu-range 0.6 1.8` — widen to ~0.55 or that marker
  is clipped.
- **The seven new bib keys are unverified** (`Dyer:1972`, `Weinberg:1976`,
  `Blandford:1986`, `Schneider:1992`, `Vietri:1983`, `Zakharov:1995`,
  `Seitz:1997`). `refs.bib` exists only on Overleaf — the draft lists them in a
  comment block at its foot. Check each against ADS before they go to Ville.
- **Post-fix regeneration** of everything listed in §5, then re-check the "+14.9%
  on ⟨κ²⟩ at z_s=5" substructure number and the `subkappathr_population` sizing —
  both were measured with clumps sitting too far out, so the substructure
  contribution likely moves up.
