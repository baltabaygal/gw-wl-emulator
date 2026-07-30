# TODO — paper, resume point (updated 2026-07-29c)

## MERGE CHECKLIST → production.tex (2026-07-29c, §III)

Everything below is **already fixed in `draft_revised_2026-07-20.tex`** and marked
with `\B{}` where it is prose. Two items cannot be carried by a `\B{}` marker
because they are defects in `production.tex` that have no counterpart in the
draft — those are flagged ⚠ and must be applied by hand.

1. ⚠ **`production.tex` does not compile.** Add `\usetikzlibrary{arrows.meta}`
   after `\usepackage{tikz}` (line 11). Without it `fig:emulator_arch` throws
   `! Package pgf Error: Unknown arrow tip kind 'Stealth'.` and no PDF is
   produced. Draft has it at line 10 with a loud comment.
2. ⚠ **`\label{fig:variance_DL}` is defined twice** in `production.tex`, at lines
   336 and 446 — `LaTeX Warning: Label 'fig:variance_DL' multiply defined`, and
   every `\ref` silently resolves to the second. Delete the whole figure block at
   332–337 (the bare-caption copy that survived the merge); keep 442–447, which
   has the full caption and the `\Gala{add Takahashi...}` note. The draft has
   only one, so there is nothing to copy across — just delete.
   - Separately, `plots/variance_D_L.pdf` **does not exist** — no file, no git
     history. Both includes are dead until the figure is generated.
3. **§III.A described the wrong checkpoint.** Was `K=6`, `B=8`, `[-3,3]`,
   Gaussian base = `data/models/conditional_nsf_backend_current.pt`. Now `K=3`,
   `B=10`, `[-16,16]`, Student-$t$(2) = `flow_smooth_lowz.pt`, which is the
   production body and the only checkpoint the composite and the `KL = 0.0073`
   headline exist for. Numbers read off the checkpoint config and state_dict.
4. **`fig:flow_transformation` is still plotted from the legacy checkpoint** and
   now carries a visible `\B{[FIGURE STILL FROM THE LEGACY CHECKPOINT ...]}` note
   in its caption. Regenerate from `flow_smooth_lowz.pt`
   (`plot_fig_flow_transformation.py --model ...`) and delete the note. Do not
   delete the note without regenerating — it is what keeps the mismatch declared
   instead of hidden.
5. **Flux is now stated consistently in four places**, all to `F_trim` (what the
   code does today) rather than to the value 1. Flip all four together when the
   sum rule is imposed — the list is in the comment block above §III.B.
6. **The KL sentence was duplicated near-verbatim** in the §III intro and in
   §III.B. The intro now forward-points instead; §III.B keeps the numbers.
7. **`Durkan:2019` → `Durkan:2019nsq`** (user confirmed `nsq` is the key that
   resolves).
8. **Notation.** `production.tex:315` uses bare `$dP/d\mu$` and `$dP/d\ln\mu$`
   while 342 and 426 use `$dP_I$`. §II.B now establishes the $I$/$S$ subscript
   convention, so 315 needs the subscript. The draft already carries it, but
   only in the part of that paragraph I did **not** rewrite — merging just my
   last sentence will not fix it.
9. Cosmetic, production only — `production.tex:445` caption reads
   `weak lensing.The curves`, missing space. Draft is fine.

Draft compiles clean after all of the above (no errors, no undefined control
sequences, no duplicate labels).

### `\B{}` re-baselined against production (2026-07-29c)

`\B{}` now means what §2 of `paper_writer.md` says it means — *differs from
production* — and nothing else. Checker committed as
`paper_prod/scripts/check_B_markers.py` (comment-aware, flags short blocks).
Before: 39 blocks, 8 of them stale. After: **19 LIVE, 6 PARTIAL, 1 unverifiable-
short, 11 inside `%` comments**.

Unwrapped as already-merged: §III.B body and its `\subsection` title, the
$\sigma_{D_L}$ paragraph, the `fig:variance_DL` caption, the §II.B opening and
feature paragraphs, the pyHalo/diffhalos footnote.

Bib keys synced to production (production is authoritative): `Weinberg:1976jq`,
`Dyer1972dis`, `Blandford:1986zz`, `Premadi:2001ez`, `Durkan:2019nsq`.

Kept whole-wrapped on purpose — `fig:magpdf_zs` and `fig:magpdf_tail` captions.
Both differ from production in five scattered places each (plane, $P_S$/$P_I$,
the $z_s$ list, "fiducial cosmology" vs "Planck 2018 values", the added
sentences). Splitting into five micro-markers per caption would be unreadable,
and a caption is the natural atomic unit for a merge.

⚠ **Near-miss worth knowing about.** `\B{\cite{Sheth:1999su}}` was classified
STALE at `frac = 1.00` and I unwrapped it — wrongly. `Sheth:1999su` appears
**zero** times in production; the lone token had aligned against something
unrelated. Production has `\cite{Baumann:2022mni} \Gala{maybe you have a better
citation}` there, so the draft's cite is the *answer* to your question, and
unwrapping it would have silently deleted the proposal. Restored. The checker
now prints `STALE?` for blocks under 8 words and refuses to recommend them, and
`paper_writer.md` §2 records this as trap 3 (plus trap 4 — `\B{}` cannot mark a
deletion, which bit the `fig:flow_transformation` caption).

### Two more production-only typos found by the sync

10. `production.tex` footnote after the diffhalos URL ends `.}.` — stray period.
11. Three leaked `\B{}` markers still corrupt production text (known, §7 of
    `paper_writer.md`): `(x/\B{x_0})` renders as `(x/)`, and
    `Ref.~\cite{Green:2021vtc})\.` has a stray `\`. These break the §2
    verification script, so they are worth fixing on the next Overleaf pass.

## 0. Session 2026-07-29b — subhalo figures cut, notation sweep completed

- **`fig:subhalo-factor-{mean,scatter}` REMOVED from the draft** (user), together
  with the sentence in §II.A.4 that referenced them. `fig:subhalo-kappa`
  (`plots/fig_subhalo_sigma_ratio_vs_subkappathr.pdf`) is now the only subhalo
  $\sigma$ figure. Generator `plot_fig_subhalo_sigma_decomposition.py` kept as a
  diagnostic. ⚠ **`production.tex:226` still carries the referencing sentence and
  therefore two dangling `\ref{}`s** — user-only file, reported not fixed.
- **`fig:subhalo-kappa` REGENERATED on the corrected profile 2026-07-29b — but
  only PARTLY. Read this before quoting the plateau.**
  - Regenerated: `playground/analytic/subkappathr_{dense,components}.json`, from
    which the figure's *shape* and *variance decomposition* are drawn. Stale
    pre-fix copies kept as `*_preprofilefix.json`; the node caches
    `tmp/subkappathr_{dense_nodes,component_nodes,component_nodes_v2}` were moved
    aside to `*_preprofilefix_<ts>` — they are keyed by path and would otherwise
    have been silently reused (CLAUDE.md §18 trap (a)).
    ⚠ Both sweep scripts short-circuit on an existing output JSON
    (`todo = [z for z in Z_SOURCES if str(z) not in results]`), so they exit 0
    with no output and no work done unless the JSON is moved away first. That
    looks exactly like a crash. Move the JSON, then run.
  - Cross-check passed: at $z_s=1$, $f=0.1$ gives 93.43 clumps/ray and a
    $\sigma_{\rm sub}$ loss of 0.079%, against CLAUDE.md §18's independently
    derived 93.6 and 0.078%. $z_s=5$ reads 0.154% vs §18's 0.152%.
  - ⚠ **The plateau amplitude is still pre-fix.** The scalar
    $A = {\rm Var}_{\rm sub}/{\rm Var}_{\rm off}$ is fitted to
    `playground/sigma_on_off_vs_subkappathr_model5_crn.json` (2026-07-27 16:08,
    3e5 rays, needs the C++ build), which was NOT regenerated. The figure
    therefore reads **+5.1%** where the corrected profile implies **~+5.9%**:
    population-weighted ${\rm Var}_{\rm sub}(0)$ rose $\times1.196/1.157/1.094$
    at $z_s=0.5/1/5$, i.e. $A: 0.105 \to 0.121$ at $z_s=1$. **Rerun that MC on
    the Mac before submission**; the shape, the decomposition and the
    threshold ledger are already correct and will not move.
  - The `<0.2\%` text claim is unaffected and comfortable: total $\sigma$ loss at
    $f=0.1$ is 0.0075% ($z_s=1$), $\sigma_{\rm sub}$ loss 0.079%/0.154% at
    $z_s=1/5$.
  - Caption still writes `\sigma_k` where the body uses `\sigma_\kappa`, and
    "decomposed" never says into what — propose a `\B{}` caption if wanted.
- **Notation swept to explicit plane subscripts** (see `paper_writer.md` §5,
  superseding block): all bare `dP/d\mu` in abstract, introduction, §III and
  conclusions are now `dP_I/d\mu`. The 2026-07-29 carve-out for those sections is
  cancelled.
- **`fig:shmf` caption footnotes moved into the body** as one `\footnote` with
  `\url{}`, mirroring production's rearrangement but without production's
  duplicated "first factor" sentence.
- Production's §II.B is deliberately one revision behind (user: "i told you to
  scan up to the magnification pdf subsection only ... i am not there yet"), so
  the Hill index / Killedar detail / vacuum-energy clause it still carries, and
  the dropped empty-beam sentence, are **not** to be treated as reversals.
  ⚠ Note for when that section is reached: draft §III item 3 says "The empty beam
  of Sec.~\ref{sec:pdf} limits how far a sightline can be demagnified", so §II.B
  must keep the one-sentence empty-beam statement or §III loses its antecedent.

---


Working file `paper_prod/draft_revised_2026-07-20.tex`. Contract in
`paper_prod/paper_writer.md`, code map in `paper_prod/paper_memo.md`.

**§1 below is DONE (2026-07-29).** All seven sub-items applied, draft compiles
clean (`revtex4-2` substituted locally, `enumitem`/`aas_macros` stubbed — the
only missing graphic is the known `plots/variance_D_L.pdf`, and there are zero
undefined `\ref{}`s). §II.B is now one column-page of text plus its two figures.
Next session starts at §2 (§III Machine learning). What was done, per item:

- **1a** empty beam — the $\bar N_l$/$e^{-\bar N_l}$ argument and footnote 5 are
  gone. §II.B now says only that the limiting case is the empty beam, so $\mu$
  cannot be made arbitrarily small. §III item 3 carries the justification for the
  low-$\mu$ cutoff and states that $\mu_{\rm min}$ is fitted to the simulations.
  No code changed. A comment at `fig:magpdf_zs` records this so the deleted
  comparison is not reintroduced by a future session.
- **1b/1c** tail block 230 → 130 words. Killedar is a bare cite; the Hill index
  and the Poisson $\mu^{-2}$-vs-$\mu^{-3}$ comparison are out of the body (kept
  in §4 here); the figure alone carries the exponent. $\sigma_8$ point kept.
- **1g** plane subscripts applied in §II.B, both captions, the
  `fig:flow_transformation` caption, §III, and **every figure carrying a
  magnification-PDF axis, all regenerated 2026-07-29**:
  - `plot_fig_magnification_pdf.py` — 3 y-axes + the in-panel `flat ⇔ …`
    annotation, which was also clipping off the right frame edge (now
    right-aligned on the last gridpoint and lifted clear of the $z_s=10$ curve).
    → `fig_magnification_pdf_{zs,tail_highstruct,ingredients}` replotted from
    cache, both cosmologies.
  - `plot_fig_flow_transformation.py` — y-axis; `--replot` from
    `flow_transformation_zs5.npz`.
  - `fig:emulator_arch` (tikz, **three** occurrences — the "Output PDF"
    layerlabel, the blue output box, and the caption). The tikz body is not one
    `\B{}` block, so each is wrapped individually, following the `$\B{\sigma_8}$`
    pattern already there.
  - `fit_tail_exponent.py` — label updated, but see below.

  Verified by `pdftotext` over all nine figures the draft `\includegraphics`:
  three carry a PDF axis and all three now read `dP_I`, six carry none.
  ⚠ No figure plots a source-plane quantity, so `\td P_S/\td\mu` appears only in
  the §II.B conversion sentence. Keep it that way — if a source-plane figure is
  ever added it must be labelled $\td P_S/\td\mu$ explicitly.

  ⚠ **`paper_prod/scripts/fit_tail_exponent.py` is a trap and is now marked as
  one in its own docstring.** It fits the binned log-density, the estimator
  `paper_writer.md` §7 warns against, and its own output shows the failure —
  $\alpha = 1.000$ with $R^2 = 1.0000$ at $z_s = 0.2/0.5/1$, a non-normalizable
  tail fitted to bins holding one count each. No number from it is in the draft.
  ⚠ **Left bare on purpose:** abstract, introduction and conclusions. There
  $\td P/\td\mu$ names the magnification distribution as a general object,
  including other authors' $N$-body work, and those lines are shared with
  `production.tex` — subscripting them means wrapping shared text in `\B{}`.
  Open question for the user.
- **1d** captions: `fig:magpdf_tail` 171 → ~55 w, `fig:magpdf_zs` 73 → ~46 w.
  Both keep their cosmology statement, so the fiducial/high-structure contrast
  stays visible.
- **1e** §II.B went from three footnotes to one. The $\langle\kappa\rangle$
  footnote moved into the body as two sentences (anchor + $\kappa\leq1$
  estimator, then the plane convention). The flux footnote lost the raw
  $\langle1/\mu\rangle = 1.08$ half — `paper_writer.md` §4 forbids quoting it as
  certified anyway — and kept the KL $0.0034 \to 0.50$ justification §III uses.
- **1f** $\psi_{\rm res}$ paragraph 250 → ~200 words, with the $m_{\rm floor}$
  convergence evidence moved to a footnote (Vaskonen-style, and `paper_writer.md`
  §3 names `m_floor` as a footnote candidate explicitly).

The `<0.2\%` of $\sigma_\kappa$ claim was re-checked against CLAUDE.md §18 and
still holds on the corrected profile (revised loss 0.059/0.078/0.152% at
$z_s = 0.5/1/5$, was 0.084/0.111/0.210%).

---

## 1. DONE 2026-07-29 — §II.B "Magnification PDF", shorter and cleaner

The subsection currently reads as three paragraphs plus two figures. It works but
it over-explains. Every item below is user feedback given verbatim at the end of
the 2026-07-28b session.

### 1a. Cut the empty-beam discussion — user "i dont like it at all"

Current text in ¶2 reads

> Its low-magnification side falls off sharply, the emptiest sightlines
> encountering the least matter between observer and source, and each curve
> terminates at that edge. The edge is not the empty beam~\cite{Dyer:1972},
> since with $\bar N_l$ lenses sampled along every sightline the empty
> configuration has probability $e^{-\bar N_l}$, so we fit its location to the
> simulated distributions rather than imposing the analytic value.[footnote 5]

**Do NOT mention $\bar N_l$ or $e^{-\bar N_l}$.** The user is not confident in
that argument and does not want it in the paper. **Delete footnote 5 too**
(the $-0.48$ vs $-1.65$ comparison).

Replace with the plain statement that **an empty-beam limit exists**, nothing
more. Then in **§III (ML)** say that there is an empty-beam limit and that this
is why we impose the low-$\mu$ cutoff on the emulator. That is the whole
argument the user wants, split across the two sections.

⚠ This deliberately drops the "our edge is not the empty beam, it sits above it"
nuance that `docs/edge_tail_flux_note.md` §1 establishes. That is the user's
call, made knowingly. Do not reintroduce it. The shipped emulator still fits the
edge from the simulation rather than using Dyer--Roeder, so the *code* is
unchanged, only what the paper claims about it.

### 1b. Shorten the tail block — "we discuss too much about the tail"

The block from "It is also the part of the distribution that is hardest to
predict..." through "...This is what makes $\td P/\td\mu$ informative about
cosmology." is too long. Specifically:

- **Killedar~\cite{Killedar:2012} — just cite it, drop the detail.** The
  sentence "A comparison of two ray-tracing treatments of the same mass
  distribution found the magnification scatter to differ by an order of
  magnitude according to whether the matter was projected onto lens planes"
  becomes a bare citation.
- The "structural constraint on the shape, no claim about the absolute event
  rate" sentence, the "broadens with source redshift" sentence, and the
  $\sigma_8$/vacuum-density sentence (Premadi) can all be compressed. User said
  "maybe leave some things here, but generally this part can benefit from the
  shortening" — so keep the $\sigma_8$ point (it motivates the paper) and trim
  around it.

### 1c. KEEP the $\mu^{-2}$ demonstration — but the FIGURE ALONE carries it

**Settled 2026-07-29.** User: *"i think just a plot and stating that it follows
the $\mu^{-2}$ is good enough."*

So `fig:magpdf_tail` plus one sentence saying the tail follows $\mu^{-2}$.
**Drop from the body** the Hill index ($1.95$--$2.03$) and the Poisson likelihood
comparison against $\mu^{-3}$ that were folded into ¶2 on 2026-07-28b. The
compensated figure is self-evident — flat line means $\mu^{-2}$ — so the
estimator numbers add words without adding conviction. Numbers retained in §4
below in case a referee asks.

### 1g. Notation — subscript the two planes (settled 2026-07-29)

$$\frac{\td P_S}{\td\mu} = \mu^{-1}\,\frac{\td P_I}{\td\mu}$$

⚠ The user's first guess was $\td P_S/\td\mu = \mu\,\td P/\td\mu$, which is
**inverted**. Three independent confirmations of the $\mu^{-1}$ direction:
1. `cpp/lensing.cpp:1396` — comment "convert from image plane to source plane
   (P_S ~ P_I/mu)", code multiplies by `exp(-lnmu)`.
2. Vaskonen 2026 p.3 — "apply a $1/\mu$ factor to convert the image-plane
   distribution into the source-plane distribution".
3. Exponent consistency — image $\mu^{-2}$ times $\mu^{-1}$ gives the source
   $\mu^{-3}$ of Vietri--Ostriker and Blandford--Narayan. The $\mu$ direction
   would give $\mu^{-1}$, which matches no known result.

Geometry: $\td\Omega_I = \mu\,\td\Omega_S$, so a random sky direction
over-weights high $\mu$ relative to a random source. Sum rules agree,
$\langle\mu\rangle_S = 1/\langle\mu^{-1}\rangle_I$, so Weinberg's source-plane
$\langle\mu\rangle=1$ is our image-plane $\langle\mu^{-1}\rangle=1$.

**Carry $P_I$ explicitly, do not leave the image plane unsubscripted.** Vaskonen
writes plain $\td P/\td\mu$ and his Fig. 2 is SOURCE plane. If we also write
plain $\td P/\td\mu$ meaning image plane, a reader arriving from his paper reads
our figure as his. Use $\td P_I/\td\mu$ in body, captions and axis labels, and
$\td P_S/\td\mu$ only where the conversion is stated.

**Knock-on work:** the y-axis label in
`paper_prod/scripts/plot_fig_magnification_pdf.py` is currently `dP/dmu` and must
become `dP_I/dmu`, then both figures regenerate (commands in the comment block at
`fig:magpdf_zs`). Check `plot_fig_clustering_field.py` and any other script that
labels a magnification PDF axis. Also sweep the draft for bare `\td P/\td\mu`.

### 1d. Both captions are too long

Measured word counts, whole draft vs production:

| caption | draft | production |
|---|---|---|
| `fig:magpdf_tail` | **171 w** | not present |
| `fig:magpdf_zs` | **73 w** | not present |
| `fig:subhalo-kappa` | 22 w | 22 w |
| `fig:kappa_convergence` | 48 w | **28 w** |
| `fig:shmf` | 117 w | **134 w** |
| `fig:variance_DL` | 36 w | **9 w** |

**The user prefers production's short captions.** Target roughly 20--50 words.
`fig:magpdf_tail` at 171 w is longer than anything in either file and is the
worst offender — the cosmology statement and the "why z_s ≤ 1 is omitted"
sentence are both needed, but the μ=8 vs μ=20 explanation and the
piecewise-power-law rationale belong in the body or nowhere.

⚠ Whatever is cut from the tail caption, **keep the fact that it uses the
high-structure cosmology** ($\Omega_M=0.38$, $\sigma_8=1$, $h=0.72$) while
`fig:magpdf_zs` is fiducial. Two body figures at different cosmologies is a
genuine trap for a reader, and the caption is the only place that says so.

### 1e. Footnotes — user has deleted almost all of them

- The $\langle\kappa\rangle$-subtraction footnote on ¶1 should either be
  **shortened Vaskonen-style and moved into the body**, or **dropped entirely**.
  User's reasoning, worth recording: *"this paper is known to be continuation of
  the vaskonen work. so some things are like that paper and thats clear."*
  Vaskonen states the mean subtraction in one main-text sentence (his p.3,
  "Next, we subtract the ensemble mean of $\{\kappa_i\}$ from each realization").
  The image-plane vs source-plane sentence may still be worth one clause
  somewhere, since our convention differs from his Fig. 2.
- Footnote 5 (empty beam) — delete, see 1a.
- The flux-sum-rule footnote (raw $\langle 1/\mu\rangle = 1.08$, KL 0.50 vs
  0.0034) — not yet commented on by the user, but assume the same pressure.

### 1f. The $\psi_{\rm res}$ paragraph is too large

User: *"i also did not mention about going down than psi_res. that one is too
large, we could benefit from decreasing that paragraph."* This is the §II.A.4
paragraph beginning "The subhalo mass function is normalized through
Eq.~\eqref{eq:fsnorm} at the calibration resolution $\psi_{\rm res}=10^{-4}$...".
It now also carries the rewritten $\kappa_{\rm thr,sub}$ material, so it has
grown further. Shorten.

---

## 2. THEN — §III Machine learning

**⚠ NEW BLOCKING ITEM (2026-07-29): the flux sum rule.** §II.B ¶3 now states that
we build the edge, the tail **and Eq. (fluxsumrule)** into the emulator (user
decision — see `paper_writer.md` §5b, which supersedes the "do not impose" block
in §5). The code does NOT do this: `smooth_model.py:134` targets the
support-restricted `F_trim(ctx)` ≈ 1.000–1.02, not 1. §III must be made
consistent, and §III is *already* self-contradictory — its intro says the
calibration "enforces ⟨μ⁻¹⟩ = 1", its item 4 says it matches the support-restricted
flux. Order of work:
1. Measure `F_trim` on the CURRENT build (the §7 claim that this cannot be done
   until a rebuild is stale — the rebuild happened 2026-07-28 13:32). Is the
   1.005–1.02 excess still there after the profile and halobias fixes?
2. If it is an artifact → fix it, then imposing the rule is free and the old
   KL 0.0034 → 0.50 result must be re-derived.
3. If it is genuine → tell the user the accuracy cost before §III is written.

### The rest of §III

Not started this session beyond the empty-beam hook in 1a. Known state from
`paper_memo.md` §III:

- §III states the KL twice (production's short paragraph vs the draft's expanded
  subsection) — pick one. Both are pre-retraining numbers anyway.
- `fig:emulator_arch` caption says "coupling transformations" but with
  `features=1` zuko's NSF is **autoregressive**. Flagged 2026-07-28, not changed.
- `fig:flow_transformation` is generated from the **legacy 1+3d checkpoint**
  while `fig:emulator_arch` advertises 7 context features. Regenerate after
  retraining.
- Add the empty-beam justification for the low-$\mu$ cutoff (item 1a).

---

## 3. Production sync notes

- **User changed a bibref in production**: `Wright:1999tv` → `Wright:1999jc`
  (NFW lensing functions). **Already synced into the draft** on 2026-07-28b,
  including the key list at the foot of the file. Neither key is verified
  against ADS.
- Production diff vs HEAD is currently 6 insertions / 6 deletions, all from the
  earlier merge of the bias/PBS passage, the SHMF $\psi\leq1$ range, and the
  radial-profile rewrite.
- Three `\B{}` markers are still **leaked into production** at `:79`, `:265`,
  `:361`, plus a stray `\.` at `:265` and a duplicated, now-contradictory
  "first factor" sentence at `:265`. Reported, not fixed — production is
  user-only. See `paper_writer.md` §7.

---

## 4. Answer to the open question — does 1.95--2.03 demonstrate the $\mu^{-2}$ tail?

Yes, but it is the weakest of the three pieces of evidence available, and the
figure is the better one.

1. **The figure itself is the strongest demonstration.** In $\mu^2\td P/\td\mu$ a
   $\mu^{-2}$ law is a horizontal line. The curves go flat above $\mu\simeq20$
   at all three source redshifts. No estimator, no fitting, just the shape.
2. **Hill index $\alpha = 1.95$--$2.03$** across source redshifts brackets 2, so
   yes it demonstrates it. ⚠ Two caveats. It is a point estimate with no quoted
   uncertainty in `docs/edge_tail_flux_note.md` §2, and it is measured in the
   asymptotic window — near $\mu=8$ the effective exponent is still $\approx2.25$
   (see `--tail-fit-mu` help in the plot script). So "1.95--2.03" is the
   asymptotic index, not the index everywhere above 8.
3. **Ruling out the alternative is arguably the most convincing.** A Poisson
   likelihood on the counts above $\mu=8$ favours $\mu^{-2}$ over $\mu^{-3}$ by
   $1.4\times10^3$--$6.8\times10^3$ in log-likelihood at every source redshift.

Recommended: lead with the figure, keep one clause for the Hill index or the
likelihood comparison, drop the other. All three numbers come from the
2026-07-11 study on the `-ar` build and were **not re-verified this session**.

---

## 5. ~~Blocking on the rebuild~~ — NOT BLOCKING, this entry was WRONG (2026-07-29)

⚠⚠ **This section claimed `make build` had not been run since the 2026-07-28
`halobias` and subhalo-profile changes. That is false, and it would have gated
work that was already unblocked.** The evidence, checked 2026-07-29:

- `build/CMakeFiles/gwcore.dir/subhalo.cpp.o` and `cosmology.cpp.o` are both
  stamped **2026-07-28 13:32**, later than the sources they compile
  (`subhalo.cpp` 11:38, `cosmology.cpp` 12:28), and `gwlensing…so` carries the
  same 13:32 stamp.
- `tests/test_cosmology_params.py` passes **11/11**, including
  `test_backward_compat_bitwise`, against the reference re-baselined *after* the
  halobias change (`reference_lnmu_pre6d.npz`, 12:46). ⚠ That test runs
  `subhalo=false`, so it proves halobias is compiled but says nothing about the
  profile — the object-file stamp is what covers that.

**Lesson: check the build products, not a prose note.** A note like this one goes
stale the moment someone runs `make build` without editing it, and it is written
in the imperative so it reads as authoritative.

Consequences, revised:

- **`fig:magpdf_tail` was already current** — the high-structure shards were run
  2026-07-28 14:58, i.e. on the post-fix build.
- **`fig:magpdf_zs` genuinely was stale**, but for a reason this entry did not
  name: its cache dated from Jul 25 and sat on the OLD 2000-linear-bin grid over
  $\mu\in[0.4,5]$. Re-run 2026-07-29 at 8 shards × 200k = 1.6M per $z_s$ on the
  current build and the log grid.
- Config drift is ruled out for both: `PRODUCTION_CONFIG` hash `0d50caf91c75`,
  and `subhalo_virial` entered it in commit `8528394` at **12:56**, before either
  run. The only uncommitted change to `ml/params.py` since is comment-only.
  The cache now records `physics_config` and its hash, so this cannot recur.
- Still genuinely open: the flux numbers and the edge quantiles quoted anywhere
  from the 2026-07-11 `-ar` study have NOT been re-measured on this build.

---

## 6. Decisions from 2026-07-28b — settled, do not re-litigate

- **No appendix.** `app:edge_tail` written then deleted (577 w). Tail evidence
  compressed into §II.B, flux numbers into a footnote, edge block dropped.
  Reopens only if the κ_thr insensitivity / model-5 gate / split-invariance
  material is ever written up.
- **§II.B follows Killedar+12 §2.1** (`papers/misc/mnras0420-0155.pdf` p.157) —
  a feature list in prose, minimal equations.
- **Plane = IMAGE throughout.**
- **Citations verified** against PDFs now in `papers/misc/`. `Seitz:1997` and
  `Zakharov:1995` were **wrong papers** and are removed; `Vietri:1983` and
  `Schneider:1992` dropped as redundant/unverified. Surviving §II.B keys needing
  `refs.bib` entries: `Dyer:1972`, `Weinberg:1976`, `Blandford:1986`,
  `Premadi:2001`, `Killedar:2012`. Also missing from production's bib and used
  elsewhere in the draft: `Durkan:2019`, `Giocoli:2006yz`, `Sheth:1999su`,
  and the literal placeholder `CITE-Mpetha`.
- **`fig:subhalo-kappa` is now referenced** in §II.A.4, which required rewriting
  the false cost claim ("costs only a modest factor... requires no resolution
  parameter"). Brute rendering is ~45 ms/ray vs ~0.11 ms/ray smooth-halo.

---

## 7. Still open, not touched

- `plots/variance_D_L.pdf` is referenced but **has no generating script in the
  repo**. User: "for later". It also needs three cumulative curves per the
  §II.B restructure, i.e. a new MC sweep rather than a replot.
- `fig:subhalo-kappa`'s caption is shared with production (not in `\B{}`), so it
  was left alone. It is thin — 22 w, uses `\sigma_k` where the axis and body use
  `\sigma_\kappa`, and does not say what the three curves are. Propose a `\B{}`
  version if wanted.
- `eq:radial`, `eq:wk`, `eq:weakcond`, `eq:sigmashot`, `eq:sigma2h`, `eq:xi1D`
  are labelled but never referenced. Normal in a physics paper, but `eq:radial`
  is odd given how much the subhalo text discusses the radial profile.
