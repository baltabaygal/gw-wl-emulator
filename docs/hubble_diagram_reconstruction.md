# hubble_diagram_reconstruction.md — the forecast pipeline downstream of $P(\mu)$

**Name adopted: "Hubble diagram reconstruction" (HDR).** Not a new coinage — it is
Vaskonen's own §3 title and it matches the function name already in the code,
`lensing::Hubble_diagram_fit`. Use it consistently; do not introduce a synonym
("forecast pipeline", "inference pipeline") in the paper or in memos.

Created 2026-07-29. Living document — date every edit, delete what turns out wrong.

Companions: `paper_prod/paper_memo.md` (draft↔code map), `paper_prod/paper_writer.md`
(writing contract), `paper_prod/TODO.md` (resume point).
**Implementation: `hubble_reconstruct/` (2026-07-29) — see its README for status,
gate results and open items. This memo is the design; that package is the code.**

---

## 1. What HDR is

Everything *downstream* of the magnification PDF. Our paper currently stops at
producing $\td P_I/\td\mu$ and emulating it; HDR is what turns that into a
cosmological forecast, and it is the step that produces Vaskonen's headline
number ($\sigma_8$ to 10% with 300 BNS) and his Fig. 5 corner plot.

Four stages:

| # | stage | output |
|---|---|---|
| 1 | magnification PDF | $p_\mu(\mu \mid z, \theta)$ |
| 2 | mock catalogue | $N$ events $\{z_j,\, D_{L,j},\, \sigma_{D_L,j}\}$ |
| 3 | likelihood | $\mathcal{L}(\theta)$ |
| 4 | MCMC | posteriors on $\theta$ |

⚠ **The PDF feeds stages 2 AND 3.** Into the catalogue, to draw a $\mu$ per fake
event; into the likelihood, as the model re-evaluated at every trial $\theta$.
This is why the emulator matters (§6) and why the selection question of §5 is not
confined to stage 2.

---

## 2. Stage-by-stage, as Vaskonen does it

Reference: Vaskonen 2026, MNRAS 547 (`papers/core/Vaskonen_2026_GWWeakLensingSigma8.pdf`),
§3 "Hubble diagram reconstruction".

### Stage 2 — mock catalogue

Two populations, both assuming EM counterparts give exact redshifts (redshift
errors assumed negligible):

| | BNS / ET | BMBH / LISA |
|---|---|---|
| $N$ | 300 | 12 |
| $z_s$ range | $< 2$ | $< 10$ |
| $\sigma_{D_L}$ | $0.03\,D_L$ | $0.003\,D_L$ |
| $p_z(z)$ | $\propto (1+z)^{2.7}\,\td V_c/\td z$, Madau–Dickinson SFR | extended Press–Schechter + Urrutia+25 MBH–galaxy co-evolution |

Per event: draw $z$, draw $\mu \sim p_\mu(\mu|z,\theta_{\rm fid})$, form
$D_L = \tilde D_L(z)/\sqrt\mu$, add Gaussian measurement noise.

⚠ He flags the distance uncertainties himself as **"intentionally optimistic"**
relative to population-averaged forecasts, justified by EM counterparts breaking
the distance–inclination degeneracy. This is the softest assumption in the paper
and is the reason for §4 below.

**Stage 2 lives in-repo (found 2026-07-29 — missing from the first version of
this memo):** `cpp/main_lensing.cpp` :99–138 (SMBH arm) and :140–183 (NS arm).
The catalogue draws $\ln\mu$ from `Plnmuf` (:120, :166) — i.e. already
**source-plane**, so the plane trap of §2 binds stage 2 as well: an
emulator-driven catalogue must apply the $\mu^{-1}$ conversion *before* drawing,
or mock and likelihood live on different planes. Three more code facts from the
same read:

- ⚠ **OOB bug, SMBH arm — UNBOUNDED, not just the last two events**
  (severity corrected 2026-07-29 after re-reading the loop):
  `zSMBHlist.erase(begin+10, end)` (:110) leaves **10** redshifts, while
  `while (jD < Nheavy = 12)` (:113) demands 12 accepted events. The counters are
  not paired: `jz++` runs on **every** iteration (:116), but `jD++` sits inside
  **two** nested guards, `if (z < zthr)` (:118) and `if (DL > 0.0)` (:130).
  Every event rejected by either guard advances `jz` without advancing `jD`, so
  `jz` can run arbitrarily far past index 9 — `vector::operator[]` UB reading
  heap garbage as a redshift, with no bound set by `Nheavy`. It will not
  necessarily crash. "Events 11–12 are suspect" understates it; the whole tail of
  the SMBH catalogue past the 10th accepted event is undefined.
  [Refinements 2026-07-29: line refs corrected (`jz++` :116, z-guard :118 — were
  :118/:122); the boundary is exhaustion of the 10 STORED entries, not the 10th
  ACCEPTED event — at 0.3% relative error `DL > 0` is a ~300σ non-event, so the
  operative guard is `z < zthr`, and any stored z ≥ 10 starts the garbage
  EARLIER: ≥2 garbage events, more in general. Termination is itself not
  guaranteed — it needs OOB garbage that happens to pass the z-guard.]
  Not yet compared against upstream `halos` (user-controlled).
  **Verify before ANY stage-2 reuse, and treat published SMBH catalogues as
  suspect until it is.**
- The NS $p_z$ (:9, :148) is the pure $(1+z)^{2.7}\,\td V_c/\td z$ rising
  branch — **no Madau–Dickinson turnover**. Harmless at $z<2$ (MD peaks at
  $z\approx1.9$); do not reuse beyond.
- The shipped driver (:231) hardcodes `DLthr = 3.0e8` kpc = 300 Gpc
  $> D_L(z=10)$, so the §5 $P_{\rm det}$ threshold machinery is present but
  **inert in every published run** — the structure exists, never exercised in
  anger.

### Stage 3 — likelihood

His Eq. (12) pushes the magnification PDF through the distance map — a change of
variables, nothing more, but it is the conceptual core: **the lensing PDF becomes
the scatter model of the Hubble diagram.**

$$p_{\rm mod}(D_L|z,\theta) = \int \td\mu \; p_\mu(\mu|z,\theta)\;
\delta\!\left(D_L - \tilde D_L(z,\theta)/\sqrt\mu\right)$$

Then Eq. (10) convolves with the Gaussian measurement error and divides by a
selection normalization $P_{\rm det}$ (Eq. 11):

$$\mathcal{L}(\theta) = \prod_j \frac{\int \td D_L\; p_{\rm obs}(D_{L,j}|D_L)\,
p_{\rm mod}(D_L|z_j,\theta)}{P_{\rm det}(z_j,\theta)}$$

He keeps the **full non-Gaussian** $p_\mu$, noting explicitly that Gaussianizing
it would collapse the whole thing to adding $\sigma_{\rm WL}^2 + \sigma_{D_L}^2$
in quadrature. The non-Gaussian shape is the information; the variance alone is
not the point.

### Stage 4 — MCMC

Metropolis–Hastings on $\theta = \{\Omega_M, h, \sigma_8\}$. Gaussian proposals
tuned to 10–28% acceptance; 4 chains × 2600 samples, 200 burn-in; uniform priors
$h \in [0.59, 0.76]$, $\Omega_M \in [0.15, 0.47]$, $\sigma_8 \in [0.4, 1.4]$;
Gelman–Rubin $\hat R < 1.05$.

Results: $\sigma_8$ to **10% (ET), 30% (LISA), 8% combined**. Strong negative
$\Omega_M$–$h$ degeneracy; $\sigma_8$ nearly uncorrelated with the other two.
Fig. 4 = the Hubble diagram (true $\tilde D_L(z)$, 99% band, catalogue points);
Fig. 5 = the corner plot, 68%/95% contours, ET vs LISA.

### Where it lives in our repo

| piece | location |
|---|---|
| Eqs. (10)–(12) | `cpp/lensing.cpp::loglikelihood` (:1410) |
| stage 4 wrapper | `cpp/lensing.cpp::Hubble_diagram_fit` (:1490) |
| MCMC engine | `cpp/basics.cpp::MCMC_sampling` (:517) |
| no-lensing control | `loglikelihood`, `lens == 0` branch |
| $P(\ln\mu)$ source | `cpp/lensing.cpp::Plnmuf` (:1364; misquoted as :1163 — fixed 2026-07-29) |
| stage 2, both catalogues | `cpp/main_lensing.cpp` (:99–183) |
| stage 2+4 driver (Zlist, chains, `DLthr`) | `cpp/main_lensing.cpp` (:185–232) |

⚠ **Neither `Hubble_diagram_fit` nor `loglikelihood` is exposed to Python** —
`grep -c` on `cpp/python_bindings.cpp` returns 0 for both (verified 2026-07-29).
Driving HDR from the emulator therefore needs either new bindings or a Python
reimplementation of the likelihood. Decide which before starting.

⚠ **The code fits a 4th parameter the paper never mentions (2026-07-29):**
`loglikelihood` sets `C.initialize(dm, pow(10, par[3]))` with `dm` ∈ {CDM, FDM,
WDM, EDM} (`main_lensing.cpp:18`), prior $[-0.6, 1.4]$ in $\log_{10}$, nonzero
step — a DM-model mass parameter, inert in CDM but random-walked by the MCMC and
present in every signature. $\theta = \{\Omega_M, h, \sigma_8\}$ describes the
paper, not the code; anyone binding or reimplementing must know.

⚠ **PLANE CONVENTION TRAP.** `loglikelihood` calls `Plnmuf`, and `Plnmuf` at
:1396 applies `exp(-lnmu)` — so the likelihood runs on the **source-plane** PDF.
That is physically right: a GW event is a random *source*, not a random sky
direction. **Our emulator returns image-plane $\td P_I/\td\mu$**, so any
emulator-driven HDR must convert, $\td P_S/\td\mu = \mu^{-1}\td P_I/\td\mu$. One
line, silently wrong if missed. The inverse ($\mu\,\td P/\td\mu$) is a
natural-looking guess and is wrong — see `paper_prod/paper_writer.md` §5 for the
three independent confirmations of the direction.
Also not literally one line (2026-07-29): `Plnmuf` weights AND **renormalizes**
(:1397–1405), and the renormalization is required, not automatic — after
$\mu^{-1}$ weighting the emulator PDF integrates to $\langle 1/\mu\rangle_I$,
which is only $\approx 1$ (the flux calibration targets $F_{\rm trim}$, not
exactly 1). The residual norm is $\theta$-dependent, so skipping it adds a
spurious $\theta$-dependent term to $\log\mathcal{L}$.

---

## 3. Why HDR is the emulator's whole justification

**One likelihood evaluation takes ~20 s on an M1**, because every step needs a
fresh Monte Carlo realization of the lens population at every likelihood z-node.
4 chains × 2600 samples ≈ $10^4$ evaluations ≈ **58 hours for one 3-parameter
run**. That is the bottleneck the emulator removes, and it is why Vaskonen could
afford 3 parameters where we have 6. When §III or the conclusions need a concrete
statement of what the emulator buys, this is the number.

Two upgrades the speed number undersells (2026-07-29 code read):

1. **The C++ likelihood is STOCHASTIC.** `Plnmuf` re-realizes the MC per call
   (fresh `mt` state, `Nreal = 1e4` rays, `main_lensing.cpp:97`), so
   Metropolis–Hastings runs on a noisy $\log\mathcal{L}$ — a pseudo-marginal
   sampler without the care, with the per-event tail bins the noisiest. The
   emulator gives a deterministic, smooth likelihood: a qualitative fix on top of
   the speed, and arguably the stronger selling point.
2. **z is quantized to 6 nodes** (10 for the combined catalogue):
   `C.Zlist = linlist(minz, maxz, 6)` (`main_lensing.cpp:206`), nearest-neighbor
   lookup in `loglikelihood` (:1459). "Every z-node" above means 6–10, NOT the
   100-pt simulation grid — which is also why 20 s is self-consistent
   (6–10 × 10⁴ rays). The emulator replaces this with continuous z.

---

## 4. The better catalogue — De Leo, Teixeira & Poulin 2026

`papers/misc/2607.20413v1.pdf`, arXiv:2607.20413, 22 Jul 2026, *"The road towards
precision measurements of $H_0$ with bright sirens in the Einstein Telescope
era"*. Flagged by Ville 2026-07-29: *"the way they generate the benchmark
catalogue looks better than what I did … so maybe eventually we want to do
something similar."*

**Take stage 2 only.** Their stages 1 and 3 are useless to us: **zero** occurrences
of "lensing", "magnification" or "$\sigma_8$" in the entire paper (verified by
grep), their likelihood is a plain Gaussian with diagonal covariance (their
Eq. 12), and their free parameters are $H_0$, $\Omega_m$ (+ $r_d$ with BAO).
Their scatter is *purely* measurement error.

### What their stage 2 does better

| ingredient | Vaskonen | De Leo+ |
|---|---|---|
| $\sigma_{D_L}$ | flat 3% / 0.3%, self-declared optimistic | **per-event Fisher forecast from GWFish** on the ET triangular 10 km Sardinia design (Branchesi+23) |
| detection | cut in redshift | **SNR > 20** |
| EM counterpart | assumed | **GRB afterglow modelled** (`afterglowpy`) against LSST sensitivity |
| inclination | not modelled | $\cos\iota$ uniform on $[-1,1]$ |
| population | analytic $p_z$ | 20 000 injections, $0.1 < z < 3.5$, Cutler & Holz merger rate, monochromatic 1.4 $\Msun$ NS |

Their useful shortcut: the full afterglow + LSST calculation is shown to be
≈ equivalent to a cut $\iota < 18^\circ$, so the cheap version is just the
inclination cut.

### What Ville meant, precisely (clarified 2026-07-29)

His remark is about **how each event's numbers are produced, not how many
events there are**. The two upgrades are (i) per-event $\sigma_{D_L}$ from a
Fisher forecast (GWFish) replacing the flat 3%, and (ii) modelled EM-counterpart
detectability (afterglowpy + LSST) replacing "assume a counterpart". $N$ is a
separate question and is NOT what he was flagging.

**Their selection splits into two parts with very different risk, so the work
can be staged:**

| part | $\mu$-coupled? | consequence |
|---|---|---|
| $\iota < 18°$ (EM-counterpart shortcut) | **No** — lensing does not change inclination | behaves exactly like Vaskonen's $z$-cut: catalogue $\mu$ stay a fair sample, no correction needed |
| SNR > 20 | **Yes** | triggers the whole §5 apparatus (matched $P_{\rm det}$, already built and gated) |

So: adopt per-event $\sigma_{D_L}$ + the inclination cut first, at essentially no
statistical risk; adopt the SNR cut second.

**Implementation status (2026-07-29):** `catalogue.generate`'s `frac_sigma_dL`
accepts a **callable** `(z, dL, w) -> sigma_dL`, so a GWFish-backed model plugs
in without touching anything downstream — `Catalogue` already stores
$\sigma_{D_L}$ per event and the likelihood already reads it per event. Verified
both paths (scalar reproduces the flat 3% exactly; callable varies per event).
⚠ The likelihood treats $\sigma_j$ as $\theta$-INDEPENDENT, which holds for a
Fisher forecast made from the observed signal but would NOT hold for a $\sigma$
computed from the trial cosmology — that would resurrect the Gaussian
normalization in the $\theta$-dependence.

### Packages (this is what Ville meant)

- **GWFish** (Dupletsa+23) — per-event Fisher uncertainties on $d_L$ for a
  detector network. Replaces the flat 3%.
- **afterglowpy** (Ryan+20) — GRB afterglow light curves for EM detectability.
- **darksirens** (Martinelli+22) — injection population / merger rate.
- **CANDI** — the wrapper tying them together, `github.com/chiaradeleo1/CANDI`.

⚠ All four are **ET/BNS-oriented**. Vaskonen's second arm is 12 BMBH with LISA to
$z_s = 10$, which is where the lensing signal is largest ($\sigma_{D_L}/D_L
\sim 0.12$ at $z_s = 10$). GWFish supports LISA; their pipeline does not. The BMBH
arm needs separate treatment — factor this into any effort estimate.

### What NOT to take

- **Their sample design.** Their headline is that $H_0$ information comes from
  *low*-$z$ events and saturates above $z \sim 1$. Our regime is the opposite —
  lensing scatter grows with $z$. Take the machinery, not the sample.
- **Their redshift range.** Stopping at $z = 3.5$, BNS only, deletes our best
  lensing signal.

One bonus in our favour: the $\iota < 18^\circ$ cut selects face-on systems, which
are the best-localized, lowest-$\sigma_{D_L}$ events. Smaller measurement error
means lensing scatter dominates the budget *more*, which should tighten
$\sigma_8$.

---

## 5. The selection coupling — why this is not a pure stage-2 swap

**The likelihood must encode the same detection rule that produced the
catalogue.** Change the rule in stage 2 and stage 3 must follow.

- **Redshift cut (Vaskonen).** Lensing does not change $z$, so the cut is
  uncorrelated with $\mu$. At a given $z$ he keeps 100% of events — magnified and
  demagnified alike — so the $\mu$ in the catalogue are a *fair sample* of
  $p_\mu$. This is precisely why he can write that selection effects do not bias
  the inference.
- **SNR cut (De Leo+).** SNR $\propto 1/d_L$ and $d_L = \tilde D_L/\sqrt\mu$, so a
  magnified event looks closer, is louder, and is detected when its unmagnified
  twin is not. The catalogue's $\mu$ are then a *biased sample*, skewed high —
  lensing-induced Malmquist bias.

Generate with the second rule, analyse with a $P_{\rm det}$ built for the first,
and the fit absorbs the mismatch into the cosmological parameters. Biased
$\sigma_8$, with nothing in the output flagging it.

⚠ Note that **neither published paper has this problem.** De Leo+ have no lensing,
so their SNR cut is harmless; Vaskonen has lensing but cuts on $z$. The coupling
is created by combining them.

Generation ordering, if adopted: draw $z$ → draw $\mu$ → form the lensed $d_L$ →
**then** test SNR. Cutting before lensing is wrong.

### How much work this actually is — SMALLER than first assessed

⚠ **Correction, 2026-07-29.** An earlier reading of this called for new machinery
in the likelihood. Wrong — most of it is already implemented. `loglikelihood`
takes a **`DLthr` argument**, and its lensing branch (:1468–1480) accumulates

```cpp
Pdet += dlnmu*Plnmu[i][1]*dY*NPDF(Y, DL0, sigmaDL);
```

inside a loop over the magnification bins `i` weighted by $P(\ln\mu)$, with the
observed-distance integral capped at `DLthr`. That **is** a distance-threshold
detection probability marginalized over magnification — exactly the structure the
De Leo selection needs. So the paper's "cutoff in redshift" is a simplification
relative to what his own code supports.

What is genuinely missing: a real SNR threshold is not a sharp cut in $d_L$ — it
also depends on inclination, masses and sky position. So the hard `DLthr` would be
replaced by a detection probability $P_{\rm det}(d_L, \iota, \ldots)$. The
$\mu$-marginalization is done.

**Lesson (same class as CLAUDE.md §15):** read the implementation before costing a
change from the paper's description. The text and the code disagreed here, and the
code was the more capable of the two.

### Two refinements (2026-07-29)

- **Conditioning caveat.** Dividing per event by $P_{\rm det}(z_j, \theta)$ is
  the likelihood *conditional on the observed $z_j$* — valid and unbiased. Under
  an SNR-type selection the detected-$z$ distribution also becomes
  $\theta$-dependent; conditioning it away is legitimate (loses a little
  information), but do NOT "upgrade" the normalization to the
  population-integrated $P(\det|\theta)$ without also modelling the $z$-term —
  that half-step is the standard way to get it wrong (Mandel–Farr–Gair 2019 is
  the reference for doing it right).
- **The tail coupling.** An SNR selection does not just change the
  normalization — it re-weights the analysis toward the high-$\mu$ tail, exactly
  the region the repo's standing rules call uncertified (and where the emulator
  tail is a fitted POT model). Cheap guard for any future run: report how much of
  each event's $\log\mathcal{L}$ comes from beyond ~q99.9 of the local PDF.

### Literature (2026-07-29) — the fix is standard; cite it

> ✅ **VERIFIED 2026-07-29b.** `fetch_hdr_refs.sh` was run by the user; all eight
> PDFs are in `papers/misc/`. Each arXiv ID was opened and checked against its
> title, authors and abstract. **All eight IDs resolve to the intended paper** —
> no repeat of the `Seitz:1997`/`Zakharov:1995` wrong-paper failure
> (`paper_writer.md` §6). **Two of the characterizations were wrong and are
> corrected below.** Still unchecked: journal volume/page numbers, since the PDFs
> are preprints — confirm against ADS before these reach `refs.bib`. (Partial,
> 2026-07-29: 2011.15109 = MNRAS 504, 3610; 2310.12764 = MNRAS 533, 36;
> 2402.19476 = PRD 110, 023502 were cross-checked against arXiv/ADS listings
> during the original search; the other five remain recall.)

**The $P_{\rm det}$-normalized likelihood** (selection in hierarchical/siren
inference):

- **Mandel, Farr & Gair 2019** (arXiv:1809.02063), *"Extracting distribution
  parameters from multiple uncertain observations with selection biases"* —
  ✅ the canonical GW reference, exactly as described.
- **Loredo 2004** (astro-ph/0409387), *"Accounting for Source Uncertainties in
  Analyses of Astronomical Survey Data"* — ⚠ **characterization corrected.** Was
  called "the older Bayesian truncation treatment"; the word "truncation" does
  not appear in it (0 hits) and "selection" appears once. It is a **Malmquist**
  paper (15 hits) about measurement uncertainty on individual sources and the
  incidental-parameter problem in survey populations, solved by Bayesian
  marginalization. Cite it for the Malmquist/measurement-uncertainty lineage, not
  for a truncated-likelihood derivation.
- **Mortlock, Feeney, Peiris, Williamson & Nissanke 2019** (arXiv:1811.11723),
  *"Unbiased Hubble constant estimation from binary neutron star mergers"* —
  ✅ consistent with "ignoring selection biases $H_0$ for BNS bright sirens".

**Magnification × SNR-limited detection** (the lensing–Malmquist coupling of this
section):

- **Cusin & Tamanini 2021** (arXiv:2011.15109), *"Characterisation of lensing
  selection effects for LISA massive black hole binary mergers"* — ✅ **the
  closest match in the list, and the abstract is near-verbatim what §5 argues**:
  "When selection effects are included, the mean of the magnification
  distribution is shifted from one to higher values for sufficiently
  high-redshift sources. This introduces an irreducible (multiplicative) bias on
  the luminosity distance reconstruction." Applied to LISA MBHBs — i.e. directly
  our 12-event arm. ⚠ Their magnification-vs-observed-distance reconstruction is
  framed for the case *without* an EM counterpart; the selection mechanism still
  applies to ours, the reconstruction does not.
- **Mpetha, Congedo, Taylor & Hendry 2024** (arXiv:2402.19476), *"Impact of weak
  lensing on bright standard siren analyses"* — ✅ and **stronger than the memo
  claimed.** They quantify it: mean bias from magnification selection effects
  $\Delta H_0 = -0.1$ km/s/Mpc for BNS, spread $\pm0.25$, comparable to the
  forecast uncertainty. They also make our pitch for us — $\sigma_\mu(z)$ "is
  dependent on the cosmological parameters that are being constrained" and "is
  also sensitive to the resolution of the simulation used for its calculation".
  Already flagged in `paper_memo.md` §IV as a comparison target; this is the
  same paper.
- **Oguri 2018** (arXiv:1807.02584), *"Effect of gravitational lensing on the
  distribution of gravitational waves from distant binary black hole mergers"* —
  ✅ title and topic as described.
- **Dai, Venumadhav & Sigurdson 2017** (arXiv:1605.09398), *"Effect of lensing
  magnification on the apparent distribution of black hole mergers"* — ✅ real and
  on topic, but ⚠ scope narrower than implied: it is about **BBH without EM
  counterparts**, where magnification is degenerate with intrinsic mass scale and
  redshift. Our sirens are bright. Cite for the mechanism, not for a bright-siren
  result.
- **Canevarolo & Chisari 2024** (arXiv:2310.12764), *"Lensing bias on
  cosmological parameters from bright standard sirens"* — ⚠ **characterization
  corrected.** Was called "precisely the '$\sigma_8$ absorbs it' statement".
  It is not: **$\sigma_8$ and $S_8$ appear zero times in the paper.** They treat
  lensing as a *systematic* and measure residual bias on $\Omega_m$ (78 hits) and
  $h$, from ET bright-siren mocks, finding the lensing bias comparable to or
  greater than the statistical uncertainty. Right mechanism, different parameter.
  The "$\sigma_8$ absorbs it" step is **our** inference and must be presented as
  ours.

Classic anchors if wanted: Malmquist 1922; Turner, Ostriker & Gott 1984 (not in
the script, not fetched, **unverified**).

Suggested sentence (keys verified; journal numbers still to confirm): selection
follows the standard treatment (Loredo 2004; Mandel et al. 2019), with the
magnification–detection coupling as characterized by Dai et al. 2017, Oguri 2018,
Cusin & Tamanini 2021, Canevarolo & Chisari 2024. ⚠ Do **not** attribute a
$\sigma_8$ statement to Canevarolo & Chisari. Mpetha et al. and Canevarolo &
Chisari remain the natural comparison points for the forecast, on $H_0$/$\Omega_m$.

---

## 6. Status and open decisions

- **IMPLEMENTED (2026-07-29) — `hubble_reconstruct/`, mock-driven.** All four
  stages now exist in Python (`../hubble_reconstruct/README.md`), gated 12/12,
  driven by an analytic MOCK P(mu) because the 1+6d retrain has not run.
  **No number from it is a forecast.** What the implementation settled or
  discovered, beyond what this memo already said:
  - **The plane conversion binds stage 2 as well**, and is not one line: the
    weighted PDF must be RENORMALIZED (sec.2). Implemented at exactly one site
    (`pmu.PDFGrid.source_plane()`).
  - ⚠ **P_det enters the NUMERATOR too, not only the normalization** — under
    selection on the TRUE lensed d_L (what an SNR trigger is), the numerator's
    mu-integrand must be restricted to detectable magnifications. Vaskonen's C++
    is self-consistent under the OTHER convention (his `DLthr` cuts on the
    OBSERVED distance, where no extra factor belongs). **Two valid schemes;
    mixing them is not** — and mixing them is not obvious, since the wrong
    version produced a LARGER bias for the "corrected" fit than the uncorrected
    one. This is the sharpest correction the build produced to sec.5.
  - **Measured, at 96 realizations:** omitting the mu-marginalized P_det under an
    SNR cut biases sigma_8 by **+0.0284 +- 0.0054 (+5.3 sigma, +3.5%)**; matched,
    +0.0023 +- 0.0048 (+0.5 sigma). So sec.5's concern is real and quantified
    (on the mock's scatter model — the SIZE will move with the real P(mu), the
    mechanism will not). ⚠ At the 24-realization default it reads only
    +1.8 sigma: enough for the sign, not for the size.
  - **A single injection-recovery pull is N(0,1)** and cannot test for bias —
    the same trap as CLAUDE.md sec.16's 2-3 sigma false positives. The bias gates
    use many realizations + a profile-likelihood estimator instead of one MCMC.
  - **A data-calibrated provider now exists (`--provider dataset`)**, fitted to
    ~3100 configs of stored engine output. It measured three things that had
    been guesses, and one provenance trap:
    (i) the analytic mock was **~4x too wide at z_s=1** -- the datasets give
    sigma(ln mu) = 0.04-0.07 there, and 0.25 is the z_s~5 value (consistent with
    CLAUDE.md's sigma_kappa = 0.031 plateau via ln mu ~ 2 kappa). The mock
    therefore OVERSTATED the sigma_8 information: 6.4% vs 9.7% on the same
    300-event catalogue.
    (ii) **P(mu) depends on h** (h^0.375), which the mock had at h^0 -- so the
    PDF carries h information and the degeneracy structure differs. Om exponent
    is 0.71 (assumed 0.5); only sigma_8 (0.89 vs 1.0) was close.
    (iii) ⚠ **`datasets/` holds TWO backend generations that must not be
    pooled**: the 2026-06-05 sets (`large_1k`, `medium`) are ~21% narrower in
    sigma(ln mu) than the 2026-06-15/17 sets. Pooling inflates the width-law
    residual 6.9% -> 12.6%. Only the newer generation is fitted.
    ⚠ All of it is pre-paper-defaults physics (subhalo_model=3, bias_model=0,
    kappa_anchor=0) and 1+3d, so this is a better stand-in, not the model.
  - **Step 0 PASSED, like-for-like (2026-07-29).** On the dataset-calibrated
    provider, 300 BNS at z<2 with 3% distance errors gives **sigma_8 to 9.7%**
    (converged: R-hat 1.006/1.018/1.004, acceptance 16-17%) against Vaskonen's
    **10%**. The comparison is fair, not lucky: `generate_dataset.py` at the
    datasets' commit passed NO physics kwargs, so they ran on the June defaults
    -- subhalo OFF, bias_model=0, kappa_anchor=0 -- which is essentially
    Vaskonen's own model, since every one of our extensions came later.
    ⚠ Still one catalogue realization, a parametric stand-in for P(mu), and a
    deterministic/continuous-z likelihood where his is stochastic on 6 nodes.
  - **sigma_8(N) scan (`--n-scan`).** 47/35/24/16.5/11.7/8.0/5.7/3.9/2.8 % at
    N = 10/20/40/75/150/300/600/1200/2400 (conditional; marginalizing widens
    ~15%). Scaling is N^{-1/2} to ~1% over a 240x range, i.e. **large N buys no
    extra information, only a bigger assumed survey.** ⚠ N is an ASSUMPTION and
    the sources differ: Vaskonen asserts 300; De Leo+ assert no absolute yield,
    drawing nested sub-catalogues of N=5-60 from a master catalogue and
    parametrising by N. Recommend quoting the CURVE (figure
    `results/sigma8_vs_N.png`), which also answers the scaling question
    Vaskonen's own conclusions raise.
  - ⚠ **Unflagged issue in the training data:** the dataset metadata records
    `invalid_fraction = 0.163` -- 884820 of 5.44M samples discarded, almost all
    `nan_gamma` (881705), plus 3115 with `detA < 0` (negative mu, i.e. strong
    lensing). Invalidity correlates with magnification by construction, so
    dropping them truncates the high-mu tail of the EMULATOR's training data,
    not just this package's calibration. Not discussed in CLAUDE.md; worth a
    look before any tail-sensitive claim.
  - **Stage 2 must draw mu at each event's EXACT z.** Caching P(mu) on a z-node
    grid in the catalogue while the likelihood used its own nodes was a ~15%
    sigma(ln mu) misspecification between truth and model, worth ~1 sigma of
    parameter bias.
- **Not scheduled beyond this.** Ville's word was "eventually". Nothing here
  blocks the current draft.
- **Target pipeline (2026-07-29):** emulator $P_I(\mu)$ ($\mu^{-1}$-converted +
  renormalized, §2) → catalogue à la De Leo+ (stage 2, same conversion before
  drawing) → likelihood with the *matching* SNR selection (§5) → MCMC over the
  6d $\vec\Theta = \{h, \Omega_M, \Omega_B, \sigma_8, n_s, z_{\rm eq}\}$
  (PRIOR_6D, `ml/params.py`). Expectation: the data constrain $\sigma_8$ + the
  $\Omega_M$–$h$ direction; $\Omega_B$, $n_s$, $z_{\rm eq}$ return
  prior-dominated. The scientific output is whether $\sigma_8$ survives
  marginalization over the extra three, not six tight contours.
- **Open: baseline vs replacement.** If the catalogue is swapped *and* the lens
  model extended at once, a changed $\sigma_8$ forecast cannot be attributed to
  either. Recommendation: keep Vaskonen's catalogue as the baseline so the
  ET/LISA numbers stay comparable to his 10%/30%/8%, and add a GWFish catalogue as
  a second forecast. Not decided.
- **Open: Python bindings vs reimplementation** for `loglikelihood` (§2).
  **Recommendation 2026-07-29: reimplement.** The C++ `loglikelihood` computes
  `Plnmuf` internally (:1446), so binding it as-is buys the slow stochastic MC
  likelihood, not an emulator-driven one — a refactor would be needed anyway. A
  ~100-line Python likelihood (plane conversion + change of variables + 1D
  Gaussian convolution + $P_{\rm det}$), validated at the fiducial against the
  C++ on the same catalogue, is the cheaper path. **Step 0 of any HDR run:
  reproduce his 10%/30%/8% with his own catalogue** — the end-to-end
  implementation gate, and the baseline arm that keeps a later catalogue swap
  attributable.
- **Open:** whether HDR appears in this paper at all, or is future work. The
  conclusions' "several simplifications remain" paragraph is the natural home for
  a one-sentence acknowledgement that the benchmark catalogue uses simplified
  distance uncertainties and a redshift-cut selection.
- **Standing rule reminder** (`paper_writer.md` §4): if an HDR forecast is ever
  run and a difference lands at the sampling floor, that *bounds* the effect, it
  does not measure it.
