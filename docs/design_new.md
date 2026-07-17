# Design: environment field without a free \(R_\perp\)

**Status:** research consensus locked (2026-07-17). Next executable step: **A0**
(analytic kernel comparison; no C++).  
**Companions (detail / history):**

| Doc | Role |
|---|---|
| `docs/nz_bias_convergence_note.md` | Problem: legacy iid bias owned by grid (\(\Delta z\), \(r_{\max}(\kappa_{\rm thr})\)) |
| `docs/bias_field_design_note.md` | Correlated LOS field redesign (model 0/1/2), KP91 pencil, \(R_\perp \ge R_L\) |
| `docs/bias_field_joint_framework.md` | Joint count + clustered weak arm; sum rule; **derived window** sketch |
| `data/results/bias_field_joint/` | Sizing ruler (`sizing.npz`, `report.md`) |
| `papers/halo_biasing/` | Seljak 0001493, Cooray & Sheth 0206508, Peacock & Smith 2000 |

This note is the **overall design summary**: problem, literature dictionary, design
space, locked target architecture, prerequisites, and the A0→A4 ladder.

---

## 0. One-paragraph summary

The production bias layer (`bias_model=1`) realizes a 1D environment field
\(\delta_{\rm 1D}(\chi)\) with a **free** transverse window \(R_\perp\) (default
\(R_L(10^{14}\,M_\odot)\approx 8.44\,{\rm Mpc}\)). That knob owns real physics
(body variance, tails). Halo-model literature never introduces \(R_\perp\) because it
computes **ensemble** 2-halo statistics in the large-cell / \(k\to 0\) limit and never
draws a finite-resolution field. We cannot take that limit if we want a Monte Carlo
magnification PDF with count–environment correlations. The fix is **not** “fit
\(R_\perp\) to a sim,” and **not** “claim zero scales.” It is: **retire free
\(R_\perp\)** in favor of a **derived** environment spectrum from the halo-model 2-halo
kernel (profiles \(\tilde u(k|m)\), mass/bias weights), one shared \(\delta_{\rm 1D}\),
and a **sum-rule split** into count modulation + clustered weak \(\kappa\). Scales that
remain live in \(R_L(M)\) and \(\tilde u\) — the same commitment as accepting the HMF
and NFW at all. `bias_Rperp` stays only as a debug/ablation override.

---

## 1. Problem statement

### 1.1 What went wrong with the legacy layer

`lensing.cpp::deltaNhfNFW` (bias_model 0) modulates each \((j_z,j_M)\) cell by an
**independent** lognormal with

\[
\sigma_b = D_g(z)\,b\,\sigma(M_b),\qquad
M_b = \text{tube segment }(\Delta z\text{-shell},\; r_{\max}(\kappa_{\rm thr})).
\]

Physical environment variance is then owned by **numerical knobs** \(\Rightarrow\) no
continuum limit in \(N_z\), \(\kappa_{\rm thr}\gtrsim 3{\times}10^{-3}\) blow-up, unphysical
\(j_M\) independence, grid-defined far tails. See `docs/nz_bias_convergence_note.md`.

### 1.2 What model 1 fixed — and what it left open

Model 1 (`BiasField1D`): one correlated pencil-projected Gaussian field, shared by all
mass bins via \(b(M,z)\), KP91 spectrum with disk window \(W(k_\perp R_\perp)\). Grid no
longer owns \(\sigma_b\). **But** \(R_\perp\) is still a free parameter that moves
\(P(\ln\mu)\). Production default \(R_\perp = R_L(10^{14})\) is a PBS-motivated prior,
not a derivation. Joint weak arm (`bias_weak`) adds clustered \(\kappa_W\) on the same
field; amplitude sizing passed (`docs/bias_field_joint_framework.md`), but the
transverse kernel of that field is still the chosen disk.

### 1.3 Success criterion (precise)

| Goal | Meaning |
|---|---|
| **No free \(R_\perp\)** | Observables not tuned by a hand-set transverse radius |
| **Not “no scales”** | UV / split scales may live in \(R_L(M)\), \(\tilde u(k|m)\), \(\kappa_{\min}\), pad \(L\) |
| **Split invariance** | Moving \(\kappa_{\rm thr}\) / \(M_{\min}\) only reshuffles arms; body \(P(\ln\mu)\) stable |
| **Keep \(S_{ew}\)** | Count–environment covariance is measured and large (\(\sim 2\times S_{ww}\)); do not drop it |

Far tails / \(f(\kappa>1)\) remain **uncertified** in this entire method class (turboGL
conclusion); “no free \(R_\perp\)” does not certify them.

---

## 2. Why the literature has no \(R_\perp\)

Folder: `papers/halo_biasing/` (Seljak; Cooray & Sheth; Peacock & Smith).

### 2.1 What they compute

Ensemble averages only: \(P(k)\), \(\xi(r)\), mean bias of a population. Canonical 2h
term (Seljak eq. 8):

\[
P_{\rm dm}^{hh}(k)
=
P_{\rm lin}(k)
\left(
\int f(\nu)\,b(\nu)\,y[k,M(\nu)]\,d\nu
\right)^2,
\]

with \(y=\tilde\rho(k,M)/M\to 1\) as \(k\to 0\). No field is realized; \(P_{\rm lin}\) is
an input function. No transverse pencil window appears.

### 2.2 Where the split scale appears and dies (Cooray & Sheth §3.3–3.4)

1. Conditional mass function inside a **cell** \((M,V)\) with overdensity \(\delta\) —
   the cell **is** a finite environment scale (via \(\sigma(M)\)).
2. **Large-cell limit:** \(\sigma(M)\to 0\), \(|\delta|\ll 1\) \(\Rightarrow\)
   \(n(m|\delta)\approx [1+b_1(m)\delta]\,\bar n(m)\), and \(b_1\) is **independent of
   cell size**.
3. Clustering uses only the number \(b_1\): \(\xi_{hh}\approx b_1 b_1'\xi_{\rm lin}\) at
   large separation. The cell scaffolding is gone.
4. Admission (§3.4): linear bias is only accurate on large scales; finite cell size
   needs \(b_2,b_3,\ldots\) / scale-dependent bias.

**One line:** they replace “draw smoothed \(\delta\) and multiply counts” by
“\(\langle nn'\rangle=\bar n\bar n'(1+bb'\xi_{\rm lin})\)” and send the environment
window to infinity so \(b\) is scale-independent.

### 2.3 Peacock & Smith Fig. 7 (common misreading)

The mass cut \(M\gtrsim 10^{11.8}\,h^{-1}M_\odot\) is a **catalogue / FoF resolution
floor** (our \(M_{\min}\) / HOD threshold), **not** \(R_\perp\). The simulation evolves
full \(\delta_{3D}\); PBS is an *explanation* of empty voids, not a filter they apply.
See §3 below.

### 2.4 Dictionary

| Halo-model literature | This codebase |
|---|---|
| Cell \((M,V)\) with \(\delta\) (C&S §3.3) | Shell with \(\delta_i\), smoothed on a transverse kernel |
| \(n(m\|\delta)=[1+b\delta]\bar n\) after \(\sigma(M)\to 0\) | Same law with **finite** \(\sigma_i\), drawn \(\delta_i\) |
| “\(b_1\) independent of cell size” | True only in their limit; residual dependence = \(R_\perp\) sensitivity |
| \(P^{hh}\propto b^2 P_{\rm lin}\) | We form one LOS and integrate \(\kappa\), not \(P(k)\) |
| Profile \(y(k,M)\) in 2h | Our \(\tilde u\); natural UV of a **derived** kernel |
| No free \(R_\perp\) | Appears only because we **realize** a stripped environment field |

---

## 3. Simulation vs our factorization

### 3.1 \(N\)-body / peak-patch (one process)

```text
δ_3D(k_all)  ──gravity──►  particles  ──FoF──►  halos in the right places
                 (long + short coupled; PBS emergent)
```

- No separate “background draw.” Long modes and collapse share one field.
- Only numerical scales: particle mass, softening, FoF \(N_{\min}\), box \(k_{\rm fund}\).
- Fitting an \(R_\perp\) to a sim **calibrates an effective parameter** in a reduced
  model; it does not “discover” a scale the sim used. turboGL’s \(k_L(z)\) fit to
  PINOCCHIO is that industry move — acceptable as calibration, wrong as a first
  principle for us.

### 3.2 Our Monte Carlo halo model (factorized)

```text
long modes   →  draw δ_env (needs a UV / 1h–2h split)
collapse     →  n̄(M,z), b(M,z)
discreteness →  N ~ Poisson(λ), λ = n̄ exp(b δ − ½ b²σ²)   [Cox process]
profiles     →  NFW on each explicit encounter
```

- Clustering of centers is a **theorem of the Cox construction**
  (\(\xi_{hh}\approx bb'\xi_{\rm env}\)), not a second Poisson.
- Free \(R_\perp\) appears because we reinserted long modes as an explicit 1D field
  after throwing away short modes (replaced by HMF+NFW). That field must not
  double-count halo-scale power.

### 3.3 Two different “scale” problems

| | Origin | Can it vanish? |
|---|---|---|
| **A. UV of zero-width pencil \(\delta\)** | Unsmoothed linear variance (log) divergent | Needs *some* UV completion |
| **B. 1h / 2h double-counting** | HMF+NFW already spent power at \(k\sim 1/R_L\) | Needs *some* split |

Classical cosmic shear integrates matter \(P(k)\) with lensing kernels and never names
\(R_\perp\). Our free radius is an artifact of factorization B, not a law of nature.

**Important non-identification:** \(R_\perp\) must **not** be tied to
\(r_{\max}(\kappa_{\rm thr})\). That re-infects the legacy disease (environment variance
owned by the counting split). \(\kappa_{\rm thr}\) is bookkeeping along the ray;
\(R_\perp\) / the environment kernel is PBS / 2h physics. Joint design: split
invariance requires the environment definition **not** to ride \(\kappa_{\rm thr}\) in
an ad hoc way (derived unresolved kernel may depend on the split only through sum-rule
bookkeeping).

---

## 4. Design space (ranked)

### Design 0 — Fixed global \(R_\perp\) (status quo)

Production: \(R_\perp = R_L(10^{14})\), optional `bias_weak`.  
**Pros:** shipped, body usable. **Cons:** free scale owns physics.  
**Role:** baseline and ablation reference only.

### Design A — Derived 2h kernel (target) ★

Do **not** low-pass \(P_{\rm lin}\) with a chosen disk radius. Build the environment
spectrum as the **halo-model 2-halo** continuum:

\[
W_L(k)
=
\frac{\displaystyle\int dm\, n(m)\,b(m)\,m\,\tilde u(k|m)}
     {\displaystyle\int dm\, n(m)\,b(m)\,m}
\,,\qquad
P_{\rm 2h}(k)=W_L(k)^2\,P_{\rm lin}(k).
\]

- \(k\to 0\): \(\tilde u\to 1\), sum rule \(\int n b m =\bar\rho\) \(\Rightarrow\)
  \(P_{\rm 2h}\to P_{\rm lin}\).
- Large \(k\): \(\tilde u\) (NFW FT) kills power — **UV from profiles**, not from
  \(R_\perp\).
- Pencil project \(P_{\rm 2h}\) (KP91, **no extra** \(W_{\rm disk}(R_\perp)\)) \(\to\)
  \(P_{\rm 1D}(k_\parallel)\) \(\to\) one shared \(\delta_{\rm 1D}(\chi)\).

Unresolved arm (joint framework §4): same idea restricted to the below-cut population;
shot piece remains Campbell \(\sigma_W\).

| Free \(R_\perp\)? | No |
| Split invariance? | Yes if both arms share \(\delta\) and sum-rule amplitudes |
| Matches Seljak 2h in ensemble? | Yes |
| Keeps \(S_{ew}\)? | Yes |

### Design B — Per-mass sharp \(R_L(M)\) only

\(\delta_M\) low-passed at \(R_L(M)=(3M/4\pi\bar\rho_m)^{1/3}\) per bin.  
No free global radius; scale owned by halo definition (PBS). Sharp-filter approximation
to A. Easy stepping stone; prefer A for production claim.

### Design C — turboGL-style: no count modulation; additive \(\kappa_L\) only

Pure Poisson explicit arm + additive clustered \(\kappa_L\).  
**Retreat:** drops \(S_{ew}\) (\(\approx 2\times S_{ww}\) in sizing — the selling point).

### Design D — Never realize \(\delta\) (ensemble 2h only)

C&S/Seljak path: moments / \(P(k)\) without a field draw.  
**Not** a magnification-PDF Monte Carlo engine. Useful for analytic acceptance targets.

### Design E — Peak-patch / WebSky / PINOCCHIO slab

Literal one-field construction; no free \(R_\perp\).  
**Calibration truth** (\(M\gtrsim 10^{12}\) slab feasible; full \(M_{\min}=10^7\) pencil
not). Not production.

### Fit \(R_\perp\) to a simulation

Valid only as **validation** of a derived kernel (“does \(R_\perp^{\rm eff}\) land near
\(R_L(10^{13}{-}10^{14})\)?”) or as an openly calibrated effective scale (turboGL).
**Not** the primary definition of the model.

---

## 5. Locked target architecture

### 5.1 Principle

> **One shared Gaussian field whose spectrum is the halo-model 2-halo spectrum
> (profile-weighted), split by mass conservation into count modulation + clustered
> weak \(\kappa\), with no independent \(R_\perp\).**

```text
                         P_lin(k)
                            │
                            ▼
          P_2h(k) = [∫ n b m ũ(k|m)/ρ̄]² P_lin     ← UV from ũ, not R_⊥
                            │
               KP91 pencil projection (no extra W_disk)
                            │
                            ▼
                      δ_1D(χ)   one draw per ray
                     ┌─────────┴─────────┐
                     ▼                   ▼
           λ_e ∝ exp(b δ − ½b²σ²)    κ_W,clust = a_w · δ
           N ~ Poisson(λ)            + κ_W,shot (Campbell)
           NFW explicit
                     └─────────┬─────────┘
                               ▼
                          κ_tot → μ
```

### 5.2 Parameters that remain (not “environment fudge”)

| Parameter | Role |
|---|---|
| Cosmology + HMF + \(b(M)\) | Structure growth; **\(b\) must be PBS of the code’s HMF** (§6) |
| NFW (ideally truncated) | Profiles + \(\tilde u\); lensing and kernel UV |
| \(\kappa_{\rm thr}\), \(M_{\min}\) | Bookkeeping split only (acceptance: split invariance) |
| \(N_z\), \(N_M\) | Quadrature; must converge; must **not** own \(\sigma_b\) |
| \(\kappa_{\min}\) / `eps_floor` | Absolute weak floor (model domain; untruncated-NFW artifact) |
| Pad \(L\approx 1.05\chi\) | Periodic field box (known wrap leakage \(\lesssim 1.6\%\)) |
| \(P_{\rm lin}\) in the transition | Halo-model class limit (C&S §3.4) |

### 5.3 What disappears from the default path

- `bias_Rperp` as a **physics** knob.  
- Keep the CLI/API argument only as **debug / ablation override** (force a disk window
  for scans).

### 5.4 Joint weak arm (unchanged logic, cleaner kernel)

\[
\kappa = \kappa_{\rm explicit}(\text{counts}\leftarrow\delta)
       + \kappa_{W,{\rm shot}}
       + \kappa_{W,{\rm clust}},
\qquad
\kappa_{W,{\rm clust}} = \sum_{j_z} a_w[j_z]\,\delta_{j_z}.
\]

- \(a_w\): bias-weighted first moment of sub-threshold population (+ sub-\(M_{\min}\)
  continuum); **prediction**, not a scan.  
- Sum-rule subtraction \(a_w \propto (d\bar\kappa_{\rm tot} I_b - W_{eb})\) is
  **diagnostic only** until NFW truncation fixes \(f_{\rm exp}>1\) (measured 1.50 at
  \(z_s=0.2\), 1.03 at \(z_s=1\)).  
- Sizing gate already passed: \((S_{ww}+2S_{ew})/\mathrm{Var}_{\rm tot} \sim 12{-}25\%\)
  at \(z_s\le 1\) (windowed); \(\rho_{ew}\approx 0.95{-}1\); \(2S_{ew}\sim 2\times S_{ww}\).  
  Ruler: `data/results/bias_field_joint/`.

### 5.5 Modulation law (orthogonal axis)

| Flag | Law |
|---|---|
| model 1 | \(\lambda = \exp(b\delta - \tfrac12 b^2\sigma^2)\) (current) |
| model 2 | Conditional first-crossing ratio of the code’s `pFC` (design note §4) |

Kernel design (this doc) is independent of 1 vs 2; model 2 still needs a finite
\(S_{\rm env}\) from the same derived field.

---

## 6. Prerequisites (non-negotiable before production claim)

These break any derived-kernel design anchored to the sum rule. Fix regardless of A0
outcome.

### 6.1 Bias–HMF \(q\) mismatch

- `pFC` uses \((p,q)=(0.3,0.8)\).  
- `halobias` is SMT01 with \(q=0.75\).  
- Mass-weighted mean bias \(\int M n b\,dM/\bar\rho\) fails the sum rule at the few-%
  level (joint report: \(I_b(0.75)\) vs \(I_b(0.8)\) at \(z=1\)).  
- **Fix:** derive \(b\) as the PBS response of the code’s own HMF (\(q=0.8\)), or adopt
  model 2 where bias coefficients are automatic.

### 6.2 Untruncated NFW / \(f_{\rm exp}\)

- Explicit projected mass within \(r_{\max}\) can exceed \(M_{200}\) when
  \(r_{\max}\gtrsim r_{200}\) \(\Rightarrow\) \(f_{\rm exp}>1\).  
- Sum-rule **subtraction** route for \(a_w\) fails; direct first-moment route inherits
  \(\lesssim\times 1.5{-}2\) inflation.  
- **Fix (A3):** truncate NFW in bookkeeping and in \(\tilde u\) (BMO / \(r_{200}\)) so
  \(f_{\rm exp}\le 1\) and the sum rule can close cleanly.

### 6.3 What stays out of scope for “done”

- Certifying \(f(\kappa>1)\) / extreme tails as \(N\)-body truth.  
- Filaments as next-order physics (PINOCCHIO gain over turboGL), not a sharper
  \(R_\perp\).

---

## 7. Implementation ladder

| Stage | What | Gate | C++? |
|---|---|---|---|
| **A0** | Analytic: \(P_{\rm 2h}\)/\(W_L\), \(P_{\rm 1D}\), \(\sigma_{\rm 1D}\), \(S_{ee}\) (and weak budget if easy) vs disk scan in \(R_\perp\) and vs joint `sizing.npz` | \(R_\perp^{\rm eff}(z_s{=}1)\) near \(R_L(10^{13}{-}10^{14})\); \(S_{W_L}/S_{\rm disk}(8.44)\sim 1\) on body | No |
| **A1** | Default environment spectrum = \(P_{\rm 2h}\) projection; `bias_Rperp` override only | \(\langle\lambda\rangle=1\); \(N_z\) flatness; bitwise flag path | Yes |
| **A2** | Joint weak arm on same derived field | Split-invariance JSD vs \(\kappa_{\rm thr}\) | Yes |
| **A3** | NFW truncation in bookkeeping + \(\tilde u\) | \(f_{\rm exp}\le 1\); sum rule ~1% | Yes |
| **q-fix** | `halobias` ↔ `pFC` | \(\int Mnb/\bar\rho = 1\) | Yes |
| **A4** | Optional true per-mass filters if A1 fails heavy-bin tests | \(\xi_{hh}(M)\) vs \(b(M)^2\xi\) | Yes |
| **E** | One-off peak-patch / PINOCCHIO slab | Body PDF / counts vs A | No (external) |

Do **not** fit \(R_\perp\) first. Do **not** start production peak-patch.

### 7.1 A0 specification (cheapest next step)

- **Script (planned):** `scripts/convergence/bias_field_A0_kernels.py`  
- **Stack:** `bias_field_joint.py` / `bias_field_prototype.py` (test env; no `build/`).  
- **Ruler:** `data/results/bias_field_joint/{sizing.npz, report.md}`.  
- **Kernels to compare** at \(z_s\in\{0.2,1,5\}\) (10 optional):

  1. **K_disk(\(R_\perp\))** — production disk; scan \(R_\perp = R_L(10^{12}),\ldots\) and
     literal 8441 kpc.  
  2. **K_WL** — derived \(W_L(k)\) / \(P_{\rm 2h}\) above; no free radius.  
  3. **K_sharp** — mass-weighted / per-mass \(R_L\) sharp filter (Design B check).

- **Headline outputs:** \(\sigma_{\rm 1D}\) or shell \(\mathrm{Var}(\delta)\); \(S_{ee}\)
  (and \(S_{ww}+2S_{ew}\) if cheap); ratio \(S_{\rm WL}/S_{\rm disk}(8.44)\);
  effective \(R_\perp^{\rm eff}\) with \(S_{\rm disk}(R^{\rm eff})\approx S_{\rm WL}\);
  reminder of \(I_b(q)\), \(f_{\rm exp}\).  
- **Deliverables:** `data/results/bias_field_A0/{kernels.npz, report.md}` (+ optional
  plot).  
- **Supervisor success pattern:** \(R_\perp^{\rm eff}(z_s{=}1)\sim 5{-}15\,{\rm Mpc}\),
  body budget ratio \(\sim 1\) \(\Rightarrow\) default PDF barely moves; weak amplitude
  stays a joint prediction; **\(R_\perp\) knob can be retired.**

### 7.2 Supervisor one-liner (if A0 lands clean)

> Environment spectrum is the halo-model 2h kernel
> \(W_L(k)=\int n b m\tilde u/\int n b m\), not a free \(R_\perp\). At \(z_s=1\) this
> matches the present disk at \(R_L(10^{14})\) to X%. Weak clustered arm amplitude
> remains the joint sum-rule prediction. `bias_Rperp` remains a debug override.

---

## 8. Locked consensus checklist

| Item | Decision |
|---|---|
| Prerequisites | **q-fix** + **NFW truncation / \(f_{\rm exp}\)** before production derived-kernel claim |
| Design C | Retreat (drops \(2S_{ew}\)) |
| Design D | Not a PDF engine |
| Design E | Calibration truth only (\(M>10^{12}\) slab) |
| Fit \(R_\perp\) to sim | Validation / optional calibration only — not the definition |
| `bias_Rperp` after A ships | Debug / ablation override only |
| Far tails | Uncertified in this method class |
| Remaining model scales | \(\kappa_{\min}\), \(L=1.05\chi\) pad, \(P_{\rm lin}\) in transition |
| Meaning of success | No **free** \(R_\perp\); scales in \(R_L(M)\) and \(\tilde u\) |
| Target | Design A + joint weak arm + sum rule |
| Next step | **A0** (analytic; no rebuild) |

---

## 9. Relation to current code flags

| Flag / default | Today | After Design A default |
|---|---|---|
| `bias_model=0` | Legacy iid (bitwise) | Unchanged legacy |
| `bias_model=1` | Correlated field + disk \(R_\perp\) | Correlated field + **\(P_{\rm 2h}\)** kernel |
| `bias_Rperp=8441` | Physics default \(R_L(10^{14})\) | Override only; ignored on default path (or forces disk ablation) |
| `bias_weak` | Clustered weak on same field (default off in some paths; check bindings) | Same; amplitudes from joint bookkeeping on derived field |
| \(\kappa_{\rm thr}\) / fixed-\(\langle N\rangle=100\) | Counting split | Counting split only; must pass split-invariance |

Reproduction note (existing): runs that assumed the old 3000 kpc placeholder must pass
`bias_Rperp=3000` explicitly; after A, disk ablations must pass an explicit override.

---

## 10. References (minimal)

- Seljak, arXiv:astro-ph/0001493 — halo-model \(P(k)\), 2h structure.  
- Cooray & Sheth, arXiv:astro-ph/0206508 — PBS, large-cell limit, scale-dependent bias.  
- Peacock & Smith, MNRAS 318, 1144 (2000) — HOD, censoring, PBS language.  
- Mo & White, arXiv:astro-ph/9512127 — classic PBS bias derivation (not in-repo).  
- Kaiser & Peacock 1991, ApJ 379, 482 — 1D projection / pencil spectra.  
- Alfradique+24, arXiv:2405.00147 — turboGL deconstruction; calibrated \(k_L\); tails.  
- Agrawal+17, arXiv:1706.09195 — lognormal field mocks.  
- Bond & Myers 1996; Stein+19 (WebSky) — peak-patch (Design E lineage).  
- In-repo: `docs/bias_field_design_note.md`, `docs/bias_field_joint_framework.md`,
  `docs/nz_bias_convergence_note.md`.

---

## 11. Changelog

| Date | Note |
|---|---|
| 2026-07-17 | Initial write-up: research consensus, Design A target, A0 ladder, prerequisites, literature dictionary. |
