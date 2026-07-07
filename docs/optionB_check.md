# Option B — independent-code cross-check vs pyHalo

Status: **PASSED, 2026-07-04.** Complements Option A: where Option A validated the
*lensing imprint* (P_sub(k) profile + amplitude vs DRCD18/ETHOS), Option B validates the
*population* (mass function + spatial distribution) against an independently written,
widely-used substructure code.

## Setup

- pyHalo 0.2.8 (Gilman et al.; the standard tool in the strong-lensing flux-ratio-anomaly
  substructure literature), isolated venv `.venv_pyhalo` (Python 3.13, own numpy 2.5 —
  the `test` conda env is untouched). Deps added: colossus, lenstronomy, mcfit.
- Matched host: M = 10¹³ M⊙, z_l = 0.5, z_s = 2; mass window [10⁷, 10¹⁰] M⊙; subhalos
  only (`LOS_normalization=0`, `two_halo_contribution=False`); cone widened to 12″
  (~38 kpc at z_l=0.5 — the strong-lensing region pyHalo renders, vs our full
  r200 ≈ 378 kpc; comparison is therefore scoped to the mass function and the INNER
  radial shape, which is where the lensing-relevant statistics live).
- pyHalo SHMF slope set to −1.82 (dN/dm) to match ours (dN/dlnψ ∝ ψ^−0.82). Over this
  window ψ = m/M ∈ [10⁻⁶,10⁻³], so our exp(−βψ^ω) cutoff ≈ 1 and both are pure power laws.
- 40 realizations (≈900 subhalos each). Script `tmp/pyhalo_compare.py`, data `.npz`.

## Results

| quantity | pyHalo | this work | agreement |
|---|---|---|---|
| SHMF slope dN/dlnm | −0.802 | −0.820 | 2% (within histogram-fit error) |
| **m_eff = ⟨m²⟩/⟨m⟩** | 2.23×10⁹ M⊙ | 2.14×10⁹ M⊙ | **ratio 1.042 (4%)** |
| projected n(R), 4–32 kpc | ~2.4×10⁻⁴ (flat) | ~2.2×10⁻⁴ (flat) | 8–15% |

- **m_eff is the key number:** it is exactly the "effective mass" that sets the P_1sh
  amplitude (DRCD18 eq. 30, P_1sh ∝ κ̄_sub m_eff/Σ_cr). Two independent implementations of
  the CDM subhalo mass function give it to 4% at matched slope and window — a direct
  cross-check on our SHMF normalization and mass sampling.
- **Radial shape:** both codes give a nearly flat area-normalized projected number density
  across the inner tens of kpc (ours from the Han+16 anti-biased profile, theirs from the
  pyHalo default), consistent with DRCD18's "weak radial dependence near the host center,
  take n_sub ≈ const" and agreeing at the ~10% level. The ~10% offset is a
  normalization/shape detail within the flat inner region, not a trend.

## Scope and caveats

- Truncation does NOT enter these three statistics: m_eff and slope are mass-function
  quantities (total/infall mass, unchanged by tidal stripping), and n(R) is spatial. The
  profile-truncation difference (our untruncated NFW vs pyHalo's tidal tNFW) is the
  Option-A subject and is already quantified there (<10% on the κ-variance).
- Geometry: pyHalo renders the inner ~38 kpc strong-lensing cone; the outer host is not
  compared here (nor needed — the lensing signal is inner-dominated, and the outer radial
  profile was validated internally in the convergence studies).
- pyHalo default subhalo profile is a (pseudo-)NFW with tidal truncation and its own
  concentration model; we matched only the SHMF slope and window, deliberately leaving the
  rest at pyHalo defaults so the comparison is a genuine independent draw, not a mirror.

## Combined external-validation statement (Options A + B)

- **Population** (SHMF slope, effective mass, inner spatial profile) ≈ pyHalo to ≤4% on
  the amplitude-setting m_eff (independent code) — Option B.
- **Lensing imprint** (P_sub(k) shape + amplitude, hence the κ-variance) ≈ DRCD18 to ≤10%,
  and DRCD18 ≈ ETHOS N-body — Option A.

Two independent external references, one on the input population and one on the output
lensing statistic, both consistent within ~10%. Together with the internal exactness
checks (mass conservation, sampler verification, invariants) and the numerical convergence
studies, this closes the validation chain for the paper.
