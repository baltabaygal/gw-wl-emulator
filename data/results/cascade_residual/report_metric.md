# #4 step-1: σ-metric recompute — does the composition interaction survive?

**Date:** 2026-07-21. Scripts: `analyze_metric.py` (lnμ metrics, existing shards),
`raw_worker.py`+`run_raw.py`+`agg_raw.py` (certified κ-core, targeted regen z_s=1,5,
10 seeds × 100k rays via `sample_lensing_raw_ml`). Tables: `tables_metric.md`,
`tables_kappacore.md`. Plot: `plots/sigma_metric_recompute.png`.

**Why:** the #3 composition test found a 37% non-additivity in σ at z_s=5, but σ was the
IQR proxy (q84−q16)/2, which disagreed in *sign* with the production clipped-Var σ for
clustering at exactly z_s=5. Before deciding whether the cascade needs a pairwise
interaction term (option b) we must re-measure the interaction in a 2nd-moment metric.

## Result 1 — the interaction is INSIGNIFICANT in every σ metric

Composition non-additivity `NA = Δσ_full − (Δσ_sub + Δσ_bias)`, SNR_NA = |NA|/seed-noise:

| z_s | metric | Δσ_bias (SNR) | NA | \|NA\|/\|Δσ_full\| | **SNR_NA** |
|---|---|---|---|---|---|
| 5.0 | IQR (q84−q16)/2 | −0.0068 (10.4) | −0.0017 | 37% | **2.0** |
| 5.0 | lnμ trim [0.1,99.9]% | −0.0104 (6.1) | −0.0010 | 114% | 0.5 |
| 5.0 | **κ_tot≤1 core (certified)** | −0.0020 (1.2) | −0.0003 | 44% | **0.2** |
| 1.0 | IQR | +0.0014 (6.0) | −0.0002 | 7% | 1.0 |
| 1.0 | **κ_tot≤1 core** | −0.0002 (0.1) | +0.0017 | 52% | 0.5 |

The eye-catching 37% was `|NA|/|Δσ_full|` — a large fraction of a *small* shift,
inflated by the IQR's insensitivity to the shoulder where clustering variance lives. On
the certified κ≤1 core the interaction's SNR_NA is **0.2** at z_s=5, ≤0.5 everywhere.
**Conclusion: there is no statistically resolved composition interaction in σ, in any
metric (max SNR_NA = 2.0, IQR; ≤1.3 in variance metrics).** "Additive at leading order"
is now on firm ground — the pairwise term stays deferred (build option **a**).

## Result 2 — a sharper caveat: clustering's Δσ is metric-fragile (design spec, not a blocker)

The *single-ingredient* clustering shift itself depends on the clip:

| z_s | IQR | lnμ-trim | κ≤1 core |
|---|---|---|---|
| 1.0 | +0.0014 | +0.0032 | −0.0002 |
| 5.0 | −0.0068 | −0.0104 | −0.0020 |

Sign and size move with the metric because **clustering's variance concentrates in the
shoulder near the κ≤1 boundary and the weak-arm sub-threshold tail** — a percentile trim
adaptively bites the very variance clustering adds (the CLAUDE.md "q99.9 trim bites a
κ_thr-dependent slice" gotcha), while the IQR ignores it. Even the certified κ-core is
contaminated at z_s≳5 by the batch-mean-κ anchoring bug (CLAUDE.md item 12: monster rays
to κ≈20 corrupt the per-batch mean used to anchor ⟨κ⟩→0), which our raw estimator
inherits.

**Design consequence:** the cascade's parametric σ operator must be defined on the
emulator's *exact production support/σ definition*, not a convenience proxy — otherwise
the clustering σ-shift is mis-sized or wrong-signed. This does not affect the subhalo
shifts (σ-shift is robust and positive across metrics; #1 stands) nor the z_s≲1 additive
verdict, but it means the clean high-precision resolution of any residual high-z_s
interaction needs: (i) the κ field, (ii) a robust anchoring (mean excluding κ>1), and
(iii) the emulator's actual σ estimator. That is exactly the deferred precision step — now
with a spec.

## Net for the (a)/(b) decision
- Interaction unresolved in every metric ⇒ **build (a): additive single-ingredient shifts,
  with a (b)-hook (empty pairwise slot).**
- Populate the slot only if the KL gate breaches budget; if it does, resolve the σ
  interaction with the κ-field + robust-anchor + production-σ estimator above, not IQR.
