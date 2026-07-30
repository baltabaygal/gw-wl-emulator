# hubble_reconstruct — HDR, everything downstream of $P(\mu)$

Stage 1 $P(\mu)$ → stage 2 mock catalogue → stage 3 likelihood → stage 4 MCMC.
A Python reimplementation of Vaskonen (2026) §3, built so the emulator can drive
it instead of the ~20 s/evaluation stochastic C++.

Design doc: **`../docs/hubble_diagram_reconstruction.md`** — read it before
touching the plane convention or the selection handling. Those are the two
documented silent-failure modes and this package is organized around them.

Created 2026-07-29. Living document — date every edit, delete what turns out wrong.

---

## Providers

| name | what it is | parameter space |
|---|---|---|
| `dataset` | **calibrated to the stored simulator output** (~3100 configs, June-2026 engine). Width law + shape template, 5.8% rms; validated on held-out configs. **Best available.** | 1+3d |
| `mock` | analytic caricature; right anatomy, guessed parameter response | 1+3d |
| `ace` | `ace_lensing.predict_pdf`; plane ASSUMED and unverified, different model | 1+3d (+w) |
| `emulator` | production flow — **stub, raises** until the 1+6d retrain | 1+6d |

```bash
python -m hubble_reconstruct.calibrate summarize   # datasets -> cache (once, ~1 min)
python -m hubble_reconstruct.calibrate fit         # cache -> calibration json
python -m hubble_reconstruct.run_forecast --provider dataset --arm bns_et
```

### What the calibration measured (2026-07-29)

$$\ln w = -0.081 + 2.325\ln z - 2.227\ln(1+z) + 0.893\ln\sigma_8
+ 0.708\ln\Omega_M + 0.375\ln h$$

($w = q_{50}-q_{05}$, the tail-free body width; 5.8% rms over 3068 configs.)
Held-out test configs: width unbiased to +0.3% with 5.9% 68th-percentile
scatter; location +0.04 widths.

Three things this exposed, all of which had been guesses:

1. **The mock was ~4× too wide at $z_s=1$.** I had anchored $\sigma(\ln\mu)=0.25$
   there; the datasets say **0.04–0.07**, and 0.25 is the $z_s\approx5$ value.
   Consistent with CLAUDE.md's $\sigma_\kappa=0.031$ plateau at $z_s=1$ via
   $\ln\mu\simeq2\kappa$. Consequence: **the mock overstated the $\sigma_8$
   information** — 6.4% vs 9.7% on the same 300-event catalogue.
2. **$P(\mu)$ depends on $h$** ($h^{0.375}$), which the mock had at $h^0$. So the
   PDF itself carries $h$ information, changing the degeneracy structure. The
   $\Omega_M$ exponent is 0.71, not the assumed 0.5; only $\sigma_8$ (0.89 vs
   1.0) was close.
3. ⚠ **The datasets are two backend generations and must not be pooled.**
   `large_1k`/`medium` (2026-06-05) run ~21% narrower in $\sigma(\ln\mu)$ than
   `backend_current_1k`/`logz_1k`/`tailrich` (2026-06-15/17). Pooling raises the
   residual 6.9% → 12.6%. Only the newer generation is fitted (`CALIB_DATASETS`);
   the older stays in the cache so the offset remains visible.

⚠⚠ **Still not the paper's physics.** These datasets predate the paper-defaults
flip: `subhalo_model=3`, `bias_model=0`, `kappa_anchor=0`. Their own flux is
violated (median $\langle1/\mu\rangle_I=1.006$, worst 1.39); `DatasetPDF`
re-anchors to $\langle1/\mu\rangle_I=1$, the production convention. And they
carry no $\Omega_b$, $n_s$ or $z_{\rm eq}$, so those stay inert in 6d.

## ⚠ STATUS: no forecast yet. The best provider is a stand-in.

The production 1+6d emulator has not been retrained (CLAUDE.md, "PIPELINE
REWIRED FOR THE RETRAIN"), so `pmu.EmulatorPDF` is a **stub that raises**. The
default provider is an analytic caricature with the right anatomy (empty-beam
edge, skew, $\mu^{-2}$ image-plane tail, flux-calibrated to
$\langle 1/\mu\rangle_I = 1$) and a width law loosely anchored to the repo's
measured $\sigma(\ln\mu)$. **Its width law is not the simulator's.** Any
$\sigma_8$ precision printed by this package describes the pipeline's behaviour
under a toy scatter model. Every figure carries a red banner saying so.

## Quick start

```bash
PY=/Users/baltabay/miniforge3/envs/test/bin/python   # or any python with numpy+scipy+matplotlib

$PY -m hubble_reconstruct.tests.run_gates            # 12 gates, ~35 s
$PY -m hubble_reconstruct.run_forecast --quick       # smoke run, seconds
$PY -m hubble_reconstruct.run_forecast --arm bns_et  # Vaskonen's ET arm, 4x2400
$PY -m hubble_reconstruct.run_forecast --demo-mismatch   # the §5 selection demo
HDR_GATE_NREAL=96 $PY -m hubble_reconstruct.tests.run_gates   # deep bias gates
```

Outputs (chains, catalogues, corner plot, Hubble diagram, `run_meta.json`) land
in `hubble_reconstruct/results/`.

## Layout

| file | role |
|---|---|
| `_repo.py` | glue to the parent repo; loads `ml/params.py` by path (no cwd assumptions) |
| `background.py` | flat FRW $H(z), D_C, D_L, \td V_c/\td z$; $\Omega_R=\Omega_M/(1+z_{\rm eq})$ |
| `pmu.py` | **the only plane-conversion site**; `MockPDF`, `ACEPDF`, `EmulatorPDF` (stub) |
| `selection.py` | `RedshiftCut`, `DistanceThreshold`, `SNRCut` — shared by stages 2 and 3 |
| `catalogue.py` | stage 2, both arms; draws $z\to\mu\to d_L\to$ **then** tests detection |
| `likelihood.py` | stage 3, Eqs. (10)–(12); `GaussianApproxLikelihood` control; `JointLikelihood` |
| `mcmc.py` | stage 4, Metropolis–Hastings + Gelman–Rubin, his settings as defaults |
| `plots.py` | corner (his Fig. 5), Hubble diagram (his Fig. 4), two-plane $P(\mu)$ panel |
| `run_forecast.py` | driver |
| `tests/run_gates.py` | 12 acceptance gates |

## The two conventions that must not drift

**1. Plane.** The emulator is image plane; the likelihood and the catalogue both
need source plane, $\td P_S/\td\mu = \mu^{-1}\td P_I/\td\mu$ **with
renormalization**. Implemented once, in `pmu.PDFGrid.source_plane()`. Everything
else calls that. Gate `plane` checks direction, the pure-$1/\mu$ reweight, and
idempotence; gate `flux` checks $\langle1/\mu\rangle_I=1$ separately, because a
flux error does **not** show up as a recovery bias (catalogue and likelihood
share the PDF) — it silently shifts the absolute distance scale.

**2. Selection.** Stage 2 and stage 3 take the *same* `SelectionRule` object.
Under a $\mu$-independent rule (Vaskonen's $z$-cut) $P_{\rm det}$ is a genuine
no-op — gate `pdet_noop` asserts bitwise equality. Under an SNR cut it is not,
and `p_det` enters **both** the numerator and the normalization, because
detection acts on the true lensed $d_L$ and the numerator integrand must be
restricted to detectable magnifications.

> ⚠ This differs from Vaskonen's C++, which is self-consistent under the *other*
> convention: his `DLthr` cuts on the **observed** distance, and there the
> numerator correctly carries no extra factor. Two valid schemes; mixing them is
> not. Getting this wrong was caught by `gate_selection` reporting a *larger*
> bias for the "corrected" fit than the uncorrected one.

## Gate results (2026-07-29, mock provider, sandbox)

```
[PASS] background   max rel err 1.1e-07 vs quad; low-z 4.9e-07
[PASS] plane        <mu>_I=1.0701 -> <mu>_S=1.0000; pre-renorm mass 1.0000
[PASS] grid         dlogL: res 0.001, range 0.016, znodes 0.04 nats
[PASS] pdf          tail slope -1.00, sigma8 body-width scaling 1.283 (want 1.286)
[PASS] flux         max |<1/mu>_I - 1| = 2.0e-04 over z and theta
[PASS] catalogue    z-cut fair to 0.004; SNR skews <mu> +1.61% (keep rate 8%)
[PASS] pdet_noop    identical to 0.0e+00 nats
[PASS] pdet_interp  n_pdet 32 vs 1024: dlogL 2.62e-03 nats over 300 events
[PASS] recovery     pulls Om -0.13, sigma8 -0.88, h +0.46; R-hat 1.004
[PASS] unbiased     sigma8 bias -0.0090 +- 0.0083 (-1.1 sigma), 24 realizations
[PASS] selection    matched -0.0027+-0.0098 (-0.3s), omitted +0.0205+-0.0112 (+1.8s)
[PASS] gaussian     curvature full/gauss = 0.90; shapes differ 2.77 nats after rescaling
12/12
```

**Headline, at depth (`HDR_GATE_NREAL=96`):** omitting the $\mu$-marginalized
$P_{\rm det}$ under an SNR cut biases $\sigma_8$ by
**$+0.0284 \pm 0.0054$ ($+5.3\sigma$, i.e. $+3.5\%$)**; the matched analysis is
$+0.0023 \pm 0.0048$ ($+0.5\sigma$, consistent with zero). This clears the repo's
$\ge 5\sigma$ standing rule; the 24-realization default reads only $+1.8\sigma$
and is enough to establish the sign, **not** to quote the size.

## $\sigma_8$ vs catalogue size

```bash
python -m hubble_reconstruct.run_forecast --n-scan --provider dataset
```

Writes `results/sigma8_vs_N.{png,json}`. Profile-likelihood curvature rather
than an MCMC per point, so the whole scan costs less than one chain.

| $N$ | 10 | 20 | 40 | 75 | 150 | **300** | 600 | 1200 | 2400 |
|---|---|---|---|---|---|---|---|---|---|
| $\sigma(\sigma_8)/\sigma_8$ | 47% | 35% | 24% | 16.5% | 11.7% | **8.0%** | 5.7% | 3.9% | 2.8% |

Scaling is $N^{-1/2}$ to ~1% over a 240× range (the $\times\sqrt{N/300}$ column
holds at 8.0–8.9%). **There is no extra information at large $N$ — only a bigger
assumed survey.**

⚠ **$N$ is a survey assumption, not a measurement, and the two source papers
handle it differently.** Vaskonen asserts 300 ET bright sirens. De Leo+ (2026)
decline to assert an absolute yield at all: they generate 20 000 injections,
cut on SNR > 20 and $\iota < 18°$, then draw nested sub-catalogues of $N=5$–60
from the master catalogue and *parametrise* by $N$. The real ceiling is merger
rate × observing time × the fraction with a detectable EM counterpart, and that
last factor is where the uncertainty lives. Plotting the curve puts the
assumption on an axis instead of burying it in one number.

⚠ These errors are **conditional** ($\Omega_M$, $h$ fixed). Marginalizing widens
them ~15%: the converged 4-chain MCMC at $N=300$ gives 9.7% where the scan gives
8.0%. The curve's *shape* is unaffected.

## Tuning

`f_step` (proposal width / prior range) defaults to **0.06**, tuned 2026-07-29 on
the 300-event ET arm: acceptance 43.8 / 26.8 / 18.1 / 11.2 % at
0.03 / 0.05 / 0.07 / 0.09, so 0.06 sits mid-band in Vaskonen's 10–28% target and
improves mixing (max $\hat R$ 1.105 → ~1.02). ⚠ The optimum depends on catalogue
size and parameter count — **retune for the 6d mode and the LISA arm**. The
runner prints the measured acceptance and warns when it leaves the band.

## Methodological notes worth keeping

- **A single injection-recovery pull is $N(0,1)$.** $|{\rm pull}| \sim 1$ means
  nothing. Bias gates therefore run many catalogue realizations and test the mean
  against its SEM (same lesson as CLAUDE.md §16's 2–3σ false positives).
- **Raw variance is tail-dominated.** The $\mu^{-2}$ tail has an $e$-folding
  length of 1 in $\ln\mu$ *independent of $\sigma$*, so $\sigma_8$ response must
  be measured with a robust/body width, not a standard deviation — a raw-σ test
  read 0.963 where the truth was 1.286. The repo's "clipped, not raw" rule again.
- **The Gaussian control is misspecified, not merely looser.** It shows a
  *larger* $\sigma_8$ curvature here, which is over-confidence, not information.
  Quantifying the information difference needs matched posteriors or Fisher —
  a study, not a gate.

## Open / not done

- **`EmulatorPDF` is a stub.** Wiring instructions in its docstring; needs the
  1+6d retrain plus the missing `*_6d.json` edge/tail caches.
- **`ACEPDF` plane is ASSUMED image and unverified**, and ACE spans only
  $(\Omega_M, h, w, \sigma_8)$ — a 6d run through it has three inert parameters.
  It also differs from this repo's simulator by 36–42% in $\langle\kappa^2\rangle$.
- **The LISA arm's $p_z$ is a placeholder.** Vaskonen's `zSMBHlist.dat` (EPS +
  Urrutia+25) is not in this repo — ask Ville. See `catalogue.SMBH_PLACEHOLDER`.
- **`SNRCut` is a caricature** (amplitude × inclination factor / $d_L$), not a
  GWFish Fisher forecast. Swapping in the real thing means replacing one method,
  `p_det_of_dL`.
- **Step 0 before any real forecast:** reproduce Vaskonen's 10%/30%/8% with his
  own catalogue and P(μ). Not attempted — the mock cannot do it by construction.
- Not yet run on the Mac `test` env (written and gated in the Linux sandbox on
  numpy 2.2/scipy 1.15, python 3.10). No compiled dependency, so it should be
  portable; `np.trapezoid` needs numpy ≥ 2.0.
