# Runtime of the paper-default config (2026-07-30)

Machine: **Apple M5**, 10 cores (4 performance). Vaskonen's quoted ~20 s is on an
**M1**, so a like-for-like statement needs the hardware factor (~1.5-2x single-core).
Scripts: `scripts/convergence/paper_config_timing.py` (two-point fit),
`build/main_vaskonen_timing` (upstream `Plnmuf` path).

## 1. What Vaskonen's "~20 s" is

His `main_lensing.cpp` has two regimes: `Nreal = 4e5` for the Fig-2 PDF and
`Nreal = 1e4` for the likelihood/catalogue scan. 4e5 realizations cannot be 20 s --
**measured 114 s on an M5 with subhalos OFF** (`main_vaskonen_timing fig2_single`),
and the M5 is the faster machine. The number that matches is **one likelihood
evaluation = 6 source redshifts x 1e4 realizations**:

| arm | this unit, M5 | implied on M1 (1.5-2x) |
|---|---|---|
| legacy (Vaskonen-like) | **14.6 s** | **22-29 s** <- matches his ~20 s |
| paper (draft defaults) | **49.9 s** | 75-100 s |

So **the paper config is 3.4x Vaskonen's cost** on the same unit and machine.
⚠ This identification is inferred from his code's `Nreal` values, not from his text.
Confirm against the paper before quoting it as his definition.

## 2. Cost model

`t(N) = t_fixed(z_s, cosmology) + N * t_ray`. `t_fixed` is the per-config table build
(sigma(M)/HMF grids, `r_thr` reach tables, model-5 restricted-intensity bins, bias
field lnT/lnV tables). Quoting a single number at small N is how the model-4 cost was
misreported ~100x in July -- a 4.5 s precompute swamped the per-ray work at N=300.

| | t_fixed [s] | t_ray [ms] @ z_s = 0.5 / 1 / 5 / 10 |
|---|---|---|
| paper | 3.2-3.3 | 0.365 / 0.231 / 0.178 / 0.175 |
| legacy | 1.4-1.5 | 0.102 / 0.114 / 0.145 / 0.162 |

## 3. ⚠ The dominant finding: cost EXPLODES at low z_s

Per-ray cost, paper config, and it runs **opposite** to legacy:

| z_s | 0.03 | 0.10 | 0.30 | 1.0 | 3.0 | 10.0 |
|---|---|---|---|---|---|---|
| paper ms/ray | **2.819** | 0.951 | 0.659 | 0.269 | 0.230 | 0.181 |
| legacy ms/ray | 0.046 | 0.098 | 0.075 | 0.115 | 0.134 | 0.154 |
| ratio | **61x** | 9.7x | 8.8x | 2.3x | 1.7x | 1.2x |

Mechanism: the threshold rule fixes `<N> = 100` explicit halos, so at low z_s -- a
short comoving path -- `kappa_thr` must fall to find 100 halos. `subhalo_kappathr` is
`0.1 * kappa_thr`, so the clump threshold is dragged down with it and the rendered
clump count per host explodes. Confirmed independently: the `likelihood6` run spent
**21.1 s at z_s=0.05** vs 5.2-6.9 s at every other redshift.

Consequences:
- Datasets sample **log-uniformly from z_min = 0.01**, so the cheap high-z decade is
  a third of the configs and the expensive low-z decade is another third. The
  log-uniform mean is **0.73 ms/ray** (paper) vs 0.104 (legacy) = ~7x.
- z_s < 0.03 is below the two-point fit's resolution (the fit returned a negative
  slope), so the very lowest decade is **unquantified** -- it could be worse. Measure
  before committing to `z_min = 0.01`.
- Cheap lever if needed: raise `z_min`, or cap the subhalo threshold below some z_s
  (e.g. an absolute `subhalo_kappathr` floor instead of the pure 0.1x factor). NOT
  recommended without a JSD gate -- it changes the physics at low z_s.

## 4. Retrain data budget

Process-level parallelism gives **~4.4x at 8 processes** (measured 2026-07-10; 8 beats
10 on a 10-core box). `subhalo_threads` is intra-host only and is 2-3x SLOWER if
forced on.

| item | rays | 1 core | 8 proc |
|---|---|---|---|
| main dataset (1000 pts -> ~1500 cfg x 1e4) | 1.5e7 | 4.4 h | 1.00 h |
| `lowz_aug` (150 cfg x 4e4, z 0.3-2.5) | 6e6 | 0.8 h | 0.17 h |
| `gen_tail_counts` (30 cfg x 1.5e6) | 4.5e7 | 9.1 h | **2.08 h** |
| `param_space_check` (15 cfg x 4e5) | 6e6 | 1.2 h | 0.28 h |
| `validate_pit` (15 cfg x 2e5) | 3e6 | 0.6 h | 0.14 h |

**Total ~3.7 h wall-clock at 8 processes**, dominated by the tail counts. Excludes
`prepare_fix` (edge) and `groundtruth.npz`, whose config counts were not enumerated.
⚠ `generate_dataset.py --num_points N` draws `4N` LHS points and filters into
train/val/test, yielding ~1.5N usable configs -- budget on that, not on N.

## 5. Single-PDF figures for the draft

At the paper defaults, one `dP/dmu`:

| rays | z_s = 0.5 | z_s = 1 | z_s = 5 |
|---|---|---|---|
| 4e5 (Vaskonen Fig 2) | 2.5 min | 1.6 min | 1.2 min |
| 1.6e6 (draft Fig magpdf) | 9.8 min | 6.2 min | 4.8 min |

The draft's Fig. `magpdf` quotes 1.6e6 realizations per source redshift; at ~5-10 min
each that figure is minutes, not hours, per redshift.
