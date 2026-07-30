# Magnification-PDF figures — run guide

Rewritten 2026-07-28. Generates the paper's magnification-PDF figures:

| output (in `paper_prod/plots/figures/`) | slot | what it shows |
|---|---|---|
| `fig_magnification_pdf_zs.{pdf,png}` | `fig:magpdf_zs`, Sec. II.B | body of `dP/dmu` vs `mu`, linear `mu`, `z_s = 0.2 … 10` at the production config (the Vaskonen 2026 Fig. 2 analog) |
| `fig_magnification_pdf_tail.{pdf,png}` | `fig:magpdf_tail`, App. `edge_tail` | **compensated** tail `mu^2 dP/dmu` vs `mu`, `z_s = 2,3,5,7,10`, flat-if-`mu^-2` |
| `fig_magnification_pdf_ingredients.{pdf,png}` | — | baseline / `+`subhalos / `+`clustering at one `z_s`. ⚠ **Not currently a paper figure**: the ingredient decomposition moved to `fig:variance_DL` (2026-07-28), and the arms are not one-variable-at-a-time (see §5). Kept as a diagnostic. |

Script: `paper_prod/scripts/plot_fig_magnification_pdf.py`.
The Monte Carlo is the expensive part. **Smoothness is bought with realizations** —
push `--nreal` up and/or shard over seeds and combine.

---

## 0. Environment

Use the **`test` conda env (Python 3.12)** — the C++ module is built for cpython-3.12.
The system `python3` (3.13) will not import `gwlensing`.

```bash
PY=/Users/baltabay/miniforge3/envs/test/bin/python
```

Run everything from the repo root (`/Users/baltabay/Desktop/gw-wl-emulator`).

⚠ **Requires a rebuild first.** The production config is imported from
`ml/params.py::PRODUCTION_CONFIG` (hash `0d50caf91c75`) and uses
`subhalo_model=5` + `subhalo_virial=True`. `make build` must postdate the
2026-07-28 `halobias` and subhalo-radial-profile changes, or the figures will
show pre-fix physics.

---

## 1. Quick smoke test (seconds–minutes, noisy)

```bash
$PY paper_prod/scripts/plot_fig_magnification_pdf.py --nreal 50000 --tag test
```

The tail will be jagged at 50k — expected. If this renders, scale up.

---

## 2. The production run

`--nreal` defaults to **4e6 per series** (decision 2026-07-28). Shard it:

```bash
for s in 1 2 3 4 5 6 7 8; do \
  $PY paper_prod/scripts/plot_fig_magnification_pdf.py \
      --nreal 500000 --seed $s --tag shard$s --no-plot & \
done; wait

$PY paper_prod/scripts/plot_fig_magnification_pdf.py \
    --combine "paper_prod/plots/data/magpdf_shard*.npz" --tag combined
```

Histograms add exactly across seeds, so shards combine without approximation.
There are 10 series (8 `z_s` + `base`/`subh` at `z_s=5`; the `z_s=5` full curve
is reused).

Why 4e6: it puts the worst compensated-tail bin at 5–7 % for `z_s >= 5` and
11–17 % at `z_s = 2–3`, against the factor-12.5 effect that distinguishes
`mu^-2` from `mu^-3` over a decade. At the old 4.8e5 those bins were 13–33 %.

> Do **not** parallelize a single `sample_lnmu` call with `subhalo_threads` —
> that is intra-host only and slower at production configs. Parallelize over
> seeds, as above.

---

## 3. Re-plot without re-running the MC

```bash
$PY paper_prod/scripts/plot_fig_magnification_pdf.py --replot --tag combined
```

Display knobs (defaults are the settled paper values):

- `--mu-range 0.5 2.5` — body x-window. Set from the data: `dP/dmu` stays above
  the y-floor out to `mu = 2.78` at `z_s = 10`.
- `--ylim 2e-2 110` — body y-range; the `z_s=0.2` peak reaches ~50–100.
- `--body-zs 0.2 0.5 1 2 5 10` — which of `ZS_LIST` to **draw**. All of
  `ZS_LIST` is still simulated; `3` and `7` are omitted from the body for
  legibility but are used by the tail figure.
- `--rebin 0` — `0` means choose the factor **per curve** from its own
  `sigma(lnmu)`. Necessary because `sigma` runs 0.009 (`z_s=0.2`) to 0.28
  (`z_s=10`): a single factor either shreds the wide curves or collapses the
  narrow ones onto ~4 bins, which reads as a truncated spike.
- `--min-counts 8` — also the target for **wing binning**: display bins are
  merged outward from the core until each holds this many rays, so the curve
  follows the density to the edge of support instead of being clipped where
  fine bins run thin.
- `--edge-q 0` — off. `0.001` restores the low-`mu` edge ticks.
- `--logx` — log-log body panel. **Not recommended**; see §5.
- `--tail-zs`, `--tail-mu 2 100`, `--tail-nbins 7`, `--tail-fit-mu 8`,
  `--tail-mu3` — compensated tail figure.
- `--smooth W` — Savitzky–Golay on `log10(dP/dmu)`. Prefer more `--nreal`.

---

## 4. What is actually being simulated

Fiducial cosmology = Planck 2018 (`OmegaM=0.315, sigma8=0.811, h=0.674`).

| arm | config |
|---|---|
| **full** | `ml/params.py::PRODUCTION_CONFIG`, imported, not restated |
| **baseline** | legacy iid clustering (`bias_model=0`), no subhalos |
| **+ subhalos** | baseline + the `subhalo*` keys of `PRODUCTION_CONFIG` |

Before 2026-07-28 the full arm hard-coded `subhalo_model=4` and omitted
`subhalo_virial` and `subhalo_kappathr_factor`, i.e. it plotted a different
subhalo population from the one the draft describes. Importing
`PRODUCTION_CONFIG` is what prevents that recurring — do not paste the flags
back in.

---

## 5. Honest caveats (read before quoting anything off these plots)

- **The body and the tail cannot share a panel.** The 1–99 % mass spans 0.07
  decades at `z_s=0.2` and 0.61 at `z_s=10`, against the ~2 decades a tail panel
  needs. That is why there are two figures and why `--logx` is not the default:
  on one log-log panel the low-`z_s` bodies degenerate into vertical spikes.
- **The tail is unsampled below `z_s ~ 2`,** and no realization count fixes it:
  `S(mu>5) = 2e-6` at `z_s=0.5`, so `mu>8` holds ~1 ray in 4.8e5 and ~20 in 1e7.
  The tail figure omits `z_s <= 1` and says so in its caption.
- The `mu^-2` **exponent** is verified (Hill 1.95–2.03; Poisson likelihood
  favours it over `mu^-3` by 1400–6800 nats above `mu=8`). The **absolute
  normalization** of the far tail is not certified in this method class.
- The compensated estimator was validated on exact samples: flat to 0.8 % on a
  true `mu^-2` draw, slope `-1.01` on a true `mu^-3` draw.
- These are **raw simulator** PDFs, not the emulated `dP/dmu`.
- ⚠ **The ingredient arms are not one-variable-at-a-time.** `CONFIG_SUBH` uses
  `bias_model=0` while `CONFIG_FULL` uses the correlated field, so the two
  differ in more than one ingredient and the effects do not add — at `z_s=5`
  the decomposition currently reads as clustering *reducing* the scatter. Fix
  the arm definitions before using `fig_magnification_pdf_ingredients` or
  `fig:variance_DL` for anything quantitative.

---

## 6. Cache format and compatibility

`paper_prod/plots/data/magpdf_<tag>.npz` holds per-series bin counts, the total
ray count, and under/overflow counts, on `STORE_EDGES` = **4000 log-spaced bins
over `mu` in [0.05, 200]** (`dlnmu = 2.1e-3`).

⚠ **Pre-2026-07-28 caches are not reusable.** They carry 2000 *linear* bins over
`mu` in [0.4, 5] — no tail window, no under/overflow — and were run with the old
subhalo config and pre-fix physics. `--combine` refuses to mix grids; `--replot`
on an old cache works (plotting uses the cache's own edges) but the tail figure
auto-skips with a warning.

---

## 7. Outputs

- Figures: `paper_prod/plots/figures/fig_magnification_pdf_{zs,tail,ingredients}.{pdf,png}`
- Cache:   `paper_prod/plots/data/magpdf_<tag>.npz`

The `\begin{figure}` blocks are wired into `draft_revised_2026-07-20.tex`:
`fig:magpdf_zs` in Sec. II.B and `fig:magpdf_tail` in App. `edge_tail`, both
referencing `plots/fig_magnification_pdf_*.pdf`. Copy the generated PDFs into
the Overleaf `plots/` directory alongside the other figures.
