# Magnification-PDF figures — run guide

Generates the two `dP/dmu` figures for **Sec. II.B (Magnification PDF)**:

| output (in `paper_prod/plots/figures/`) | what it shows |
|---|---|
| `fig_magnification_pdf_zs.{pdf,png}` | `dP/dmu` vs `mu` for `z_s = 0.5, 1, 2, 5, 10` at the full production config (Vaskonen 2026 Fig. 2 analog) |
| `fig_magnification_pdf_ingredients.{pdf,png}` | `dP/dmu` at fixed `z_s = 5` for baseline / `+`subhalos / `+`clustering (what the paper's two new ingredients do) |

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

Requirement: the `gwlensing` module must be the **post-2026-07-23 Mac build**
(`make build`) that includes `subhalo_model=4` + the mass-conserving carve. If
`sample_lnmu(..., subhalo_model=4)` throws, rebuild first with `make build`.

Run everything from the repo root (`/Users/baltabay/Desktop/gw-wl-emulator`).

---

## 1. Quick smoke test (seconds–minutes, noisy)

Confirms the configs run and the plots render before you commit to a long job.

```bash
$PY paper_prod/scripts/plot_fig_magnification_pdf.py --nreal 50000 --tag test
```

The curves will be jagged in the tail at 50k — that is expected. If this works,
scale up.

---

## 2. The smooth run

### Option A — one big single-process run (simplest)

```bash
$PY paper_prod/scripts/plot_fig_magnification_pdf.py --nreal 3000000 --tag run1
```

`3e6` realizations per series gives a smooth body and a clean `mu^-2` tail out to
`mu ~ 1.8`. Go higher (`5e6`–`1e7`) if you want the tail smoother. There are 7
series total (5 `z_s` for figure 1 + `base`/`subh` for figure 2; the `z_s=5` full
curve is reused), so wall-time is `7 x (per-series MC)`.

### Option B — shard over seeds, then combine (recommended for speed)

Histograms just add, so independent seeds combine exactly. Run 8 shards in
parallel (process-level parallelism is the efficient lever — see CLAUDE.md item 11),
then sum them.

```bash
for s in 1 2 3 4 5 6 7 8; do \
  $PY paper_prod/scripts/plot_fig_magnification_pdf.py \
      --nreal 1000000 --seed $s --tag shard$s --no-plot & \
done; wait

$PY paper_prod/scripts/plot_fig_magnification_pdf.py \
    --combine "paper_prod/plots/data/magpdf_shard*.npz" --tag combined
```

That is `8e6` realizations per series total, produced ~`min(8, ncores)x` faster
than Option A. The `--combine` call writes the merged cache and renders the
figures. Adjust the number of shards / `--nreal` to your core count and patience.

> Do **not** try to parallelize a single `sample_lnmu` call with `subhalo_threads`
> — that is intra-host only and is slower at production configs. Parallelize over
> seeds (separate processes), as above.

---

## 3. Re-plot / restyle without re-running the MC

Every MC run caches tiny histograms to `paper_prod/plots/data/magpdf_<tag>.npz`,
so tweaking axis ranges, colors, or smoothing is instant:

```bash
$PY paper_prod/scripts/plot_fig_magnification_pdf.py --replot --tag run1 \
    --mu-range 0.6 1.8 --ylim 1e-2 3e1 --rebin 8
```

Useful display knobs (all optional):

- `--mu-range LO HI`  x-axis window (default `0.6 1.8`, matches Vaskonen Fig. 2).
- `--ylim LO HI`      y-axis (log) range (default `1e-2 3e1`).
- `--rebin N`         merge `N` stored fine bins per display bin (default `8`,
  i.e. ~250 display bins from the 2000 stored). Larger `N` = smoother, coarser.
- `--smooth W`        odd Savitzky–Golay window on `log10(dP/dmu)` for display
  (default `0` = off). **Prefer more `--nreal` over cosmetic smoothing.**
- `--zs-ingredients Z` fixed `z_s` for figure 2 (default `5`).

---

## 4. What is actually being simulated

Fiducial cosmology = Planck 2018 (`OmegaM=0.315, sigma8=0.811, h=0.674`), the
Vaskonen 2026 benchmark.

| curve | config |
|---|---|
| **baseline** (Vaskonen 2026) | legacy iid clustering (`bias_model=0`), no subhalos |
| **+ subhalos** | baseline + `subhalo_model=4` (every subhalo to `m_floor=1e7`, host carved) |
| **+ clustering (full)** | correlated field `bias_model=1`, `bias_window=1` (top-hat), `bias_Rperp=20000` (R_s=20 Mpc), `bias_weak=True`, `fil_bias=True`, + subhalos |

Figure 1 uses the **full** config at every `z_s`. Figure 2 stacks the three
configs at one `z_s`.

Flux anchor is `kappa_anchor=1` (robust `<kappa>=0` over rays with `kappa<=1`),
which avoids the monster-ray batch-mean shift. This is the physically clean choice
for the simulator PDF.

---

## 5. Honest caveats (read before quoting anything off these plots)

- The plotted window stops at `mu ~ 1.8`. That is deliberate. The **far tail**
  (`mu` well above ~2, `q>99.9` at `z_s>=5`) is not certified in this method class
  (needs N-body) — do not extend the x-range and read numbers off the extreme tail.
  The `mu^-2` slope shown in the plotted range **is** verified (Hill fit 1.95–2.03).
- These are **raw simulator** PDFs, not the emulated `dP/dmu`. An emulator-vs-simulator
  overlay belongs in Sec. III, not here.
- If you change `z_s` for figure 2 to a value not in `{0.5,1,2,5,10}`, the script
  runs an extra `full` MC at that `z_s` (one more series).

---

## 6. Outputs

- Figures: `paper_prod/plots/figures/fig_magnification_pdf_{zs,ingredients}.{pdf,png}`
- Cache:   `paper_prod/plots/data/magpdf_<tag>.npz` (histograms + counts; a few kB)

The `\begin{figure}` blocks are already wired into Sec. II.B of
`draft_revised_2026-07-20.tex`, referencing `plots/fig_magnification_pdf_zs.pdf`
and `plots/fig_magnification_pdf_ingredients.pdf`. Copy the generated PDFs into the
Overleaf `plots/` directory (same as the other figures).
