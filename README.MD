# GW Weak Lensing Emulator

Fast ML emulator for the weak gravitational lensing magnification PDF of
gravitational waves, built on the C++ Monte Carlo engine from
[Vaskonen (2026), arXiv:2601.06023](papers/core/Vaskonen_2026_GWWeakLensingSigma8.pdf).

The emulator reproduces dP/d(ln μ | z, σ₈, Ω_M, h) orders of magnitude faster than the
C++ sampler, enabling efficient MCMC inference of σ₈ from bright standard sirens detected
by ET and LISA. The parameter space has since been extended to a 1+6d cosmology
(Θ = {h, z_eq, Ω_M, Ω_B, A_s, n_s}).

**Status:** C++ baseline operational; NSF emulator trained and validated (production model
in the `gw-wl-emulator-ar` worktree); subhalo substructure gate complete. See
[`CLAUDE.md`](CLAUDE.md) for the detailed, up-to-date development log.

---

## Structure

Each folder has its own `README.md` with details — this is the map.

```
gw-wl-emulator/
├── cpp/           # C++ Monte Carlo lensing engine + pybind11 bindings (gwlensing)
├── python/        # Dataset generation, validation, phase-1 plotting
├── ml/            # Emulator: models, training, evaluation, posteriors
├── scripts/       # Workflows: build/ · subhalo_gate/ · figures/ · comparisons/
├── tests/         # Pytest suite (make pytest)
├── datasets/      # Training/val/test HDF5 datasets  (local, gitignored)
├── data/          # Derived numeric data, model checkpoints, results, mock catalogs
│   ├── raw/       #   immutable C++ sampler output
│   ├── models/    #   saved checkpoints
│   ├── cache/     #   regenerable intermediates (gitignored)
│   └── results/   #   benchmark / analysis outputs
├── papers/        # Reference PDFs + notes (core / methods / context / misc)
├── docs/          # Reports & guides: reference/ · backend_current/ · phase2/ · phase3/ · subhalo/
├── plots/         # Generated figures (regenerable)
├── paper_prod/    # Curated, provenance-tracked figures for paper submission
├── docker/        # Reproducible build environment (Dockerfile)
├── notebooks/     # Jupyter exploration
├── plans/         # Planning / design notes
├── research/      # Exploratory scratch (archive candidate)
├── ace_lensing/   # External ACE fork (own git repo; gitignored here)
├── build/         # CMake build output (gitignored)
├── dataL/         # C++ run output (gitignored)
├── tmp/           # Throwaway scratch — nothing you want to keep
└── playground/    # Subhalo-factor experiment scratch (analytic/ · dgate/ · paired/ · …)
```

---

## Environment

Use the **`test` conda env (Python 3.12)** for anything touching the C++ module or ACE —
the system `python3` is 3.13 and will not load the compiled `gwlensing` module.

```bash
source /Users/baltabay/miniforge3/bin/activate test
# or: PY=/Users/baltabay/miniforge3/envs/test/bin/python
```

See [`CLAUDE.md`](CLAUDE.md) for import gotchas, the C++ public API, and the 1+6d
parameterization.

---

## Quick Start

```bash
make build                   # build the C++ + pybind module
make test                    # smoke-test the API (python/testing_api.py)
make pytest                  # run the test suite
make generate-small-dataset  # generate a tiny dataset
```

Docker-based reproducible build: see [`docs/reference/docker.md`](docs/reference/docker.md).

---

## Roadmap

**Phase 0 (done):** Reproducible C++ baseline.
**Phase 1 (done):** pybind11 bindings; training dataset over (z, σ₈, Ω_M, h), later
extended to 1+6d.
**Phase 2 (done):** Conditional Neural Spline Flow density estimator, validated against
C++ PDFs.
**Phase 3 (done):** Emulator in MCMC inference; σ₈ recovery benchmarked on mock ET/LISA
catalogs. Subhalo substructure gate complete.

---

## Conventions

- Raw sampler data is immutable — write to `data/raw/`, never edit in place.
- Reproducible commands live in `scripts/` and the `Makefile`.
- Plots → `plots/`; throwaway work → `tmp/`.
- Substructure reference numbers come from the PDFs in `papers/misc/`, not from memory.

`CLAUDE.md` is the living development document — the authoritative record of the current
state, environment, and design decisions.
