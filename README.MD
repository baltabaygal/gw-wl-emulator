# GW Weak Lensing Emulator

Fast ML emulator for the weak gravitational lensing magnification PDF of
gravitational waves, based on the C++ Monte Carlo engine from
[Vaskonen (2026), arXiv:2601.06023](papers/core/Vaskonen_2026_GWWeakLensingSigma8.pdf).

The emulator reproduces dP/d(ln μ | z, σ₈, Ω_M, h) orders of magnitude faster
than the C++ sampler, enabling efficient MCMC inference of σ₈ from bright
standard sirens detected by ET and LISA.

**Current status**: repo structure initialized; C++ baseline operational.
ML integration begins next.

---

## Structure

```
gw-wl-emulator/
├── cpp/              # Original C++ lensing engine (CMake + Docker)
├── python/           # Initial pybind11 bindings and plotting
├── papers/           # PDFs, notes, and paper index
│   ├── core/         # Primary reference papers
│   ├── methods/      # ML and statistics methods papers
│   ├── context/      # Background cosmology papers
│   └── misc/         # Unsorted
├── research/         # Experiments and exploratory ML work
│   └── shared/       # Shared helpers (data loaders, metrics)
├── src/              # Stable, reusable production modules (promoted)
├── data/
│   ├── raw/          # Immutable C++ output (lnmu samples, PDFs)
│   ├── processed/    # Training/test sets
│   └── models/       # Saved checkpoints
├── plots/
│   ├── style/        # Shared plot style
│   └── figures/      # Final publication figures
├── scripts/          # Reproducible build and run workflows
└── docs/             # Architecture, workflows, rulebook
```

---

## Quick Start (C++ engine)

```bash
# Build Docker image (once)
make image

# Enter dev shell
make shell

# Inside container: build and run
./scripts/build.sh
mkdir -p dataL
./build/main_lensing 0 1 2 0.05
```

See [docs/workflows.md](docs/workflows.md) for all workflows.

---

## Roadmap

**Phase 0 (done)**: Reproducible C++ baseline in Docker.

**Phase 1**: Expose raw ln(μ) samples via pybind11; build training dataset
covering a grid of (z, σ₈, Ω_M, h).

**Phase 2**: Train conditional density estimator (Neural Spline Flow);
validate against C++ PDFs.

**Phase 3**: Plug emulator into MCMC inference; benchmark σ₈ recovery on
mock ET/LISA catalogues.

---

## Core Rules

- `research/` is for experiments; `src/` is for promoted production code.
- `src/` must not import from `research/`.
- Raw data is immutable — write to `data/raw/`, never edit in place.
- All reproducible commands live in `scripts/`.

See [AGENTS.md](AGENTS.md) and [docs/rulebook.md](docs/rulebook.md) for
operating rules.
