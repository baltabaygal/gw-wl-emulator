# cpp/ — C++ Monte Carlo lensing engine

The original C++ weak-lensing engine from Vaskonen (2026), plus the pybind11 bindings
that expose it to Python as the `gwlensing` module. This is the ground-truth sampler the
ML emulator is trained against.

Build with `make build` (runs `scripts/build.sh`, CMake + pybind11). Rebuild against the
Python 3.12 interpreter in the `test` conda env so the `.so` matches — see the root
`CLAUDE.md` for environment details.

## Key files

| File | Role |
|------|------|
| `cosmology.cpp` / `.h` | Halo mass function, NFW profiles, growth factor, P(k) normalization (σ₈ and A_s modes) |
| `lensing.cpp` / `.h` | Ray-tracing / convergence sampling; produces ln(μ) realizations |
| `subhalo.cpp` / `.h` | Substructure gate: subhalo populations and the resolved/unresolved split |
| `basics.cpp` / `.h` | Shared numerics and utilities |
| `lnmu_wrapper.cpp` / `.h` | Thin wrapper around the sampler used by the bindings |
| `python_bindings.cpp` | pybind11 module definition (`gwlensing` public API) |
| `invalid_stats.h`, `test_stats.cpp` | Diagnostics and a small C++ test |

## Entry points (`main_*.cpp`)

`main_lensing.cpp` (primary CLI), `main_sample_lnmu.cpp`, `main_subhalo_profile.cpp`,
`main_vaskonen_timing.cpp`.

## Related

The clean production port of the subhalo model lives in the separate `~/Desktop/halos`
fork (see `CLAUDE.md` and `docs/subhalo/subhalo_combining.md`). Internal units are kpc, M⊙, Gyr.
