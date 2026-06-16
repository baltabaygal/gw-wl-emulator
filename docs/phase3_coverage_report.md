# Phase 3 Coverage Report

Coverage execution was deferred.

Reason: the simulator-vs-NSF smoke posterior comparison failed on the central mixed catalog. Running 20 NSF-only catalogs would test calibration of the NSF with respect to its own likelihood, but it would not answer the required simulator-replacement question while the simulator reference posterior disagrees with NSF in the smoke gate.

Next coverage run after Phase 3B smoke repair:

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=build:. conda run -n test python ml/run_phase3_coverage.py --n_catalogs 20 --n_events 1000 --grid_resolution 12
```
