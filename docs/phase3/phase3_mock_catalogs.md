# Phase 3 Mock Catalogs

`ml/generate_mock_catalogs.py` creates reproducible mock weak-lensing / bright-standard-siren catalogs under `data/mock_catalogs/phase3/`.

Each catalog stores `z`, `lnmu`, and JSON metadata containing `catalog_id`, catalog type, redshift distribution, `N`, seed, true cosmology, simulator git commit, generation timestamp, and `catalog_hash`.

Smoke run:

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=build:. conda run -n test python ml/generate_mock_catalogs.py --smoke
```

