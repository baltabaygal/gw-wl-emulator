
---

### `docs/troubleshooting.md` (copy/paste)
```md
# Troubleshooting

## ImportError: No module named 'gwlensing'
Check module exists:
```bash
find /work -maxdepth 3 -name "gwlensing*.so" -print


PYTHONPATH=/work/build python3 -c "import gwlensing; print(gwlensing.__file__)"

if not finding pybind11:

export pybind11_DIR=$(python3 -m pybind11 --cmakedir)
./scripts/build.sh
