"""Gate: the PRODUCTION C++ BiasField1D covariance vs the numpy replica, per
bias_window (0 disk, 1 spherical top-hat, 2 Gaussian).

validate_field_covariance.py's layer 1 compares two python integrals against
each other — it validates the derivation. This script validates the SHIPPED
C++: it runs build/bias_window_field_probe (which #includes cpp/lensing.cpp,
so its BiasField1D is production's) and compares sig2 and the full shell
covariance against cpp_field().

Build the probe first (repo root):
  c++ -std=c++17 -O2 -Icpp -I/opt/homebrew/include \
    playground/bias_window_field_probe.cpp cpp/cosmology.cpp cpp/basics.cpp \
    cpp/subhalo.cpp -L/opt/homebrew/lib -lgsl -lgslcblas \
    -o build/bias_window_field_probe

Run: python3 playground/bias_field/check_cpp_vs_replica.py
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "convergence"))
sys.path.insert(0, str(REPO / "playground" / "bias_field"))

from bias_field_prototype import Cosmo                     # noqa: E402
from validate_field_covariance import cpp_field, WINDOW_NAME  # noqa: E402

PROBE = REPO / "build" / "bias_window_field_probe"
# gate: the C++ and the replica differ only by table-interpolation-level noise
# (same algorithm, different summation order + the Cholesky jitter)
GATE_COV = 5e-6      # max |dCov| / max(diag)
GATE_SIG = 5e-6      # max |sigma_cpp/sigma_py - 1|


def run_probe(zs, Rperp, window):
    out = subprocess.run([str(PROBE), repr(zs), repr(Rperp), str(window)],
                         capture_output=True, text=True, cwd=REPO, check=True).stdout
    lines = out.strip().split("\n")
    hdr = dict(kv.split("=") for kv in lines[0].lstrip("# ").split())
    n = int(hdr["n"])
    sig2 = np.array([float(x) for x in lines[1:1 + n]])
    cov = np.array([[float(x) for x in r.split()] for r in lines[1 + n:1 + 2 * n]])
    return sig2, cov, int(hdr["Nmax"])


def main():
    if not PROBE.exists():
        sys.exit(f"probe not built: {PROBE} (see the docstring for the build line)")
    C = Cosmo(Nz=100)
    ok = True
    for zs in (1.0, 5.0):
        for Rperp in (3000.0, 8441.0, 20000.0):
            for window in (0, 1, 2):
                s2c, covc, nmax = run_probe(zs, Rperp, window)
                f = cpp_field(C, zs, Rperp, window)
                scale = float(np.max(np.diag(f["Cov"])))
                dcov = float(np.max(np.abs(covc - f["Cov"]))) / scale
                dsig = float(np.max(np.abs(np.sqrt(s2c / f["sig2"]) - 1.0)))
                good = dcov < GATE_COV and dsig < GATE_SIG
                ok &= good
                print(f"zs={zs:g} win={WINDOW_NAME[window]:6s} Rperp={Rperp:8.4g} "
                      f"Nmax={nmax:>7d}  |dCov|/diag={dcov:.2e}  dsig={dsig:.2e}  "
                      f"sig_max(cpp)={np.sqrt(s2c.max()):.4f}  [{'ok' if good else 'FAIL'}]")
    print("CPP-vs-REPLICA " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
