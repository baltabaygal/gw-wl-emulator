#!/usr/bin/env bash
set -euo pipefail

# If pybind11_DIR not set, auto-detect from python package
if [ -z "${pybind11_DIR:-}" ]; then
  export pybind11_DIR="$(python3 -m pybind11 --cmakedir)"
fi

rm -rf /work/build
mkdir -p /work/build
cd /work/build
cmake ../cpp -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
