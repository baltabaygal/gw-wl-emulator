#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"

# Auto-detect pybind11 CMake dir if not set
if [ -z "${pybind11_DIR:-}" ]; then
  export pybind11_DIR="$(python3 -m pybind11 --cmakedir)"
fi

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake "$REPO_ROOT/cpp" -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
echo "Build completed."
