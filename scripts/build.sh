#!/usr/bin/env bash
set -euo pipefail

# Auto-detect pybind11 CMake dir if not set
if [ -z "${pybind11_DIR:-}" ]; then
  export pybind11_DIR="$(python3 -m pybind11 --cmakedir)"
fi

rm -rf ./build
mkdir -p ./build
cd ./build

cmake ../cpp -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=/Users/baltabay/miniforge3/envs/test/bin/python
cmake --build . -j
echo "Build completed."