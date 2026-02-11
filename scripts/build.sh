#!/usr/bin/env bash
set -euo pipefail

mkdir -p /work/build /work/dist
cmake -S /work/cpp -B /work/build -DCMAKE_BUILD_TYPE=Release
cmake --build /work/build --config Release -j "$(nproc)"
cp /work/build/main_leinsing /work/dist/