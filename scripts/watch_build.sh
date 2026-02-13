#!/usr/bin/env bash
set -euo pipefail

INTERVAL="${1:-1}"  # seconds

hash_tree() {
  # Hash filenames + mtimes + sizes to detect changes quickly
  find /work/cpp -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.hpp" -o -name "CMakeLists.txt" \) -print0 \
    | sort -z \
    | xargs -0 stat -c "%n %Y %s" \
    | sha256sum \
    | awk '{print $1}'
}

echo "[watch_build] Watching /work/cpp for changes (poll=${INTERVAL}s)"
LAST="$(hash_tree)"
echo "[watch_build] Initial hash: $LAST"
echo "[watch_build] Building once..."
/work/scripts/build.sh
echo "[watch_build] Ready."

while true; do
  sleep "$INTERVAL"
  NOW="$(hash_tree)"
  if [ "$NOW" != "$LAST" ]; then
    echo
    echo "[watch_build] Change detected -> rebuilding..."
    LAST="$NOW"
    if /work/scripts/build.sh; then
      echo "[watch_build] Build OK."
    else
      echo "[watch_build] Build FAILED (fix code; will retry on next change)."
    fi
  fi
done
