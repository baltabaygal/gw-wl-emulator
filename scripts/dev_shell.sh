#!/usr/bin/env bash

IMAGE=gw-wl-emulator:py

# Build image if it doesn't exist
if [[ -z "$(docker images -q $IMAGE 2> /dev/null)" ]]; then
  echo "Docker image not found. Building..."
  docker build -t $IMAGE -f docker/Dockerfile .
fi

echo "Starting development shell..."
docker run --rm -it \
  -v "$(pwd):/work" \
  -w /work \
  $IMAGE \
  bash
