# docker/ — Reproducible build environment

`Dockerfile` for the containerized C++ build/run environment used to reproduce the
Vaskonen baseline. See `docs/reference/docker.md` for usage and the root `README.MD` quick-start.
Typical flow (from repo root): build the image, enter a dev shell, then
`./scripts/build/build.sh` inside the container.
