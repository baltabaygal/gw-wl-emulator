# GW WL Emulator – Docker Dev Workflow (Windows + PowerShell)

This project contains a C++ weak-lensing engine and a Python module (`gwlensing`) built with pybind11.
Docker is used to provide a consistent Linux build environment (compiler, GSL, Python deps).

## Key Concepts

- **Docker image**: a reusable “blueprint” that contains all dependencies and the compiled tools.
- **Docker container**: a running instance of the image.
- **Volume mount (`-v`)**: maps your local project folder into the container as `/work`.
  - This means edits on your laptop appear instantly inside Docker.
  - Builds and generated files created inside Docker are saved on your laptop.

## Main Commands (PowerShell)

All commands are run from the repo root using:

```powershell
.\scripts\run_make.ps1 <target>



---

### `docs/dev_workflow.md` (copy/paste)
```md
# Dev workflow

There are two modes:

## A) Reproducible build (recommended for “it just works”)
Use the Dockerfile build:

```bash
docker build -t gw-wl-emulator:py -f docker/Dockerfile .
docker run --rm -it gw-wl-emulator:py
