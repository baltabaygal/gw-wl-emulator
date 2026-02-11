# Fast ML Emulator for GW Weak Lensing PDFs

This project aims to build a fast machine-learning emulator for weak gravitational lensing of gravitational waves, based on an existing C++ Monte-Carlo implementation from a published paper.

**Current status:** Will be updated! 
We are in the **baseline C++ replication phase**.  
No ML components have been added yet.

---

# Project Structure

```
gw-wl-emulator/
│
├── cpp/              # Original C++ lensing simulation code
│   ├── basics.*
│   ├── cosmology.*
│   ├── lensing.*
│   └── main_lensing.cpp
│
├── docker/           # Docker build configuration
│   └── Dockerfile
│
├── scripts/          # Helper scripts
│   └── build.sh
│
├── build/            # CMake build directory (generated)
├── dataL/            # Lensing simulation outputs (generated)
└── README.md
```

---

# Requirements

- Docker Desktop installed
- Windows / Mac / Linux host supported

No local C++ or GSL installation is required.

---

# Building the Project (Docker-only Workflow)

## 1️ Build the Docker image

From project root:

```bash
docker build -t gw-wl-dev -f docker/Dockerfile .
```

This installs:
- GCC
- CMake
- GNU Scientific Library (GSL)

---

## 2️ Run container with volume mount

Important: use volume mount so outputs persist on host machine.

Linux / Git Bash:
```bash
docker run --rm -it -v ${PWD}:/work gw-wl-dev
```

PowerShell:
```powershell
docker run --rm -it -v ${PWD}:/work gw-wl-dev
```

---

## 3️ Build inside container

Inside the container:

```bash
./scripts/build.sh
```

This generates:

```
build/main_lensing
```

---

## 4️ Run the lensing simulation

Inside the container:

```bash
mkdir -p dataL
./build/main_lensing 0 1 2 0.05
```

Arguments:

| Argument | Meaning |
|----------|----------|
| 0 | CDM cosmology |
| 1 | Lensing enabled |
| 2 | NS catalogue |
| 0.05 | Step size |

---

# Output

The simulation generates files in:

```
dataL/
```

Key output:

- `Plnmuz.dat` → Tabulated PDF of ln(\mu) vs redshift
- `PDL.dat` → Lensed luminosity distance distribution
- Catalogue files (depending on options)

These are produced by Monte Carlo lensing calculations.

---

# What We Have Achieved

- Reproducible Docker-based build  
- Successful compilation of C++ lensing code  
- Monte Carlo lensing PDFs generated  
- Output persisted to host filesystem  

---

# Next Steps (Planned)

Phase 1:
- Isolate Monte Carlo lensing sampler
- Expose raw ln(\mu) samples (not only binned PDFs)
- Add pybind11 bindings
- Create Python streaming interface

Phase 2:
- Implement conditional density estimator (Neural Spline Flow)
- Validate against C++ PDFs

Phase 3:
- Extend to lens-model nuisance parameters
- Extend to WDM/FDM cosmologies

---

# Notes

- `build/` and `dataL/` are intentionally excluded from version control.
- Always run Docker with `-v ${PWD}:/work` to persist outputs.
- First simulation run is slow (table generation).

---

# Current Status

Baseline C++ lensing pipeline is operational and reproducible.

ML integration begins next.
